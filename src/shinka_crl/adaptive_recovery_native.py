"""Runtime-only adapter for the declared failed adaptive-search checkpoint.

The native sampler, prompts, patches, evaluator and database are unchanged.
Only slot accounting and a serial, fail-closed completion barrier are adapted.
The caller must validate the frozen recovery plan before invoking this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib
import importlib.metadata
import math
import os
from pathlib import Path
import random
from typing import Any

from .pilot import read_json, require, sha256, write_json
from .reference_timing import artifact_hashes
from .search import database_snapshot


def _reserve(work: Path, generation: int) -> None:
    """Durably consume the slot before native proposal code can execute."""
    import json

    directory = work / "recovery_slots"
    directory.mkdir(exist_ok=True)
    path = directory / f"gen_{generation}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"generation": generation,
                   "started_at": datetime.now(timezone.utc).isoformat()}, stream)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _runner_type(base: type, work: Path, start: int, missing: tuple[int, ...],
                 instances: list[Any]) -> type:
    class RecoveryRunner(base):
        def __init__(self, **kwargs):
            require(kwargs["db_config"].num_islands == 1, "Recovery requires one island")
            for name in ("max_evaluation_jobs", "max_proposal_jobs", "max_db_workers"):
                require(kwargs.get(name) == 1, f"Recovery requires {name}=1")
            self.recovery_failure: str | None = None
            self.recovery_attempted: list[int] = []
            self.recovery_verified: set[int] = set()
            super().__init__(**kwargs)
            instances.append(self)

        def _fail(self, reason: str) -> None:
            if self.recovery_failure is None:
                self.recovery_failure = reason
            self.should_stop.set()
            self.slot_available.set()
            self.finalization_complete.set()

        async def _persisted(self) -> set[int]:
            ids = await self.async_db.get_persisted_generation_ids_async()
            count = await self.async_db.get_total_program_count_async()
            require(len(set(ids)) == count, "Duplicate native generation rows")
            require(all(0 <= value < self.evo_config.num_generations for value in ids),
                    "Native generation outside the declared budget")
            require(not set(ids).intersection(missing), "Consumed no-row slot acquired a row")
            return set(ids)

        async def _count_completed_generations_from_db(self) -> int:
            return len((await self._persisted()).union(missing))

        async def _get_missing_persisted_generations(self) -> list[int]:
            return sorted(set(range(self.evo_config.num_generations))
                          - (await self._persisted()).union(missing))

        async def _restore_resume_progress(self) -> None:
            require(await self._persisted() == set(range(start)) - set(missing),
                    "Recovery database does not match the declared starting checkpoint")
            self.completed_generations = start
            self.next_generation_to_submit = start
            self.assigned_generations.update(range(start))

        async def _update_completed_generations_fallback(self, running_generations):
            # The native fallback counts rows and would silently lose consumed slot 15.
            self._fail("Native completion accounting failed; row-count fallback is disabled")

        async def _verify_generation(self, generation: int) -> None:
            rows = await self.async_db.get_programs_by_generation_async(generation)
            require(len(rows) == 1 and rows[0].correct == 1,
                    f"Generation {generation} has no single correct persisted result")
            row = rows[0]
            root = work / "shinka" / f"gen_{generation}"
            result = root / "results"
            evidence = result / "evaluation"
            require(read_json(evidence / "receipt.json") == artifact_hashes(evidence),
                    f"Generation {generation} evaluation receipt mismatch")
            request = read_json(evidence / "request.json")
            require(request["status"] == "complete" and request["slot_consumed"] is True,
                    f"Generation {generation} evaluation request is incomplete")
            require(request["evaluator_model_calls"] == 0,
                    "Evaluator unexpectedly reports model calls")
            require(row.code == (root / "main.py").read_text(), "Native program changed")
            require(sha256(root / "main.py") == sha256(evidence / "program.py")
                    == request["candidate"]["source_sha256"], "Evaluated source mismatch")
            metrics = read_json(evidence / "metrics.json")
            require(read_json(evidence / "correct.json")["correct"] is True,
                    "Sealed evaluator result is incorrect")
            require(read_json(result / "metrics.json") == metrics,
                    "Native metrics differ from sealed evaluator metrics")
            require(read_json(result / "correct.json") == read_json(evidence / "correct.json"),
                    "Native correctness differs from sealed evaluator result")
            require(math.isfinite(row.combined_score)
                    and row.combined_score == metrics["combined_score"]
                    == request["aggregate"]["scores"]["combined_score"]["mean"],
                    "Native score differs from sealed evaluator score")
            require(row.public_metrics == metrics["public"]
                    and row.private_metrics == metrics["private"], "Native metrics changed")
            self.recovery_verified.add(generation)

        async def _start_proposals(self, count: int) -> None:
            # The native monitor removes completed jobs before persisting their rows.
            # Its processing lock closes that gap; no external polling is involved.
            async with self.processing_lock:
                if self.should_stop.is_set() or self.recovery_failure or count <= 0:
                    return
                if (self.running_jobs or self.active_proposal_tasks
                        or self.failed_jobs_for_retry
                        or getattr(self, "_completed_jobs_pending", 0)):
                    return
                generation = self.next_generation_to_submit
                if generation >= self.evo_config.num_generations:
                    return
                try:
                    require(generation >= start, "Attempt to reuse a historical generation")
                    if generation > start:
                        await self._verify_generation(generation - 1)
                    require(not (work / "shinka" / f"gen_{generation}").exists(),
                            "Attempt to reuse an existing generation directory")
                    _reserve(work, generation)
                    self.recovery_attempted.append(generation)
                    await super()._start_proposals(1)
                except Exception as exc:
                    self._fail(f"Proposal barrier: {type(exc).__name__}: {exc}")

        async def _generate_proposal_async(self, generation: int, task_id: str):
            prior = os.environ.get("SHINKA_RECOVERY_GENERATION")
            os.environ["SHINKA_RECOVERY_GENERATION"] = str(generation)
            try:
                result = await super()._generate_proposal_async(generation, task_id)
                if result is None and not self.recovery_failure:
                    self._fail(f"Generation {generation} produced no evaluation job")
                return result
            finally:
                if prior is None:
                    os.environ.pop("SHINKA_RECOVERY_GENERATION", None)
                else:
                    os.environ["SHINKA_RECOVERY_GENERATION"] = prior

        async def _record_terminal_failed_proposal(self, **kwargs):
            try:
                await super()._record_terminal_failed_proposal(**kwargs)
            finally:
                self._fail(f"Generation {kwargs['generation']} terminal proposal failure: "
                           f"{kwargs.get('failure_reason', 'unknown')}")

        async def _process_completed_jobs_safely(self, jobs):
            await super()._process_completed_jobs_safely(jobs)
            for job in jobs:
                if job.generation >= start:
                    try:
                        await self._verify_generation(job.generation)
                    except Exception as exc:
                        self._fail(f"Generation {job.generation} completion barrier: "
                                   f"{type(exc).__name__}: {exc}")

        async def _cleanup_async(self):
            # Native scheduler shutdown closes its executor but does not kill jobs.
            for job in list(self.running_jobs):
                await self.scheduler.cancel_job_async(job.job_id)
            await super()._cleanup_async()

    return RecoveryRunner


def native_main(work: Path, target: int, start: int = 16,
                missing: tuple[int, ...] = (15,)) -> int:
    """Run a plan-validated working copy; return nonzero for every incomplete stop.

    A graceful terminal failure saves the current host RNG, clearly marked in the
    receipt. An abrupt process termination cannot produce that receipt; durable
    reservations still prevent reuse of the attempted slots.
    """
    work = work.resolve()
    require(0 < start < target and len(set(missing)) == len(missing)
            and all(0 <= value < start for value in missing), "Invalid recovery slot range")
    receipt_path = work / "native-recovery.json"
    require(not receipt_path.exists(), "Native recovery receipt already exists")
    db_path = work / "shinka/programs.sqlite"
    initial = database_snapshot(db_path)
    require({row["generation"] for row in initial} == set(range(start)) - set(missing)
            and len(initial) == start - len(missing), "Unexpected starting native rows")
    rng_path = work / "shinka/rng_state.json"
    initial_rng = sha256(rng_path)
    root = Path(importlib.metadata.distribution("shinka-evolve").locate_file("shinka"))
    require(not any(path.exists() for path in (root.parent / ".env", Path.cwd() / ".env")),
            "Shinka dotenv override appeared after preparation")
    import numpy as np

    cli = importlib.import_module("shinka.cli.run")
    saved = read_json(rng_path)
    state = saved["python"]
    random.setstate((state[0], tuple(state[1]), state[2]))
    state = saved["numpy"]
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    original = cli.ShinkaEvolveRunner
    instances: list[Any] = []
    cli.ShinkaEvolveRunner = _runner_type(original, work, start, missing, instances)
    graceful = False
    rng_saved = False
    error = None
    result = 1
    try:
        result = cli.main(["--task-dir", str(work / "task"),
                           "--config-fname", "shinka-subscription.yaml",
                           "--results_dir", str(work / "shinka"),
                           "--num_generations", str(target)])
        graceful = True
        state = np.random.get_state()
        write_json(rng_path, {"python": random.getstate(),
                              "numpy": [state[0], state[1].tolist(), *state[2:]]})
        rng_saved = True
    except (Exception, SystemExit) as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        cli.ShinkaEvolveRunner = original
    runner = instances[0] if instances else None
    rows = database_snapshot(db_path)
    ids = [row["generation"] for row in rows]
    attempted = runner.recovery_attempted if runner else []
    # Include earlier durable reservations even when this invocation refused
    # their reuse before scheduling a proposal.
    reserved = [int(path.stem.removeprefix("gen_"))
                for path in (work / "recovery_slots").glob("gen_*.json")
                if path.stem.removeprefix("gen_").isdigit()]
    consumed = sorted(set(ids).union(missing, attempted, reserved))
    historical = [row for row in rows if row["generation"] < start]
    failure = error or (runner.recovery_failure if runner else "Native runner not created")
    if historical != initial:
        failure = "Historical native program rows changed"
    complete = (result == 0 and graceful and failure is None and runner is not None
                and runner.completed_generations == target
                and set(ids).union(missing) == set(range(target))
                and runner.recovery_verified == set(range(start, target)))
    receipt = {"schema_version": 1, "status": "complete" if complete else "failed",
               "start": start, "target": target, "missing": list(missing),
               "attempted_generations": attempted, "persisted_generations": ids,
               "consumed_generations": consumed,
               "completed_generations": runner.completed_generations if runner else None,
               "next_generation_to_submit": runner.next_generation_to_submit if runner else start,
               "verified_generations": sorted(runner.recovery_verified) if runner else [],
               "terminal_reason": failure or (None if complete else "Native stopped before target"),
               "graceful_return": graceful, "rng_saved": rng_saved,
               "initial_rng_sha256": initial_rng, "final_rng_sha256": sha256(rng_path)}
    write_json(receipt_path, receipt)
    return 0 if complete else 1
