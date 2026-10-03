"""Evaluate a resource-limited PPO prefix without retraining or rewriting its source.

The original full-budget development attempt remains interrupted. A verified
native whole-state checkpoint is copied to a fresh directory; the unchanged
upstream trainer finalizes it with its loop endpoint equal to the saved step.
This produces a separate, explicitly shorter development experiment.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from shinka_crl.analysis import run_analysis, validate_analysis
from shinka_crl.baseline_contract import validate_baseline_config
from shinka_crl.experiment import (
    UPSTREAM_COMMIT, build_command, load_profile, nominal_training_steps, score_curve,
    validate_profile, verify_upstream,
)
from shinka_crl.pilot import read_json, require, sha256, write_json
from shinka_crl.reference_comparison import (
    THREAD_ENV, _pid_identity, artifact_hashes, command_for, monitor_training, source_hashes,
    utc_now,
)
from shinka_crl.reference_timing import runtime_probe


def prefix_profile(original: dict, endpoint: int) -> dict:
    """Retain complete task cycles and the original per-phase training budgets."""
    validate_profile(original)
    interval = original["ppo"]["task_interval"]
    require(type(endpoint) is int and 0 < endpoint < original["ppo"]["num_updates"]
            and endpoint % interval == 0, "Endpoint must be a proper complete-phase prefix")
    phases = endpoint // interval
    require(phases % original["num_tasks"] == 0, "Prefix must retain complete task cycles")
    result = deepcopy(original)
    result.update(num_phases=phases, purpose="Resource-limited completed prefix of development "
                  "training; not the full-budget reference comparison or reporting trials")
    result["ppo"]["num_updates"] = endpoint
    result["ne"]["num_generations"] = phases * result["ne"]["task_interval"]
    validate_profile(result)
    return result


@contextmanager
def terminal_source_lock(suite: Path):
    """Read-only lock acquisition rejects an active controller or native trainer."""
    with (suite / "controller.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Source controller is still active") from exc
        active = suite / "active-process.json"
        if active.exists():
            row = read_json(active)
            require(_pid_identity(row["pid"]) != row["start_ticks"],
                    "Source native trainer is still active")
        require(read_json(suite / "suite.json")["status"] in {"failed", "interrupted"},
                "Source suite must be terminal and incomplete")
        yield


def inspect_source(suite: Path, endpoint: int) -> dict:
    """Verify the sealed, pinned development attempt before trusting its pickle."""
    suite = Path(suite).resolve()
    plan = read_json(suite / "plan.json")
    require(plan["mode"] == "development" and plan["methods"] == ["ppo"],
            "Only the isolated PPO development calibration may be finalized as a prefix")
    expected = load_profile("paper-cartpole")
    expected.update(seeds=[1001], purpose="Full-budget development runtime calibration; not reporting")
    require(plan["profiles"] == {"paper-cartpole": expected}, "Original development profile differs")
    job = {"profile": "paper-cartpole", "method": "ppo", "seed": 1001,
           "trial": 1002, "eval_seed": 901001}
    require(plan["jobs"] == [job] and plan["eval_episodes"] == 10,
            "Development and reporting identities must remain disjoint")
    require(plan["upstream_commit"] == UPSTREAM_COMMIT
            and plan["source_sha256"] == source_hashes(), "Frozen source changed")
    verify_upstream(Path(plan["upstream"]))
    profile = prefix_profile(expected, endpoint)
    attempts = sorted((suite / "trials/ppo/seed_1001/training").glob("attempt_*"))
    require(bool(attempts), "Missing original training attempt")
    costs, receipts = [], {}
    for attempt in attempts:
        require(not attempt.is_symlink(), "Source attempt must not be a symlink")
        receipt = read_json(attempt / "receipt.json")
        require(receipt == artifact_hashes(attempt), "Original attempt receipt mismatch")
        manifest = read_json(attempt / "manifest.json")
        require(manifest["status"] in {"failed", "interrupted"},
                "Only stopped, incomplete development attempts are eligible")
        require(manifest["profile"] == expected and manifest["method"] == "ppo"
                and manifest["seed"] == 1001 and manifest["trial"] == 1002
                and manifest["ga_settings"] is None
                and manifest["upstream_commit"] == UPSTREAM_COMMIT
                and manifest["command"] == command_for(plan, expected, job, attempt),
                "Original training identity or command differs")
        measurement = read_json(attempt / "process-measurement.json")
        require(measurement["trainer_cpu_affinity"] == plan["cpu_affinity"]
                and measurement["wall_seconds"] == manifest["wall_seconds"] >= 0,
                "Original measured compute differs")
        costs.append({"attempt": str(attempt.relative_to(suite)), **measurement})
        receipts[str(attempt.relative_to(suite))] = receipt
    source = attempts[-1]
    measurement = costs[-1]
    events = [json.loads(line) for line in (source / "phase-events.jsonl").read_text().splitlines()]
    require(events == measurement["phase_events"] and events
            and events[-1]["kind"] == "native_checkpoint_written"
            and events[-1]["completed_updates"] == endpoint,
            "The last recorded native checkpoint must be the requested endpoint")
    previous_update, previous_time = measurement["start_update"], 0
    for event in events:
        require(event["kind"] == "native_checkpoint_written"
                and event["completed_updates"] == previous_update + expected["ppo"]["task_interval"]
                and math.isfinite(event["elapsed_seconds"])
                and previous_time <= event["elapsed_seconds"] <= measurement["wall_seconds"],
                "Native checkpoint journal is not contiguous or has invalid timing")
        previous_update, previous_time = event["completed_updates"], event["elapsed_seconds"]
    require(source.joinpath("resume.pkl").is_file(), "Missing native resume checkpoint")
    return {"suite": suite, "source": source, "plan": plan, "job": job, "profile": profile,
            "checkpoint_sha256": sha256(source / "resume.pkl"), "attempt_costs": costs,
            "attempt_receipts": receipts}


# Executed only by the source run's pinned interpreter after source provenance
# and the copied checkpoint hash have been verified. NumPy/JAX stay out of the
# lightweight reporting environment; the pickle is never an arbitrary input.
PROBE = r'''
import hashlib, json, pickle, sys
from pathlib import Path
import numpy as np

def digest_array(value):
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
            "finite": bool(np.isfinite(array).all())}

if sys.argv[1] == "checkpoint":
    with open(sys.argv[2], "rb") as stream:
        saved = pickle.load(stream)
    records = saved["records"]
    result = {"step": int(saved["step"]), "phase_tasks": saved["phase_tasks"],
              "phase_agents": [digest_array(value) for value in saved["phase_agents"]]}
    result["original_records_sha256"] = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    # The native writer adds figure-facing aliases only when it emits JSON.
    # Apply that exact pinned transform to the in-memory copy, while checking
    # that every originally recorded key and value is retained unchanged.
    originals = [dict(row) for row in records]
    from source.runners.common import add_figure_columns
    add_figure_columns(records, is_rl=True)
    assert all(all(json.dumps(row[key]) == json.dumps(value) for key, value in old.items())
               for old, row in zip(originals, records))
    result["record_serialization"] = "pinned upstream add_figure_columns(is_rl=True); original columns preserved"
else:
    root = Path(sys.argv[2])
    records = json.loads((root / "training_metrics.json").read_text())
    with np.load(root / "checkpoints.npz", allow_pickle=False) as saved:
        result = {"phase_agents": [digest_array(value) for value in saved["final"]],
                  "noise_vectors": saved["noise_vectors"].tolist()}
result["record_count"] = len(records)
result["records_sha256"] = hashlib.sha256(
    json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
print(json.dumps(result, allow_nan=False))
'''


def probe(plan: dict, kind: str, path: Path) -> dict:
    return json.loads(subprocess.check_output(
        [plan["python"], "-c", PROBE, kind, str(path)], cwd=plan["upstream"],
        text=True, timeout=60))


def finalize_checkpoint(*, checkpoint: Path, checkpoint_sha256: str, profile: dict,
                        plan: dict, job: dict, output: Path, timeout: int = 600) -> dict:
    """Finalize a trusted native checkpoint with zero additional PPO updates."""
    require(sha256(checkpoint) == checkpoint_sha256, "Native checkpoint hash mismatch")
    require(not output.exists(), "Finalization requires a fresh output directory")
    output.mkdir(parents=True)
    copied = output / "resume.pkl"
    shutil.copyfile(checkpoint, copied)
    require(sha256(copied) == checkpoint_sha256, "Copied checkpoint differs")
    before = probe(plan, "checkpoint", copied)
    endpoint, phases = profile["ppo"]["num_updates"], profile["num_phases"]
    require(before["step"] == before["record_count"] == endpoint,
            "Native endpoint must equal the saved step; additional training is forbidden")
    require(before["phase_tasks"] == [i % profile["num_tasks"] for i in range(phases)]
            and len(before["phase_agents"]) == phases
            and all(item["finite"] for item in before["phase_agents"]),
            "Checkpoint lacks the exact finite completed phase agents")
    write_json(output / "checkpoint-probe.json", before)
    command = build_command(profile=profile, method="ppo", seed=job["seed"], trial=job["trial"],
                            output_dir=output, upstream=Path(plan["upstream"]),
                            python=plan["python"]) + ["--checkpoint_every", str(profile["ppo"]["task_interval"])]
    manifest = {"status": "running", "profile": profile, "method": "ppo", "seed": job["seed"],
                "trial": job["trial"], "upstream_commit": UPSTREAM_COMMIT, "ga_settings": None,
                "command": command, "started_at_utc": utc_now(),
                "derivation": {"kind": "native_checkpoint_completed_prefix",
                               "source_checkpoint_sha256": checkpoint_sha256,
                               "completed_updates": endpoint, "additional_training_updates": 0}}
    write_json(output / "manifest.json", manifest)
    started = time.monotonic()
    try:
        measured = monitor_training(command=command, cwd=Path(plan["upstream"]), path=output,
                                    profile=profile, method="ppo", timeout=timeout,
                                    active_path=output / "active-process.json", start_update=endpoint)
        write_json(output / "process-measurement.json", measured)
        require(measured["status"] == "complete" and measured["returncode"] == 0
                and measured["phase_events"] == [], "Native zero-update finalization failed")
        after = probe(plan, "artifacts", output)
        require(after["record_count"] == before["record_count"]
                and after["records_sha256"] == before["records_sha256"]
                and after["phase_agents"] == before["phase_agents"],
                "Finalization changed trained policies or recorded history")
        results = read_json(output / "results.json")
        cfg = results["config"]
        validate_baseline_config(cfg, "ppo")
        expected_cfg = {"num_generations": endpoint, "task_interval": profile["ppo"]["task_interval"],
                        "task_sequence": before["phase_tasks"], "seed": job["seed"],
                        "trial": job["trial"], "num_tasks": profile["num_tasks"],
                        "episode_length": profile["episode_length"], "eval_episodes": profile["eval_episodes"],
                        **{key: profile["ppo"][key] for key in ("num_envs", "num_steps", "num_minibatches")}}
        require(all(cfg.get(key) == value for key, value in expected_cfg.items()),
                "Finalized configuration differs from prefix protocol")
        require(results["env_steps"] == cfg["num_timesteps"]
                == nominal_training_steps(profile, "ppo"), "Finalized nominal budget differs")
        require(after["noise_vectors"] == [results["noise_vectors"][task]
                                         for task in before["phase_tasks"]],
                "Finalized checkpoint task vectors differ")
        summary = score_curve(read_json(output / "training_metrics.json"), profile=profile, method="ppo")
        write_json(output / "summary.json", summary)
        manifest.update(status="complete", metrics_sha256=sha256(output / "training_metrics.json"),
                        wall_seconds=measured["wall_seconds"])
        require(sha256(checkpoint) == checkpoint_sha256, "Original checkpoint changed")
        return measured
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest.setdefault("wall_seconds", time.monotonic() - started)
        manifest["finished_at_utc"] = utc_now()
        write_json(output / "manifest.json", manifest)
        write_json(output / "receipt.json", artifact_hashes(output))


def finalize_prefix(*, suite: Path, output: Path, endpoint: int = 6000,
                    allocation_seconds: int = 28800, prior_auxiliary_seconds: float = 0) -> dict:
    """Create and analyze a separate development prefix under the eight-hour cap."""
    require(type(allocation_seconds) is int and 0 < allocation_seconds <= 28800,
            "PPO allocation must not exceed the authorized eight hours")
    require(type(prior_auxiliary_seconds) in (int, float)
            and math.isfinite(prior_auxiliary_seconds) and prior_auxiliary_seconds >= 0,
            "Prior auxiliary compute must be finite nonnegative seconds")
    suite, output = Path(suite).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(suite),
            "Use a fresh output outside the original suite")
    with terminal_source_lock(suite):
        source = inspect_source(suite, endpoint)
        plan, job, profile = (source[key] for key in ("plan", "job", "profile"))
        training_spent = sum(row["wall_seconds"] for row in source["attempt_costs"])
        spent = training_spent + prior_auxiliary_seconds
        require(spent < allocation_seconds - 120, "Insufficient PPO allocation for finalization and evaluation")
        output.mkdir(parents=True)
        adapter_started = time.monotonic()
        provenance = {"kind": "resource_limited_development_prefix", "status": "running",
                      "original_suite": str(suite), "original_attempt": str(source["source"].relative_to(suite)),
                      "original_plan_sha256": sha256(suite / "plan.json"),
                      "original_attempt_receipts": source["attempt_receipts"],
                      "source_checkpoint_sha256": source["checkpoint_sha256"],
                      "profile": profile, "job": job, "upstream_commit": UPSTREAM_COMMIT,
                      "source_sha256": source_hashes(), "adapter_sha256": sha256(Path(__file__)),
                      "allocation_seconds": allocation_seconds, "original_attempt_costs": source["attempt_costs"],
                      "original_attempt_total_wall_seconds": training_spent,
                      "prior_auxiliary_seconds": prior_auxiliary_seconds,
                      "training_budget_deviation": f"{profile['num_phases']} complete phases retained from "
                      "the declared twenty-phase development trial; unchanged per-phase budget and baseline settings. "
                      "Not a completed full-budget or reporting comparison.",
                      "started_at_utc": utc_now()}
        write_json(output / "provenance.json", provenance)
        previous_affinity = os.sched_getaffinity(0)
        previous_env = {key: os.environ.get(key) for key in THREAD_ENV}
        try:
            os.sched_setaffinity(0, plan["cpu_affinity"])
            os.environ.update(THREAD_ENV)
            runtime = runtime_probe(plan["python"], dict(os.environ))
            runtime.update(upstream_commit=UPSTREAM_COMMIT, source_sha256=plan["source_sha256"])
            require(runtime == read_json(suite / "environment.json"), "Runtime differs from original calibration")
            write_json(output / "environment.json", runtime)
            def remaining():
                return int(allocation_seconds - spent - (time.monotonic() - adapter_started))
            measured = finalize_checkpoint(checkpoint=source["source"] / "resume.pkl",
                                           checkpoint_sha256=source["checkpoint_sha256"],
                                           profile=profile, plan=plan, job=job,
                                           output=output / "training", timeout=min(600, remaining()))
            require(remaining() > 0, "PPO allocation exhausted before evaluation")
            run_analysis(run_dir=output / "training", output_dir=output / "analysis",
                         eval_seed=job["eval_seed"], upstream=Path(plan["upstream"]),
                         python=plan["python"], episodes=plan["eval_episodes"],
                         timeout=min(plan["analysis_timeout_seconds"], remaining()))
            analyzed = validate_analysis(run_dir=output / "training", output_dir=output / "analysis",
                                         episodes=plan["eval_episodes"], eval_seed=job["eval_seed"])
            require(source_hashes() == plan["source_sha256"], "Frozen sources changed during analysis")
            verify_upstream(Path(plan["upstream"]))
            for relative, receipt in source["attempt_receipts"].items():
                require(artifact_hashes(suite / relative) == receipt, "Original attempt changed")
            summary = {"status": "complete", "kind": provenance["kind"], "reporting_trials": 0,
                       "completed_prefix_phases": profile["num_phases"], "original_planned_phases": 20,
                       "completed_prefix_updates": endpoint, "original_planned_updates": 30000,
                       "nominal_training_steps": nominal_training_steps(profile, "ppo"),
                       "additional_training_updates": 0, "analysis": analyzed,
                       "compute": {"original_attempt_total_wall_seconds": training_spent,
                                   "prior_auxiliary_seconds": prior_auxiliary_seconds,
                                   "finalizer_wall_seconds": measured["wall_seconds"],
                                   "finalizer_maximum_rss_kib": measured["maximum_trainer_rss_kib"],
                                   "analysis_wall_seconds": read_json(output / "analysis/manifest.json")["wall_seconds"],
                                   "original_training_maximum_rss_kib": max(
                                       row["maximum_trainer_rss_kib"] for row in source["attempt_costs"]),
                                   "total_accounted_wall_seconds": spent + time.monotonic() - adapter_started,
                                   "timing_basis": "measured monotonic process duration; original unsuccessful "
                                   "and beyond-checkpoint work retained in compute accounting"}}
            require(summary["compute"]["total_accounted_wall_seconds"] <= allocation_seconds,
                    "Measured PPO allocation exceeded")
            write_json(output / "summary.json", summary)
            provenance["status"] = "complete"
            return summary
        except BaseException as exc:
            provenance.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            provenance.update(finished_at_utc=utc_now(), adapter_wall_seconds=time.monotonic() - adapter_started)
            write_json(output / "provenance.json", provenance)
            write_json(output / "receipt.json", artifact_hashes(output))
            os.sched_setaffinity(0, previous_affinity)
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
