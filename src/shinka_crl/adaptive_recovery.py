"""Reviewed recovery of the stopped adaptive archive, in a separate working copy.

The original search, evaluator, task and provider remain frozen. This protocol
adds consumed-slot accounting and supervision; it does not repair proposals.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import runpy
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time

from shinka_crl import adaptive_search as search
from shinka_crl.adaptive_evaluation import digest, read_plan, runtime_scope
from shinka_crl.experiment import REPO_ROOT
from shinka_crl.pilot import read_json, require, sha256, write_json
from shinka_crl.reference_timing import utc_now
from shinka_crl.search import artifact_receipts, database_snapshot, validate_saved_state
from shinka_crl.recovery_process import run_supervised

VERSION = "adaptive-shinka-reviewed-recovery-v1"
START, TARGET = 16, 25
MISSING = [15]
RNG_POLICY = "restore-sealed-stale-stage13-RNG-once-before-native-initialization"
SOURCE_FILES = (
    "src/shinka_crl/adaptive_recovery.py", "src/shinka_crl/adaptive_recovery_native.py",
    "src/shinka_crl/recovery_process.py", "scripts/run_adaptive_recovery.py",
    "scripts/report_adaptive_recovery.py", "scripts/recovery_headless.py",
)
TEXT_SUFFIXES = {".json", ".jsonl", ".py", ".yaml", ".yml", ".md", ".log",
                 ".out", ".err", ".txt", ".diff", ".patch"}


def inventory(root: Path) -> dict[str, str]:
    """Hash files, rejecting symlinks; SQLite SHM/WAL are connection sidecars.

    Database logical contents are checked separately and copied with backup().
    Never copy writable hardlinks or follow links out of the source archive.
    """
    require(root.is_dir() and not root.is_symlink(), "Expected a real archive directory")
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), f"Archive symlink is unsupported: {path}")
        if (path.is_file() and "__pycache__" not in path.parts
                and not path.name.endswith((".sqlite-wal", ".sqlite-shm"))):
            result[str(path.relative_to(root))] = sha256(path)
    return result


def database_digest(path: Path) -> str:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        return digest(list(connection.iterdump()))


def ledger_groups(path: Path) -> list[list[dict]]:
    groups = {}
    for line in path.read_text().splitlines():
        event = json.loads(line)
        groups.setdefault(event["request_id"], []).append(event)
    return list(groups.values())


def validate_source(source: Path, *, runtime: bool = False) -> tuple[dict, dict]:
    """Fail closed for this reviewed checkpoint, not arbitrary interrupted runs."""
    original = read_json(source / "plan.json")
    state = read_json(source / "state.json")
    summary = search.summarize(source)
    require(original["protocol"] == "adaptive-shinka-v1"
            and state["status"] == "failed" and summary["slots_consumed"] == START,
            "Recovery requires the reviewed stopped sixteen-slot archive")
    rows = state["programs"]
    require([r["generation"] for r in rows] == list(range(15))
            and all(r["correct"] for r in rows[:14]) and not rows[14]["correct"],
            "Unexpected persisted source slots")
    require([r["generation"] for r in summary["failed_requests"]] == [14]
            and summary["failed_requests"][0]["receipt_verified"]
            and summary["failed_requests"][0]["error"]
            == "ProgramValidationError: AST exceeds 512 nodes"
            and not summary["unpersisted_requests"]
            and summary["incomplete_training_attempts"] == 0,
            "Source failure differs from the reviewed grammar rejection")
    canceled = source / "shinka/gen_15"
    require(sorted(str(p.relative_to(canceled)) for p in canceled.rglob("*") if p.is_file())
            == [".generation_lock"] and (canceled / ".generation_lock").stat().st_size == 0
            and all(p == canceled / "results" for p in canceled.rglob("*") if p.is_dir()),
            "Interrupted slot 15 has unexpected artifacts")
    groups = ledger_groups(source / "model_requests.jsonl")
    require(len(groups) == 15 and all([e["event"] for e in g]
            == ["started", "codex_exec", "finished"] and g[-1]["outcome"] == "success"
            for g in groups[:14]), "Source proposal ledger differs from reviewed requests")
    require([e["event"] for e in groups[-1]] == ["started", "finished"]
            and groups[-1][-1]["outcome"] == "provider_failure"
            and groups[-1][-1]["returncode"] == -15,
            "Interrupted request must precede any CLI launch")
    if runtime:
        evaluation = read_plan(Path(original["evaluation_study"]))
        env = search.search_environment(Path(original["evaluation_study"]), evaluation,
                                        original["model"], original["proposal_timeout_seconds"])
        require(search.make_plan(Path(original["evaluation_study"]), evaluation, env,
                                 original["model"], original["proposal_timeout_seconds"])
                == original, "Original frozen runtime or plan changed")
    return original, summary


def recovery_sources() -> dict:
    return {name: sha256(REPO_ROOT / name) for name in SOURCE_FILES}


def implementation_revision() -> str:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    for name in SOURCE_FILES:
        committed = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=REPO_ROOT)
        require(committed == (REPO_ROOT / name).read_bytes(),
                "Commit the tested recovery implementation before preparing its plan")
    return revision


def wrapper_preflight(original: dict) -> dict:
    study = Path(original["evaluation_study"])
    evaluation = read_plan(study)
    env = search.search_environment(study, evaluation, original["model"],
                                    original["proposal_timeout_seconds"])
    result = json.loads(subprocess.check_output(
        [sys.executable, str(REPO_ROOT / "scripts/recovery_headless.py"), "--check"],
        env=env, text=True, timeout=60))
    require(result == original["runtime"]["subscription_preflight"],
            "Recovery wrapper preflight differs from frozen subscription route")
    return result


def _independent_paths(source: Path, output: Path, study: Path) -> None:
    require(not output.exists(), "Recovery requires a fresh results directory")
    require(not any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
                    for a, b in ((source, output), (study, output))),
            "Keep source archive, recovery copy and evaluation cache separate")


def prepare(*, source: Path, output: Path) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    original, summary = validate_source(source, runtime=True)
    wrapper_check = wrapper_preflight(original)
    study = Path(original["evaluation_study"])
    _independent_paths(source, output, study)
    before, db_before = inventory(source), database_digest(source / "shinka/programs.sqlite")
    plan = {
        "schema_version": 1, "protocol": VERSION, "source_archive": str(source),
        "implementation_revision": implementation_revision(),
        "source_archive_sha256": before, "source_database_logical_sha256": db_before,
        "original_plan_sha256": sha256(source / "plan.json"),
        "original_state_sha256": sha256(source / "state.json"),
        "source_summary_sha256": digest(summary), "source_sha256": recovery_sources(),
        "original_protocol": original["protocol"], "evaluation_study": str(study),
        "wrapper_preflight": wrapper_check,
        "original_consumed_generations": list(range(START)), "no_row_consumed_generations": MISSING,
        "allowed_generations": list(range(START, TARGET)), "target_slots": TARGET,
        "rng_policy": RNG_POLICY, "rng_input_sha256": sha256(source / "shinka/rng_state.json"),
        "rng_deviation": "Saved RNG predates generations 13–15. Restoring it is a declared "
                         "recovery deviation, not reconstruction of uninterrupted sampling.",
        "max_model_requests_per_slot": 1, "max_additional_training_trials": 27,
        "max_additional_nominal_training_steps": 207360000,
        "session_timeout_seconds": 14400,
        "terminal_finalization_grace_seconds": 10,
        "failure_policy": "Stop on any new terminal failure; never retry or backfill consumed slots",
        "execution_policy": "One execution from prepared state; failed/running states require a new review",
        "transport_change": "Exclusive per-slot request reservation wrapping unchanged subscription guard",
        "reporting": "Development recovery; reserved validation and final reporting remain unused",
    }
    output.mkdir(parents=True)
    work = output / "archive"
    shutil.copytree(source, work, ignore=shutil.ignore_patterns("__pycache__", "*.sqlite*"))
    with sqlite3.connect(f"file:{source / 'shinka/programs.sqlite'}?mode=ro", uri=True) as src:
        with sqlite3.connect(work / "shinka/programs.sqlite") as dst:
            src.backup(dst)
    require(database_digest(work / "shinka/programs.sqlite") == db_before,
            "Copied database differs from source")
    require(inventory(source) == before and database_digest(source / "shinka/programs.sqlite")
            == db_before, "Source changed during recovery preparation")
    plan["prepared_archive_sha256"] = inventory(work)
    write_json(output / "plan.json", plan)
    write_json(output / "plan-receipt.json", {"plan_sha256": sha256(output / "plan.json")})
    state = {"schema_version": 1, "status": "prepared", "sessions": [],
             "created_at": utc_now(), "plan_sha256": sha256(output / "plan.json")}
    seal(output, state)
    return state


def read_recovery_plan(output: Path, *, runtime: bool = False) -> dict:
    plan = read_json(output / "plan.json")
    require(read_json(output / "plan-receipt.json") == {"plan_sha256": sha256(output / "plan.json")},
            "Recovery plan receipt changed")
    require(plan["protocol"] == VERSION and plan["source_sha256"] == recovery_sources(),
            "Recovery implementation changed; use its frozen source revision")
    require(plan["allowed_generations"] == list(range(START, TARGET))
            and plan["no_row_consumed_generations"] == MISSING
            and plan["target_slots"] == TARGET and plan["rng_policy"] == RNG_POLICY
            and plan["max_model_requests_per_slot"] == 1
            and plan["max_additional_training_trials"] == 27
            and plan["max_additional_nominal_training_steps"] == 207360000
            and plan["session_timeout_seconds"] == 14400
            and plan["terminal_finalization_grace_seconds"] == 10,
            "Recovery budget or RNG policy changed")
    source = Path(plan["source_archive"])
    require(inventory(source) == plan["source_archive_sha256"]
            and database_digest(source / "shinka/programs.sqlite")
            == plan["source_database_logical_sha256"], "Original archive changed")
    original, summary = validate_source(source, runtime=runtime)
    if runtime:
        require(wrapper_preflight(original) == plan["wrapper_preflight"],
                "Recovery provider preflight changed")
    require(digest(summary) == plan["source_summary_sha256"]
            and sha256(source / "plan.json") == plan["original_plan_sha256"]
            and sha256(source / "state.json") == plan["original_state_sha256"]
            and sha256(source / "shinka/rng_state.json") == plan["rng_input_sha256"]
            and original["evaluation_study"] == plan["evaluation_study"],
            "Recovery source binding changed")
    return plan


def verify_inherited(output: Path, plan: dict) -> None:
    work, source = output / "archive", Path(plan["source_archive"])
    base = read_json(source / "state.json")
    rows = database_snapshot(work / "shinka/programs.sqlite")
    require(rows[:15] == base["programs"], "Inherited native programs changed")
    for name, value in base["artifact_sha256"].items():
        require(sha256(work / name) == value, f"Inherited program artifact changed: {name}")
    for name in ("plan.json", "state.json", "summary.json"):
        require(sha256(work / name) == plan["source_archive_sha256"][name],
                f"Inherited archive record changed: {name}")
    search.validate_task(work)
    require(inventory(work / "preflight") == inventory(source / "preflight"),
            "Inherited scheduler preflight changed")
    require((work / "model_requests.jsonl").read_bytes().startswith(
            (source / "model_requests.jsonl").read_bytes()), "Inherited proposal ledger changed")
    generations = {int(p.name[4:]) for p in (work / "shinka").glob("gen_*") if p.is_dir()}
    require(set(range(START)) <= generations <= set(range(TARGET)), "Recovery slot budget exceeded")


def seal(output: Path, state: dict) -> None:
    work = output / "archive"
    state.update(programs=database_snapshot(work / "shinka/programs.sqlite"),
                 artifact_sha256=artifact_receipts(work), rng_sha256=sha256(work / "shinka/rng_state.json"),
                 model_requests_sha256=sha256(work / "model_requests.jsonl"),
                 recovery_artifacts={name: inventory(work / name) if (work / name).exists() else {}
                                     for name in ("recovery_slots", "recovery_requests")},
                 native_receipt_sha256=sha256(work / "native-recovery.json")
                 if (work / "native-recovery.json").is_file() else None)
    write_json(output / "state.json", state)


def validate_state(output: Path, plan: dict) -> dict:
    state, work = read_json(output / "state.json"), output / "archive"
    require(state["plan_sha256"] == sha256(output / "plan.json"), "State plan binding changed")
    validate_saved_state(work, state)
    verify_inherited(output, plan)
    for name, expected in state["recovery_artifacts"].items():
        require((inventory(work / name) if (work / name).exists() else {}) == expected,
                "Recovery reservation receipts changed")
    receipt = work / "native-recovery.json"
    require((sha256(receipt) if receipt.exists() else None) == state["native_receipt_sha256"],
            "Native recovery receipt changed")
    if state["status"] == "prepared":
        require(inventory(work) == plan["prepared_archive_sha256"], "Prepared copy changed")
    return state


@contextmanager
def execution_lock(output: Path):
    # Lock file is outside the immutable working-copy receipt. Never unlink a
    # lock another process may already have opened.
    with (output / ".controller.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another recovery controller holds this archive") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def stop_reason(work: Path) -> str | None:
    if any(not r["correct"] for r in database_snapshot(work / "shinka/programs.sqlite")
           if r["generation"] >= START):
        return "new terminal evaluation failure"
    for generation in range(START, TARGET):
        if (work / f"shinka/gen_{generation}/failure.json").is_file():
            return "new terminal proposal failure"
    return None


def terminal_monitor(work: Path, grace_seconds: float):
    """Let the synchronous native barrier seal a failure before forced cleanup."""
    detected_at = None
    reason = None

    def check():
        nonlocal detected_at, reason
        current = stop_reason(work)
        if current and detected_at is None:
            detected_at, reason = time.monotonic(), current
        if detected_at is not None and time.monotonic() - detected_at >= grace_seconds:
            return f"{reason}; native finalization grace expired"
        return None

    return check


def summarize(output: Path) -> dict:
    plan = read_recovery_plan(output)
    state = validate_state(output, plan)
    require(state["status"] != "running", "Do not summarize an active recovery controller")
    work = output / "archive"
    # The inherited state stays failed and immutable. Our own seal has just
    # validated the growing copy; legacy summary verifies evaluator receipts.
    summary = search.summarize(work, check_state=False)
    base = search.summarize(Path(plan["source_archive"]))
    slot_records = reservation_records(work / "recovery_slots")
    request_records = reservation_records(work / "recovery_requests")
    reserved = [r["generation"] for r in slot_records]
    requested = [r["generation"] for r in request_records]
    require(len(reserved) == len(set(reserved)) and set(reserved) <= set(plan["allowed_generations"])
            and len(requested) == len(set(requested)) and set(requested) <= set(reserved),
            "Invalid recovery slot/request reservations")
    # A reservation itself consumes a slot even if shutdown precedes mkdir or
    # the guarded provider's first ledger write.
    actual = {int(p.name[4:]) for p in (work / "shinka").glob("gen_*") if p.is_dir()}
    consumed = actual | set(reserved)
    require(actual - set(range(START)) <= set(reserved), "Unreserved native generation")
    verify_native_receipt(work, plan, state, reserved, summary)
    new_groups = ledger_groups(work / "model_requests.jsonl")[15:]
    require(len(new_groups) <= len(requested), "More guarded requests than reserved invocations")
    require(all(sum(e["event"] == "started" for e in g) == 1
                and sum(e["event"] == "codex_exec" for e in g) <= 1
                and sum(e["event"] == "finished" for e in g) <= 1 for g in new_groups),
            "Repeated proposal or CLI launch")
    delta_keys = ("new_training_trials", "new_training_trials_allocated",
                  "new_nominal_training_steps", "new_nominal_training_steps_allocated",
                  "new_scored_training_trials", "training_wall_seconds", "evaluation_wall_seconds")
    added = {key: summary[key] - base[key] for key in delta_keys}
    require(0 <= added["new_training_trials_allocated"] <= 3 * len(reserved)
            and added["new_nominal_training_steps_allocated"] <= plan["max_additional_nominal_training_steps"],
            "Recovery training allocation exceeds reserved slots")
    if state["status"] == "complete":
        require(consumed == set(range(TARGET)) and set(reserved) == set(requested)
                == set(range(START, TARGET)) and len(new_groups) == TARGET - START,
                "Incomplete recovery endpoint")
        require(all(r["receipt_complete"] for r in slot_records + request_records),
                "Incomplete recovery reservation receipt")
        require(all([e["event"] for e in g] == ["started", "codex_exec", "finished"]
                    and g[-1]["outcome"] == "success" for g in new_groups),
                "Incomplete successful proposal ledger")
        require({r["generation"] for r in summary["programs"]}
                == set(range(14)) | set(range(START, TARGET)), "Endpoint lacks valid new evaluations")
        require(not stop_reason(work) and not summary["unpersisted_requests"]
                and summary["incomplete_training_attempts"] == 0, "Unresolved recovery work")
    summary.update(protocol=VERSION, original_protocol=base["protocol"], status=state["status"],
                   original_search_status="failed", slots_consumed=len(consumed),
                   proposal_slots_consumed=len(consumed) - 1, target_slots=TARGET,
                   recovery_reserved_generations=sorted(reserved),
                   recovery_requested_generations=sorted(requested), recovery_added=added,
                   recovery_slot_receipts=slot_records, recovery_request_receipts=request_records,
                   remaining_generations=sorted(set(range(TARGET)) - consumed),
                   inherited_no_row_consumed_generations=MISSING, rng_policy=plan["rng_policy"],
                   rng_deviation=plan["rng_deviation"],
                   sessions=[*base["sessions"], *state["sessions"]],
                   session_wall_seconds=base["session_wall_seconds"]
                   + sum(s.get("wall_seconds", 0.) for s in state["sessions"]))
    return summary


def verify_native_receipt(work: Path, plan: dict, state: dict, reserved: list[int], summary: dict) -> None:
    path = work / "native-recovery.json"
    if not path.exists():
        require(state["status"] != "complete", "Completed recovery lacks native receipt")
        return
    receipt = read_json(path)
    persisted = [r["generation"] for r in database_snapshot(work / "shinka/programs.sqlite")]
    valid = {r["generation"] for r in summary["programs"] if r["generation"] >= START}
    attempted = receipt["attempted_generations"]
    require(receipt["start"] == START and receipt["target"] == TARGET and receipt["missing"] == MISSING
            and attempted == sorted(set(attempted)) and set(attempted) <= set(reserved)
            and receipt["persisted_generations"] == persisted
            and receipt["consumed_generations"] == sorted(set(persisted) | set(MISSING) | set(reserved)),
            "Native recovery receipt slot accounting changed")
    require(receipt["initial_rng_sha256"] == plan["rng_input_sha256"]
            and receipt["final_rng_sha256"] == sha256(work / "shinka/rng_state.json"),
            "Native recovery RNG binding changed")
    require(set(receipt["verified_generations"]) <= valid, "Native receipt claims unverified scores")
    if state["status"] == "complete":
        require(receipt["status"] == "complete" and receipt["graceful_return"] is True
                and receipt["rng_saved"] is True and receipt["terminal_reason"] is None
                and attempted == sorted(reserved)
                and receipt["completed_generations"] == receipt["next_generation_to_submit"] == TARGET
                and receipt["verified_generations"] == list(range(START, TARGET)),
                "Incomplete native recovery completion receipt")


def reservation_records(directory: Path) -> list[dict]:
    records = []
    for path in sorted(directory.glob("*")):
        match = re.fullmatch(r"gen_(\d+)\.json", path.name)
        require(match is not None and path.is_file(), "Unexpected recovery reservation artifact")
        generation = int(match.group(1))
        try:
            record = read_json(path)
        except (json.JSONDecodeError, UnicodeDecodeError):
            record = None  # An exclusive file creation itself consumes the slot.
        require(record is None or record.get("generation") == generation,
                "Recovery reservation generation changed")
        records.append({"generation": generation, "receipt_complete": record is not None,
                        "sha256": sha256(path)})
    return records


def run(output: Path) -> dict:
    output = Path(output).resolve()
    with execution_lock(output):
        plan = read_recovery_plan(output, runtime=True)
        state = validate_state(output, plan)
        require(state["status"] == "prepared" and not state["sessions"],
                "Only a prepared recovery may execute; stopped work requires review")
        work, study = output / "archive", Path(plan["evaluation_study"])
        original = read_json(work / "plan.json")
        evaluation = read_plan(study)
        env = search.search_environment(study, evaluation, original["model"],
                                        original["proposal_timeout_seconds"])
        env.update(SHINKA_SUBSCRIPTION_LEDGER=str(work / "model_requests.jsonl"),
                   SHINKA_RECOVERY_WORK=str(work),
                   SHINKA_HEADLESS_COMMAND=shlex.join([sys.executable, str(REPO_ROOT / "scripts/recovery_headless.py")]))
        command = [sys.executable, "-m", "shinka_crl.adaptive_recovery", "--native", str(output)]
        session = {"index": 1, "target": TARGET, "status": "running", "started_at": utc_now(),
                   "command": command, "controller_pid": os.getpid()}
        state.update(status="running", sessions=[session])
        write_json(output / "state.json", state)
        started = time.monotonic()
        try:
            with runtime_scope(evaluation["cpu_affinity"]):
                result = run_supervised(command, log=output / "session_001.log", env=env,
                                        timeout=plan["session_timeout_seconds"],
                                        stop_check=terminal_monitor(work, plan["terminal_finalization_grace_seconds"]))
            session.update(result)
            require(result["returncode"] == 0 and result["stop_reason"] is None
                    and result["cleanup"]["complete"] is True,
                    f"Recovery stopped: {result}")
            receipt = read_json(work / "native-recovery.json")
            require(receipt["status"] == "complete", "Native recovery did not complete")
            session["status"] = state["status"] = "complete"
        except BaseException as exc:
            session.update(getattr(exc, "supervision_result", {}))
            session.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            state["status"] = "failed"
            raise
        finally:
            session.setdefault("wall_seconds", time.monotonic() - started)
            session["finished_at"] = utc_now()
            seal(output, state)
            # Failed evidence is published as well; a verification failure is
            # explicit rather than silently leaving a stale completed summary.
            try:
                write_json(output / "summary.json", summarize(output))
            except BaseException as exc:
                session.update(status="failed", verification_error=f"{type(exc).__name__}: {exc}")
                state["status"] = "failed"
                seal(output, state)
                raise
        return state


def reserve_provider_request(work: Path, generation: int) -> Path:
    require(START <= generation < TARGET, "Provider generation is outside recovery budget")
    slot = work / "recovery_slots" / f"gen_{generation}.json"
    require(read_json(slot)["generation"] == generation, "Provider lacks a matching slot reservation")
    directory = work / "recovery_requests"
    directory.mkdir(exist_ok=True)
    path = directory / f"gen_{generation}.json"
    with path.open("x") as stream:
        json.dump({"generation": generation, "started_at": utc_now(), "status": "reserved",
                   "ledger_offset_bytes": (work / "model_requests.jsonl").stat().st_size}, stream)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def provider_main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    guard = runpy.run_path(str(REPO_ROOT / "scripts/subscription_headless.py"))
    if argv == ["--check"]:
        # Native Headless probes availability before a generation exists. The
        # unchanged guard's check has no inference or request-ledger side effects.
        return guard["main"](argv)
    work = Path(os.environ["SHINKA_RECOVERY_WORK"])
    require(work.is_absolute(), "Recovery working path must be absolute")
    generation = int(os.environ["SHINKA_RECOVERY_GENERATION"])
    reserve_provider_request(work, generation)
    # Keep the reservation immutable. Completion and interruption are recorded
    # by the unchanged guard's append-only request ledger, after cleanup.
    return guard["main"](argv)


def export(*, output: Path, report: Path) -> dict:
    output, report = Path(output).resolve(), Path(report).resolve()
    with execution_lock(output):
        return _export_locked(output=output, report=report)


def _export_locked(*, output: Path, report: Path) -> dict:
    plan = read_recovery_plan(output)
    require(not report.exists() and not any(report == p or report.is_relative_to(p)
            or p.is_relative_to(report) for p in (output, Path(plan["source_archive"]),
                                                Path(plan["evaluation_study"]))),
            "Use a fresh independent recovery report path")
    summary = summarize(output)
    output_before = inventory(output)
    work, study = output / "archive", Path(plan["evaluation_study"])
    prepared_only = summary["status"] == "prepared"
    if prepared_only:
        # A pre-execution freeze needs bindings and zero-work accounting, not
        # another copy of every already-published development trajectory.
        files = [(output / name, "raw/recovery/" + name)
                 for name in ("plan.json", "plan-receipt.json", "state.json")]
    else:
        files = [(p, "raw/recovery/" + str(p.relative_to(output))) for p in output.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts
                 and not p.name.endswith((".sqlite-wal", ".sqlite-shm"))]
        files += [(study / name, "raw/evaluation/" + name) for name in ("plan.json", "plan-receipt.json")]
        for origin in summary["evidence_cache_origins"]:
            files += [(p, "raw/evaluation/" + str(p.relative_to(study)))
                      for p in (study / origin).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    before = {str(p): sha256(p) for p, _ in files}
    replacements = ((str(work), "$RECOVERY_ARCHIVE"), (str(output), "$RECOVERY"),
                    (plan["source_archive"], "$SOURCE_ARCHIVE"), (str(study), "$EVALUATION_STUDY"),
                    (str(REPO_ROOT), "$REPO_ROOT"))

    def portable(value: str) -> str:
        for old, new in replacements:
            value = value.replace(old, new)
        return value

    report.mkdir(parents=True)
    for path, name in files:
        if path.suffix in TEXT_SUFFIXES:
            target = report / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(portable(path.read_text()))
    write_json(report / "summary.json", json.loads(portable(json.dumps(summary, allow_nan=False))))
    write_json(report / "recovery-plan.json", json.loads(portable(json.dumps(plan, allow_nan=False))))
    require(before == {str(p): sha256(p) for p, _ in files}, "Recovery evidence changed during export")
    require(inventory(output) == output_before, "Recovery file set changed during export")
    write_json(report / "checksums.json", {"original_sha256": {n: sha256(p) for p, n in files},
               "published_sha256": inventory(report),
               "prepared_only": prepared_only,
               "note": "Paths redacted; binary and empty lock evidence retained locally with original hashes. "
                       "Inherited archive state/summary describe the stopped source; top-level summary is current. "
                       "Prepared exports contain the freeze and bindings only; historical results remain "
                       "in the stopped-source publication."})
    return summary


if __name__ == "__main__":
    require(len(sys.argv) == 3 and sys.argv[1] == "--native", "Use scripts/run_adaptive_recovery.py")
    destination = Path(sys.argv[2]).resolve()
    saved_plan = read_recovery_plan(destination)
    saved_state = read_json(destination / "state.json")
    require(saved_state["status"] == "running" and saved_state["sessions"][-1]["controller_pid"]
            == os.getppid(), "Native recovery must be launched by its active controller")
    from shinka_crl.adaptive_recovery_native import native_main

    raise SystemExit(native_main(destination / "archive", target=TARGET, start=START, missing=tuple(MISSING)))
