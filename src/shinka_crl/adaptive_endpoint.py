"""Explicitly reviewed attempts for the unused slots after generation 20.

Every attempt requires a fresh independent copy and published freeze. A new
terminal failure remains terminal; a later prepare must independently validate
that exact checkpoint. Allocation exhaustion is distinct from controller success.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time

from shinka_crl import adaptive_continuation as previous
from shinka_crl import adaptive_recovery as recovery
from shinka_crl import adaptive_search as search
from shinka_crl.adaptive_evaluation import digest, read_plan, runtime_scope
from shinka_crl.experiment import REPO_ROOT
from shinka_crl.pilot import read_json, require, sha256, write_json
from shinka_crl.reference_timing import utc_now
from shinka_crl.recovery_process import run_supervised
from shinka_crl.search import database_snapshot, validate_saved_state

VERSION = "adaptive-shinka-reviewed-endpoint-v3"
FIRST, TARGET = 21, 25
MISSING = [15]
RNG_POLICY = "restore-reviewed-gracefully-saved-RNG-once-before-native-initialization"
SOURCE_FILES = (*previous.SOURCE_FILES,
                "src/shinka_crl/adaptive_endpoint.py",
                "scripts/run_adaptive_endpoint.py",
                "scripts/report_adaptive_endpoint.py", "scripts/endpoint_headless.py")
HISTORY_NAMES = ("native-recovery.json", "recovery_slots", "recovery_requests")
DELTA_KEYS = ("new_training_trials", "new_training_trials_allocated",
              "new_nominal_training_steps", "new_nominal_training_steps_allocated",
              "new_scored_training_trials", "training_wall_seconds", "evaluation_wall_seconds")
inventory = previous.inventory
database_digest = previous.database_digest
ledger_groups = previous.ledger_groups
reservation_records = previous.reservation_records
execution_lock = previous.execution_lock


def source_ancestors(source: Path) -> list[Path]:
    """Bound source traversal and reject cyclic/nondecreasing endpoint chains."""
    ancestors = []
    upper = TARGET
    current = source.resolve()
    while True:
        require(current not in ancestors, "Cyclic endpoint source chain")
        ancestors.append(current)
        plan = read_json(current / "plan.json")
        if plan["protocol"] == VERSION:
            start = plan["start_generation"]
            require(type(start) is int and FIRST <= start < upper,
                    "Endpoint source chain must have strictly earlier starts")
            upper = start
        elif plan["protocol"] not in (previous.VERSION, recovery.VERSION, "adaptive-shinka-v1"):
            raise ValueError("Unknown endpoint source protocol")
        if "source_archive" not in plan:
            return ancestors
        current = Path(plan["source_archive"]).resolve()


def source_summary(source: Path, *, runtime: bool = False) -> tuple[dict, dict]:
    source_ancestors(source)
    protocol = read_json(source / "plan.json")["protocol"]
    if protocol == previous.VERSION:
        plan = previous.read_continuation_plan(source, runtime=runtime)
        return plan, previous.summarize(source)
    require(protocol == VERSION, "Endpoint requires a reviewed continuation or endpoint source")
    plan = read_endpoint_plan(source, runtime=runtime)
    return plan, summarize(source)


def verify_grammar_stop(output: Path, plan: dict, state: dict, summary: dict,
                        reserved: list[int], requested: list[int]) -> None:
    """A fully sealed grammar-only native stop may be independently reviewed."""
    work = output / "archive"
    start = plan.get("start_generation", previous.START)
    end = summary["slots_consumed"]
    rows = state["programs"]
    require(state["status"] == "failed" and start < end <= TARGET
            and [r["generation"] for r in rows] == [g for g in range(end) if g not in MISSING]
            and all(bool(r["correct"]) == (r["generation"] != end - 1)
                    for r in rows if r["generation"] >= start),
            "Source must end at one persisted grammar rejection with contiguous consumed slots")
    require(reserved == requested == list(range(start, end))
            and summary["remaining_generations"] == list(range(end, TARGET))
            and not summary["terminal_proposal_failures"]
            and not summary["unpersisted_requests"]
            and summary["incomplete_training_attempts"] == 0,
            "Source has unresolved work or unexpected consumed slots")
    require(all(r["receipt_complete"] for directory in ("recovery_slots", "recovery_requests")
                for r in reservation_records(work / directory)), "Source reservation receipt is incomplete")
    failures = summary["failed_requests"]
    require([r["generation"] for r in failures] == [r["generation"] for r in rows if not r["correct"]]
            and all(r["receipt_verified"] and r["status"] == "failed"
                    and r["error"] == "ProgramValidationError: AST exceeds 512 nodes"
                    and not r.get("cache_origin") for r in failures),
            "Source failures differ from reviewed grammar rejections")
    receipt = read_json(work / "native-recovery.json")
    require(receipt["status"] == "failed" and receipt["start"] == start
            and receipt["target"] == TARGET and receipt["missing"] == MISSING
            and receipt["attempted_generations"] == list(range(start, end))
            and receipt["verified_generations"] == list(range(start, end - 1))
            and receipt["persisted_generations"] == [r["generation"] for r in rows]
            and receipt["consumed_generations"] == list(range(end))
            and receipt["completed_generations"] == receipt["next_generation_to_submit"] == end
            and receipt["graceful_return"] is True and receipt["rng_saved"] is True
            and receipt["terminal_reason"]
            and receipt["initial_rng_sha256"] == plan["rng_input_sha256"]
            and receipt["final_rng_sha256"] == sha256(work / "shinka/rng_state.json"),
            "Source lacks fresh RNG and complete native finalization")
    require(len(state["sessions"]) == 1 and state["sessions"][0]["status"] == "failed"
            and state["sessions"][0]["returncode"] == 1
            and state["sessions"][0]["stop_reason"] is None
            and state["sessions"][0]["cleanup"]["complete"] is True
            and not state["sessions"][0]["cleanup"].get("survivors")
            and not state["sessions"][0]["cleanup"].get("errors")
            and not state["sessions"][0].get("verification_error"),
            "Source session did not stop gracefully with complete cleanup")
    groups = ledger_groups(work / "model_requests.jsonl")
    require(len(groups) == end - 1 and all([e["event"] for e in group]
            == ["started", "codex_exec", "finished"] and group[-1]["outcome"] == "success"
            for group in groups[plan["inherited_model_request_groups"]:]),
            "Source proposal ledger changed")


def validate_source(source: Path, *, runtime: bool = False) -> tuple[dict, dict]:
    prior_plan, summary = source_summary(source, runtime=runtime)
    state = read_json(source / "state.json")
    prefix = "continuation" if prior_plan["protocol"] == previous.VERSION else "endpoint"
    verify_grammar_stop(source, prior_plan, state, summary,
                        summary[prefix + "_reserved_generations"],
                        summary[prefix + "_requested_generations"])
    end = summary["slots_consumed"]
    require(FIRST <= end < TARGET, "No unused slots remain for an endpoint attempt")
    if prior_plan["protocol"] == previous.VERSION:
        require(end == FIRST and summary["programs_evaluated"] == 17
                and [r["generation"] for r in summary["failed_requests"]] == [14, 17, 20],
                "Endpoint requires the reviewed generation-20 continuation checkpoint")
    return read_json(source / "archive/plan.json"), summary


def endpoint_sources() -> dict:
    return {name: sha256(REPO_ROOT / name) for name in SOURCE_FILES}


def implementation_revision() -> str:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    for name in SOURCE_FILES:
        committed = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=REPO_ROOT)
        require(committed == (REPO_ROOT / name).read_bytes(),
                "Commit the tested endpoint implementation before preparing its plan")
    return revision


def wrapper_preflight(original: dict) -> dict:
    study = Path(original["evaluation_study"])
    evaluation = read_plan(study)
    env = search.search_environment(study, evaluation, original["model"],
                                    original["proposal_timeout_seconds"])
    result = json.loads(subprocess.check_output(
        [sys.executable, str(REPO_ROOT / "scripts/endpoint_headless.py"), "--check"],
        env=env, text=True, timeout=60))
    require(result == original["runtime"]["subscription_preflight"],
            "Endpoint wrapper preflight differs from frozen subscription route")
    return result


def prepare(*, source: Path, output: Path) -> dict:
    source, output = Path(source).resolve(), Path(output).resolve()
    original, summary = validate_source(source, runtime=True)
    start = summary["slots_consumed"]
    wrapper_check = wrapper_preflight(original)
    study, source_work = Path(original["evaluation_study"]), source / "archive"
    for ancestor in source_ancestors(source):
        recovery._independent_paths(ancestor, output, study)
    before = inventory(source)
    db_before = database_digest(source_work / "shinka/programs.sqlite")
    plan = {
        "schema_version": 1, "protocol": VERSION, "source_archive": str(source),
        "implementation_revision": implementation_revision(),
        "source_archive_sha256": before, "source_database_logical_sha256": db_before,
        "source_plan_sha256": sha256(source / "plan.json"),
        "source_state_sha256": sha256(source / "state.json"),
        "source_native_receipt_sha256": sha256(source_work / "native-recovery.json"),
        "source_summary_sha256": digest(summary), "source_sha256": endpoint_sources(),
        "original_protocol": original["protocol"], "previous_protocol": summary["protocol"],
        "start_generation": start,
        "evaluation_study": str(study), "wrapper_preflight": wrapper_check,
        "original_consumed_generations": list(range(start)), "no_row_consumed_generations": MISSING,
        "allowed_generations": list(range(start, TARGET)), "target_slots": TARGET,
        "inherited_valid_generations": [r["generation"] for r in summary["programs"]],
        "inherited_invalid_generations": [r["generation"] for r in summary["failed_requests"]],
        "inherited_model_request_groups": len(ledger_groups(source_work / "model_requests.jsonl")),
        "rng_policy": RNG_POLICY, "rng_input_sha256": sha256(source_work / "shinka/rng_state.json"),
        "rng_deviation": "Restore the independently reviewed graceful checkpoint once before native "
                         "initialization; initialization may consume RNG draws. The inherited stage-13 "
                         "RNG rollback remains a deviation from uninterrupted sampling.",
        "max_model_requests_per_slot": 1, "max_additional_training_trials": (TARGET - start) * 3,
        "max_additional_nominal_training_steps": (TARGET - start) * 23040000,
        "session_timeout_seconds": 14400, "terminal_finalization_grace_seconds": 10,
        "failure_policy": "Stop on any new terminal failure; never retry or backfill consumed slots",
        "execution_policy": "One execution from prepared state; failed/running states require a new review",
        "history_policy": "Prior controller records copied to history/controller; prior native receipt and "
                          "reservation directories moved from the copied archive to history/archive",
        "reporting": "Development endpoint; reserved validation and final reporting remain unused",
    }
    output.mkdir(parents=True)
    work = output / "archive"
    shutil.copytree(source_work, work, ignore=shutil.ignore_patterns("__pycache__", "*.sqlite*"))
    with sqlite3.connect(f"file:{source_work / 'shinka/programs.sqlite'}?mode=ro", uri=True) as src:
        with sqlite3.connect(work / "shinka/programs.sqlite") as dst:
            src.backup(dst)
    require(database_digest(work / "shinka/programs.sqlite") == db_before,
            "Copied database differs from source")
    def controller_ignore(directory, names):
        omitted = {"__pycache__", ".controller.lock"}
        if Path(directory) == source:
            omitted.add("archive")
        return set(names) & omitted

    # Only the source's active archive is copied separately. In particular,
    # history/archive in every previous attempt must survive recursively.
    shutil.copytree(source, output / "history/controller", ignore=controller_ignore)
    history = output / "history/archive"
    history.mkdir()
    for name in HISTORY_NAMES:
        shutil.move(str(work / name), history / name)
    require(inventory(source) == before
            and database_digest(source_work / "shinka/programs.sqlite") == db_before,
            "Source changed during endpoint preparation")
    plan["prepared_archive_sha256"] = inventory(work)
    plan["history_sha256"] = inventory(output / "history")
    write_json(output / "plan.json", plan)
    write_json(output / "plan-receipt.json", {"plan_sha256": sha256(output / "plan.json")})
    state = {"schema_version": 1, "status": "prepared", "sessions": [],
             "created_at": utc_now(), "plan_sha256": sha256(output / "plan.json")}
    seal(output, state)
    return state


def read_endpoint_plan(output: Path, *, runtime: bool = False) -> dict:
    plan = read_json(output / "plan.json")
    start = plan["start_generation"]
    require(type(start) is int and FIRST <= start < TARGET, "Invalid endpoint start")
    source_ancestors(output)
    require(read_json(output / "plan-receipt.json") == {"plan_sha256": sha256(output / "plan.json")},
            "Endpoint plan receipt changed")
    require(plan["protocol"] == VERSION and plan["source_sha256"] == endpoint_sources(),
            "Endpoint implementation changed; use its frozen source revision")
    require(plan["allowed_generations"] == list(range(start, TARGET))
            and plan["original_consumed_generations"] == list(range(start))
            and plan["no_row_consumed_generations"] == MISSING and plan["target_slots"] == TARGET
            and plan["rng_policy"] == RNG_POLICY
            and plan["max_model_requests_per_slot"] == 1
            and plan["max_additional_training_trials"] == (TARGET - start) * 3
            and plan["max_additional_nominal_training_steps"] == (TARGET - start) * 23040000
            and plan["session_timeout_seconds"] == 14400
            and plan["terminal_finalization_grace_seconds"] == 10,
            "Endpoint budget or RNG policy changed")
    source = Path(plan["source_archive"])
    source_work = source / "archive"
    require(inventory(source) == plan["source_archive_sha256"]
            and database_digest(source_work / "shinka/programs.sqlite")
            == plan["source_database_logical_sha256"], "Stopped recovery archive changed")
    original, summary = validate_source(source, runtime=runtime)
    require(start == summary["slots_consumed"]
            and plan["previous_protocol"] == summary["protocol"]
            and plan["inherited_valid_generations"] == [r["generation"] for r in summary["programs"]]
            and plan["inherited_invalid_generations"] == [r["generation"] for r in summary["failed_requests"]]
            and plan["inherited_model_request_groups"]
            == len(ledger_groups(source_work / "model_requests.jsonl")), "Inherited slot accounting changed")
    if runtime:
        require(wrapper_preflight(original) == plan["wrapper_preflight"],
                "Endpoint provider preflight changed")
    require(digest(summary) == plan["source_summary_sha256"]
            and sha256(source / "plan.json") == plan["source_plan_sha256"]
            and sha256(source / "state.json") == plan["source_state_sha256"]
            and sha256(source_work / "native-recovery.json") == plan["source_native_receipt_sha256"]
            and sha256(source_work / "shinka/rng_state.json") == plan["rng_input_sha256"]
            and original["evaluation_study"] == plan["evaluation_study"],
            "Endpoint source binding changed")
    return plan


def verify_inherited(output: Path, plan: dict) -> None:
    start = plan["start_generation"]
    work, source = output / "archive", Path(plan["source_archive"])
    source_work = source / "archive"
    base = read_json(source / "state.json")
    rows = database_snapshot(work / "shinka/programs.sqlite")
    require(rows[:len(base["programs"])] == base["programs"], "Inherited native programs changed")
    for name, value in base["artifact_sha256"].items():
        require(sha256(work / name) == value, f"Inherited program artifact changed: {name}")
    for generation in range(start):
        prefix = f"archive/shinka/gen_{generation}/"
        expected = {name.removeprefix(prefix): value
                    for name, value in plan["source_archive_sha256"].items() if name.startswith(prefix)}
        require(inventory(work / f"shinka/gen_{generation}") == expected,
                f"Inherited generation file set changed: {generation}")
    for name in ("plan.json", "state.json", "summary.json"):
        require(sha256(work / name) == plan["source_archive_sha256"]["archive/" + name],
                f"Inherited archive record changed: {name}")
    require(inventory(output / "history") == plan["history_sha256"], "Historical recovery evidence changed")
    search.validate_task(work)
    require(inventory(work / "preflight") == inventory(source_work / "preflight"),
            "Inherited scheduler preflight changed")
    require((work / "model_requests.jsonl").read_bytes().startswith(
            (source_work / "model_requests.jsonl").read_bytes()), "Inherited proposal ledger changed")
    generations = {int(p.name[4:]) for p in (work / "shinka").glob("gen_*") if p.is_dir()}
    require(set(range(start)) <= generations <= set(range(TARGET)), "Endpoint slot budget exceeded")


def seal(output: Path, state: dict) -> None:
    # Extend the frozen shared seal to cover every file in this working archive.
    previous.seal(output, state)
    state["archive_inventory_sha256"] = inventory(output / "archive")
    write_json(output / "state.json", state)


def validate_state(output: Path, plan: dict) -> dict:
    state, work = read_json(output / "state.json"), output / "archive"
    require(state["plan_sha256"] == sha256(output / "plan.json"), "State plan binding changed")
    require(inventory(work) == state["archive_inventory_sha256"], "Sealed endpoint archive changed")
    validate_saved_state(work, state)
    verify_inherited(output, plan)
    for name, expected in state["recovery_artifacts"].items():
        require((inventory(work / name) if (work / name).exists() else {}) == expected,
                "Endpoint reservation receipts changed")
    receipt = work / "native-recovery.json"
    require((sha256(receipt) if receipt.exists() else None) == state["native_receipt_sha256"],
            "Native endpoint receipt changed")
    if state["status"] == "prepared":
        require(inventory(work) == plan["prepared_archive_sha256"], "Prepared copy changed")
    return state


def stop_reason(work: Path) -> str | None:
    start = read_json(work.parent / "plan.json")["start_generation"]
    if any(not r["correct"] for r in database_snapshot(work / "shinka/programs.sqlite")
           if r["generation"] >= start):
        return "new terminal evaluation failure"
    if any((work / f"shinka/gen_{generation}/failure.json").is_file()
           for generation in range(start, TARGET)):
        return "new terminal proposal failure"
    return None


def terminal_monitor(work: Path, grace_seconds: float):
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


def verify_native_receipt(work: Path, plan: dict, state: dict, reserved: list[int], summary: dict) -> None:
    start = plan["start_generation"]
    path = work / "native-recovery.json"
    if not path.exists():
        require(state["status"] != "complete", "Completed endpoint lacks native receipt")
        return
    receipt = read_json(path)
    persisted = [r["generation"] for r in database_snapshot(work / "shinka/programs.sqlite")]
    valid = {r["generation"] for r in summary["programs"] if r["generation"] >= start}
    attempted = receipt["attempted_generations"]
    require(receipt["start"] == start and receipt["target"] == TARGET and receipt["missing"] == MISSING
            and attempted == sorted(set(attempted)) and set(attempted) <= set(reserved)
            and receipt["persisted_generations"] == persisted
            and receipt["consumed_generations"] == sorted(set(persisted) | set(MISSING) | set(reserved)),
            "Native endpoint receipt slot accounting changed")
    require(receipt["initial_rng_sha256"] == plan["rng_input_sha256"]
            and receipt["final_rng_sha256"] == sha256(work / "shinka/rng_state.json"),
            "Native endpoint RNG binding changed")
    require(set(receipt["verified_generations"]) <= valid, "Native receipt claims unverified scores")
    if state["status"] == "complete":
        require(receipt["status"] == "complete" and receipt["graceful_return"] is True
                and receipt["rng_saved"] is True and receipt["terminal_reason"] is None
                and attempted == sorted(reserved)
                and receipt["completed_generations"] == receipt["next_generation_to_submit"] == TARGET
                and receipt["verified_generations"] == list(range(start, TARGET)),
                "Incomplete native endpoint completion receipt")


def summarize(output: Path) -> dict:
    plan = read_endpoint_plan(output)
    start = plan["start_generation"]
    state = validate_state(output, plan)
    require(state["status"] != "running", "Do not summarize an active endpoint controller")
    work = output / "archive"
    summary = search.summarize(work, check_state=False)
    _, base = source_summary(Path(plan["source_archive"]))
    slot_records = reservation_records(work / "recovery_slots")
    request_records = reservation_records(work / "recovery_requests")
    reserved = [r["generation"] for r in slot_records]
    requested = [r["generation"] for r in request_records]
    require(len(reserved) == len(set(reserved)) and set(reserved) <= set(plan["allowed_generations"])
            and len(requested) == len(set(requested)) and set(requested) <= set(reserved),
            "Invalid endpoint slot/request reservations")
    require(reserved == list(range(start, start + len(reserved)))
            and requested == list(range(start, start + len(requested))),
            "Endpoint slots and requests must be sequential")
    actual = {int(p.name[4:]) for p in (work / "shinka").glob("gen_*") if p.is_dir()}
    consumed = actual | set(reserved)
    require(actual - set(range(start)) <= set(reserved), "Unreserved native generation")
    verify_native_receipt(work, plan, state, reserved, summary)
    new_groups = ledger_groups(work / "model_requests.jsonl")[plan["inherited_model_request_groups"]:]
    require(len(new_groups) <= len(requested), "More guarded requests than reserved invocations")
    require(all(sum(e["event"] == "started" for e in group) == 1
                and sum(e["event"] == "codex_exec" for e in group) <= 1
                and sum(e["event"] == "finished" for e in group) <= 1 for group in new_groups),
            "Repeated proposal or CLI launch")
    added = {key: summary[key] - base[key] for key in DELTA_KEYS}
    require(0 <= added["new_training_trials_allocated"] <= 3 * len(reserved)
            and added["new_training_trials_allocated"] <= plan["max_additional_training_trials"]
            and 0 <= added["new_nominal_training_steps_allocated"]
            <= plan["max_additional_nominal_training_steps"],
            "Endpoint training allocation exceeds reserved slots")
    if state["status"] == "complete":
        require(consumed == set(range(TARGET)) and set(reserved) == set(requested)
                == set(range(start, TARGET)) and len(new_groups) == TARGET - start,
                "Incomplete allocation endpoint")
        require(all(r["receipt_complete"] for r in slot_records + request_records),
                "Incomplete endpoint reservation receipt")
        require(all([e["event"] for e in group] == ["started", "codex_exec", "finished"]
                    and group[-1]["outcome"] == "success" for group in new_groups),
                "Incomplete successful proposal ledger")
        require({r["generation"] for r in summary["programs"]}
                == set(plan["inherited_valid_generations"]) | set(range(start, TARGET)),
                "Endpoint lacks valid new evaluations")
        require(not stop_reason(work) and not summary["unpersisted_requests"]
                and summary["incomplete_training_attempts"] == 0, "Unresolved endpoint work")
    summary.update(protocol=VERSION, original_protocol=plan["original_protocol"],
                   previous_protocol=base["protocol"], status=state["status"],
                   original_search_status="failed", slots_consumed=len(consumed),
                   proposal_slots_consumed=len(consumed) - 1, target_slots=TARGET,
                   endpoint_reserved_generations=sorted(reserved),
                   endpoint_requested_generations=sorted(requested), endpoint_added=added,
                   endpoint_slot_receipts=slot_records, endpoint_request_receipts=request_records,
                   inherited_attempts=[*base.get("inherited_attempts", []),
                       {"protocol": base["protocol"], "source_archive": plan["source_archive"],
                        "slots_consumed": base["slots_consumed"]}],
                   remaining_generations=sorted(set(range(TARGET)) - consumed),
                   inherited_no_row_consumed_generations=MISSING, rng_policy=plan["rng_policy"],
                   rng_deviation=plan["rng_deviation"],
                   sessions=[*base["sessions"], *state["sessions"]],
                   session_wall_seconds=base["session_wall_seconds"]
                   + sum(s.get("wall_seconds", 0.) for s in state["sessions"]))
    summary.update(allocation_status="exhausted" if len(consumed) == TARGET else "open",
                   endpoint_status="unresolved", endpoint_resolution_required=True)
    if len(consumed) == TARGET and state["status"] == "complete":
        summary["endpoint_status"] = "allocation_exhausted_complete"
    elif len(consumed) == TARGET and state["status"] == "failed":
        try:
            verify_grammar_stop(output, plan, state, summary, sorted(reserved), sorted(requested))
        except (ValueError, KeyError, OSError):
            pass  # Allocation alone never upgrades an incomplete controller stop.
        else:
            summary["endpoint_status"] = "allocation_exhausted_with_grammar_stop"
    return summary


def run(output: Path) -> dict:
    output = Path(output).resolve()
    with execution_lock(output):
        plan = read_endpoint_plan(output, runtime=True)
        state = validate_state(output, plan)
        require(state["status"] == "prepared" and not state["sessions"],
                "Only a prepared endpoint may execute; stopped work requires review")
        work, study = output / "archive", Path(plan["evaluation_study"])
        original = read_json(work / "plan.json")
        evaluation = read_plan(study)
        env = search.search_environment(study, evaluation, original["model"],
                                        original["proposal_timeout_seconds"])
        env.update(SHINKA_SUBSCRIPTION_LEDGER=str(work / "model_requests.jsonl"),
                   SHINKA_RECOVERY_WORK=str(work),
                   SHINKA_HEADLESS_COMMAND=shlex.join(
                       [sys.executable, str(REPO_ROOT / "scripts/endpoint_headless.py")]))
        command = [sys.executable, "-m", "shinka_crl.adaptive_endpoint", "--native", str(output)]
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
                    f"Endpoint stopped: {result}")
            require(read_json(work / "native-recovery.json")["status"] == "complete",
                    "Native endpoint did not complete")
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
            try:
                write_json(output / "summary.json", summarize(output))
            except BaseException as exc:
                session.update(status="failed", verification_error=f"{type(exc).__name__}: {exc}")
                state["status"] = "failed"
                seal(output, state)
                raise
        return state


def reserve_provider_request(work: Path, generation: int) -> Path:
    plan = read_json(work.parent / "plan.json")
    require(read_json(work.parent / "plan-receipt.json")
            == {"plan_sha256": sha256(work.parent / "plan.json")}, "Endpoint plan receipt changed")
    start = plan["start_generation"]
    require(plan["protocol"] == VERSION and FIRST <= start <= generation < TARGET
            and plan["allowed_generations"] == list(range(start, TARGET)),
            "Provider generation is outside endpoint budget")
    requests = reservation_records(work / "recovery_requests")
    require([r["generation"] for r in requests] == list(range(start, generation)),
            "Provider request is not the first unused slot")
    rows = database_snapshot(work / "shinka/programs.sqlite")
    require([r["generation"] for r in rows] == [g for g in range(generation) if g not in MISSING]
            and all(r["correct"] for r in rows if r["generation"] >= start),
            "Provider requires all earlier new slots to have completed successfully")
    return recovery.reserve_provider_request(work, generation)


def provider_main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    guard = runpy.run_path(str(REPO_ROOT / "scripts/subscription_headless.py"))
    if argv == ["--check"]:
        return guard["main"](argv)
    work = Path(os.environ["SHINKA_RECOVERY_WORK"])
    require(work.is_absolute(), "Endpoint working path must be absolute")
    reserve_provider_request(work, int(os.environ["SHINKA_RECOVERY_GENERATION"]))
    return guard["main"](argv)


def export(*, output: Path, report: Path) -> dict:
    output, report = Path(output).resolve(), Path(report).resolve()
    with execution_lock(output):
        return _export_locked(output=output, report=report)


def _export_locked(*, output: Path, report: Path) -> dict:
    plan = read_endpoint_plan(output)
    source = Path(plan["source_archive"])
    require(not report.exists() and not any(report == p or report.is_relative_to(p)
            or p.is_relative_to(report) for p in (output, *source_ancestors(source), Path(plan["evaluation_study"]))),
            "Use a fresh independent endpoint report path")
    summary = summarize(output)
    output_before = inventory(output)
    work, study = output / "archive", Path(plan["evaluation_study"])
    prepared_only = summary["status"] == "prepared"
    if prepared_only:
        files = [(output / name, "raw/endpoint/" + name)
                 for name in ("plan.json", "plan-receipt.json", "state.json")]
    else:
        files = [(p, "raw/endpoint/" + str(p.relative_to(output))) for p in output.rglob("*")
                 if p.is_file() and "__pycache__" not in p.parts
                 and not p.name.endswith((".sqlite-wal", ".sqlite-shm"))]
        files += [(study / name, "raw/evaluation/" + name) for name in ("plan.json", "plan-receipt.json")]
        for origin in summary["evidence_cache_origins"]:
            files += [(p, "raw/evaluation/" + str(p.relative_to(study)))
                      for p in (study / origin).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    before = {str(p): sha256(p) for p, _ in files}
    replacements = ((str(work), "$CONTINUATION_ARCHIVE"), (str(output), "$CONTINUATION"),
                    (plan["source_archive"], "$SOURCE_RECOVERY"), (str(study), "$EVALUATION_STUDY"),
                    (str(REPO_ROOT), "$REPO_ROOT"))

    def portable(value: str) -> str:
        for old, new in replacements:
            value = value.replace(old, new)
        return value

    report.mkdir(parents=True)
    for path, name in files:
        if path.suffix in recovery.TEXT_SUFFIXES:
            target = report / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(portable(path.read_text()))
    write_json(report / "summary.json", json.loads(portable(json.dumps(summary, allow_nan=False))))
    write_json(report / "endpoint-plan.json", json.loads(portable(json.dumps(plan, allow_nan=False))))
    require(before == {str(p): sha256(p) for p, _ in files}, "Endpoint evidence changed during export")
    require(inventory(output) == output_before, "Endpoint file set changed during export")
    write_json(report / "checksums.json", {"original_sha256": {name: sha256(path) for path, name in files},
               "published_sha256": inventory(report), "prepared_only": prepared_only,
               "note": "Paths redacted; binary and empty lock evidence retained locally with original hashes. "
                       "History and inherited archive records describe prior stopped stages; top-level summary "
                       "is current. Prepared exports contain the freeze and bindings only; historical results "
                       "remain in the stopped-recovery publication."})
    return summary


if __name__ == "__main__":
    require(len(sys.argv) == 3 and sys.argv[1] == "--native", "Use scripts/run_adaptive_endpoint.py")
    destination = Path(sys.argv[2]).resolve()
    saved_plan = read_endpoint_plan(destination)
    saved_state = read_json(destination / "state.json")
    require(saved_state["status"] == "running" and saved_state["sessions"][-1]["controller_pid"]
            == os.getppid(), "Native endpoint must be launched by its active controller")
    from shinka_crl.adaptive_recovery_native import native_main

    raise SystemExit(native_main(destination / "archive", target=TARGET, start=saved_plan["start_generation"], missing=tuple(MISSING)))
