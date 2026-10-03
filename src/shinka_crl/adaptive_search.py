"""Bounded native Shinka search using the unchanged adaptive evaluator and cache."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import time

from shinka_crl.adaptive_evaluation import (
    CONTROL_IDS, digest, read_plan, runtime_fingerprint, runtime_scope, validate_request,
)
from shinka_crl.experiment import REPO_ROOT
from shinka_crl.pilot import read_json, require, sha256, write_json
from shinka_crl.reference_timing import artifact_files, artifact_hashes, utc_now
from shinka_crl.search import (
    RANDOM_SEED, artifact_receipts, database_snapshot, monitored_run, runtime_identity,
    subscription_environment, validate_saved_state,
)

TASK_FILES = {
    "initial.py": "tasks/cartpole_adaptive/initial.py",
    "evaluate.py": "tasks/cartpole_adaptive/shinka_evaluate.py",
    "shinka-subscription.yaml": "tasks/cartpole_adaptive/shinka-subscription.yaml",
}
SOURCE_FILES = (
    "src/shinka_crl/adaptive_search.py", "scripts/run_adaptive_search.py",
    "scripts/report_adaptive_search.py", "scripts/subscription_headless.py",
    *TASK_FILES.values(),
)
TARGETS = (5, 13, 25)


def search_environment(study: Path, plan: dict, model: str, timeout: int) -> dict:
    env = subscription_environment(model, timeout)
    # The static search sets PYTHONHASHSEED. Adaptive training must instead match
    # its previously frozen context exactly; native_main seeds host sampling RNG.
    for key, value in plan["runtime"]["numerical_environment"].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    env["SHINKA_ADAPTIVE_STUDY"] = str(study.resolve())
    return env


def native_command(output: Path, target: int) -> list[str]:
    return [sys.executable, "-m", "shinka_crl.search", "--native",
            str(output / "shinka/rng_state.json"), "--task-dir", str(output / "task"),
            "--config-fname", "shinka-subscription.yaml", "--results_dir",
            str(output / "shinka"), "--num_generations", str(target)]


def make_plan(study: Path, evaluation: dict, env: dict, model: str, timeout: int) -> dict:
    require(timeout >= 90, "Proposal timeout must be at least 90 seconds")
    controls = {name: validate_request(study, study / "requests" / name, evaluation)
                for name in CONTROL_IDS}
    with runtime_scope(evaluation["cpu_affinity"]):
        require(runtime_fingerprint(evaluation["python"]) == evaluation["runtime"],
                "Frozen adaptive evaluation runtime changed")
        runtime = runtime_identity(env)
    return {
        "schema_version": 1, "protocol": "adaptive-shinka-v1",
        "evaluation_study": str(study), "evaluation_context_sha256": digest(evaluation),
        "evaluation_plan_sha256": sha256(study / "plan.json"),
        "evaluation_source_sha256": evaluation["source_sha256"],
        "source_sha256": {name: sha256(REPO_ROOT / name) for name in SOURCE_FILES},
        "controls": {name: {"receipt_sha256": sha256(study / "requests" / name / "receipt.json"),
                            "aggregate": row["aggregate"]} for name, row in controls.items()},
        "model": model, "reasoning_effort": "medium", "auth": "chatgpt",
        "proposal_timeout_seconds": timeout, "runtime": runtime,
        "cpu_affinity": evaluation["cpu_affinity"], "stages": list(TARGETS),
        "outer_random_seed": RANDOM_SEED, "proposal_slots": 24,
        "max_model_requests_per_slot": 1,
        "duplicate_policy": "Each request consumes a slot; only verified canonical-AST cache reuse",
        "failure_policy": "Stop on terminal proposal/evaluation failure; preserve failed slots; no automatic retry",
        "rng_policy": "Native Python/NumPy state checkpoint; model responses and restart scheduling "
                      "do not guarantee the uninterrupted trajectory",
        "reporting": "Development search; reserved validation and final reporting remain unused",
    }


def validate_task(output: Path) -> None:
    for target, source in TASK_FILES.items():
        require(sha256(output / "task" / target) == sha256(REPO_ROOT / source),
                f"Frozen task changed: {target}")


def verified_rows(output: Path, plan: dict, evaluation: dict, *, allow_failed: bool = False) -> list[dict]:
    """Bind native ancestry/scores/code to independently validated evaluator receipts."""
    study = Path(plan["evaluation_study"])
    rows = database_snapshot(output / "shinka/programs.sqlite")
    require(len({r["generation"] for r in rows}) == len(rows), "Duplicate native generation")
    ids = {row["id"]: row["generation"] for row in rows}
    verified = []
    for row in rows:
        if not row["correct"] and allow_failed:
            continue
        require(bool(row["correct"]), "Failed scored candidate requires review")
        directory = output / "shinka" / f"gen_{row['generation']}"
        request_dir = directory / "results/evaluation"
        request = validate_request(study, request_dir, evaluation)
        require(row["code"] == (directory / "main.py").read_text()
                and sha256(directory / "main.py") == request["candidate"]["source_sha256"],
                "Native code differs from evaluated source")
        for name in ("correct.json", "metrics.json"):
            require(read_json(directory / "results" / name) == read_json(request_dir / name),
                    "Native result differs from immutable evaluation")
        metrics = read_json(request_dir / "metrics.json")
        require(row["combined_score"] == metrics["combined_score"]
                and json.loads(row["public_metrics"]) == metrics["public"]
                and json.loads(row["private_metrics"]) == metrics["private"],
                "Native database score or feedback differs from verified evaluation")
        generation, parent = row["generation"], row["parent_id"]
        require((generation == 0 and parent is None)
                or (parent in ids and ids[parent] < generation), "Invalid native ancestry")
        if generation == 0:
            identity = validate_request(study, study / "requests/identity", evaluation)
            require(request["cache_hit"] and request["new_training_trials"] == 0
                    and request["cache_key"] == identity["cache_key"]
                    and request["cache_origin"] == identity["cache_origin"]
                    and request["aggregate"] == identity["aggregate"],
                    "Initial slot must reuse the verified identity control")
        verified.append({"generation": generation, "id": row["id"], "parent_id": parent,
                         "program_sha256": request["candidate"]["source_sha256"],
                         "canonical_ast_sha256": request["candidate"]["program"]["canonical_ast_sha256"],
                         "request": request})
    return verified


def scheduler_preflight(output: Path, study: Path, evaluation: dict, env: dict) -> None:
    """Exercise actual native logging + evaluator context before any proposal."""
    directory = output / "preflight"
    require(not directory.exists(), "Preflight evidence already exists; inspect before retry")
    directory.mkdir()
    code = """
import sys
from shinka.launch.scheduler import JobScheduler, LocalJobConfig
scheduler = JobScheduler(job_type='local', config=LocalJobConfig(
    eval_program_path=sys.argv[1], numeric_threads_per_job=1, time='00:02:00'), verbose=False)
results, duration = scheduler.run(sys.argv[2], sys.argv[3])
if not results.get('correct', {}).get('correct'):
    raise SystemExit('Native scheduler identity preflight failed')
"""
    command = [sys.executable, "-c", code, str(output / "task/evaluate.py"),
               str(output / "task/initial.py"), str(directory / "results")]
    with runtime_scope(evaluation["cpu_affinity"]):
        result = monitored_run(command, log=directory / "scheduler.log", env=env, timeout=150)
    require(result["returncode"] == 0 and result["stop_reason"] is None,
            "Native scheduler preflight failed before model proposals")
    request = validate_request(study, directory / "results/evaluation", evaluation)
    identity = validate_request(study, study / "requests/identity", evaluation)
    require(request["cache_hit"] and request["new_training_trials"] == 0
            and request["cache_origin"] == identity["cache_origin"]
            and request["aggregate"] == identity["aggregate"], "Preflight failed to reuse identity")
    for name in ("correct.json", "metrics.json"):
        require(read_json(directory / "results" / name)
                == read_json(directory / "results/evaluation" / name), "Preflight result mirror changed")
    write_json(directory / "contract.json", {"status": "passed", "purpose": "Native scheduler cache check",
               "new_training_trials": 0, "model_calls": 0, "command": command, **result})


def proposal_usage(output: Path) -> dict:
    path = output / "model_requests.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    grouped = {}
    for event in events:
        grouped.setdefault(event["request_id"], []).append(event)
    launches = sum(e["event"] == "codex_exec" for e in events)
    successful = sum(e["event"] == "finished" and e.get("outcome") == "success" for e in events)
    return {"guarded_requests": len(grouped), "codex_cli_launches": launches,
            "successful_requests": successful,
            "failed_or_incomplete_requests": len(grouped) - successful,
            "backend_model_requests": None, "paid_api_calls": 0,
            "note": "Guard enforces ChatGPT authentication; launches are not backend request counts "
                    "or a remaining-subscription-allowance meter"}


def summarize(output: Path, *, check_state: bool = True) -> dict:
    plan, state = read_json(output / "plan.json"), read_json(output / "state.json")
    require(plan["source_sha256"] == {name: sha256(REPO_ROOT / name) for name in SOURCE_FILES},
            "Search sources changed; use recorded revision")
    evaluation = read_plan(Path(plan["evaluation_study"]))
    require(digest(evaluation) == plan["evaluation_context_sha256"], "Evaluator context changed")
    validate_task(output)
    if check_state:
        validate_saved_state(output, state)
        if state.get("preflight_sha256") is not None:
            require(artifact_hashes(output / "preflight") == state["preflight_sha256"],
                    "Scheduler preflight artifacts changed")
    rows = verified_rows(output, plan, evaluation, allow_failed=state["status"] != "complete")
    directories = sorted((output / "shinka").glob("gen_*"))
    slots = [int(p.name.removeprefix("gen_")) for p in directories if p.is_dir()]
    failures = [str(p.relative_to(output)) for p in (output / "shinka").glob("gen_*/failure.json")]
    requests = [row["request"] for row in rows]
    failed_requests, unpersisted_requests, attempts, training_attempts = [], [], [], []
    known_generations = {row["generation"] for row in rows}
    study = Path(plan["evaluation_study"])
    origins, new_origins = set(), set()
    for directory in directories:
        request_dir = directory / "results/evaluation"
        request_file = request_dir / "request.json"
        if not request_file.is_file():
            if request_dir.exists():
                failed_requests.append({"generation": int(directory.name.removeprefix("gen_")),
                                        "status": "missing_request", "receipt_verified": False,
                                        "path": str(request_dir.relative_to(output))})
            continue
        request = read_json(request_file)
        generation = int(directory.name.removeprefix("gen_"))
        receipt = request_dir / "receipt.json"
        if receipt.is_file():
            require(artifact_hashes(request_dir) == read_json(receipt), "Failed request evidence changed")
        require(request["context_sha256"] == digest(evaluation)
                and request["slot_consumed"] is True and request["evaluator_model_calls"] == 0
                and request["request_id"] == request_dir.name,
                "Request context or slot provenance changed")
        if request["status"] == "complete":
            if generation not in known_generations:
                request = validate_request(study, request_dir, evaluation)
                requests.append(request)
                unpersisted_requests.append({"generation": generation, "request": request})
        else:
            require(request["status"] in ("failed", "running"), "Unknown request status")
            failed_requests.append({"generation": generation, "path": str(request_dir.relative_to(output)),
                                    "receipt_verified": receipt.is_file(), **request})
        if request.get("cache_origin"):
            origin = (study / request["cache_origin"]).resolve()
            require(origin.is_relative_to(study / "cache") and not origin.is_symlink(),
                    "Request cache origin escapes study")
            relative = str(origin.relative_to(study))
            origins.add(relative)
            if not request["cache_hit"]:
                new_origins.add(relative)
    # Training can finish before post-hoc scoring fails. Count its manifest even
    # when there is no valid candidate score or native database row.
    for relative in sorted(new_origins):
        attempt = study / relative
        receipt = attempt / "receipt.json"
        if receipt.is_file():
            require(artifact_hashes(attempt) == read_json(receipt), "Cache attempt evidence changed")
        saved = read_json(attempt / "summary.json") if (attempt / "summary.json").is_file() else {}
        if saved:
            require(saved["context_sha256"] == digest(evaluation), "Cache attempt context changed")
        attempts.append({"path": relative, "status": saved.get("status", "missing_summary"),
                         "receipt_verified": receipt.is_file(), "scored_trials": len(saved.get("trials", []))})
        for path in sorted(attempt.glob("seed_*/training/manifest.json")):
            manifest = read_json(path)
            training_attempts.append({"path": str(path.relative_to(study)), "status": manifest["status"],
                                      "seed": manifest["seed"], "wall_seconds": manifest.get("wall_seconds", 0.)})
    from shinka_crl.experiment import nominal_training_steps

    steps = nominal_training_steps(evaluation["profile"], "ga")
    completed_training = sum(row["status"] == "complete" for row in training_attempts)
    usage = proposal_usage(output)
    require(usage["guarded_requests"] <= max(0, len(slots) - 1), "More model requests than proposal slots")
    if state["status"] == "complete":
        require(sorted(slots) == [row["generation"] for row in rows] == list(range(len(rows)))
                and not failures and not failed_requests and not unpersisted_requests,
                "Completed stage has missing or failed slots")
        require(usage["guarded_requests"] == usage["successful_requests"] == len(rows) - 1
                and usage["codex_cli_launches"] == len(rows) - 1, "Proposal ledger differs from stage")
    return {"schema_version": 1, "status": state["status"], "protocol": plan["protocol"],
            "evaluation_profile": evaluation["profile"], "objective": evaluation["objective_version"],
            "slots_consumed": len(slots), "programs_evaluated": len(rows),
            "proposal_slots_consumed": sum(s > 0 for s in slots),
            "distinct_canonical_programs": len({r["canonical_ast_sha256"] for r in rows}),
            "cache_hits": sum(r["cache_hit"] for r in requests),
            "new_training_trials": completed_training,
            "new_nominal_training_steps": completed_training * steps,
            "new_training_trials_allocated": len(training_attempts),
            "new_nominal_training_steps_allocated": len(training_attempts) * steps,
            "new_scored_training_trials": sum(row["scored_trials"] for row in attempts),
            "incomplete_training_attempts": len(training_attempts) - completed_training,
            "training_wall_seconds": sum(row["wall_seconds"] for row in training_attempts),
            "cost_accounting_note": "Completed nominal steps exclude incomplete trials; allocated steps "
                                    "count their full budget. Training manifests are retained even when "
                                    "post-hoc scoring fails. Unsealed interruptions require review.",
            "avoided_nominal_training_steps": sum(r["avoided_training_steps_nominal"] for r in requests),
            "evaluation_wall_seconds": sum(r.get("wall_seconds", 0.) for r in requests + failed_requests),
            "session_wall_seconds": sum(s.get("wall_seconds", 0.) for s in state["sessions"]),
            "proposal_usage": usage, "terminal_proposal_failures": failures,
            "failed_requests": failed_requests, "unpersisted_requests": unpersisted_requests,
            "cache_attempts": attempts, "training_attempts": training_attempts,
            "evidence_cache_origins": sorted(origins),
            "programs": rows, "controls": plan["controls"], "sessions": state["sessions"],
            "interpretation": "First adaptive program search on three development seeds; "
                              "no reserved validation or general search-method comparison"}


def run_search(*, output: Path, study: Path, target: int = 5, model: str = "gpt-6.1-sol",
               timeout: int = 600, prepare_only: bool = False, resume: bool = False) -> dict:
    require(target in TARGETS, "Use a declared cumulative target: 5, 13, 25")
    output, study = Path(output).resolve(), Path(study).resolve()
    require(not output.is_relative_to(study) and not study.is_relative_to(output),
            "Keep search archive and evaluation cache separate")
    evaluation = read_plan(study)
    env = search_environment(study, evaluation, model, timeout)
    env["SHINKA_SUBSCRIPTION_LEDGER"] = str(output / "model_requests.jsonl")
    plan = make_plan(study, evaluation, env, model, timeout)
    if output.exists():
        require(resume, "Existing archive requires explicit --resume")
        require(read_json(output / "plan.json") == plan, "Frozen search plan or runtime changed")
        state = read_json(output / "state.json")
        validate_task(output)
        validate_saved_state(output, state)
        if state.get("preflight_sha256") is not None:
            require(artifact_hashes(output / "preflight") == state["preflight_sha256"],
                    "Scheduler preflight artifacts changed")
    else:
        require(not resume, "Cannot resume a missing archive")
        (output / "task").mkdir(parents=True)
        for name, source in TASK_FILES.items():
            shutil.copyfile(REPO_ROOT / source, output / "task" / name)
        write_json(output / "plan.json", plan)
        state = {"schema_version": 1, "status": "prepared", "sessions": [],
                 "programs": [], "artifact_sha256": {}}
        write_json(output / "state.json", state)
    if prepare_only:
        return state
    require(state["status"] in ("prepared", "complete"), "Interrupted or failed stage requires review")
    require(all(s["status"] == "complete" for s in state["sessions"]), "Failed stage cannot be retried automatically")
    count = len(state["programs"])
    require(count == 0 or count >= 2, "Generation-zero-only resume is unsupported by pinned Shinka")
    require(target > count, "Target must exceed completed slots")
    if count:
        summarize(output)
    command = native_command(output, target)
    session = {"index": len(state["sessions"]) + 1, "target": target, "command": command,
               "status": "running", "started_at": utc_now()}
    state["sessions"].append(session)
    state["status"] = "running"
    write_json(output / "state.json", state)
    log = output / "sessions" / f"session_{session['index']:03d}.log"
    print(f"Running adaptive Shinka to {target} total slots; log: {log}", flush=True)
    started = time.monotonic()
    try:
        if not state["programs"]:
            scheduler_preflight(output, study, evaluation, env)
        with runtime_scope(evaluation["cpu_affinity"]):
            result = monitored_run(command, log=log, env=env,
                                   timeout=(target-count) * (timeout + 4 * evaluation["timeout_seconds"]),
                                   shinka=output / "shinka")
        session.update(result)
        require(result["returncode"] == 0 and result["stop_reason"] is None,
                f"Adaptive search stopped; inspect {log}: {result}")
        rows = verified_rows(output, plan, evaluation)
        require([row["generation"] for row in rows] == list(range(target)), "Incomplete bounded stage")
        require((output / "shinka/rng_state.json").is_file(), "Missing native host RNG checkpoint")
        session["status"] = state["status"] = "complete"
    except BaseException as exc:
        session.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        state["status"] = "failed"
        raise
    finally:
        session.setdefault("wall_seconds", time.monotonic()-started)
        session["finished_at"] = utc_now()
        state["programs"] = database_snapshot(output / "shinka/programs.sqlite")
        state["artifact_sha256"] = artifact_receipts(output)
        state["preflight_sha256"] = artifact_hashes(output / "preflight")
        for key, name in (("rng_sha256", "shinka/rng_state.json"),
                          ("model_requests_sha256", "model_requests.jsonl")):
            path = output / name
            state[key] = sha256(path) if path.is_file() else None
        write_json(output / "state.json", state)
    try:
        summary = summarize(output)
    except BaseException as exc:
        state["status"] = "failed"
        session.update(status="failed", error=f"Post-stage verification: {type(exc).__name__}: {exc}")
        write_json(output / "state.json", state)
        raise
    write_json(output / "summary.json", summary)
    return state


def export_search(*, output: Path, report: Path) -> dict:
    output, report = Path(output).resolve(), Path(report).resolve()
    require(not report.exists() and not report.is_relative_to(output), "Use a fresh external report path")
    summary = summarize(output)
    plan = read_json(output / "plan.json")
    study = Path(plan["evaluation_study"])
    require(not report.is_relative_to(study), "Do not export into evaluation cache")
    origins = summary["evidence_cache_origins"]
    files = [(path, "raw/search/" + str(path.relative_to(output)))
             for path in artifact_files(output)]
    files += [(study / name, "raw/evaluation/" + name) for name in ("plan.json", "plan-receipt.json")]
    for origin in origins:
        files += [(path, "raw/evaluation/" + str(path.relative_to(study)))
                  for path in artifact_files(study / origin)]
    redactions = ((str(output), "$SEARCH"), (str(study), "$EVALUATION_STUDY"),
                  (str(REPO_ROOT), "$REPO_ROOT"))

    def portable(text):
        for source, replacement in redactions:
            text = text.replace(source, replacement)
        return text

    before = {str(path): sha256(path) for path, _ in files}
    report.mkdir(parents=True)
    originals = {}
    for path, relative in files:
        originals[relative] = sha256(path)
        if path.suffix.lower() not in {".json", ".jsonl", ".py", ".yaml", ".yml", ".md", ".log", ".out", ".err", ".txt", ".diff", ".patch"}:
            continue
        target_path = report / relative
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(portable(path.read_text()))
    write_json(report / "summary.json", json.loads(portable(json.dumps(summary, allow_nan=False))))
    require(before == {str(path): sha256(path) for path, _ in files}, "Evidence changed while exporting")
    write_json(report / "checksums.json", {
        "original_sha256": originals, "published_sha256": artifact_hashes(report),
        "note": "Binary checkpoints/databases remain local; original hashes retained. "
                "Published paths are redacted; local receipt hashes refer to original artifacts."})
    return summary
