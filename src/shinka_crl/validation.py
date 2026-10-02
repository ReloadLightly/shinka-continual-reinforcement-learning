"""Execute a frozen finalist comparison once, resuming only whole trial attempts."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import runpy
import shutil
import statistics
import subprocess
import time

from shinka_crl.analysis import run_analysis, validate_analysis
from shinka_crl.baseline_contract import validate_baseline_config
from shinka_crl.experiment import (
    DEFAULT_PYTHON, DEFAULT_UPSTREAM, REPO_ROOT, UPSTREAM_COMMIT,
    build_command, nominal_training_steps, run_experiment, score_curve, verify_upstream,
)
from shinka_crl.pilot import (
    TRAINING_FILES, completed_attempt, next_attempt, read_json, require, sha256, write_json,
)
from shinka_crl.search import SEARCH_THREAD_ENV, effective_key

FREEZER = REPO_ROOT / "scripts/freeze_finalists.py"
SOURCES = (
    "src/shinka_crl/validation.py", "scripts/run_validation.py", "scripts/report_validation.py",
    "scripts/freeze_finalists.py", "scripts/report_search.py", "tasks/cartpole_ga/evaluate.py",
    "src/shinka_crl/experiment.py", "src/shinka_crl/analysis.py",
    "src/shinka_crl/baseline_contract.py", "src/shinka_crl/pilot.py", "src/shinka_crl/search.py",
    "src/shinka_crl/profiles/cartpole-validation.json", "requirements/cpu.lock", "upstream.lock.json",
)


def finalists(path: Path) -> dict:
    return runpy.run_path(str(FREEZER))["validate_finalists"](path)


def validate_trial(path: Path, *, profile: dict, candidate: dict, seed: int,
                   receipt: bool = True) -> dict:
    """Check the exact tuned settings while preserving all other baseline invariants."""
    path = path.resolve()
    hashes = {name: sha256(path / name) for name in TRAINING_FILES}
    if receipt:
        require(read_json(path / "receipt.json") == hashes, "Training artifact changed")
    manifest = read_json(path / "manifest.json")
    expected = {"status": "complete", "profile": profile, "method": "ga", "seed": seed,
                "trial": seed + 1, "upstream_commit": UPSTREAM_COMMIT,
                "ga_settings": candidate["settings"]}
    require(all(manifest.get(k) == v for k, v in expected.items()), "Training identity mismatch")
    command = manifest["command"]
    require(command == build_command(profile=profile, method="ga", seed=seed, trial=seed + 1,
                                     output_dir=path, upstream=Path(command[1]).parent.parent,
                                     python=command[0], ga_settings=candidate["settings"]),
            "Training command changed")
    require(manifest["metrics_sha256"] == hashes["training_metrics.json"], "Curve hash mismatch")
    score = score_curve(read_json(path / "training_metrics.json"), profile=profile, method="ga")
    require(read_json(path / "summary.json") == {
        **score, "profile": profile["name"], "method": "ga", "seed": seed,
        "trial": seed + 1, "upstream_commit": UPSTREAM_COMMIT}, "Training score mismatch")
    result = read_json(path / "results.json")
    config = result["config"]
    settings = candidate["settings"]
    ne = profile["ne"]
    elites = effective_key(settings, ne["pop_size"])[1]
    resolved = {"refresh": True, "num_elites": elites, "num_offspring": ne["pop_size"] - elites,
                "variation": "gaussian", "sigma": settings["sigma"], "cross_over_rate": 0.0}
    require(config.get("sigma") == settings["sigma"]
            and config.get("searcher_kwargs") == {"elite_ratio": settings["elite_ratio"],
                                                 "init_around_mean": False}
            and config.get("searcher_resolved") == resolved, "Tuned GA settings mismatch")
    # The established baseline validator owns architecture and non-search settings.
    baseline = {**config, "sigma": 0.5,
                "searcher_kwargs": {"elite_ratio": 0.5, "init_around_mean": False},
                "searcher_resolved": {**resolved, "sigma": 0.5,
                                     "num_elites": ne["pop_size"] // 2,
                                     "num_offspring": ne["pop_size"] // 2}}
    validate_baseline_config(baseline, "ga")
    expected_config = {"seed": seed, "trial": seed + 1, "schedule": "switch", "noise_range": 0.5,
                       "num_tasks": 2, "task_sequence": [0, 1, 0, 1],
                       "episode_length": 500, "eval_episodes": 10, **ne}
    for key, value in expected_config.items():
        require(config.get(key) == value, f"Resolved protocol mismatch: {key}")
    require(result["env_steps"] == nominal_training_steps(profile, "ga"), "Training budget mismatch")
    vectors = result["noise_vectors"]
    require(isinstance(vectors, list) and len(vectors) == 2 and vectors[0] == [0, 0, 0, 0]
            and all(isinstance(v, list) and len(v) == 4 and all(
                type(x) in (int, float) and math.isfinite(x) for x in v) for v in vectors),
            "Invalid task vectors")
    wall = manifest["wall_seconds"]
    require(type(wall) in (int, float) and math.isfinite(wall) and wall >= 0, "Invalid duration")
    return {"normalized_score": score["normalized_score"], "artifact_sha256": hashes,
            "noise_vectors": vectors, "wall_seconds": wall}


def describe(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values)
            if len(values) > 1 else None, "n": len(values)}


def aggregate(rows: list[dict], manifest: dict) -> dict:
    """Select within each frozen arm by validation active return, never by forgetting."""
    groups = []
    for candidate in manifest["candidates"]:
        trials = [row for row in rows if row["candidate_id"] == candidate["id"]]
        if not trials:
            continue
        groups.append({"candidate_id": candidate["id"], "settings": candidate["settings"],
                       "program_sha256": candidate["program_sha256"],
                       "memberships": candidate["memberships"], "completed_seeds": len(trials),
                       "score": describe([row["normalized_score"] for row in trials]),
                       "metrics": {name: describe([row["analysis"]["metrics"][name] for row in trials])
                                   for name in ("learning_accuracy", "forgetting",
                                                "learning_minus_forgetting", "zero_shot_transfer",
                                                "normalized_curve_average")}})
    complete = len(rows) == len(manifest["candidates"]) * len(manifest["profile"]["seeds"])
    winners = {}
    if complete:
        for arm in ("shinka", "random"):
            eligible = []
            for group in groups:
                memberships = [m for m in group["memberships"] if m["arm"] == arm]
                if memberships:
                    member = min(memberships, key=lambda m: (m["development_rank"],
                                                            m["program_sha256"]))
                    eligible.append((group, member))
            require(bool(eligible), f"No finalist from {arm}")
            best, member = min(eligible, key=lambda item: (-item[0]["score"]["mean"],
                                                          item[1]["development_rank"],
                                                          item[1]["program_sha256"]))
            winners[arm] = {"candidate_id": best["candidate_id"],
                            "program_sha256": member["program_sha256"],
                            "program_path": member["program_path"],
                            "settings": member["settings"],
                            "validation_representative_program_sha256": best["program_sha256"],
                            "validation_score": best["score"]["mean"]}
    return {"groups": groups, "selected_winners": winners,
            "selection_complete": complete,
            "interpretation": "One reserved finalist comparison. No feedback to proposal search; "
                              "active-return selection does not establish reduced forgetting."}


def run_validation(*, frozen: Path, output: Path, upstream: Path = DEFAULT_UPSTREAM,
                   python: str = str(DEFAULT_PYTHON), cpus: int = 2, timeout: int = 1800,
                   resume: bool = False, max_trials: int | None = None) -> dict:
    original_affinity = os.sched_getaffinity(0)
    original_environment = {key: os.environ.get(key) for key in SEARCH_THREAD_ENV}
    try:
        return _run_validation(frozen=frozen, output=output, upstream=upstream, python=python,
                               cpus=cpus, timeout=timeout, resume=resume, max_trials=max_trials)
    finally:
        os.sched_setaffinity(0, original_affinity)
        for key, value in original_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _run_validation(*, frozen: Path, output: Path, upstream: Path,
                    python: str, cpus: int, timeout: int,
                    resume: bool, max_trials: int | None) -> dict:
    require(cpus > 0 and timeout > 0 and (max_trials is None or max_trials > 0),
            "CPU count, timeout, and trial limit must be positive")
    frozen, output, upstream = (Path(p).resolve() for p in (frozen, output, upstream))
    manifest = finalists(frozen)
    profile = manifest["profile"]
    require(profile == read_json(REPO_ROOT / "src/shinka_crl/profiles/cartpole-validation.json"),
            "Frozen finalist profile differs from the executable validation profile")
    verify_upstream(upstream)
    interpreter = shutil.which(python)
    require(interpreter is not None, "Missing training interpreter")
    python = str(Path(interpreter).absolute())
    affinity = sorted(os.sched_getaffinity(0))[:cpus]
    require(len(affinity) == cpus, "Requested CPUs unavailable")
    os.sched_setaffinity(0, affinity)
    os.environ.update(SEARCH_THREAD_ENV)
    source_hashes = {name: sha256(REPO_ROOT / name) for name in SOURCES}
    plan = {"schema_version": 1, "finalist_manifest_sha256": sha256(frozen / "manifest.json"),
            "profile": profile, "candidate_ids": [c["id"] for c in manifest["candidates"]],
            "upstream_commit": UPSTREAM_COMMIT, "source_sha256": source_hashes,
            "python": python, "cpu_affinity": affinity, "thread_environment": SEARCH_THREAD_ENV,
            "posthoc_episodes": 10, "eval_seed_offset": 900000}
    probe = (
        "import importlib.metadata as m,json,jax,platform,os;"
        "print(json.dumps({'python':platform.python_version(),'jax_backend':jax.default_backend(),"
        "'jax_devices':[str(d) for d in jax.devices()],"
        "'cpu_affinity':sorted(os.sched_getaffinity(0)),"
        "'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))")
    environment = json.loads(subprocess.check_output([python, "-c", probe], text=True, timeout=60))
    environment.update(platform=platform.platform(), thread_environment=SEARCH_THREAD_ENV)
    require(environment["jax_backend"] == "cpu" and environment["cpu_affinity"] == affinity,
            "Unexpected training runtime")
    if output.exists():
        require(resume, "Use explicit --resume for an existing comparison")
        require(read_json(output / "plan.json") == plan, "Validation plan or sources changed")
        require(sha256(output / "finalists/manifest.json") == plan["finalist_manifest_sha256"],
                "Copied finalist manifest changed")
        finalists(output / "finalists")
        sessions = read_json(output / "suite.json")["sessions"]
    else:
        require(not resume, "Cannot resume a missing comparison")
        output.mkdir(parents=True)
        shutil.copytree(frozen, output / "finalists")
        write_json(output / "plan.json", plan)
        sessions = []
    if (output / "environment.json").exists():
        require(read_json(output / "environment.json") == environment, "Training runtime changed")
    else:
        write_json(output / "environment.json", environment)
    started, added = time.monotonic(), 0
    suite = {"schema_version": 1, "status": "running", "rows": [], "sessions": sessions,
             "planned_trials": len(manifest["candidates"]) * len(profile["seeds"]),
             "completed_trials": 0}
    write_json(output / "suite.json", suite)
    vectors = {}
    try:
        for seed in profile["seeds"]:
            for candidate in manifest["candidates"]:
                base = output / "trials" / candidate["id"] / f"seed_{seed}"
                training = completed_attempt(base / "training")
                analysis = completed_attempt(base / "analysis")
                existing = training is not None and analysis is not None
                if not existing and max_trials is not None and added >= max_trials:
                    continue
                print(f"{'Checking' if existing else 'Running'} {candidate['id']}/seed_{seed}",
                      flush=True)
                if training is None:
                    training = next_attempt(base / "training")
                    run_experiment(profile=profile, method="ga", seed=seed, trial=seed + 1,
                                   ga_settings=candidate["settings"], output_dir=training,
                                   upstream=upstream, python=python, timeout=timeout)
                if not (training / "receipt.json").exists():
                    # Recover the narrow crash window after completed training, before its receipt.
                    # Existing receipts are always verified, never replaced.
                    checked = validate_trial(training, profile=profile, candidate=candidate,
                                             seed=seed, receipt=False)
                    write_json(training / "receipt.json", checked["artifact_sha256"])
                checked = validate_trial(training, profile=profile, candidate=candidate, seed=seed)
                if seed in vectors:
                    require(vectors[seed] == checked["noise_vectors"], "Unmatched task vectors")
                vectors[seed] = checked["noise_vectors"]
                if analysis is None:
                    analysis = next_attempt(base / "analysis")
                    run_analysis(run_dir=training, output_dir=analysis, eval_seed=900000 + seed,
                                 upstream=upstream, python=python, episodes=10, timeout=timeout)
                analyzed = validate_analysis(run_dir=training, output_dir=analysis, episodes=10,
                                             eval_seed=900000 + seed)
                phases = analyzed["phase_returns"]
                switches = [{"from_phase": i, "to_phase": i + 1,
                             "task": phases[i]["task"], "before": phases[i]["own_mean"],
                             "after": phases[i + 1]["previous_mean"],
                             "forgetting": phases[i]["own_mean"] - phases[i + 1]["previous_mean"]}
                            for i in range(len(phases) - 1)]
                suite["rows"].append({"candidate_id": candidate["id"], "seed": seed,
                                      "trial": seed + 1, "eval_seed": 900000 + seed,
                                      "normalized_score": checked["normalized_score"],
                                      "training_wall_seconds": checked["wall_seconds"],
                                      "analysis_wall_seconds": read_json(
                                          analysis / "manifest.json")["wall_seconds"],
                                      "training_path": str(training.relative_to(output)),
                                      "analysis_path": str(analysis.relative_to(output)),
                                      "analysis": analyzed, "switches": switches})
                suite["completed_trials"] = len(suite["rows"])
                added += int(not existing)
                write_json(output / "suite.json", suite)
        suite["status"] = ("complete" if suite["completed_trials"] == suite["planned_trials"]
                           else "partial")
        suite.update(aggregate(suite["rows"], manifest))
    except BaseException as exc:
        suite.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        sessions.append({"wall_seconds": time.monotonic() - started, "new_completed_trials": added,
                         "timeout_per_process": timeout, "status": suite["status"]})
        suite["wall_seconds"] = sum(s["wall_seconds"] for s in sessions)
        write_json(output / "suite.json", suite)
    return suite


def export_validation(runs_root: Path, output: Path, *, allow_partial: bool = False) -> dict:
    """Reconstruct finalized validation claims and preserve all attempted work."""
    import hashlib
    import tempfile

    runs_root, output = runs_root.resolve(), output.resolve()
    require(not output.exists(), "Refusing to overwrite validation evidence")
    require(not output.is_relative_to(runs_root), "Export must be outside the result directory")
    plan, suite = read_json(runs_root / "plan.json"), read_json(runs_root / "suite.json")
    require(suite["status"] != "running", "Wait for a saved validation stage")
    require(allow_partial or suite["status"] == "complete", "Validation is incomplete")
    frozen = finalists(runs_root / "finalists")
    require(sha256(runs_root / "finalists/manifest.json") == plan["finalist_manifest_sha256"],
            "Finalist handoff changed")
    require(plan["profile"] == frozen["profile"], "Validation profile changed")
    for name, digest in plan["source_sha256"].items():
        require(sha256(REPO_ROOT / name) == digest, f"Validation source changed: {name}")
    candidates = {c["id"]: c for c in frozen["candidates"]}
    require(list(candidates) == plan["candidate_ids"], "Candidate order changed")
    jobs = [(c, seed) for seed in plan["profile"]["seeds"] for c in candidates]
    rows = suite["rows"]
    require([(r["candidate_id"], r["seed"]) for r in rows] == jobs[:len(rows)],
            "Validation rows are not the frozen job prefix")
    require(len(rows) == suite["completed_trials"] <= len(jobs) == suite["planned_trials"],
            "Validation count mismatch")
    require((suite["status"] == "complete") == (len(rows) == len(jobs)),
            "Incorrect completion status")
    vectors = {}
    for row in rows:
        candidate, seed = candidates[row["candidate_id"]], row["seed"]
        expected_root = runs_root / "trials" / candidate["id"] / f"seed_{seed}"
        training, analysis = (runs_root / row[k] for k in ("training_path", "analysis_path"))
        require(training.resolve() == completed_attempt(expected_root / "training").resolve()
                and analysis.resolve() == completed_attempt(expected_root / "analysis").resolve(),
                "Trial artifact path mismatch")
        checked = validate_trial(training, profile=plan["profile"], candidate=candidate, seed=seed)
        analyzed = validate_analysis(run_dir=training, output_dir=analysis, episodes=10,
                                     eval_seed=900000 + seed)
        require(row["normalized_score"] == checked["normalized_score"]
                and row["training_wall_seconds"] == checked["wall_seconds"]
                and row["analysis"] == analyzed
                and row["trial"] == seed + 1 and row["eval_seed"] == 900000 + seed
                and row["analysis_wall_seconds"] == read_json(
                    analysis / "manifest.json")["wall_seconds"], "Stale trial summary")
        phases = analyzed["phase_returns"]
        switches = [{"from_phase": i, "to_phase": i + 1, "task": phases[i]["task"],
                     "before": phases[i]["own_mean"], "after": phases[i + 1]["previous_mean"],
                     "forgetting": phases[i]["own_mean"] - phases[i + 1]["previous_mean"]}
                    for i in range(len(phases) - 1)]
        require(row["switches"] == switches, "Switch differences changed")
        if seed in vectors:
            require(checked["noise_vectors"] == vectors[seed], "Task vectors differ across finalists")
        vectors[seed] = checked["noise_vectors"]
    computed = aggregate(rows, frozen)
    if suite["status"] in ("partial", "complete"):
        require(all(suite[k] == v for k, v in computed.items()), "Finalist aggregation changed")
    costs = []
    steps = nominal_training_steps(plan["profile"], "ga")
    for path in sorted((runs_root / "trials").glob("*/seed_*/training/attempt_*/manifest.json")):
        attempt = read_json(path)
        wall = attempt["wall_seconds"]
        require(type(wall) in (int, float) and math.isfinite(wall) and wall >= 0,
                "Invalid attempted training duration")
        costs.append({"path": str(path.parent.relative_to(runs_root)),
                      "status": attempt["status"], "wall_seconds": wall,
                      "nominal_steps_allocated": steps,
                      "nominal_steps_completed": steps if attempt["status"] == "complete" else None})
    wall = sum(x["wall_seconds"] for x in suite["sessions"])
    require(wall == suite["wall_seconds"]
            and all(math.isfinite(x["wall_seconds"]) and x["wall_seconds"] >= 0
                    for x in suite["sessions"]), "Invalid session duration")
    report = {"schema_version": 1, "status": suite["status"], "profile": plan["profile"],
              "upstream_commit": UPSTREAM_COMMIT,
              "finalist_manifest_sha256": plan["finalist_manifest_sha256"],
              "completed_trials": len(rows), "planned_trials": len(jobs), "rows": rows, **computed,
              "training_attempts": costs, "session_wall_seconds": wall,
              "nominal_steps_completed": sum(x["nominal_steps_completed"] or 0 for x in costs),
              "nominal_steps_allocated": sum(x["nominal_steps_allocated"] for x in costs),
              "incomplete_training_attempts": sum(x["status"] != "complete" for x in costs),
              "validation": {"source_hashes_match": True, "artifact_receipts_match": True,
                             "scores_rederived": True, "posthoc_metrics_rederived": True,
                             "switch_differences_rederived": True, "task_vectors_match": True},
              "artifact_policy": "Exact text artifacts with local paths redacted; original and "
                                 "exported hashes retained. Binary checkpoints remain local. "
                                 "Failed attempts remain charged separately."}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        checksums = {}
        for path in sorted(runs_root.rglob("*")):
            if not path.is_file():
                continue
            require(path.resolve().is_relative_to(runs_root), "Artifact escapes result root")
            data = path.read_bytes()
            relative = path.relative_to(runs_root)
            entry = {"source": str(relative), "original_sha256": hashlib.sha256(data).hexdigest(),
                     "original_bytes": len(data)}
            if path.suffix in {".json", ".jsonl", ".txt", ".log", ".py"}:
                require(b"\0" not in data, "NUL byte in text artifact")
                exported = data
                for original, replacement in ((str(runs_root), "<runs>"),
                                              (str(REPO_ROOT), "<repo>"),
                                              (str(Path.home()), "<home>")):
                    exported = exported.replace(original.encode(), replacement.encode())
                destination = staging / "raw" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(exported)
                entry.update(exported=True, exported_sha256=hashlib.sha256(exported).hexdigest(),
                             exported_bytes=len(exported), redacted=exported != data)
            else:
                entry.update(exported=False, reason="Binary artifact retained locally")
            checksums[str(Path("raw") / relative)] = entry
        write_json(staging / "summary.json", report)
        write_json(staging / "checksums.json", checksums)
        require(not output.exists(), "Refusing to overwrite report")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report
