"""Frozen adaptive selection studies with verified, canonical-AST evaluation reuse."""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

from shinka_crl.adaptive import GRAMMAR_VERSION, SOURCE_MAX_BYTES, load_program
from shinka_crl.adaptive_objective import score_adaptive_trial
from shinka_crl.analysis import run_analysis, validate_analysis
from shinka_crl.baseline_contract import validate_baseline_config
from shinka_crl.experiment import (
    DEFAULT_PYTHON, DEFAULT_UPSTREAM, REPO_ROOT, UPSTREAM_COMMIT, build_command, load_profile,
    nominal_training_steps, run_experiment, score_curve, verify_upstream,
)
from shinka_crl.pilot import read_json, require, sha256, write_json
from shinka_crl.reference_timing import artifact_files, artifact_hashes, runtime_probe, utc_now
from shinka_crl.search import SEARCH_THREAD_ENV

VERSION = "adaptive-cartpole-v1"
PARTITION_VERSION = "adaptive-cartpole-v2"
HISTORICAL_ALLOCATION = "docs/adaptive-seed-allocation-20261003.json"
WINNERS = "reports/validation-static-20261002/summary.json"
FINALISTS = "reports/finalists-static-20261002"
CONTROL_IDS = ("identity", "arithmetic", "focus", "static_shinka11", "static_random24")
LAUNCHER_SOURCES = {
    "scripts/adaptive_train.py", "src/shinka_crl/adaptive.py", "src/shinka_crl/experiment.py",
    "src/shinka_crl/baseline_contract.py", "src/shinka_crl/search.py", "src/shinka_crl/pilot.py",
    "requirements/cpu.lock", "upstream.lock.json",
}
SOURCE_FILES = (
    "src/shinka_crl/adaptive_evaluation.py", "src/shinka_crl/adaptive_objective.py",
    "src/shinka_crl/adaptive.py", "scripts/adaptive_train.py",
    "scripts/run_adaptive_controls.py", "scripts/report_adaptive_controls.py",
    "tasks/cartpole_adaptive/evaluate.py", "tasks/cartpole_adaptive/initial.py",
    "tasks/cartpole_adaptive/arithmetic.py", "src/shinka_crl/experiment.py",
    "src/shinka_crl/analysis.py", "src/shinka_crl/baseline_contract.py",
    "src/shinka_crl/pilot.py", "src/shinka_crl/search.py", "src/shinka_crl/reference_timing.py",
    "src/shinka_crl/profiles/adaptive-search.json",
    "src/shinka_crl/profiles/adaptive-validation.json", "requirements/cpu.lock", "upstream.lock.json",
    HISTORICAL_ALLOCATION,
    WINNERS,
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def source_hashes() -> dict:
    return {name: sha256(REPO_ROOT / name) for name in SOURCE_FILES}


def runtime_fingerprint(python):
    runtime = runtime_probe(python, dict(os.environ))
    runtime.update(platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                   numerical_environment={key: os.environ.get(key) for key in
                   ("XLA_FLAGS", "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION",
                    "JAX_NUM_CPU_DEVICES", "PYTHONHASHSEED", "PYTHONPATH")})
    return runtime


@contextmanager
def runtime_scope(affinity):
    previous = os.sched_getaffinity(0)
    environment = {key: os.environ.get(key) for key in SEARCH_THREAD_ENV}
    require(set(affinity).issubset(previous) and len(affinity) == 2, "Frozen CPUs unavailable")
    try:
        os.sched_setaffinity(0, affinity)
        os.environ.update(SEARCH_THREAD_ENV)
        yield
    finally:
        try:
            os.sched_setaffinity(0, previous)
        finally:
            for key, value in environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def controls() -> list[dict]:
    declared = []
    for name, filename in (("identity", "initial.py"), ("arithmetic", "arithmetic.py")):
        path = f"tasks/cartpole_adaptive/{filename}"
        declared.append({"id": name, "variant": "ga_adaptive", "source": path,
                         "source_sha256": sha256(REPO_ROOT / path), "settings": None})
    declared.append({"id": "focus", "variant": "ga_focus", "source": None,
                     "source_sha256": None, "settings": None})
    winners = read_json(REPO_ROOT / WINNERS)["selected_winners"]
    for arm, name in (("shinka", "static_shinka11"), ("random", "static_random24")):
        winner = winners[arm]
        source = f"{FINALISTS}/{winner['program_path']}"
        require(sha256(REPO_ROOT / source) == winner["program_sha256"], "Frozen static source changed")
        declared.append({"id": name, "variant": "ga_static", "source": source,
                         "source_sha256": winner["program_sha256"], "settings": winner["settings"]})
    return declared


def _partition(seeds, size, name):
    require(isinstance(seeds, list) and len(seeds) == size
            and all(type(seed) is int and 0 <= seed < 2**32 - 900000 for seed in seeds)
            and len(set(seeds)) == size,
            f"{name} requires {size} distinct seeds with valid derived trial/evaluation IDs")
    return {"seeds": seeds, "trials": [seed + 1 for seed in seeds],
            "eval_seeds": [seed + 900000 for seed in seeds]}


def _validate_allocation(evidence, development_seeds, validation_seeds):
    """Check task identities as well as training/evaluation RNG identities."""
    partitions = {"adaptive_search": _partition(development_seeds, 3, "Development"),
                  "adaptive_validation": _partition(validation_seeds, 5, "Validation")}
    require(evidence.get("schema_version") == 1 and evidence.get("invalid_json") == []
            and evidence.get("proposed_existing_hits") == []
            and evidence.get("proposed") == partitions,
            "Seed allocation audit is incomplete or differs from the requested partitions")
    observed = evidence.get("observed", {})
    for field in ("seed", "seeds", "trial", "eval_seed"):
        require(isinstance(observed.get(field), list)
                and all(type(value) is int and value >= 0 for value in observed[field]),
                f"Seed allocation audit lacks observed {field} identities")
    history = read_json(REPO_ROOT / HISTORICAL_ALLOCATION)
    historical = history["observed"]
    reserved = {
        "seeds": set(historical["seed"] + historical["seeds"] + list(range(42, 52))),
        "trials": set(historical["trial"] + list(range(1, 11))),
        "eval_seeds": set(historical["eval_seed"] + list(range(900042, 900052))),
    }
    for partition in history["proposed"].values():
        for field in reserved:
            reserved[field].update(partition[field])
    reserved["seeds"].update(observed["seed"] + observed["seeds"])
    reserved["trials"].update(observed["trial"])
    reserved["eval_seeds"].update(observed["eval_seed"])
    for field in reserved:
        development = set(partitions["adaptive_search"][field])
        validation = set(partitions["adaptive_validation"][field])
        require(development.isdisjoint(validation), f"Development and validation {field} overlap")
        require((development | validation).isdisjoint(reserved[field]),
                f"New {field} overlap existing or reserved experiment identities")
    return partitions


def freeze_study(*, output: Path, upstream: Path = DEFAULT_UPSTREAM,
                 python: str = str(DEFAULT_PYTHON), timeout: int = 1800,
                 development_seeds: list[int] | None = None,
                 validation_seeds: list[int] | None = None,
                 seed_allocation: Path | None = None) -> dict:
    require(type(timeout) is int and timeout > 0, "Timeout must be a positive integer")
    output, upstream = Path(output).resolve(), Path(upstream).resolve()
    require(not output.exists(), "Study must use a fresh directory")
    partitioned = any(value is not None for value in
                      (development_seeds, validation_seeds, seed_allocation))
    allocation_path = None
    if partitioned:
        require(all(value is not None for value in
                    (development_seeds, validation_seeds, seed_allocation)),
                "Explicit development and validation seeds require a seed allocation audit")
        allocation_path = Path(seed_allocation).resolve()
        _validate_allocation(read_json(allocation_path), development_seeds, validation_seeds)
        allocation_sha256 = sha256(allocation_path)
    verify_upstream(upstream)
    interpreter = shutil.which(python)
    require(interpreter is not None, "Missing training interpreter")
    python = str(Path(interpreter).absolute())
    affinity = sorted(os.sched_getaffinity(0))[:2]
    with runtime_scope(affinity):
        runtime = runtime_fingerprint(python)
    require(runtime["jax_backend"] == "cpu", "Adaptive protocol requires CPU JAX")
    profile, reserved = load_profile("adaptive-search"), load_profile("adaptive-validation")
    require(profile["seeds"] == [4001, 4002, 4003] and reserved["seeds"] == list(range(5001, 5006)),
            "Adaptive seed partition changed")
    if partitioned:
        profile["seeds"], reserved["seeds"] = list(development_seeds), list(validation_seeds)
    plan = {"schema_version": 1,
            "protocol_version": PARTITION_VERSION if partitioned else VERSION, "profile": profile,
            "reserved_validation_profile": reserved, "trial_offset": 1, "eval_seed_offset": 900000,
            "posthoc_episodes": 10, "objective_version": "adaptive-active-previous-v1",
            "objective_weights": {"active": .5, "previous": .5}, "grammar_version": GRAMMAR_VERSION,
            "upstream_commit": UPSTREAM_COMMIT, "upstream": str(upstream), "python": python,
            "timeout_seconds": timeout, "cpu_affinity": affinity, "thread_environment": SEARCH_THREAD_ENV,
            "runtime": runtime, "source_sha256": source_hashes(), "controls": controls(),
            "cache_policy": "Complete verified candidate evaluations only; canonical AST plus full context; "
                            "duplicate requests consume slots, never extra training; failed attempts retained",
            "outer_search_policy": "Separate Shinka archive; staged 5/13/25 total slots; each repeated or "
                                   "invalid proposal consumes its slot; no model calls in control study"}
    if partitioned:
        plan["seed_allocation"] = {"source_path": str(allocation_path),
                                   "frozen_path": "seed-allocation.json", "sha256": allocation_sha256}
    output.mkdir(parents=True)
    if partitioned:
        shutil.copyfile(allocation_path, output / "seed-allocation.json")
        require(sha256(output / "seed-allocation.json") == allocation_sha256,
                "Seed allocation audit changed while freezing")
    (output / "programs").mkdir()
    for control in plan["controls"]:
        if control["source"]:
            shutil.copyfile(REPO_ROOT / control["source"], output / "programs" / f"{control['id']}.py")
    write_json(output / "plan.json", plan)
    write_json(output / "plan-receipt.json", {"plan_sha256": sha256(output / "plan.json"),
               "program_sha256": {p.name: sha256(p) for p in sorted((output / "programs").glob("*.py"))}})
    return plan


def read_plan(study: Path) -> dict:
    plan = read_json(study / "plan.json")
    receipt = read_json(study / "plan-receipt.json")
    require(receipt == {"plan_sha256": sha256(study / "plan.json"),
                       "program_sha256": {p.name: sha256(p) for p in sorted((study / "programs").glob("*.py"))}},
            "Study plan or frozen controls changed")
    require(plan["protocol_version"] in (VERSION, PARTITION_VERSION)
            and plan["source_sha256"] == source_hashes(),
            "Evaluation sources changed; use the recorded revision")
    profile, reserved = load_profile("adaptive-search"), load_profile("adaptive-validation")
    if plan["protocol_version"] == PARTITION_VERSION:
        allocation = plan.get("seed_allocation", {})
        require(set(allocation) == {"source_path", "frozen_path", "sha256"}
                and allocation["frozen_path"] == "seed-allocation.json"
                and isinstance(allocation["source_path"], str)
                and sha256(study / "seed-allocation.json") == allocation["sha256"],
                "Frozen seed allocation audit changed")
        _validate_allocation(read_json(study / "seed-allocation.json"),
                             plan["profile"]["seeds"], plan["reserved_validation_profile"]["seeds"])
        profile["seeds"] = plan["profile"]["seeds"]
        reserved["seeds"] = plan["reserved_validation_profile"]["seeds"]
    else:
        require("seed_allocation" not in plan, "Legacy study cannot override its seed allocation")
    require(plan["profile"] == profile and plan["reserved_validation_profile"] == reserved
            and plan["controls"] == controls(), "Frozen study protocol changed")
    require(plan["objective_weights"] == {"active": .5, "previous": .5}
            and plan["objective_version"] == "adaptive-active-previous-v1"
            and plan["grammar_version"] == GRAMMAR_VERSION
            and plan["posthoc_episodes"] == 10 and plan["trial_offset"] == 1
            and plan["eval_seed_offset"] == 900000 and plan["upstream_commit"] == UPSTREAM_COMMIT
            and plan["thread_environment"] == SEARCH_THREAD_ENV, "Unsupported evaluation protocol")
    return plan


def candidate_identity(candidate: dict) -> dict:
    if candidate["variant"] == "ga_adaptive":
        return {"variant": "ga_adaptive", "canonical_ast_sha256": candidate["program"]["canonical_ast_sha256"]}
    return {"variant": candidate["variant"], "settings": candidate["settings"]}


def cache_key(plan: dict, candidate: dict) -> str:
    return digest({"context_sha256": digest(plan), "candidate": candidate_identity(candidate)})


def command_for(plan, candidate, seed, training, program):
    command = build_command(profile=plan["profile"], method="ga_focus" if candidate["variant"] == "ga_focus" else "ga",
                            seed=seed, trial=seed + 1, output_dir=training,
                            upstream=Path(plan["upstream"]), python=plan["python"], ga_settings=candidate["settings"])
    if candidate["variant"] == "ga_adaptive":
        command = [plan["python"], str(REPO_ROOT / "scripts/adaptive_train.py"),
                   "--upstream", plan["upstream"], "--program-path", str(program),
                   "--receipt-path", str(training / "adaptive-manifest.json"), "--", *command[2:]]
    return command


def _run_trial(plan, candidate, seed, training, program):
    if candidate["variant"] != "ga_adaptive":
        return run_experiment(profile=plan["profile"], method="ga_focus" if candidate["variant"] == "ga_focus" else "ga",
                              seed=seed, trial=seed + 1, output_dir=training,
                              upstream=Path(plan["upstream"]), python=plan["python"],
                              ga_settings=candidate["settings"], timeout=plan["timeout_seconds"])
    command = command_for(plan, candidate, seed, training, program)
    training.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "profile": plan["profile"], "method": "ga", "algorithm_variant": "ga_adaptive",
                "seed": seed, "trial": seed + 1, "upstream_commit": UPSTREAM_COMMIT,
                "ga_settings": None, "program": candidate["program"], "command": command,
                "source_sha256": plan["source_sha256"], "jax_platforms": "cpu"}
    write_json(training / "manifest.json", manifest)
    started = time.monotonic()
    try:
        with (training / "process.log").open("w") as log:
            subprocess.run(command, cwd=plan["upstream"], stdout=log, stderr=subprocess.STDOUT,
                           timeout=plan["timeout_seconds"], check=True,
                           env={**os.environ, "PYTHONUNBUFFERED": "1"})
        score = score_curve(read_json(training / "training_metrics.json"), profile=plan["profile"], method="ga")
        write_json(training / "summary.json", {**score, "profile": plan["profile"]["name"], "method": "ga",
                   "seed": seed, "trial": seed + 1, "upstream_commit": UPSTREAM_COMMIT})
        manifest.update(status="complete", metrics_sha256=sha256(training / "training_metrics.json"))
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        write_json(training / "manifest.json", manifest)


def _validate_settings(config: dict, candidate: dict, population: int):
    if candidate["variant"] != "ga_static":
        validate_baseline_config(config, "ga_focus" if candidate["variant"] == "ga_focus" else "ga")
        return
    settings = candidate["settings"]
    elites = max(1, int(population * settings["elite_ratio"]))
    resolved = {"refresh": True, "num_elites": elites, "num_offspring": population - elites,
                "variation": "gaussian", "sigma": settings["sigma"], "cross_over_rate": 0.}
    require(config.get("sigma") == settings["sigma"] and config.get("searcher_kwargs") == {
        "elite_ratio": settings["elite_ratio"], "init_around_mean": False}
        and config.get("searcher_resolved") == resolved, "Static winner settings changed")
    normalized = {**config, "sigma": .5, "searcher_kwargs": {"elite_ratio": .5, "init_around_mean": False},
                  "searcher_resolved": {**resolved, "sigma": .5, "num_elites": population // 2,
                                        "num_offspring": population // 2}}
    validate_baseline_config(normalized, "ga")


def _score_trial(plan, candidate, seed, training, analysis, program):
    manifest = read_json(training / "manifest.json")
    method = "ga_focus" if candidate["variant"] == "ga_focus" else "ga"
    require(manifest["status"] == "complete" and manifest["profile"] == plan["profile"]
            and manifest["seed"] == seed and manifest["trial"] == seed + 1
            and manifest["method"] == method and manifest["upstream_commit"] == UPSTREAM_COMMIT
            and manifest["ga_settings"] == candidate["settings"]
            and manifest["command"] == command_for(plan, candidate, seed, training, program),
            "Training protocol differs from frozen candidate")
    require(sha256(training / "training_metrics.json") == manifest["metrics_sha256"], "Training metrics changed")
    results = read_json(training / "results.json")
    require(read_json(training / "config.json") == results["config"], "Resolved configuration copies differ")
    _validate_settings(results["config"], candidate, plan["profile"]["ne"]["pop_size"])
    require(results["config"]["schedule"] == "switch" and results["config"]["noise_range"] == .5,
            "Task generation settings changed")
    if candidate["variant"] == "ga_adaptive":
        receipt = read_json(training / "adaptive-manifest.json")
        command = command_for(plan, candidate, seed, training, program)
        require(receipt["status"] == "complete" and receipt["program"] == candidate["program"]
                and receipt["program_raw_sha256"] == candidate["source_sha256"]
                and receipt["program_path"] == str(program)
                and receipt["launcher_invocation"] == command
                and receipt["native_args"] == command[command.index("--") + 1:]
                and receipt["algorithm_variant"] == "ga_adaptive"
                and receipt["upstream_commit"] == UPSTREAM_COMMIT,
                "Adaptive launcher receipt mismatch")
        require(receipt["jax_backend"] == "cpu" and receipt["cpu_affinity"] == plan["cpu_affinity"]
                and receipt["thread_environment"] == plan["thread_environment"]
                and receipt["runtime"]["python"] == plan["runtime"]["python"], "Adaptive runtime mismatch")
        for name in ("jax", "jaxlib", "numpy"):
            require(receipt["runtime"][name] == plan["runtime"]["packages"][name], "Numerical runtime mismatch")
        require(set(receipt["source_sha256"]) == LAUNCHER_SOURCES and all(plan["source_sha256"].get(name) == value
                for name, value in receipt["source_sha256"].items()), "Adaptive source receipt mismatch")
        require(set(receipt["artifact_sha256"]) == {"results.json", "config.json", "training_metrics.json",
                                                  "checkpoints.npz", "trajectory.npz"}
                and all(sha256(training / name) == value for name, value in receipt["artifact_sha256"].items()),
                "Adaptive native artifacts changed")
        widths = [r["sigma"] for r in read_json(training / "training_metrics.json")]
        count = plan["profile"]["ne"]["num_generations"]
        require(len(widths) == count and all(type(v) in (int, float) and math.isfinite(v)
                and .001 <= v <= 2 for v in widths), "Adaptive raw mutation width is invalid")
        require(receipt["completed_generations"] == count and receipt["invalid_update"] is False
                and receipt["state_initialized"] is True and receipt["memory_final_finite"] is True
                and len(receipt["memory_final"]) == 4
                and all(type(v) in (int, float) and math.isfinite(v) for v in receipt["memory_final"])
                and receipt["sigma_next_final_finite"] is True
                and receipt["sigma_next_final"] == widths[-1], "Adaptive final state is incomplete or invalid")
        validation = receipt["width_validation"]
        require(validation["completed_generations"] == count == validation["raw_rows"]
                and validation["matched_host_observations"] is True
                and validation["final_update_validated"] is True
                and validation["raw_sigma_timing"] == "after_tell_next_generation"
                and validation["sigma_used_initial"] == .5
                and validation["sigma_used_final"] == ([.5, *widths[:-1]])[-1]
                and validation["sigma_next_final"] == widths[-1]
                and validation["sigma_next_min"] == min(widths) and validation["sigma_next_max"] == max(widths)
                and validation["sigma_used_sha256"] == hashlib.sha256(json.dumps([.5, *widths[:-1]],
                                                               separators=(",", ":")).encode()).hexdigest(),
                "Adaptive width evidence mismatch")
    verified = validate_analysis(run_dir=training, output_dir=analysis,
                                 episodes=plan["posthoc_episodes"], eval_seed=seed + 900000)
    return score_adaptive_trial(profile=plan["profile"], records=read_json(training / "training_metrics.json"),
                                analysis=verified, evaluation=read_json(analysis / "evaluation.json"),
                                method=method, seed=seed, trial=seed + 1, eval_seed=seed + 900000,
                                episodes=plan["posthoc_episodes"])


def describe(values):
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values)}


def aggregate(rows):
    return {"scores": {key: describe([row["score"][key] for row in rows]) for key in
                       ("combined_score", "active_score", "previous_score")},
            "reference_metrics": {key: describe([row["score"]["reference_metrics"][key] for row in rows])
                                  for key in ("learning_accuracy", "forgetting", "learning_minus_forgetting",
                                              "zero_shot_transfer", "normalized_curve_average")}}


def validate_cache(attempt: Path, plan: dict, expected_key: str) -> dict:
    require(artifact_hashes(attempt) == read_json(attempt / "receipt.json"), "Cache artifact receipt mismatch")
    summary = read_json(attempt / "summary.json")
    require(summary["status"] == "complete" and summary["context_sha256"] == digest(plan)
            and summary["cache_key"] == expected_key == cache_key(plan, summary["candidate"]),
            "Cache context or candidate changed")
    candidate = summary["candidate"]
    program = attempt / "program.py"
    if candidate["source_sha256"]:
        require(sha256(program) == candidate["source_sha256"], "Cached source changed")
    if candidate["variant"] == "ga_adaptive":
        require(load_program(program).metadata() == candidate["program"], "Cached AST identity changed")
    require([row["seed"] for row in summary["trials"]] == plan["profile"]["seeds"], "Cache seeds incomplete")
    for row in summary["trials"]:
        seed = row["seed"]
        training, analysis = attempt / f"seed_{seed}/training", attempt / f"seed_{seed}/analysis"
        require(_score_trial(plan, candidate, seed, training, analysis, program) == row["score"],
                "Cached score differs from raw verified evidence")
        require(row["training_wall_seconds"] == read_json(training / "manifest.json")["wall_seconds"]
                and row["analysis_wall_seconds"] == read_json(analysis / "manifest.json")["wall_seconds"],
                "Cached execution cost changed")
    require(aggregate(summary["trials"]) == summary["aggregate"], "Cached aggregate changed")
    return summary


def _train_candidate(attempt, plan, candidate, source):
    attempt.mkdir(parents=True, exist_ok=False)
    if source:
        shutil.copyfile(source, attempt / "program.py")
    summary = {"status": "running", "cache_key": cache_key(plan, candidate), "context_sha256": digest(plan),
               "candidate": candidate, "trials": [], "started_at": utc_now()}
    write_json(attempt / "summary.json", summary)
    started = time.monotonic()
    try:
        for seed in plan["profile"]["seeds"]:
            print(f"  {candidate['id']}: training seed {seed}", flush=True)
            training, analysis = attempt / f"seed_{seed}/training", attempt / f"seed_{seed}/analysis"
            _run_trial(plan, candidate, seed, training, attempt / "program.py")
            run_analysis(run_dir=training, output_dir=analysis, eval_seed=seed + 900000,
                         upstream=Path(plan["upstream"]), python=plan["python"],
                         episodes=plan["posthoc_episodes"], timeout=plan["timeout_seconds"])
            score = _score_trial(plan, candidate, seed, training, analysis, attempt / "program.py")
            summary["trials"].append({"seed": seed, "trial": seed + 1, "eval_seed": seed + 900000,
                                      "score": score,
                                      "training_wall_seconds": read_json(training / "manifest.json")["wall_seconds"],
                                      "analysis_wall_seconds": read_json(analysis / "manifest.json")["wall_seconds"]})
            write_json(attempt / "summary.json", summary)
        require(plan["source_sha256"] == source_hashes(), "Sources changed during evaluation")
        verify_upstream(Path(plan["upstream"]))
        summary.update(status="complete", aggregate=aggregate(summary["trials"]))
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary.update(wall_seconds=time.monotonic() - started, finished_at=utc_now())
        write_json(attempt / "summary.json", summary)
        write_json(attempt / "receipt.json", artifact_hashes(attempt))
    return summary


def evaluate_candidate(*, study: Path, request_dir: Path, program_path: Path | None = None,
                       control: str | None = None) -> dict:
    """One consumed evaluation slot. Cache reuse still produces a fresh request receipt."""
    study, request_dir = Path(study).resolve(), Path(request_dir).resolve()
    require((program_path is None) != (control is None), "Specify one program or fixed control")
    require(not request_dir.exists() or not any(request_dir.iterdir()), "Request output must be fresh or empty")
    plan = read_plan(study)
    request_dir.mkdir(parents=True, exist_ok=True)
    request = {"status": "running", "request_id": request_dir.name, "context_sha256": digest(plan), "cache_hit": False,
               "slot_consumed": True, "started_at": utc_now(), "evaluator_model_calls": 0}
    started = time.monotonic()
    try:
        if control is not None:
            candidates = [c for c in plan["controls"] if c["id"] == control]
            require(len(candidates) == 1, "Unknown frozen control")
            candidate = copy.deepcopy(candidates[0])
            source = study / "programs" / f"{control}.py" if candidate["source"] else None
        else:
            source = Path(program_path).resolve()
            require(source.stat().st_size <= SOURCE_MAX_BYTES, "Adaptive source exceeds the frozen byte limit")
            candidate = {"id": "proposal", "variant": "ga_adaptive", "source": None,
                         "source_sha256": sha256(source), "settings": None}
        if source:
            shutil.copyfile(source, request_dir / "program.py")
            source = request_dir / "program.py"
            require(sha256(source) == candidate["source_sha256"], "Candidate changed while copying")
        if candidate["variant"] == "ga_adaptive":
            candidate["program"] = load_program(source).metadata()
        request["candidate"] = candidate
        key = cache_key(plan, candidate)
        request["cache_key"] = key
        write_json(request_dir / "request.json", request)
        with (study / "evaluation.lock").open("a") as lock, runtime_scope(plan["cpu_affinity"]):
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            verify_upstream(Path(plan["upstream"]))
            require(runtime_fingerprint(plan["python"]) == plan["runtime"], "Evaluation runtime changed")
            cache_root = study / "cache" / key
            attempts = sorted(cache_root.glob("attempt_*")) if cache_root.exists() else []
            for saved in attempts:
                require((saved / "receipt.json").exists()
                        and artifact_hashes(saved) == read_json(saved / "receipt.json"),
                        "Existing cache attempt is incomplete or corrupted; preserve it for review")
            completed = [a for a in attempts if (a / "summary.json").exists()
                         and read_json(a / "summary.json").get("status") == "complete"]
            if completed:
                attempt = completed[0]
                result = validate_cache(attempt, plan, key)
                request["cache_hit"] = True
            else:
                attempt = cache_root / f"attempt_{len(attempts) + 1:04d}"
                request["cache_origin"] = str(attempt.relative_to(study))
                write_json(request_dir / "request.json", request)
                _train_candidate(attempt, plan, candidate, source)
                result = validate_cache(attempt, plan, key)
            require(plan["source_sha256"] == source_hashes(), "Sources changed during request")
            request.update(status="complete", cache_origin=str(attempt.relative_to(study)),
                           cache_receipt_sha256=sha256(attempt / "receipt.json"), aggregate=result["aggregate"],
                           evaluated_source_sha256=result["candidate"]["source_sha256"])
            steps = nominal_training_steps(plan["profile"], "ga") * len(plan["profile"]["seeds"])
            request.update(new_training_trials=0 if request["cache_hit"] else len(plan["profile"]["seeds"]),
                           new_training_steps_nominal=0 if request["cache_hit"] else steps,
                           avoided_training_steps_nominal=steps if request["cache_hit"] else 0)
    except BaseException as exc:
        request.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        request.update(wall_seconds=time.monotonic() - started, finished_at=utc_now())
        write_json(request_dir / "request.json", request)
        correct = request["status"] == "complete"
        write_json(request_dir / "correct.json", {"correct": correct, "error": request.get("error")})
        write_json(request_dir / "metrics.json", {
            "combined_score": request["aggregate"]["scores"]["combined_score"]["mean"] if correct else 0.,
            "public": {"profile": "adaptive-search", "objective": plan["objective_version"],
                       "cache_hit": request["cache_hit"], "scores": request.get("aggregate", {}).get("scores", {})},
            "private": {"request_file": "request.json", "cache_origin": request.get("cache_origin"),
                        "reference_metrics": request.get("aggregate", {}).get("reference_metrics", {})}})
        write_json(request_dir / "receipt.json", artifact_hashes(request_dir))
    return request


def validate_request(study, request_dir, plan):
    require(artifact_hashes(request_dir) == read_json(request_dir / "receipt.json"), "Request artifacts changed")
    request = read_json(request_dir / "request.json")
    require(request["status"] == "complete" and request["context_sha256"] == digest(plan), "Incomplete or stale request")
    require(request["request_id"] == request_dir.name and request["slot_consumed"] is True
            and type(request["cache_hit"]) is bool and request["evaluator_model_calls"] == 0,
            "Request slot provenance changed")
    candidate = request["candidate"]
    if candidate["source_sha256"]:
        require(sha256(request_dir / "program.py") == candidate["source_sha256"], "Request source changed")
    if candidate["variant"] == "ga_adaptive":
        require(load_program(request_dir / "program.py").metadata() == candidate["program"], "Request program changed")
    require(request["cache_key"] == cache_key(plan, candidate), "Request cache identity changed")
    origin = (study / request["cache_origin"]).resolve()
    require(origin.is_relative_to(study / "cache") and not origin.is_symlink(), "Cache origin escapes study")
    result = validate_cache(origin, plan, request["cache_key"])
    require(sha256(origin / "receipt.json") == request["cache_receipt_sha256"]
            and result["aggregate"] == request["aggregate"]
            and request["evaluated_source_sha256"] == result["candidate"]["source_sha256"],
            "Request differs from cache evidence")
    seeds = len(plan["profile"]["seeds"])
    steps = nominal_training_steps(plan["profile"], "ga") * seeds
    require(request["new_training_trials"] == (0 if request["cache_hit"] else seeds)
            and request["new_training_steps_nominal"] == (0 if request["cache_hit"] else steps)
            and request["avoided_training_steps_nominal"] == (steps if request["cache_hit"] else 0),
            "Request cache work accounting changed")
    require(read_json(request_dir / "correct.json") == {"correct": True, "error": None}
            and read_json(request_dir / "metrics.json")["combined_score"] == result["aggregate"]["scores"]["combined_score"]["mean"],
            "Shinka result contract differs from verified score")
    return request


def run_controls(*, study: Path, max_controls: int | None = None) -> dict:
    """Resume completed controls, then prove a formatting-only identity cache hit."""
    study = Path(study).resolve()
    plan = read_plan(study)
    require(max_controls is None or type(max_controls) is int and max_controls > 0, "Invalid control limit")
    completed = 0
    for control in CONTROL_IDS:
        existing = [p for p in sorted((study / "requests").glob(f"{control}*"))
                    if p.name == control or p.name.startswith(control + "_retry_")]
        successful = [p for p in existing if (p / "request.json").exists()
                      and read_json(p / "request.json")["status"] == "complete"]
        if successful:
            validate_request(study, successful[0], plan)
            continue
        if max_controls is not None and completed >= max_controls:
            break
        print(f"Evaluating frozen control: {control}", flush=True)
        request_dir = study / "requests" / (control if not existing else f"{control}_retry_{len(existing):04d}")
        evaluate_candidate(study=study, request_dir=request_dir, control=control)
        completed += 1
    state = summarize_study(study)
    if set(state["control_requests"]) == set(CONTROL_IDS):
        request_dir = study / "requests" / "identity_format_duplicate"
        if not request_dir.exists():
            formatted = study / "formatted-identity.py"
            require(not formatted.exists(), "Diagnostic formatted source already exists")
            formatted.write_text("# Formatting-only duplicate for verified cache exercise\n\ndef update_sigma( sigma, stats, memory ):\n    return (sigma, memory)\n")
            evaluate_candidate(study=study, request_dir=request_dir, program_path=formatted)
        duplicate = validate_request(study, request_dir, plan)
        identity = state["control_requests"]["identity"]
        require(duplicate["cache_hit"] and duplicate["cache_origin"] == identity["cache_origin"]
                and duplicate["candidate"]["source_sha256"] != identity["candidate"]["source_sha256"],
                "Formatting duplicate did not reuse the original verified evaluation")
    return summarize_study(study)


def summarize_study(study):
    plan = read_plan(study)
    requests, failed_requests, selected = [], [], {}
    for path in sorted((study / "requests").glob("*")):
        if not path.is_dir():
            continue
        require(artifact_hashes(path) == read_json(path / "receipt.json"), "Request evidence changed")
        saved = read_json(path / "request.json")
        if saved["status"] != "complete":
            failed_requests.append(saved)
            continue
        request = validate_request(study, path, plan)
        requests.append(request)
        name = request["candidate"]["id"]
        if name in CONTROL_IDS:
            control = next(c for c in plan["controls"] if c["id"] == name)
            require(all(request["candidate"].get(key) == value for key, value in control.items()),
                    "Control request differs from the frozen source")
            selected.setdefault(name, request)
    duplicates = [r for r in requests if r["request_id"] == "identity_format_duplicate"]
    duplicate_ok = bool(duplicates and "identity" in selected
                        and duplicates[0]["cache_hit"] and duplicates[0]["cache_origin"] == selected["identity"]["cache_origin"]
                        and duplicates[0]["candidate"]["source_sha256"] != selected["identity"]["candidate"]["source_sha256"])
    attempts, training_attempts = [], []
    for path in sorted((study / "cache").glob("*/attempt_*/summary.json")):
        require(artifact_hashes(path.parent) == read_json(path.parent / "receipt.json"), "Cache attempt evidence changed")
        attempts.append(read_json(path))
        for training_path in sorted(path.parent.glob("seed_*/training/manifest.json")):
            manifest = read_json(training_path)
            training_attempts.append({"path": str(training_path.relative_to(study)), "status": manifest["status"],
                                      "seed": manifest["seed"], "wall_seconds": manifest["wall_seconds"]})
    completed_training = sum(t["status"] == "complete" for t in training_attempts)
    steps = nominal_training_steps(plan["profile"], "ga")
    return {"schema_version": 1, "status": "complete" if set(selected) == set(CONTROL_IDS) and duplicate_ok else "partial",
            "protocol_version": plan["protocol_version"], "profile": plan["profile"], "requests": requests,
            "control_requests": selected, "failed_requests": failed_requests,
            "cache_format_check_passed": duplicate_ok, "training_attempts": training_attempts,
            "cache_attempts": len(attempts), "failed_attempts": sum(a["status"] != "complete" for a in attempts),
            "completed_training_trials": completed_training,
            "scored_training_trials": sum(len(a["trials"]) for a in attempts),
            "nominal_steps_completed": completed_training * steps,
            "nominal_steps_allocated": len(training_attempts) * steps,
            "incomplete_training_attempts": len(training_attempts) - completed_training,
            "cache_hits": sum(r["cache_hit"] for r in requests),
            "evaluation_wall_seconds": sum(r["wall_seconds"] for r in requests + failed_requests),
            "interpretation": "Fixed-control development comparison and cache verification; no evolved adaptive programs"}


def export_study(*, study: Path, report_dir: Path):
    study, report_dir = Path(study).resolve(), Path(report_dir).resolve()
    require(not report_dir.exists() and not report_dir.is_relative_to(study), "Report must be new and outside study")
    summary = summarize_study(study)
    before = artifact_hashes(study)
    report_dir.mkdir(parents=True)
    published = {}
    for path in artifact_files(study):
        if path.suffix not in {".json", ".jsonl", ".log", ".py"}:
            continue
        target = report_dir / "raw" / path.relative_to(study)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path.read_text().replace(str(study), "$STUDY").replace(str(REPO_ROOT), "$REPO_ROOT"))
        published[str(target.relative_to(report_dir))] = sha256(target)
    write_json(report_dir / "summary.json", summary)
    published["summary.json"] = sha256(report_dir / "summary.json")
    require(before == artifact_hashes(study), "Evidence changed while exporting")
    write_json(report_dir / "checksums.json", {"original_artifact_sha256": before, "published_sha256": published})
    return summary
