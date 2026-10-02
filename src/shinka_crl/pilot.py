"""Sequential, resumable development pilot with immutable trial attempts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

from shinka_crl.baseline_contract import validate_baseline_config
from shinka_crl.experiment import (
    DEFAULT_PYTHON, DEFAULT_UPSTREAM, REPO_ROOT, UPSTREAM_COMMIT,
    build_command, load_profile, nominal_training_steps as training_steps,
    run_experiment, score_curve, verify_upstream,
)

PROFILES = ("pilot-switching", "pilot-stationary")
METHODS = ("ga", "es", "ppo")
THREAD_ENV = {"JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1",
              "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
TRAINING_FILES = ("manifest.json", "summary.json", "config.json", "results.json",
                  "training_metrics.json", "checkpoints.npz", "process.log", "train.log")


def write_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_training(path: Path, *, profile: dict, method: str, seed: int,
                      trial: int, require_receipt: bool = True) -> dict:
    """Validate resolved protocol and all persisted training artifact hashes."""
    path = Path(path).resolve()
    for name in TRAINING_FILES:
        require((path / name).is_file(), f"Missing training artifact: {path / name}")
    hashes = {name: sha256(path / name) for name in TRAINING_FILES}
    if require_receipt:
        require(read_json(path / "receipt.json") == hashes,
                f"Training artifact changed after validation: {path}")
    manifest = read_json(path / "manifest.json")
    require(manifest.get("status") == "complete", f"Incomplete training: {path}")
    require(manifest.get("profile") == profile, "Training profile mismatch")
    for key, value in {"method": method, "seed": seed, "trial": trial,
                       "upstream_commit": UPSTREAM_COMMIT, "ga_settings": None}.items():
        require(manifest.get(key) == value, f"Training manifest mismatch: {key}")
    require(manifest.get("metrics_sha256") == hashes["training_metrics.json"],
            "Training metric hash mismatch")
    command = manifest.get("command")
    require(isinstance(command, list) and len(command) > 1, "Missing training command")
    upstream = Path(command[1]).parent.parent
    expected_command = build_command(profile=profile, method=method, seed=seed, trial=trial,
                                     output_dir=path, upstream=upstream, python=command[0])
    require(command == expected_command, "Recorded training command does not match protocol")
    records = read_json(path / "training_metrics.json")
    computed = score_curve(records, profile=profile, method=method)
    summary = read_json(path / "summary.json")
    expected_summary = {**computed, "profile": profile["name"], "method": method,
                        "seed": seed, "trial": trial, "upstream_commit": UPSTREAM_COMMIT}
    require(summary == expected_summary, "Training summary differs from raw trajectory")
    results = read_json(path / "results.json")
    config = results["config"]
    validate_baseline_config(config, method)
    budget = profile["ppo"] if method == "ppo" else profile["ne"]
    steps = budget["num_updates" if method == "ppo" else "num_generations"]
    switching = "_sigma" in profile["env"]
    task_sequence = [phase % profile["num_tasks"] for phase in range(profile["num_phases"])]
    expected = {
        "env": "CartPole-v1", "method": method, "seed": seed, "trial": trial,
        "schedule": "switch" if switching else "task0", "num_tasks": profile["num_tasks"],
        "noise_range": float(profile["env"].split("_sigma")[1]) if switching else 0.0,
        "num_generations": steps, "task_interval": budget["task_interval"],
        "task_sequence": task_sequence, "episode_length": profile["episode_length"],
        "eval_episodes": profile["eval_episodes"], "hidden_dims": [16, 16],
    }
    keys = ("num_envs", "num_steps", "num_minibatches") if method == "ppo" else (
        "pop_size", "num_evals")
    expected.update({key: budget[key] for key in keys})
    for key, value in expected.items():
        require(config.get(key) == value, f"Resolved training config mismatch: {key}")
    nominal = training_steps(profile, method)
    require(results.get("env_steps") == nominal, "Nominal training budget mismatch")
    vectors = results.get("noise_vectors")
    require(isinstance(vectors, list) and len(vectors) == profile["num_tasks"],
            "Missing task vectors")
    require(all(isinstance(v, list) and len(v) == 4 and all(
        type(x) in (float, int) and math.isfinite(x) for x in v) for v in vectors),
        "Invalid CartPole task vectors")
    require(vectors[0] == [0, 0, 0, 0], "First task must be clean")
    wall = manifest.get("wall_seconds")
    require(type(wall) in (int, float) and math.isfinite(wall) and wall >= 0,
            "Invalid training duration")
    for name in ("process.log", "train.log"):
        require(b"\0" not in (path / name).read_bytes(), f"Corrupt log: {name}")
    return {"summary": summary, "config": config, "noise_vectors": vectors,
            "nominal_training_steps": nominal, "wall_seconds": wall,
            "artifact_sha256": hashes}


def source_hashes() -> dict:
    names = ["src/shinka_crl/pilot.py", "src/shinka_crl/analysis.py",
             "src/shinka_crl/baseline_contract.py",
             "src/shinka_crl/experiment.py", "scripts/run_pilot.py", "requirements/cpu.lock"]
    names.extend(f"src/shinka_crl/profiles/{name}.json" for name in PROFILES)
    return {name: sha256(REPO_ROOT / name) for name in names}


def make_jobs(profiles: dict) -> list[dict]:
    require(profiles[PROFILES[0]]["seeds"] == profiles[PROFILES[1]]["seeds"],
            "Pilot profiles must share the same ordered seeds")
    jobs = []
    for seed in profiles[PROFILES[0]]["seeds"]:
        for name in PROFILES:
            require(seed in profiles[name]["seeds"], "Pilot profiles must share seeds")
            for method in METHODS:
                jobs.append({"profile": name, "method": method, "seed": seed,
                             "trial": seed + 1, "eval_seed": 900000 + seed})
    return jobs


def next_attempt(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    attempts = list(root.glob("attempt_*"))
    number = max((int(p.name.removeprefix("attempt_")) for p in attempts), default=0) + 1
    return root / f"attempt_{number:03d}"


def completed_attempt(root: Path) -> Path | None:
    for path in sorted(root.glob("attempt_*"), reverse=True):
        try:
            manifest = read_json(path / "manifest.json")
        except (OSError, ValueError):
            continue  # Preserve interrupted attempts and start a fresh directory.
        if manifest.get("status") == "complete":
            return path
    return None


def run_pilot(*, output: Path, upstream: Path = DEFAULT_UPSTREAM,
              python: str = str(DEFAULT_PYTHON), cpus: int = 2, timeout: int = 600,
              resume: bool = False, max_trials: int | None = None) -> dict:
    """Run whole trials sequentially; max_trials bounds new completed trials per session."""
    from shinka_crl.analysis import run_analysis, validate_analysis

    require(cpus > 0 and timeout > 0, "CPU count and timeout must be positive")
    require(max_trials is None or max_trials > 0, "max_trials must be positive")
    output, upstream = Path(output).resolve(), Path(upstream).resolve()
    verify_upstream(upstream)
    interpreter = shutil.which(python)
    require(interpreter is not None, f"Missing Python interpreter: {python}")
    python = str(Path(interpreter).absolute())
    affinity = sorted(os.sched_getaffinity(0))[:cpus]
    os.sched_setaffinity(0, affinity)
    os.environ.update(THREAD_ENV)
    profiles = {name: load_profile(name) for name in PROFILES}
    jobs = make_jobs(profiles)
    plan = {"schema_version": 1, "profiles": profiles, "methods": list(METHODS),
            "jobs": jobs, "eval_episodes": 10, "eval_seed_offset": 900000,
            "upstream_commit": UPSTREAM_COMMIT, "source_sha256": source_hashes(),
            "python": python, "cpu_affinity": affinity, "thread_environment": THREAD_ENV}
    if output.exists():
        require(resume, f"Output exists; explicitly use --resume: {output}")
        require(read_json(output / "plan.json") == plan,
                "Frozen pilot plan/source/runtime changed; use a new results directory")
        sessions = read_json(output / "suite.json").get("sessions", [])
    else:
        require(not resume, "Cannot resume a nonexistent pilot")
        output.mkdir(parents=True)
        write_json(output / "plan.json", plan)
        sessions = []

    probe = (
        "import importlib.metadata as m,json,jax,platform;"
        "print(json.dumps({'python':platform.python_version(),"
        "'jax_backend':jax.default_backend(),'jax_devices':[str(d) for d in jax.devices()],"
        "'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"
    )
    environment = json.loads(subprocess.check_output([python, "-c", probe], text=True, timeout=60))
    require(environment["jax_backend"] == "cpu", "Pilot requires CPU JAX")
    if (output / "environment.json").exists():
        old = read_json(output / "environment.json")
        require(all(old.get(k) == v for k, v in environment.items()),
                "Runtime packages or devices changed since pilot started")
    else:
        environment.update(platform=platform.platform(), upstream_commit=UPSTREAM_COMMIT,
                           cpu_affinity=affinity, source_sha256=plan["source_sha256"],
                           thread_environment=THREAD_ENV)
        environment["harness_commit"] = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
        environment["harness_dirty"] = bool(subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True).strip())
        write_json(output / "environment.json", environment)

    started = time.monotonic()
    suite = {"schema_version": 1, "status": "running", "rows": [], "sessions": sessions,
             "planned_trials": len(jobs), "completed_trials": 0, "wall_seconds": 0.0}
    new_trials = 0
    try:
        for job in jobs:
            name, method, seed = job["profile"], job["method"], job["seed"]
            root = output / "trials" / name / method / f"seed_{seed}"
            training = completed_attempt(root / "training")
            analysis = completed_attempt(root / "analysis")
            existing = training is not None and analysis is not None
            if not existing and max_trials is not None and new_trials >= max_trials:
                continue
            print(f"{'Checking' if existing else 'Running'} {name}/{method}/seed_{seed}", flush=True)
            kwargs = {"profile": profiles[name], "method": method, "seed": seed,
                      "trial": job["trial"]}
            if training is None:
                training = next_attempt(root / "training")
                run_experiment(**kwargs, output_dir=training, upstream=upstream,
                               python=python, timeout=timeout)
                validated = validate_training(training, **kwargs, require_receipt=False)
                write_json(training / "receipt.json", validated["artifact_sha256"])
            validated = validate_training(training, **kwargs)
            if analysis is None:
                analysis = next_attempt(root / "analysis")
                run_analysis(run_dir=training, output_dir=analysis, upstream=upstream,
                             python=python, episodes=10, eval_seed=job["eval_seed"], timeout=timeout)
            analyzed = validate_analysis(run_dir=training, output_dir=analysis,
                                         episodes=10, eval_seed=job["eval_seed"])
            analysis_manifest = read_json(analysis / "manifest.json")
            suite["rows"].append({**job, "training_path": str(training.relative_to(output)),
                                  "analysis_path": str(analysis.relative_to(output)),
                                  "training_wall_seconds": validated["wall_seconds"],
                                  "analysis_wall_seconds": analysis_manifest["wall_seconds"],
                                  "analysis": analyzed})
            suite["completed_trials"] = len(suite["rows"])
            if not existing:
                new_trials += 1
            write_json(output / "suite.json", suite)
        suite["status"] = "complete" if len(suite["rows"]) == len(jobs) else "partial"
    except BaseException as exc:
        suite.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        sessions.append({"wall_seconds": time.monotonic() - started,
                         "new_completed_trials": new_trials, "timeout_per_process": timeout,
                         "status": suite["status"]})
        suite["wall_seconds"] = sum(s["wall_seconds"] for s in sessions)
        write_json(output / "suite.json", suite)
    return suite
