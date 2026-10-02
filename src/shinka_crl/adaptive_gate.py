"""Seven diagnostic trials establishing native identity and adaptive actuation."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import time

from shinka_crl.adaptive import load_program
from shinka_crl.analysis import run_analysis, validate_analysis
from shinka_crl.baseline_contract import validate_baseline_config
from shinka_crl.experiment import (
    DEFAULT_PYTHON, DEFAULT_UPSTREAM, REPO_ROOT, UPSTREAM_COMMIT, build_command,
    load_profile, nominal_training_steps, run_experiment, score_curve, verify_upstream,
)
from shinka_crl.pilot import read_json, require, sha256, write_json
from shinka_crl.reference_timing import artifact_files, artifact_hashes, runtime_probe, utc_now
from shinka_crl.search import SEARCH_THREAD_ENV

SEED, TRIAL, EVAL_SEED = 3001, 3002, 903001
TRIALS = (
    ("plain_stationary", "stationary", None),
    ("identity_stationary", "stationary", "initial.py"),
    ("plain_switching", "switching", None),
    ("identity_switching", "switching", "initial.py"),
    ("halving_switching", "switching", "halving.py"),
    ("arithmetic_switching", "switching", "arithmetic.py"),
    ("focus_switching", "switching", None),
)
SOURCE_FILES = (
    "src/shinka_crl/adaptive.py", "src/shinka_crl/adaptive_gate.py",
    "scripts/adaptive_entrypoint.py", "scripts/check_adaptive_trace.py",
    "scripts/run_adaptive_gate.py", "scripts/report_adaptive_gate.py",
    "src/shinka_crl/experiment.py", "src/shinka_crl/analysis.py",
    "src/shinka_crl/baseline_contract.py", "src/shinka_crl/pilot.py",
    "src/shinka_crl/search.py", "src/shinka_crl/reference_timing.py",
    "src/shinka_crl/profiles/adaptive-gate-stationary.json",
    "src/shinka_crl/profiles/adaptive-gate-switching.json",
    "tasks/cartpole_adaptive/initial.py", "tasks/cartpole_adaptive/halving.py",
    "tasks/cartpole_adaptive/arithmetic.py", "requirements/cpu.lock", "upstream.lock.json",
)


def source_hashes() -> dict:
    return {name: sha256(REPO_ROOT / name) for name in SOURCE_FILES}


def make_plan(*, output: Path, upstream: Path = DEFAULT_UPSTREAM,
              python: str = str(DEFAULT_PYTHON), cpus: int = 2, timeout: int = 600) -> dict:
    require(type(cpus) is int and cpus == 2, "Gate requires exactly two logical CPUs")
    require(type(timeout) is int and timeout > 0, "Timeout must be a positive integer")
    affinity = sorted(os.sched_getaffinity(0))
    require(len(affinity) >= cpus, "Insufficient CPU affinity")
    output, upstream = Path(output).resolve(), Path(upstream).resolve()
    jobs = []
    for name, condition, program in TRIALS:
        profile = load_profile(f"adaptive-gate-{condition}")
        require(profile["seeds"] == [SEED], "Diagnostic seed changed")
        method = "ga_focus" if name == "focus_switching" else "ga"
        training = output / name / "training"
        command = build_command(profile=profile, method=method, seed=SEED, trial=TRIAL,
                                output_dir=training, upstream=upstream, python=python)
        if method == "ga":
            native_args = command[2:] + ["--ne_override", "population_snapshot_interval=1",
                                         "snapshot_members=0"]
            command = [str(python), str(REPO_ROOT / "scripts/adaptive_entrypoint.py"),
                       "--upstream", str(upstream), "--variant", "adaptive" if program else "plain",
                       "--trace-dir", str(training / "adapter-trace")]
            if program:
                command += ["--program-path", str(output / "programs" / program)]
            command += ["--", *native_args]
        jobs.append({"id": name, "profile": profile, "method": method,
                     "algorithm_variant": "ga_adaptive" if program else method,
                     "program": program, "training_dir": str(training),
                     "analysis_dir": str(output / name / "analysis"), "command": command,
                     "nominal_training_steps": nominal_training_steps(profile, method)})
    return {"schema_version": 1, "purpose": "Identity and actuation diagnostics; no method ranking",
            "seed": SEED, "trial": TRIAL, "eval_seed": EVAL_SEED,
            "upstream_commit": UPSTREAM_COMMIT, "upstream": str(upstream), "python": str(python),
            "cpu_affinity": affinity[:cpus], "thread_environment": SEARCH_THREAD_ENV,
            "timeout_seconds": timeout, "trials": jobs,
            "planned_training_steps_nominal": sum(j["nominal_training_steps"] for j in jobs),
            "source_sha256": source_hashes(), "resume_policy": "Fresh directory only"}


def _run_wrapped(job: dict, plan: dict, output: Path) -> None:
    training = Path(job["training_dir"])
    training.mkdir(parents=True, exist_ok=False)
    spec = load_program(output / "programs" / job["program"]) if job["program"] else None
    manifest = {"status": "running", "profile": job["profile"], "method": job["method"],
                "algorithm_variant": job["algorithm_variant"], "program": spec.metadata() if spec else None,
                "seed": SEED, "trial": TRIAL, "upstream_commit": UPSTREAM_COMMIT,
                "ga_settings": None, "command": job["command"],
                "source_sha256": plan["source_sha256"],
                "jax_platforms": os.environ.get("JAX_PLATFORMS"),
                "python_version": subprocess.check_output([plan["python"], "--version"], text=True).strip()}
    write_json(training / "manifest.json", manifest)
    started = time.monotonic()
    try:
        with (training / "process.log").open("w") as log:
            subprocess.run(job["command"], cwd=plan["upstream"], check=True,
                           stdout=log, stderr=subprocess.STDOUT, timeout=plan["timeout_seconds"],
                           env={**os.environ, "PYTHONUNBUFFERED": "1"})
        validate_baseline_config(read_json(training / "results.json")["config"], "ga")
        score = score_curve(read_json(training / "training_metrics.json"),
                            profile=job["profile"], method="ga")
        write_json(training / "summary.json", {**score, "profile": job["profile"]["name"],
                   "method": "ga", "algorithm_variant": job["algorithm_variant"],
                   "seed": SEED, "trial": TRIAL, "upstream_commit": UPSTREAM_COMMIT})
        manifest.update(status="complete", metrics_sha256=sha256(training / "training_metrics.json"))
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        write_json(training / "manifest.json", manifest)


def check_traces(suite: Path, output: Path, python: str) -> dict:
    subprocess.run([python, str(REPO_ROOT / "scripts/check_adaptive_trace.py"),
                    "--suite", str(suite), "--output", str(output)], check=True, timeout=120)
    return read_json(output)


def run_gate(*, output: Path, upstream: Path = DEFAULT_UPSTREAM,
             python: str = str(DEFAULT_PYTHON), cpus: int = 2, timeout: int = 600) -> dict:
    output, upstream = Path(output).resolve(), Path(upstream).resolve()
    plan = make_plan(output=output, upstream=upstream, python=python, cpus=cpus, timeout=timeout)
    verify_upstream(upstream)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    (output / "programs").mkdir()
    for program in sorted({j["program"] for j in plan["trials"] if j["program"]}):
        shutil.copyfile(REPO_ROOT / "tasks/cartpole_adaptive" / program, output / "programs" / program)
    original_affinity = os.sched_getaffinity(0)
    original_env = {key: os.environ.get(key) for key in SEARCH_THREAD_ENV}
    suite = {**plan, "status": "running", "started_at": utc_now(), "trials": []}
    started = time.monotonic()
    write_json(output / "suite.json", suite)
    try:
        os.sched_setaffinity(0, plan["cpu_affinity"])
        os.environ.update(SEARCH_THREAD_ENV)
        suite["runtime"] = runtime_probe(python, dict(os.environ))
        require(suite["runtime"]["jax_backend"] == "cpu", "CPU backend required")
        for job in plan["trials"]:
            print(f"Starting {job['id']} ({job['nominal_training_steps']:,} nominal steps)", flush=True)
            row = {**job, "status": "running"}
            suite["trials"].append(row)
            write_json(output / "suite.json", suite)
            if job["method"] == "ga_focus":
                run_experiment(profile=job["profile"], method="ga_focus", seed=SEED, trial=TRIAL,
                               output_dir=Path(job["training_dir"]), upstream=upstream,
                               python=python, timeout=timeout)
            else:
                _run_wrapped(job, plan, output)
            row["analysis"] = run_analysis(run_dir=Path(job["training_dir"]),
                                           output_dir=Path(job["analysis_dir"]),
                                           upstream=upstream, python=python,
                                           eval_seed=EVAL_SEED, episodes=10, timeout=timeout)
            row["training_wall_seconds"] = read_json(Path(job["training_dir"]) / "manifest.json")["wall_seconds"]
            row["analysis_wall_seconds"] = read_json(Path(job["analysis_dir"]) / "manifest.json")["wall_seconds"]
            row["status"] = "complete"
            write_json(output / "suite.json", suite)
            print(f"Completed {job['id']}", flush=True)
        require(source_hashes() == plan["source_sha256"], "Gate sources changed during execution")
        verify_upstream(upstream)
        suite["status"] = "complete"
        write_json(output / "suite.json", suite)
        suite["verification"] = check_traces(output / "suite.json", output / "verification.json", python)
    except BaseException as exc:
        suite.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if suite["trials"] and suite["trials"][-1]["status"] == "running":
            suite["trials"][-1]["status"] = "failed"
        raise
    finally:
        try:
            suite.update(finished_at=utc_now(), wall_seconds=time.monotonic() - started)
            write_json(output / "suite.json", suite)
            write_json(output / "receipt.json", artifact_hashes(output))
        finally:
            try:
                os.sched_setaffinity(0, original_affinity)
            finally:
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
    return suite


def export_gate(*, run_dir: Path, report_dir: Path, python: str = str(DEFAULT_PYTHON)) -> dict:
    """Revalidate receipt, protocol, analyses and array comparisons before publication."""
    import tempfile

    run_dir, report_dir = Path(run_dir).resolve(), Path(report_dir).resolve()
    require(not report_dir.exists() and not report_dir.is_relative_to(run_dir), "Report must be new and outside run")
    require(artifact_hashes(run_dir) == read_json(run_dir / "receipt.json"), "Gate artifact receipt mismatch")
    suite, plan = read_json(run_dir / "suite.json"), read_json(run_dir / "plan.json")
    require(plan["source_sha256"] == source_hashes(), "Source changed since gate; use recorded revision")
    require(suite["status"] in {"complete", "failed"}, "Only terminal attempts can be exported")
    expected = make_plan(output=run_dir, upstream=Path(plan["upstream"]), python=plan["python"],
                         timeout=plan["timeout_seconds"])
    # Caller affinity may differ from the two cores frozen at launch.
    expected["cpu_affinity"] = plan["cpu_affinity"]
    require(plan == expected, "Frozen gate plan differs from protocol")
    require(all(suite.get(key) == value for key, value in plan.items() if key != "trials"),
            "Suite protocol metadata differs from plan")
    if suite["status"] == "complete":
        require(len(suite["trials"]) == len(TRIALS), "Incomplete trial suite")
        for job, row in zip(plan["trials"], suite["trials"], strict=True):
            require(all(row.get(k) == v for k, v in job.items()), "Trial identity changed")
            require(row.get("status") == "complete", "Incomplete trial marked complete")
            training, analysis = Path(row["training_dir"]), Path(row["analysis_dir"])
            require(read_json(training / "manifest.json")["command"] == job["command"], "Training command changed")
            computed = validate_analysis(run_dir=training, output_dir=analysis, eval_seed=EVAL_SEED)
            require(computed == row["analysis"], "Analysis differs from raw evidence")
        with tempfile.TemporaryDirectory(prefix="adaptive-recheck-") as temporary:
            check = check_traces(run_dir / "suite.json", Path(temporary) / "verification.json", python)
        require(check == suite["verification"] == read_json(run_dir / "verification.json"), "Trace verification changed")
    report_dir.mkdir(parents=True)
    published = {}
    for path in artifact_files(run_dir):
        if path.suffix not in {".json", ".jsonl", ".log", ".py"}:
            continue
        target = report_dir / "raw" / path.relative_to(run_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path.read_text().replace(str(run_dir), "$RUN").replace(str(REPO_ROOT), "$REPO_ROOT"))
        published[str(target.relative_to(report_dir))] = sha256(target)
    summary = {key: suite[key] for key in ("status", "purpose", "seed", "trial", "eval_seed",
               "started_at", "finished_at", "wall_seconds", "planned_training_steps_nominal")}
    summary["trials"] = [{key: row[key] for key in ("id", "status", "algorithm_variant",
                          "nominal_training_steps", "training_wall_seconds", "analysis_wall_seconds") if key in row}
                         for row in suite["trials"]]
    summary["verification"] = suite.get("verification")
    summary["error"] = suite.get("error")
    write_json(report_dir / "summary.json", summary)
    published["summary.json"] = sha256(report_dir / "summary.json")
    write_json(report_dir / "checksums.json", {"original_artifact_sha256": artifact_hashes(run_dir),
                                               "published_sha256": published})
    return summary
