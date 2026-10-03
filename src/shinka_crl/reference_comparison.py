"""Full CartPole reference trials using the unchanged upstream trainers and evaluator.

Development and reporting suites have separate identities. Native phase checkpoints
are retained, completed training is reused after analysis failures, and each failed
process remains an immutable attempt. An explicit resume can carry its verified
native checkpoint into a fresh attempt; this module never loads a pickle.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import math
import os
from pathlib import Path
import re
import selectors
import shutil
import signal
import subprocess
import time

from shinka_crl.analysis import run_analysis, validate_analysis
from shinka_crl.baseline_contract import validate_baseline_config
from shinka_crl.experiment import (
    DEFAULT_PYTHON, DEFAULT_UPSTREAM, REPO_ROOT, UPSTREAM_COMMIT, build_command,
    load_profile, nominal_training_steps, score_curve, verify_upstream,
)
from shinka_crl.pilot import next_attempt, read_json, require, sha256, write_json
from shinka_crl.reference_timing import runtime_probe
from shinka_crl.search import SEARCH_THREAD_ENV as THREAD_ENV

METHODS = ("ga", "es", "ppo")
SOURCE_FILES = (
    "src/shinka_crl/reference_comparison.py", "scripts/run_reference_comparison.py",
    "src/shinka_crl/reference_timing.py", "src/shinka_crl/experiment.py",
    "src/shinka_crl/analysis.py", "src/shinka_crl/baseline_contract.py",
    "src/shinka_crl/pilot.py", "src/shinka_crl/search.py",
    "src/shinka_crl/profiles/paper-cartpole.json", "src/shinka_crl/profiles/smoke.json",
    "requirements/cpu.lock", "upstream.lock.json",
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    return {name: sha256(REPO_ROOT / name) for name in SOURCE_FILES}


def make_plan(*, mode: str, cpus: int = 2, timeout: int = 21600,
              analysis_timeout: int = 1800, methods=None,
              python: str = str(DEFAULT_PYTHON), upstream: Path = DEFAULT_UPSTREAM) -> dict:
    require(mode in {"development", "diagnostic", "reporting"}, "Unknown experiment mode")
    for name, value in (("cpus", cpus), ("timeout", timeout),
                        ("analysis_timeout", analysis_timeout)):
        require(type(value) is int and value > 0, f"{name} must be a positive integer")
    affinity = sorted(os.sched_getaffinity(0))
    require(cpus <= len(affinity), "Requested CPUs exceed available affinity")
    selected = list(methods) if methods is not None else (
        ["es", "ppo"] if mode == "development" else list(METHODS))
    require(selected and len(set(selected)) == len(selected)
            and set(selected) <= set(METHODS), "Invalid or duplicate methods")
    require(mode != "diagnostic" or selected == list(METHODS),
            "Diagnostic requires GA, ES, PPO in the declared order")
    require(mode != "reporting" or selected in (list(METHODS), ["ga", "es"]),
            "Reporting requires GA, ES, PPO or the amended GA, ES scope in the declared order")
    profile = load_profile("smoke" if mode == "diagnostic" else "paper-cartpole")
    if mode != "reporting":
        profile["seeds"] = [3001 if mode == "diagnostic" else 1001]
        profile["purpose"] = ("Reduced execution diagnostic; not reporting" if mode == "diagnostic"
                              else "Full-budget development runtime calibration; not reporting")
    jobs = [{"profile": profile["name"], "method": method, "seed": seed,
             "trial": index if mode == "reporting" else seed + 1,
             "eval_seed": 900000 + seed}
            for index, seed in enumerate(profile["seeds"], 1) for method in selected]
    return {"schema_version": 1, "mode": mode, "profiles": {profile["name"]: profile},
            "methods": selected, "jobs": jobs, "eval_episodes": 10,
            "upstream_commit": UPSTREAM_COMMIT, "source_sha256": source_hashes(),
            "python": str(Path(python).absolute()), "upstream": str(Path(upstream).resolve()),
            "cpu_affinity": affinity[:cpus], "thread_environment": THREAD_ENV,
            "timeout_seconds": timeout, "analysis_timeout_seconds": analysis_timeout,
            "checkpoint_policy": "Unchanged native whole-state checkpoint every phase",
            "resume_policy": "Explicit resume only; verified completed stages are reused; "
                             "verified native checkpoints are copied into fresh attempts",
            "nominal_training_steps_per_method": {
                method: nominal_training_steps(profile, method) for method in selected}}


def command_for(plan, profile, job, path):
    budget = profile["ppo"] if job["method"] == "ppo" else profile["ne"]
    return build_command(profile=profile, method=job["method"], seed=job["seed"],
                         trial=job["trial"], output_dir=path, upstream=Path(plan["upstream"]),
                         python=plan["python"]) + ["--checkpoint_every", str(budget["task_interval"])]


def artifact_hashes(path):
    hashes = {}
    for item in sorted(path.rglob("*")):
        require(not item.is_symlink(), "Artifact symlinks are forbidden")
        if item.is_file() and item != path / "receipt.json":
            hashes[str(item.relative_to(path))] = sha256(item)
    return hashes


def phase_event(line, elapsed, profile, method):
    budget = profile["ppo"] if method == "ppo" else profile["ne"]
    count = budget["num_updates" if method == "ppo" else "num_generations"]
    unit, log_unit = ("update", "update") if method == "ppo" else ("generation", "gen")
    match = re.match(rf"\s*checkpoint: {unit} (\d+) -> ", line)
    if match:
        completed, kind = int(match[1]), "native_checkpoint_written"
        require(0 < completed < count and completed % budget["task_interval"] == 0,
                "Unexpected native phase checkpoint")
    else:
        match = re.match(rf"\s*{log_unit}\s+(\d+)\s+task=", line)
        if not match or int(match[1]) != count - 1:
            return None
        completed, kind = count, "last_update_logged"
    return {"completed_updates": completed, "elapsed_seconds": elapsed, "kind": kind}


def _pid_identity(pid):
    try:
        # The comm field can contain spaces and parentheses; parse after its last ')'.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except (OSError, IndexError):
        return None


@contextmanager
def suite_lock(output):
    with (output / "controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another reference comparison controller holds this suite") from exc
        active = output / "active-process.json"
        if active.exists():
            record = read_json(active)
            require(_pid_identity(record["pid"]) != record["start_ticks"],
                    "A previously launched trainer is still active; do not duplicate it")
        yield


def monitor_training(*, command, cwd, path, profile, method, timeout, active_path,
                     start_update=0):
    """Measure this exact native child with wait4; persist phase progress as it arrives."""
    started, events, buffer = time.monotonic(), [], b""
    status, usage, stop_at = "complete", None, None
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, start_new_session=True, bufsize=0,
                               env={**os.environ, "PYTHONUNBUFFERED": "1"})
    write_json(active_path, {"pid": process.pid, "start_ticks": _pid_identity(process.pid),
                            "command": command, "started_at_utc": utc_now()})
    affinity = sorted(os.sched_getaffinity(process.pid))
    selector = selectors.DefaultSelector()
    os.set_blocking(process.stdout.fileno(), False)
    selector.register(process.stdout, selectors.EVENT_READ)

    def stop(reason):
        nonlocal status, stop_at
        status, stop_at = reason, time.monotonic()
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    try:
        with (path / "process.log").open("wb") as log, (path / "phase-events.jsonl").open("w") as journal:
            import json
            while True:
                try:
                    ready = selector.select(0.1)
                except KeyboardInterrupt:
                    stop("interrupted")
                    ready = []
                for key, _ in ready:
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    log.write(data)
                    log.flush()
                    buffer += data
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        event = phase_event(line.decode(errors="replace"),
                                            time.monotonic() - started, profile, method)
                        if event:
                            previous = events[-1]["completed_updates"] if events else start_update
                            budget = profile["ppo"] if method == "ppo" else profile["ne"]
                            require(event["completed_updates"] == previous + budget["task_interval"],
                                    "Phase boundaries must be contiguous")
                            events.append(event)
                            journal.write(json.dumps(event) + "\n")
                            journal.flush()
                            print(f"{method}: completed native step {event['completed_updates']}", flush=True)
                if usage is None:
                    pid, wait_status, measured = os.wait4(process.pid, os.WNOHANG)
                    if pid:
                        process.returncode = os.waitstatus_to_exitcode(wait_status)
                        usage = measured
                if usage is not None and not selector.get_map():
                    break
                now = time.monotonic()
                if stop_at is None and now - started >= timeout:
                    stop("timed_out")
                if stop_at is not None and now - stop_at >= 10:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
    finally:
        selector.close()
        process.stdout.close()
        if usage is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            _, wait_status, usage = os.wait4(process.pid, 0)
            process.returncode = os.waitstatus_to_exitcode(wait_status)
        active_path.unlink(missing_ok=True)
    if status == "complete" and process.returncode:
        status = "failed"
    return {"status": status, "returncode": process.returncode,
            "wall_seconds": time.monotonic() - started, "maximum_trainer_rss_kib": usage.ru_maxrss,
            "user_cpu_seconds": usage.ru_utime, "system_cpu_seconds": usage.ru_stime,
            "trainer_cpu_affinity": affinity, "start_update": start_update,
            "phase_events": events,
            "timing_note": "Parent-observed flushed native boundaries include JIT, evaluation, "
                           "diagnostics and checkpoint I/O; RSS is this trainer's Linux wait4 peak"}


def validate_training(path, *, plan, profile, job, require_receipt=True):
    """Check the instrumented exact command, baseline settings, budget and raw curve."""
    hashes = artifact_hashes(path)
    if require_receipt:
        require(read_json(path / "receipt.json") == hashes, "Training artifact receipt mismatch")
    for name in ("manifest.json", "summary.json", "config.json", "results.json",
                 "training_metrics.json", "checkpoints.npz", "process.log", "train.log",
                 "process-measurement.json"):
        require(name in hashes, f"Missing training artifact: {name}")
    manifest, results = read_json(path / "manifest.json"), read_json(path / "results.json")
    require(manifest["status"] == "complete", "Training is incomplete")
    for key, value in {"profile": profile, "method": job["method"], "seed": job["seed"],
                       "trial": job["trial"], "ga_settings": None,
                       "upstream_commit": UPSTREAM_COMMIT}.items():
        require(manifest.get(key) == value, f"Training identity mismatch: {key}")
    require(manifest["command"] == command_for(plan, profile, job, path), "Training command changed")
    require(manifest["metrics_sha256"] == hashes["training_metrics.json"], "Metric hash mismatch")
    cfg, method = results["config"], job["method"]
    validate_baseline_config(cfg, method)
    budget = profile["ppo"] if method == "ppo" else profile["ne"]
    count = budget["num_updates" if method == "ppo" else "num_generations"]
    expected = {"seed": job["seed"], "trial": job["trial"], "schedule": "switch",
                "num_tasks": 2, "noise_range": 0.5, "num_generations": count,
                "task_interval": budget["task_interval"], "checkpoint_every": budget["task_interval"],
                "task_sequence": [phase % 2 for phase in range(profile["num_phases"])],
                "episode_length": profile["episode_length"], "eval_episodes": profile["eval_episodes"]}
    keys = ("num_envs", "num_steps", "num_minibatches") if method == "ppo" else ("pop_size", "num_evals")
    expected.update({key: budget[key] for key in keys})
    for key, value in expected.items():
        require(cfg.get(key) == value, f"Resolved training config mismatch: {key}")
    nominal = nominal_training_steps(profile, method)
    require(results["env_steps"] == nominal, "Training budget mismatch")
    vectors = results.get("noise_vectors")
    require(isinstance(vectors, list) and len(vectors) == 2 and vectors[0] == [0, 0, 0, 0]
            and all(isinstance(v, list) and len(v) == 4 and all(
                type(x) in (int, float) and math.isfinite(x) for x in v) for v in vectors),
            "Invalid task vectors")
    summary = {**score_curve(read_json(path / "training_metrics.json"), profile=profile, method=method),
               "profile": profile["name"], "method": method, "seed": job["seed"],
               "trial": job["trial"], "upstream_commit": UPSTREAM_COMMIT}
    require(summary == read_json(path / "summary.json"), "Training summary differs from raw curve")
    measurement = read_json(path / "process-measurement.json")
    require(measurement["status"] == "complete" and measurement["returncode"] == 0,
            "Training process did not complete")
    require(measurement["trainer_cpu_affinity"] == plan["cpu_affinity"], "Training affinity changed")
    require(measurement["phase_events"][-1]["completed_updates"] == count, "Missing final boundary")
    require(manifest["wall_seconds"] == measurement["wall_seconds"] >= 0, "Training duration changed")
    return {"summary": summary, "config": cfg, "noise_vectors": vectors,
            "nominal_training_steps": nominal, "wall_seconds": manifest["wall_seconds"],
            "maximum_trainer_rss_kib": measurement["maximum_trainer_rss_kib"],
            "artifact_sha256": hashes}


def _completed(root):
    for path in sorted(root.glob("attempt_*"), reverse=True):
        manifest = path / "manifest.json"
        if manifest.exists() and read_json(manifest).get("status") == "complete":
            return path
    return None


def run_training(*, path, plan, profile, job, output, resume_source=None):
    path.mkdir(parents=True)
    start_update, resume_record = 0, None
    if resume_source is not None:
        require(read_json(resume_source / "receipt.json") == artifact_hashes(resume_source),
                "Interrupted training artifacts changed; checkpoint reuse rejected")
        prior = read_json(resume_source / "manifest.json")
        require(prior.get("profile") == profile and all(prior.get(key) == job[key]
                for key in ("method", "seed", "trial")), "Checkpoint identity mismatch")
        require(prior.get("upstream_commit") == UPSTREAM_COMMIT and prior.get("ga_settings") is None
                and prior.get("command") == command_for(plan, profile, job, resume_source),
                "Checkpoint command or pinned implementation differs")
        prior_measurement = read_json(resume_source / "process-measurement.json")
        checkpoints = [event for event in prior_measurement["phase_events"]
                       if event["kind"] == "native_checkpoint_written"]
        if checkpoints:
            start_update = checkpoints[-1]["completed_updates"]
        else:
            previous_resume = prior.get("resume_from") or {}
            start_update = previous_resume.get("start_update", 0)
            require(start_update > 0 and prior_measurement["start_update"] == start_update
                    and sha256(resume_source / "resume.pkl") == previous_resume.get("sha256"),
                    "Checkpoint has no verified native boundary")
        shutil.copyfile(resume_source / "resume.pkl", path / "resume.pkl")
        resume_record = {"source": str(resume_source.relative_to(output)),
                         "sha256": sha256(path / "resume.pkl"), "start_update": start_update}
    manifest = {"status": "running", "profile": profile, "method": job["method"],
                "seed": job["seed"], "trial": job["trial"], "upstream_commit": UPSTREAM_COMMIT,
                "ga_settings": None, "command": command_for(plan, profile, job, path),
                "started_at_utc": utc_now(), "resume_from": resume_record}
    write_json(path / "manifest.json", manifest)
    started = time.monotonic()
    try:
        measurement = monitor_training(command=manifest["command"], cwd=Path(plan["upstream"]),
                                       path=path, profile=profile, method=job["method"],
                                       timeout=plan["timeout_seconds"],
                                       active_path=output / "active-process.json", start_update=start_update)
        write_json(path / "process-measurement.json", measurement)
        manifest.update(status=measurement["status"], wall_seconds=measurement["wall_seconds"])
        require(measurement["status"] == "complete", f"Native trainer {measurement['status']}")
        records = read_json(path / "training_metrics.json")
        summary = {**score_curve(records, profile=profile, method=job["method"]),
                   "profile": profile["name"], "method": job["method"], "seed": job["seed"],
                   "trial": job["trial"], "upstream_commit": UPSTREAM_COMMIT}
        write_json(path / "summary.json", summary)
        manifest["metrics_sha256"] = sha256(path / "training_metrics.json")
        write_json(path / "manifest.json", manifest)
        validate_training(path, plan=plan, profile=profile, job=job, require_receipt=False)
    except BaseException as exc:
        manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest.setdefault("wall_seconds", time.monotonic() - started)
        manifest["finished_at_utc"] = utc_now()
        write_json(path / "manifest.json", manifest)
        write_json(path / "receipt.json", artifact_hashes(path))


def run_comparison(*, output: Path, mode: str, resume=False, max_trials=None, **kwargs):
    require(max_trials is None or type(max_trials) is int and max_trials > 0, "Invalid max_trials")
    output = Path(output).resolve()
    require(output.exists() == resume, "Use a fresh result directory, or explicit --resume")
    plan = make_plan(mode=mode, **kwargs)
    verify_upstream(Path(plan["upstream"]))
    require(shutil.which(plan["python"]) is not None, "Missing upstream Python")
    output.mkdir(parents=True, exist_ok=True)
    with suite_lock(output):
        if resume:
            require(read_json(output / "plan.json") == plan, "Frozen plan, source or runtime changed")
            sessions = read_json(output / "suite.json").get("sessions", [])
        else:
            write_json(output / "plan.json", plan)
            sessions = []
        original_affinity = os.sched_getaffinity(0)
        original_environment = {key: os.environ.get(key) for key in THREAD_ENV}
        started, new_trials = time.monotonic(), 0
        suite = {"schema_version": 1, "mode": mode, "status": "running", "rows": [],
                 "planned_trials": len(plan["jobs"]), "completed_trials": 0, "sessions": sessions,
                 "wall_seconds": sum(s["wall_seconds"] for s in sessions), "controller_pid": os.getpid()}
        write_json(output / "suite.json", suite)
        try:
            os.sched_setaffinity(0, plan["cpu_affinity"])
            os.environ.update(THREAD_ENV)
            runtime = runtime_probe(plan["python"], dict(os.environ))
            require(runtime["jax_backend"] == "cpu" and runtime["cpu_affinity"] == plan["cpu_affinity"]
                    and runtime["thread_environment"] == THREAD_ENV, "Runtime differs from CPU plan")
            runtime.update(upstream_commit=UPSTREAM_COMMIT, source_sha256=plan["source_sha256"])
            if resume:
                require(runtime == read_json(output / "environment.json"), "Runtime packages changed")
            else:
                write_json(output / "environment.json", runtime)
            for job in plan["jobs"]:
                profile = plan["profiles"][job["profile"]]
                root = output / "trials" / job["method"] / f"seed_{job['seed']}"
                training, analysis = _completed(root / "training"), _completed(root / "analysis")
                existing = training is not None and analysis is not None
                if not existing and max_trials is not None and new_trials >= max_trials:
                    continue
                suite["active_job"] = job
                write_json(output / "suite.json", suite)
                print(f"{'Checking' if existing else 'Running'} {mode}/{job['method']}/seed_{job['seed']}", flush=True)
                if training is None:
                    prior = sorted((root / "training").glob("attempt_*"))
                    resume_source = prior[-1] if prior and (prior[-1] / "resume.pkl").exists() else None
                    training = next_attempt(root / "training")
                    run_training(path=training, plan=plan, profile=profile, job=job,
                                 output=output, resume_source=resume_source)
                validated = validate_training(training, plan=plan, profile=profile, job=job)
                if analysis is None:
                    analysis = next_attempt(root / "analysis")
                    run_analysis(run_dir=training, output_dir=analysis, upstream=Path(plan["upstream"]),
                                 python=plan["python"], episodes=plan["eval_episodes"],
                                 eval_seed=job["eval_seed"], timeout=plan["analysis_timeout_seconds"])
                analyzed = validate_analysis(run_dir=training, output_dir=analysis,
                                             episodes=plan["eval_episodes"], eval_seed=job["eval_seed"])
                require(source_hashes() == plan["source_sha256"], "Harness source changed during run")
                verify_upstream(Path(plan["upstream"]))
                for row in suite["rows"]:
                    if row["seed"] == job["seed"]:
                        require(row["noise_vectors"] == validated["noise_vectors"], "Task draws differ by method")
                attempts = [read_json(path / "manifest.json") for path in (root / "training").glob("attempt_*")
                            if (path / "manifest.json").exists()]
                analysis_attempts = [read_json(path / "manifest.json") for path in (root / "analysis").glob("attempt_*")
                                     if (path / "manifest.json").exists()]
                suite["rows"].append({**job, "training_path": str(training.relative_to(output)),
                                      "analysis_path": str(analysis.relative_to(output)),
                                      "training_wall_seconds": validated["wall_seconds"],
                                      "analysis_wall_seconds": read_json(analysis / "manifest.json")["wall_seconds"],
                                      "maximum_trainer_rss_kib": validated["maximum_trainer_rss_kib"],
                                      "training_attempts": len(attempts),
                                      "training_total_wall_seconds": sum(a.get("wall_seconds", 0) for a in attempts),
                                      "analysis_attempts": len(analysis_attempts),
                                      "analysis_total_wall_seconds": sum(a.get("wall_seconds", 0) for a in analysis_attempts),
                                      "noise_vectors": validated["noise_vectors"], "analysis": analyzed})
                suite["completed_trials"] = len(suite["rows"])
                new_trials += not existing
                write_json(output / "suite.json", suite)
            suite["status"] = "complete" if len(suite["rows"]) == len(plan["jobs"]) else "partial"
        except BaseException as exc:
            suite.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                         error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            sessions.append({"wall_seconds": time.monotonic() - started, "status": suite["status"],
                             "new_completed_trials": new_trials, "finished_at_utc": utc_now()})
            suite["wall_seconds"] = sum(s["wall_seconds"] for s in sessions)
            suite.pop("active_job", None)
            write_json(output / "suite.json", suite)
            os.sched_setaffinity(0, original_affinity)
            for key, value in original_environment.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    return suite
