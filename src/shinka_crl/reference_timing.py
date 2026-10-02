"""One instrumented, unchanged-upstream GA development reference trial.

Phase times are parent-observed stdout boundary times, not GPU/kernel timings.
The native checkpoint writer provides exact generation boundaries without source
patching. Its serialization cost is included. No saved pickle is loaded here.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import platform
import re
import selectors
import shutil
import signal
import subprocess
import time

from shinka_crl.analysis import run_analysis, validate_analysis
from shinka_crl.baseline_contract import validate_baseline_config
from shinka_crl.experiment import (
    DEFAULT_PYTHON, DEFAULT_UPSTREAM, REPO_ROOT, UPSTREAM_COMMIT,
    build_command, load_profile, nominal_training_steps, score_curve, verify_upstream,
)
from shinka_crl.pilot import read_json, require, sha256, write_json
from shinka_crl.search import SEARCH_THREAD_ENV as THREAD_ENV

PROFILE = "paper-cartpole-timing"
SEED, TRIAL, EVAL_SEED = 1001, 1002, 901001
SOURCE_FILES = (
    "src/shinka_crl/reference_timing.py", "scripts/run_reference_timing.py",
    "scripts/report_reference_timing.py", "src/shinka_crl/experiment.py",
    "src/shinka_crl/analysis.py", "src/shinka_crl/baseline_contract.py",
    "src/shinka_crl/pilot.py", "src/shinka_crl/search.py",
    "src/shinka_crl/profiles/paper-cartpole-timing.json",
    "src/shinka_crl/profiles/smoke.json", "requirements/cpu.lock", "upstream.lock.json",
)
TIMING_NOTE = (
    "Monotonic parent receipt of flushed native stdout. Intermediate boundaries follow "
    "checkpoint serialization; phase 1 includes process startup and JIT compilation. "
    "The final boundary is the last-generation log, before final artifact writes. "
    "Phase intervals include training, in-loop evaluations/diagnostics and checkpoint I/O. "
    "They are observed wall intervals, not isolated training or exact device times."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_hashes() -> dict:
    return {name: sha256(REPO_ROOT / name) for name in SOURCE_FILES}


def make_plan(*, output: Path, upstream: Path = DEFAULT_UPSTREAM,
              python: str = str(DEFAULT_PYTHON), cpus: int = 2,
              timeout: int = 7200, analysis_timeout: int = 600,
              profile_name: str = PROFILE) -> dict:
    for name, value in {"cpus": cpus, "timeout": timeout,
                        "analysis_timeout": analysis_timeout}.items():
        require(type(value) is int and value > 0, f"{name} must be a positive integer")
    require(profile_name in {PROFILE, "smoke"}, "Only development timing or smoke is allowed")
    profile = load_profile(profile_name)
    require(profile["seeds"] == [SEED], "Reference trial must use development seed 1001")
    output, upstream = Path(output).resolve(), Path(upstream).resolve()
    affinity = sorted(os.sched_getaffinity(0))
    require(len(affinity) >= cpus, "Requested CPU count exceeds available affinity")
    command = build_command(profile=profile, method="ga", seed=SEED, trial=TRIAL,
                            output_dir=output / "training", upstream=upstream, python=python)
    command += ["--checkpoint_every", str(profile["ne"]["task_interval"])]
    return {
        "schema_version": 1, "purpose": profile["purpose"], "profile": profile,
        "method": "ga", "seed": SEED, "trial": TRIAL, "eval_seed": EVAL_SEED,
        "upstream_commit": UPSTREAM_COMMIT, "command": command,
        "cpu_affinity": affinity[:cpus], "thread_environment": THREAD_ENV,
        "timeout_seconds": timeout, "analysis_timeout_seconds": analysis_timeout,
        "planned_training_steps_nominal": nominal_training_steps(profile, "ga"),
        "timing_note": TIMING_NOTE, "resume_policy": "fresh directory only; no automatic resume",
        "source_sha256": source_hashes(),
    }


def boundary_event(line: str, *, elapsed: float, profile: dict) -> dict | None:
    checkpoint = re.match(r"\s*checkpoint: generation (\d+) -> ", line)
    generation = re.match(r"\s*gen\s+(\d+)\s+task=", line)
    count, interval = profile["ne"]["num_generations"], profile["ne"]["task_interval"]
    if checkpoint:
        completed, kind = int(checkpoint.group(1)), "native_checkpoint_written"
        require(0 < completed < count and completed % interval == 0,
                "Unexpected native checkpoint boundary")
    elif generation and int(generation.group(1)) == count - 1:
        completed, kind = count, "last_generation_logged"
    else:
        return None
    require(math.isfinite(elapsed) and elapsed >= 0, "Invalid boundary timestamp")
    return {"completed_generations": completed, "elapsed_seconds": elapsed, "kind": kind}


def phase_intervals(events: list[dict], profile: dict) -> list[dict]:
    """Reject missing, duplicate, unordered or fabricated phase boundaries."""
    previous, phases = 0.0, []
    interval = profile["ne"]["task_interval"]
    for index, event in enumerate(events):
        require(isinstance(event, dict) and set(event) == {
            "completed_generations", "elapsed_seconds", "kind"}, "Invalid phase event schema")
        require(event["completed_generations"] == (index + 1) * interval,
                "Phase events must be contiguous and unique")
        require(index < profile["num_phases"], "Too many phase events")
        expected_kind = ("last_generation_logged" if index + 1 == profile["num_phases"]
                         else "native_checkpoint_written")
        require(event["kind"] == expected_kind, "Incorrect phase boundary event kind")
        elapsed = event["elapsed_seconds"]
        require(type(elapsed) in (float, int) and math.isfinite(elapsed)
                and elapsed >= previous, "Invalid or unordered phase timestamp")
        phases.append({"phase": index, "task": index % profile["num_tasks"],
                       "completed_generations": event["completed_generations"],
                       "observed_wall_seconds": elapsed - previous,
                       "boundary_elapsed_seconds": elapsed, "boundary_kind": event["kind"]})
        previous = elapsed
    return phases


def monitor_training(*, command: list[str], cwd: Path, output: Path,
                     profile: dict, timeout: int, environment: dict) -> dict:
    """Observe stdout and use wait4 for this trainer's own Linux peak RSS."""
    started, events = time.monotonic(), []
    status, stop_at, buffer = "complete", None, b""
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, env=environment,
                               start_new_session=True, bufsize=0)
    trainer_affinity = sorted(os.sched_getaffinity(process.pid))
    assert process.stdout is not None
    os.set_blocking(process.stdout.fileno(), False)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    usage = None

    def stop(reason):
        nonlocal status, stop_at
        status, stop_at = reason, time.monotonic()
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    with (output / "process.log").open("wb") as log, (
        output / "phase-events.jsonl"
    ).open("w") as journal:
        try:
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
                        event = boundary_event(line.decode(errors="replace"),
                                               elapsed=time.monotonic() - started,
                                               profile=profile)
                        if event:
                            events.append(event)
                            phase_intervals(events, profile)
                            journal.write(json.dumps(event, allow_nan=False) + "\n")
                            journal.flush()
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
    if status == "complete" and process.returncode:
        status = "failed"
    return {"status": status, "returncode": process.returncode,
            "wall_seconds": time.monotonic() - started,
            "trainer_cpu_affinity": trainer_affinity,
            "maximum_trainer_rss_kib": usage.ru_maxrss,
            "memory_note": "Linux wait4 ru_maxrss for this trainer only; excludes post-hoc analysis",
            "phase_events": events, "phase_timings": phase_intervals(events, profile)}


def runtime_probe(python: str, environment: dict) -> dict:
    probe = (
        "import importlib.metadata as m,json,jax,os,platform;"
        "print(json.dumps({'python':platform.python_version(),'jax_backend':jax.default_backend(),"
        "'jax_devices':[str(d) for d in jax.devices()],'cpu_affinity':sorted(os.sched_getaffinity(0)),"
        "'thread_environment':{k:os.environ.get(k) for k in "
        f"{list(THREAD_ENV)!r}" + "},"
        "'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"
    )
    return json.loads(subprocess.check_output([python, "-c", probe], text=True,
                                             timeout=60, env=environment))


def artifact_files(output: Path) -> list[Path]:
    paths = []
    for path in sorted(output.rglob("*")):
        require(not path.is_symlink() and path.resolve().is_relative_to(output.resolve()),
                "Artifact escapes result root or is a symlink")
        if path.is_file():
            paths.append(path)
    return paths


def artifact_hashes(output: Path) -> dict:
    return {str(path.relative_to(output)): sha256(path) for path in artifact_files(output)
            if path != output / "receipt.json"}


def run_reference_timing(*, output: Path, upstream: Path = DEFAULT_UPSTREAM,
                         python: str = str(DEFAULT_PYTHON), cpus: int = 2,
                         timeout: int = 7200, analysis_timeout: int = 600,
                         profile_name: str = PROFILE) -> dict:
    """Execute a fresh trial and retain artifacts for every terminal outcome."""
    require(platform.system() == "Linux", "Timing runner requires Linux affinity and wait4")
    output, upstream = Path(output).resolve(), Path(upstream).resolve()
    require(not output.exists(), "Result directory must be new; existing attempts are immutable")
    verify_upstream(upstream)
    interpreter = shutil.which(python)
    require(interpreter is not None, f"Missing Python interpreter: {python}")
    python = str(Path(interpreter).absolute())
    plan = make_plan(output=output, upstream=upstream, python=python, cpus=cpus,
                     timeout=timeout, analysis_timeout=analysis_timeout, profile_name=profile_name)
    output.mkdir(parents=True)
    training = output / "training"
    training.mkdir()
    write_json(output / "plan.json", plan)
    profile, original_affinity = plan["profile"], os.sched_getaffinity(0)
    original_environment = {key: os.environ.get(key) for key in THREAD_ENV}
    environment = {**os.environ, **THREAD_ENV, "PYTHONUNBUFFERED": "1"}
    started = time.monotonic()
    summary = {"status": "running", "started_at_utc": utc_now(), "profile": profile["name"],
               "seed": SEED, "trial": TRIAL, "timing_note": TIMING_NOTE,
               "planned_training_steps_nominal": plan["planned_training_steps_nominal"],
               "training": None, "analysis": None}
    manifest = {"status": "running", "profile": profile, "method": "ga", "seed": SEED,
                "trial": TRIAL, "upstream_commit": UPSTREAM_COMMIT, "ga_settings": None,
                "command": plan["command"]}
    write_json(training / "manifest.json", manifest)
    write_json(output / "summary.json", summary)
    try:
        os.sched_setaffinity(0, plan["cpu_affinity"])
        os.environ.update(THREAD_ENV)
        runtime = runtime_probe(python, environment)
        require(runtime["jax_backend"] == "cpu", "Reference timing requires CPU JAX")
        require(runtime["cpu_affinity"] == plan["cpu_affinity"], "Runtime affinity mismatch")
        require(runtime["thread_environment"] == THREAD_ENV, "Runtime numerical threads mismatch")
        runtime.update(platform=platform.platform(), recorded_at_utc=utc_now(),
                       harness_commit=subprocess.check_output(
                           ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
                       harness_dirty=bool(subprocess.check_output(
                           ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()))
        write_json(output / "environment.json", runtime)
        timing = monitor_training(command=plan["command"], cwd=upstream, output=training,
                                  profile=profile, timeout=timeout, environment=environment)
        summary["training"] = timing
        require(timing["trainer_cpu_affinity"] == plan["cpu_affinity"],
                "Trainer process affinity differs from the frozen runtime")
        manifest.update(status=timing["status"], wall_seconds=timing["wall_seconds"])
        write_json(training / "manifest.json", manifest)
        summary["status"] = timing["status"]
        if timing["status"] != "complete":
            return summary
        require(len(timing["phase_timings"]) == profile["num_phases"],
                "Completed trainer lacks complete phase-boundary evidence")
        records = read_json(training / "training_metrics.json")
        score = score_curve(records, profile=profile, method="ga")
        results = read_json(training / "results.json")
        validate_baseline_config(results["config"], "ga")
        require(results["config"].get("checkpoint_every") == profile["ne"]["task_interval"],
                "Native checkpoint instrumentation was not applied")
        require(results.get("env_steps") == plan["planned_training_steps_nominal"],
                "Completed nominal training budget mismatch")
        write_json(training / "summary.json", {**score, "profile": profile["name"], "method": "ga",
                   "seed": SEED, "trial": TRIAL, "upstream_commit": UPSTREAM_COMMIT})
        manifest.update(metrics_sha256=sha256(training / "training_metrics.json"))
        write_json(training / "manifest.json", manifest)
        summary["active_score"] = score
        summary["training"]["finalization_wall_seconds"] = (
            timing["wall_seconds"] - timing["phase_events"][-1]["elapsed_seconds"])
        summary["analysis"] = run_analysis(run_dir=training, output_dir=output / "analysis",
                                           eval_seed=EVAL_SEED, upstream=upstream, python=python,
                                           episodes=10, timeout=analysis_timeout)
        validate_analysis(run_dir=training, output_dir=output / "analysis", eval_seed=EVAL_SEED)
        require(source_hashes() == plan["source_sha256"], "Harness source changed during trial")
        verify_upstream(upstream)
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
        journal = training / "phase-events.jsonl"
        if summary["training"] is None and journal.exists():
            events = [json.loads(line) for line in journal.read_text().splitlines()]
            summary["training"] = {
                "status": summary["status"], "wall_seconds": None,
                "maximum_trainer_rss_kib": None, "phase_events": events,
                "phase_timings": phase_intervals(events, profile),
                "measurement_error": "Supervisor interrupted; complete process measurement unavailable",
            }
        if manifest["status"] == "running":
            manifest.update(status=summary["status"], error=summary["error"])
            write_json(training / "manifest.json", manifest)
    finally:
        os.sched_setaffinity(0, original_affinity)
        for key, value in original_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        timing = summary.get("training") or {}
        phases = timing.get("phase_timings", [])
        summary.update(finished_at_utc=utc_now(), total_wall_seconds=time.monotonic() - started,
                       completed_phases_observed=len(phases),
                       training_steps_nominal_observed_lower_bound=(
                           phases[-1]["completed_generations"] if phases else 0)
                       * profile["ne"]["pop_size"] * profile["ne"]["num_evals"]
                       * profile["episode_length"])
        write_json(output / "summary.json", summary)
        write_json(output / "receipt.json", artifact_hashes(output))
    return summary


def export_reference_timing(*, run_dir: Path, report_dir: Path) -> dict:
    """Verify immutable evidence and export text artifacts; binaries stay local."""
    run_dir, report_dir = Path(run_dir).resolve(), Path(report_dir).resolve()
    require(not report_dir.exists(), "Report directory must be new")
    require(not report_dir.is_relative_to(run_dir), "Report must be outside the result directory")
    require(artifact_hashes(run_dir) == read_json(run_dir / "receipt.json"),
            "Reference evidence changed after receipt")
    plan, summary = read_json(run_dir / "plan.json"), read_json(run_dir / "summary.json")
    require(summary["status"] in {"complete", "failed", "timed_out", "interrupted"},
            "Only terminal attempts may be exported")
    require(plan["source_sha256"] == source_hashes(), "Source changed since reference trial")
    profile = load_profile(plan["profile"]["name"])
    require(plan["profile"] == profile and profile["seeds"] == [SEED]
            and profile["name"] in {PROFILE, "smoke"}, "Reference profile changed")
    require(plan["seed"] == summary["seed"] == SEED and plan["trial"] == summary["trial"] == TRIAL
            and plan["eval_seed"] == EVAL_SEED and plan["method"] == "ga"
            and plan["upstream_commit"] == UPSTREAM_COMMIT
            and summary["profile"] == profile["name"], "Reference identity changed")
    command = plan["command"]
    require(command == build_command(profile=profile, method="ga", seed=SEED, trial=TRIAL,
                                     output_dir=run_dir / "training", python=command[0],
                                     upstream=Path(command[1]).parent.parent)
            + ["--checkpoint_every", str(profile["ne"]["task_interval"])],
            "Reference command differs from protocol")
    require(read_json(run_dir / "training/manifest.json")["command"] == command,
            "Training command differs from plan")
    planned_steps = nominal_training_steps(profile, "ga")
    require(summary["planned_training_steps_nominal"] == plan["planned_training_steps_nominal"]
            == planned_steps, "Allocated training budget changed")
    events = []
    if summary.get("training"):
        events = [json.loads(line) for line in
                  (run_dir / "training/phase-events.jsonl").read_text().splitlines()]
        require(events == summary["training"]["phase_events"], "Phase journal differs from summary")
        require(phase_intervals(events, plan["profile"]) == summary["training"]["phase_timings"],
                "Phase timing summary differs from journal")
    completed = events[-1]["completed_generations"] if events else 0
    observed_steps = (completed * profile["ne"]["pop_size"] * profile["ne"]["num_evals"]
                      * profile["episode_length"])
    require(summary["completed_phases_observed"] == len(events)
            and summary["training_steps_nominal_observed_lower_bound"] == observed_steps,
            "Observed phase count or training budget changed")
    if summary["status"] == "complete":
        require(len(events) == profile["num_phases"]
                and summary["training"]["status"] == "complete"
                and summary["training"]["returncode"] == 0, "Incomplete training marked complete")
        active_score = score_curve(read_json(run_dir / "training/training_metrics.json"),
                                   profile=profile, method="ga")
        require(summary["active_score"] == active_score, "Active score differs from raw trajectory")
        require(summary["training"]["finalization_wall_seconds"]
                == summary["training"]["wall_seconds"] - events[-1]["elapsed_seconds"]
                >= 0, "Finalization timing differs from boundary evidence")
        analysis = validate_analysis(run_dir=run_dir / "training", output_dir=run_dir / "analysis",
                                     eval_seed=EVAL_SEED)
        require(analysis == summary["analysis"], "Reference analysis differs from summary")
        results = read_json(run_dir / "training/results.json")
        validate_baseline_config(results["config"], "ga")
    report_dir.mkdir(parents=True)
    published = {}
    for path in artifact_files(run_dir):
        if path.suffix not in {".json", ".jsonl", ".log"}:
            continue
        relative = path.relative_to(run_dir)
        target = report_dir / "raw" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(path.read_text().replace(str(run_dir), "$RUN")
                          .replace(str(REPO_ROOT), "$REPO_ROOT"))
        published[str(Path("raw") / relative)] = sha256(target)
    write_json(report_dir / "summary.json", summary)
    published["summary.json"] = sha256(report_dir / "summary.json")
    write_json(report_dir / "checksums.json", {"original_artifact_sha256": artifact_hashes(run_dir),
                                               "published_sha256": published})
    return summary
