"""Time one search-profile candidate on CPU, without model or API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import signal
import subprocess
import sys
import time

from shinka_crl.experiment import DEFAULT_PYTHON, REPO_ROOT, load_profile, verify_upstream
from shinka_crl.experiment import DEFAULT_UPSTREAM


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600, help="Total evaluator timeout, seconds")
    args = parser.parse_args()
    if args.cpus < 1 or args.timeout < 1:
        parser.error("CPU count and timeout must be positive")
    output, report = args.results_dir.resolve(), args.report_dir.resolve()
    if output.exists() or report.exists():
        parser.error("Result and report paths must both be new")
    revision = verify_upstream(DEFAULT_UPSTREAM)
    affinity = sorted(os.sched_getaffinity(0))[: args.cpus]
    os.sched_setaffinity(0, affinity)
    thread_environment = {
        "JAX_PLATFORMS": "cpu", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
    }
    os.environ.update(thread_environment)
    output.mkdir(parents=True)
    probe = (
        "import importlib.metadata as m,json,jax,platform;"
        "print(json.dumps({'python':platform.python_version(),"
        "'jax_backend':jax.default_backend(),'jax_devices':[str(d) for d in jax.devices()],"
        "'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"
    )
    environment = json.loads(subprocess.check_output(
        [str(DEFAULT_PYTHON), "-c", probe], text=True, timeout=60
    ))
    if environment["jax_backend"] != "cpu":
        raise RuntimeError("Benchmark requires CPU JAX backend")
    source_paths = [
        "scripts/benchmark_search.py", "src/shinka_crl/experiment.py",
        "src/shinka_crl/profiles/search.json", "tasks/cartpole_ga/evaluate.py",
        "tasks/cartpole_ga/initial.py", "requirements/cpu.lock",
    ]
    environment.update({
        "platform": platform.platform(), "cpu_affinity": affinity,
        "upstream_commit": revision, "thread_environment": thread_environment,
        "harness_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, cwd=REPO_ROOT
        ).strip(),
        "harness_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, cwd=REPO_ROOT
        ).strip()),
        "source_sha256": {
            path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
            for path in source_paths
        },
    })
    write_json(output / "environment.json", environment)
    profile = load_profile("search")
    write_json(output / "protocol.json", profile)
    command = [
        sys.executable, str(REPO_ROOT / "tasks/cartpole_ga/evaluate.py"),
        "--program_path", str(REPO_ROOT / "tasks/cartpole_ga/initial.py"),
        "--results_dir", str(output / "candidate"),
    ]
    status = {
        "status": "running", "purpose": "runtime calibration; no performance comparison",
        "profile": "search", "seeds": profile["seeds"],
        "nominal_training_steps_per_seed": (
            profile["ne"]["num_generations"] * profile["ne"]["pop_size"]
            * profile["ne"]["num_evals"] * profile["episode_length"]
        ),
        "timing_boundary": "evaluator subprocess; includes imports, compilation, training and scoring",
        "timeout_seconds": args.timeout,
        "training_process_timeout_seconds": 600,
        "command": command,
    }
    write_json(output / "timing.json", status)
    started = time.monotonic()
    with (output / "evaluator.log").open("w") as log:
        process = subprocess.Popen(
            command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "SHINKA_CRL_PROFILE": "search", "SHINKA_CRL_TIMEOUT": "600"},
        )
        try:
            returncode = process.wait(timeout=args.timeout)
            status.update(status="complete" if returncode == 0 else "failed", returncode=returncode)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
            status.update(status="timed_out", returncode=process.returncode)
    status["wall_seconds"] = time.monotonic() - started
    status["maximum_child_process_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    status["memory_note"] = "Linux RUSAGE_CHILDREN ru_maxrss; maximum child-process RSS, not aggregate"
    status["seed_timings"] = []
    for path in sorted((output / "candidate").glob("seed_*/manifest.json")):
        manifest = json.loads(path.read_text())
        status["seed_timings"].append({
            "seed": manifest["seed"], "status": manifest["status"],
            "wall_seconds": manifest.get("wall_seconds"),
        })
    write_json(output / "timing.json", status)
    report.mkdir(parents=True)
    published, raw = {}, {}
    for source in sorted(output.rglob("*")):
        if not source.is_file() or source.suffix not in {".json", ".log"}:
            continue
        relative = source.relative_to(output)
        payload = source.read_bytes()
        raw[str(relative)] = hashlib.sha256(payload).hexdigest()
        destination = report / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload.decode().replace(str(REPO_ROOT), "$REPO_ROOT"))
        published[str(relative)] = hashlib.sha256(destination.read_bytes()).hexdigest()
    write_json(report / "checksums.json", {"raw_sha256": raw, "published_sha256": published})
    print(json.dumps(status, indent=2), flush=True)
    return 0 if status["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
