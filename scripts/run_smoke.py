"""Run actual CPU GA/ES/PPO training and the initial Shinka evaluator, sequentially."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from shinka_crl.experiment import (
    DEFAULT_PYTHON,
    DEFAULT_UPSTREAM,
    REPO_ROOT,
    load_profile,
    run_experiment,
    verify_upstream,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per training process")
    args = parser.parse_args()
    if args.cpus < 1 or args.timeout < 1:
        parser.error("--cpus and --timeout must be positive")
    output = args.results_dir.resolve()
    if output.exists():
        parser.error(f"Results directory already exists: {output}")
    revision = verify_upstream(args.upstream)
    # Affinity limits both compilation and execution pools without altering JAX arithmetic.
    affinity = None
    if hasattr(os, "sched_getaffinity"):
        affinity = sorted(os.sched_getaffinity(0))[: args.cpus]
        os.sched_setaffinity(0, affinity)
    for key, value in {
        "JAX_PLATFORMS": "cpu",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }.items():
        os.environ[key] = value
    output.mkdir(parents=True)
    status = {"status": "running", "completed": [], "purpose": "pipeline validation only"}
    started = time.monotonic()
    try:
        probe = (
            "import importlib.metadata as m, json, jax, platform; "
            "print(json.dumps({'python':platform.python_version(), "
            "'jax_devices':[str(d) for d in jax.devices()], "
            "'jax_backend':jax.default_backend(), "
            "'packages':{d.metadata['Name']:d.version for d in m.distributions()}}))"
        )
        environment = json.loads(
            subprocess.check_output([args.python, "-c", probe], text=True, timeout=args.timeout)
        )
        if environment["jax_backend"] != "cpu":
            raise RuntimeError("Smoke suite requires the CPU JAX backend")
        environment.update(
            {
                "platform": platform.platform(),
                "cpu_affinity": affinity,
                "upstream_commit": revision,
                "harness_commit": subprocess.check_output(
                    ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
                ).strip(),
                "thread_environment": {
                    key: os.environ[key]
                    for key in (
                        "JAX_PLATFORMS",
                        "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS",
                    )
                },
            }
        )
        source_paths = [
            "scripts/run_smoke.py",
            "src/shinka_crl/experiment.py",
            "src/shinka_crl/profiles/smoke.json",
            "tasks/cartpole_ga/evaluate.py",
            "tasks/cartpole_ga/initial.py",
            "requirements/cpu.lock",
        ]
        environment["source_sha256"] = {
            path: hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
            for path in source_paths
        }
        environment["harness_dirty"] = bool(
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain"], text=True
            ).strip()
        )
        (output / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
        profile = load_profile("smoke")
        seed = profile["seeds"][0]
        trial = seed + 1
        for method in ("ga", "es", "ppo"):
            print(f"Running {method.upper()} smoke trial {trial} on CPU...", flush=True)
            summary = run_experiment(
                profile=profile,
                method=method,
                seed=seed,
                trial=trial,
                output_dir=output / "baselines" / "smoke" / method / f"trial_{trial}",
                upstream=args.upstream,
                python=args.python,
                timeout=args.timeout,
            )
            status["completed"].append(method)
            print(json.dumps(summary), flush=True)
        print("Evaluating the initial Shinka candidate (no model calls)...", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tasks/cartpole_ga/evaluate.py"),
                "--program_path",
                str(REPO_ROOT / "tasks/cartpole_ga/initial.py"),
                "--results_dir",
                str(output / "shinka"),
            ],
            check=True,
            cwd=REPO_ROOT,
            timeout=args.timeout + 60,
            env={
                **os.environ,
                "SHINKA_CRL_PROFILE": "smoke",
                "SHINKA_CRL_UPSTREAM": str(args.upstream.resolve()),
                "SHINKA_CRL_PYTHON": args.python,
                "SHINKA_CRL_TIMEOUT": str(args.timeout),
            },
        )
        status["completed"].append("shinka_initial")
        status["status"] = "complete"
    except Exception as exc:
        status.update(status="failed", error=str(exc))
        raise
    finally:
        status["wall_seconds"] = time.monotonic() - started
        (output / "suite.json").write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
