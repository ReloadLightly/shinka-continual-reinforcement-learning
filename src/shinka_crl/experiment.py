"""Launch unchanged upstream trainers and score their recorded evaluation curves."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from importlib.resources import files

REPO_ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_COMMIT = "821570eb6a22db0f7aa77111b2ea541fe8fa795b"
DEFAULT_UPSTREAM = REPO_ROOT / ".upstream" / "continual_neuroevolution"
DEFAULT_PYTHON = DEFAULT_UPSTREAM / ".venv" / "bin" / "python"


def load_profile(name: str) -> dict:
    if name not in {"smoke", "search", "paper-cartpole"}:
        raise ValueError(f"Unknown profile: {name}")
    return json.loads(files("shinka_crl").joinpath("profiles", f"{name}.json").read_text())


def validate_ga_settings(settings: dict) -> dict:
    bounds = {"sigma": (0.001, 2.0), "elite_ratio": (0.05, 0.95)}
    if not isinstance(settings, dict) or set(settings) != set(bounds):
        raise ValueError("GA settings must contain exactly sigma and elite_ratio")
    clean = {}
    for key, (low, high) in bounds.items():
        value = settings[key]
        if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key} must be a finite number in [{low}, {high}]")
        clean[key] = float(value)
    return clean


def verify_upstream(upstream: Path) -> str:
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_COMMIT:
        raise ValueError(f"Upstream revision mismatch: {revision}; expected {UPSTREAM_COMMIT}")
    dirty = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise ValueError("Upstream has modified tracked files; use an unchanged pinned checkout")
    # Untracked source files can shadow imports even when tracked files are clean.
    untracked = subprocess.check_output(
        ["git", "-C", str(upstream), "ls-files", "--others", "--exclude-standard",
         "source", "scripts", "third_party/kinetix"], text=True
    ).strip()
    if untracked:
        raise ValueError("Upstream has untracked source files; use a clean pinned checkout")
    return revision


def build_command(*, profile: dict, method: str, seed: int, output_dir: Path,
                  upstream: Path = DEFAULT_UPSTREAM, python: str = str(DEFAULT_PYTHON),
                  ga_settings: dict | None = None, trial: int = 1) -> list[str]:
    if method not in {"ga", "es", "ppo"}:
        raise ValueError("Supported baseline methods: ga, es, ppo")
    if ga_settings is not None and method != "ga":
        raise ValueError("GA overrides are valid only for method ga")
    command = [str(python), str(upstream.resolve() / "source/run.py"),
               "--suite", "gymnax", "--env", profile["env"], "--method", method,
               "--seed", str(seed), "--trial", str(trial),
               "--output_dir", str(output_dir.resolve()),
               "--num_phases", str(profile["num_phases"]),
               "--num_tasks", str(profile["num_tasks"]), "--task_type", "noise",
               "--eval_episodes", str(profile["eval_episodes"]),
               "--episode_length", str(profile["episode_length"])]
    if method in {"ga", "es"}:
        for key, value in profile["ne"].items():
            command.extend([f"--{key}", str(value)])
    else:
        budget = profile["ppo"]
        if budget is None:
            raise ValueError("This development profile does not define a PPO comparison")
        for key in ("num_updates", "task_interval"):
            command.extend([f"--{key}", str(budget[key])])
        # Paper runs use the unchanged upstream PPO hyperparameters.
        if profile["name"] != "paper-cartpole":
            command.append("--ppo_override")
            command.extend(f"{key}={budget[key]}" for key in
                           ("num_envs", "num_steps", "num_minibatches"))
    if ga_settings is not None:
        settings = validate_ga_settings(ga_settings)
        command.extend(["--ne_override", f"sigma={settings['sigma']}",
                        f"searcher_kwargs.elite_ratio={settings['elite_ratio']}"])
    return command


def score_curve(records: list[dict], *, profile: dict, method: str) -> dict:
    """Mean active-task centroid return / episode cap, a development objective.

    This is not the paper's full suite of continual-learning metrics. Each
    generation/update contributes once; missing, duplicate or corrupt rows fail.
    """
    budget = profile["ppo"] if method == "ppo" else profile["ne"]
    expected = budget["num_updates" if method == "ppo" else "num_generations"]
    if not isinstance(records, list) or len(records) != expected:
        raise ValueError(f"Expected {expected} metric rows, got {len(records)}")
    values = []
    for index, record in enumerate(records):
        task = (index // budget["task_interval"]) % profile["num_tasks"]
        if record.get("task") != task:
            raise ValueError(f"Unexpected task schedule at row {index}")
        # Both upstream loops serialize their zero-based step as generation.
        step_key = "generation"
        if record.get(step_key) != index:
            raise ValueError(f"Unexpected {step_key} at row {index}")
        value = record.get(f"centroid_task{task}")
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"Missing or non-finite centroid return at row {index}")
        if not 0 <= value <= profile["episode_length"]:
            raise ValueError(f"CartPole return outside the episode cap at row {index}")
        values.append(value)
    mean = sum(values) / len(values)
    return {"mean_return": mean, "normalized_score": mean / profile["episode_length"],
            "metric_rows": len(values)}


def run_experiment(*, profile: dict, method: str, seed: int, output_dir: Path,
                   upstream: Path = DEFAULT_UPSTREAM, python: str = str(DEFAULT_PYTHON),
                   ga_settings: dict | None = None, timeout: int = 1800,
                   trial: int = 1) -> dict:
    upstream, output_dir = Path(upstream).resolve(), Path(output_dir).resolve()
    revision = verify_upstream(upstream)
    interpreter = shutil.which(str(python))
    if interpreter is None:
        raise FileNotFoundError(f"Upstream Python not found: {python}; install its environment first")
    # Preserve virtualenv symlinks while making relative paths independent of cwd.
    python = str(Path(interpreter).absolute())
    command = build_command(profile=profile, method=method, seed=seed, output_dir=output_dir,
                            upstream=upstream, python=python, ga_settings=ga_settings, trial=trial)
    python_version = subprocess.check_output([str(python), "--version"], text=True).strip()
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {"status": "running", "profile": profile, "method": method,
                "seed": seed, "trial": trial, "upstream_commit": revision,
                "ga_settings": ga_settings, "command": command,
                "python_version": python_version,
                "jax_platforms": os.environ.get("JAX_PLATFORMS"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.monotonic()
    try:
        with (output_dir / "train.log").open("w") as log:
            subprocess.run(command, cwd=upstream, stdout=log, stderr=subprocess.STDOUT,
                           timeout=timeout, check=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
        metric_path = output_dir / "training_metrics.json"
        summary = score_curve(json.loads(metric_path.read_text()), profile=profile, method=method)
        summary.update({"profile": profile["name"], "method": method, "seed": seed,
                        "trial": trial, "upstream_commit": revision})
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        manifest.update(status="complete", metrics_sha256=hashlib.sha256(
            metric_path.read_bytes()).hexdigest())
        return summary
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
