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
PROFILE_NAMES = ("smoke", "search", "pilot-stationary", "pilot-switching", "paper-cartpole",
                 "paper-cartpole-timing", "cartpole-validation")


def _positive_integer(value: object, label: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")


def nominal_training_steps(profile: dict, method: str) -> int:
    """Configured training transitions, excluding evaluation and diagnostics.

    NE counts episode caps, including the masked steps after termination. This
    matches the reference budget convention, not useful transitions or FLOPs.
    """
    if method in {"ga", "es"}:
        ne = profile["ne"]
        return (ne["num_generations"] * ne["pop_size"] * ne["num_evals"]
                * profile["episode_length"])
    if method != "ppo":
        raise ValueError("Supported baseline methods: ga, es, ppo")
    if profile["ppo"] is None:
        raise ValueError("This development profile does not define a PPO comparison")
    ppo = profile["ppo"]
    return ppo["num_updates"] * ppo["num_envs"] * ppo["num_steps"]


def validate_profile(profile: dict) -> None:
    """Reject inconsistent budgets before an upstream process is launched."""
    keys = {"name", "purpose", "env", "num_phases", "num_tasks", "episode_length",
            "eval_episodes", "seeds", "ne", "ppo"}
    if not isinstance(profile, dict) or set(profile) != keys:
        raise ValueError("Profile must contain exactly the documented protocol fields")
    if profile["name"] not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile: {profile['name']}")
    if not isinstance(profile["purpose"], str) or not profile["purpose"].strip():
        raise ValueError("Profile purpose must be a nonempty string")
    for key in ("num_phases", "num_tasks", "episode_length", "eval_episodes"):
        _positive_integer(profile[key], key)
    if profile["episode_length"] > 500:
        raise ValueError("CartPole episode_length must not exceed 500")
    task_counts = {"CartPole-v1": 1, "CartPole-v1_sigma0.5": 2}
    if (not isinstance(profile["env"], str) or profile["env"] not in task_counts
            or profile["num_tasks"] != task_counts[profile["env"]]):
        raise ValueError("Profile environment and number of tasks are inconsistent")
    if profile["num_phases"] % profile["num_tasks"]:
        raise ValueError("Profile phases must contain complete task cycles")
    seeds = profile["seeds"]
    if (not isinstance(seeds, list) or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("Profile seeds must be distinct nonnegative 32-bit integers")
    expected_keys = {
        "ne": {"num_generations", "task_interval", "pop_size", "num_evals"},
        "ppo": {"num_updates", "task_interval", "num_envs", "num_steps", "num_minibatches"},
    }
    for family, fields in expected_keys.items():
        budget = profile[family]
        if family == "ppo" and budget is None:
            if profile["name"] not in {"search", "paper-cartpole-timing", "cartpole-validation"}:
                raise ValueError("Only GA development profiles may omit a PPO budget")
            continue
        if not isinstance(budget, dict) or set(budget) != fields:
            raise ValueError(f"Unexpected {family} budget fields")
        for key, value in budget.items():
            _positive_integer(value, f"{family}.{key}")
        length = budget["num_generations" if family == "ne" else "num_updates"]
        if length != profile["num_phases"] * budget["task_interval"]:
            raise ValueError(f"{family} budget does not match the phase grid")
    if profile["ne"]["pop_size"] % 2:
        raise ValueError("NE population must be even for the ES antithetic pairs")
    ppo = profile["ppo"]
    if ppo is not None:
        if (ppo["num_envs"] * ppo["num_steps"]) % ppo["num_minibatches"]:
            raise ValueError("PPO rollout batch must divide evenly into minibatches")
        if (profile["name"] != "smoke"
                and nominal_training_steps(profile, "ga") != nominal_training_steps(profile, "ppo")):
            raise ValueError("Comparison profiles require matched nominal training budgets")


def load_profile(name: str) -> dict:
    if name not in PROFILE_NAMES:
        raise ValueError(f"Unknown profile: {name}")
    profile = json.loads(files("shinka_crl").joinpath("profiles", f"{name}.json").read_text())
    validate_profile(profile)
    if profile["name"] != name:
        raise ValueError("Profile resource name does not match its declared name")
    return profile


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
    validate_profile(profile)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be a nonnegative 32-bit integer")
    _positive_integer(trial, "trial")
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
    validate_profile(profile)
    nominal_training_steps(profile, method)  # Validate the method and presence of its budget.
    budget = profile["ppo"] if method == "ppo" else profile["ne"]
    expected = budget["num_updates" if method == "ppo" else "num_generations"]
    if not isinstance(records, list):
        raise ValueError("Metric records must be a list")
    if len(records) != expected:
        raise ValueError(f"Expected {expected} metric rows, got {len(records)}")
    values = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Metric row {index} must be an object")
        task = (index // budget["task_interval"]) % profile["num_tasks"]
        if type(record.get("task")) is not int or record["task"] != task:
            raise ValueError(f"Unexpected task schedule at row {index}")
        # Both upstream loops serialize their zero-based step as generation.
        step_key = "generation"
        if type(record.get(step_key)) is not int or record[step_key] != index:
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
        # Upstream creates train.log itself; sharing that path corrupts both
        # streams when its Tee reopens the file. Capture the process separately.
        with (output_dir / "process.log").open("w") as log:
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
