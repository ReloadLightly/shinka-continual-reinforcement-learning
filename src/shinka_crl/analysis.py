"""Evaluate saved phase agents with the pinned upstream evaluator.

Training artifacts are immutable inputs. The upstream evaluator writes into a
fresh copy directory; the small reporting layer uses only the standard library.
"""

from __future__ import annotations

from bisect import bisect_right
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from shinka_crl.experiment import DEFAULT_PYTHON, DEFAULT_UPSTREAM, score_curve, verify_upstream

INPUT_FILES = ("manifest.json", "results.json", "config.json", "training_metrics.json",
               "checkpoints.npz")
OUTPUT_FILES = ("evaluation.json", "checkpoint-metadata.json", "summary.json", "process.log",
                "evaluation-input/results.json", "evaluation-input/checkpoints.npz")
UPSTREAM_EVALUATOR = "scripts/analysis/evaluate_continual.py"
UPSTREAM_INTEGRATOR = "source/metrics/continual_metrics.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _returns(entry: dict, key: str, episodes: int, cap: int) -> float:
    values = entry.get(key)
    if not isinstance(values, list) or len(values) != episodes:
        raise ValueError(f"Expected {episodes} evaluation episodes in {key}")
    if any(type(value) not in (int, float) or not math.isfinite(value)
           or not 0 <= value <= cap for value in values):
        raise ValueError(f"Invalid CartPole evaluation return in {key}")
    return _mean(values)


def _interpolate(x: list[float], y: list[float], point: float) -> float:
    """Linear interpolation with the pinned evaluator's constant edge values."""
    if point <= x[0]:
        return y[0]
    if point >= x[-1]:
        return y[-1]
    right = bisect_right(x, point)
    left = right - 1
    fraction = (point - x[left]) / (x[right] - x[left])
    return y[left] + fraction * (y[right] - y[left])


def integrate_curve(values: list[float], *, steps_per_update: int,
                    reference_steps_per_generation: int) -> dict:
    """Pinned Cum discretization, plus the exact logged-curve area.

The reference plots put dense Gymnax rows at completed-update steps, then
convert that clock to NE generation equivalents. ``cumulative_reward`` samples
this curve on ``linspace(0, G, int(G)+1)`` before trapezoidal integration. In
particular, this subsamples PPO: integrating its dense log directly differs.
"""
    if len(values) < 2 or steps_per_update <= 0 or reference_steps_per_generation <= 0:
        raise ValueError("Cumulative return needs at least two rows and positive step sizes")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Cumulative return requires finite values")
    total_steps = len(values) * steps_per_update
    x = [(index + 1) * steps_per_update / reference_steps_per_generation
         for index in range(len(values))]
    upper = total_steps / reference_steps_per_generation
    points = max(int(upper) + 1, 2)
    grid = [index * upper / (points - 1) for index in range(points)]
    sampled = [_interpolate(x, values, point) for point in grid]
    generation_integral = sum(
        (grid[i + 1] - grid[i]) * (sampled[i] + sampled[i + 1]) / 2
        for i in range(points - 1))
    exact = steps_per_update * (values[0] + sum(
        (values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)))
    return {"cumulative_reward_steps": generation_integral * reference_steps_per_generation,
            "cumulative_reward_generation_equivalents": generation_integral,
            "cumulative_reward_steps_exact": exact,
            "integration_grid_points": points}


def summarize_trial(*, manifest: dict, results: dict, records: list[dict],
                    evaluation: dict, checkpoint_metadata: dict, episodes: int,
                    eval_seed: int) -> dict:
    """Validate identities and derive reward-unit metrics from raw evidence."""
    if type(episodes) is not int or episodes <= 0:
        raise ValueError("episodes must be a positive integer")
    if type(eval_seed) is not int or eval_seed < 0 or eval_seed == manifest["seed"]:
        raise ValueError("Use a recorded nonnegative evaluation seed distinct from training")
    if manifest.get("status") != "complete":
        raise ValueError("Training must be complete before checkpoint analysis")
    profile, method = manifest["profile"], manifest["method"]
    if method not in {"ga", "es", "ppo"}:
        raise ValueError("Unsupported analysis method")
    cfg = {**results.get("config", {}),
           **{key: value for key, value in results.items() if key != "config"}}
    phase_count, cap = profile["num_phases"], profile["episode_length"]
    phase_sequence = [phase % profile["num_tasks"] for phase in range(phase_count)]
    expected_cfg = {"method": method, "seed": manifest["seed"], "trial": manifest["trial"],
                    "env": profile["env"].split("_sigma")[0], "episode_length": cap,
                    "num_tasks": profile["num_tasks"], "task_sequence": phase_sequence,
                    "eval_episodes": profile["eval_episodes"]}
    budget = profile["ppo"] if method == "ppo" else profile["ne"]
    count_key = "num_updates" if method == "ppo" else "num_generations"
    updates = budget[count_key]
    expected_cfg.update(num_generations=updates, task_interval=budget["task_interval"])
    if updates != phase_count * budget["task_interval"] or cfg.get("task_warmup", 0):
        raise ValueError("Analysis requires equal complete phases with no task warmup")
    if method == "ppo":
        expected_cfg.update({key: budget[key] for key in
                             ("num_envs", "num_steps", "num_minibatches")})
        step_size = budget["num_envs"] * budget["num_steps"]
    else:
        expected_cfg.update(pop_size=budget["pop_size"], num_evals=budget["num_evals"])
        step_size = budget["pop_size"] * budget["num_evals"] * cap
    for key, expected in expected_cfg.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Resolved configuration mismatch for {key}")
    total_steps = updates * step_size
    if (cfg.get("num_timesteps") if method == "ppo" else results.get("env_steps")) != total_steps:
        raise ValueError("Resolved nominal training budget mismatch")
    score_curve(records, profile=profile, method=method)
    source = "final" if method == "ppo" else "centroid"
    expected_eval = {"method": method, "env": cfg["env"], "trial": manifest["trial"],
                     "pop_size": cfg["pop_size"], "episodes": episodes,
                     "eval_seed": eval_seed, "num_tasks": phase_count}
    for key, expected in expected_eval.items():
        if evaluation.get(key) != expected:
            raise ValueError(f"Post-hoc evaluation identity mismatch for {key}")
    checkpoint_sources = checkpoint_metadata.get("sources", {})
    if source not in checkpoint_sources:
        raise ValueError(f"Missing checkpoint source: {source}")
    if any(shape != [phase_count, cfg["num_params"]]
           for shape in checkpoint_sources.values()):
        raise ValueError("Checkpoint source shape does not match phase count or policy")
    vectors = results.get("noise_vectors")
    if not isinstance(vectors, list) or len(vectors) != profile["num_tasks"]:
        raise ValueError("Missing distinct task vectors")
    expected_vectors = [vectors[task] for task in phase_sequence]
    if checkpoint_metadata.get("noise_vectors") != expected_vectors:
        raise ValueError("Checkpoint task vectors do not match phase identities")
    if checkpoint_metadata.get("finite") is not True:
        raise ValueError("Non-finite checkpoint policy or task vector")
    declared_sources = evaluation.get("agent_sources", [])
    if len(declared_sources) != len(set(declared_sources)) or set(declared_sources) != set(
            checkpoint_sources):
        raise ValueError("Evaluation sources do not match saved checkpoint sources")
    entries = {}
    for entry in evaluation.get("per_task", []):
        phase = entry.get("task_idx")
        key = (entry.get("source"), phase)
        if (type(phase) is not int or not 0 <= phase < phase_count
                or key[0] not in declared_sources or key in entries):
            raise ValueError("Duplicate or invalid evaluation source/phase identity")
        parsed = {"own_mean": _returns(entry, "returns", episodes, cap),
                  "previous_mean": None, "next_mean": None}
        for name, field, required in (("previous_mean", "prev_returns", phase > 0),
                                      ("next_mean", "zero_shot_next_returns",
                                       phase < phase_count - 1)):
            if required:
                parsed[name] = _returns(entry, field, episodes, cap)
            elif field in entry:
                raise ValueError(f"Unexpected {field} at phase {phase}")
        entries[key] = parsed
    if len(entries) != phase_count * len(declared_sources):
        raise ValueError("Missing checkpoint evaluation source/phase entries")
    phases = [{"phase": phase, "task": task,
               "checkpoint_completed_steps": (phase + 1) * budget["task_interval"] * step_size,
               **entries[(source, phase)]}
              for phase, task in enumerate(phase_sequence)]
    learning = _mean([phase["own_mean"] for phase in phases])
    stationary = profile["num_tasks"] == 1
    forgetting = transfer = None
    if not stationary and phase_count > 1:
        forgetting = _mean([phases[i]["own_mean"] - phases[i + 1]["previous_mean"]
                            for i in range(phase_count - 1)])
        transfer = _mean([phases[i]["next_mean"] for i in range(phase_count - 1)])
    active = [record[f"centroid_task{record['task']}"] for record in records]
    reference_size = profile["ne"]["pop_size"] * profile["ne"]["num_evals"] * cap
    integral = integrate_curve(active, steps_per_update=step_size,
                               reference_steps_per_generation=reference_size)
    return {"analysis_version": 1, "method": method, "profile": profile["name"],
            "condition": "stationary" if stationary else "switching",
            "seed": manifest["seed"], "trial": manifest["trial"], "agent_source": source,
            "episodes": episodes, "eval_seed": eval_seed, "num_phases": phase_count,
            "phase_sequence": phase_sequence, "nominal_training_steps": total_steps,
            "steps_per_update": step_size,
            "metrics": {"learning_accuracy": learning, "forgetting": forgetting,
                        "zero_shot_transfer": transfer,
                        "learning_minus_forgetting": None if stationary else learning - forgetting,
                        **{key: value for key, value in integral.items()
                           if key != "integration_grid_points"},
                        "normalized_curve_average": integral["cumulative_reward_steps"]
                        / total_steps / cap},
            "phase_returns": phases,
            "phase_training_returns": [_mean(active[start:start + budget["task_interval"]])
                                       for start in range(0, updates, budget["task_interval"])],
            "clock": {"record_timing": "after_update", "raw_generation_labels": "zero_based",
                      "completed_steps": "(generation + 1) * steps_per_update",
                      "integration": "pinned_unit_NE_generation_grid_trapezoid",
                      "reference_steps_per_generation": reference_size,
                      "integration_grid_points": integral["integration_grid_points"],
                      "edge_values": "constant_first_and_last_logged_return",
                      "initial_untrained_policy_evaluated": False},
            "transfer_interpretation": "not_applicable_stationary" if stationary
            else "consecutive_switches_including_both_directions"}


def _summarize_paths(run_dir: Path, output_dir: Path, episodes: int, eval_seed: int) -> dict:
    def read(path: Path):
        return json.loads(path.read_text())
    return summarize_trial(manifest=read(run_dir / "manifest.json"),
                           results=read(run_dir / "results.json"),
                           records=read(run_dir / "training_metrics.json"),
                           evaluation=read(output_dir / "evaluation.json"),
                           checkpoint_metadata=read(output_dir / "checkpoint-metadata.json"),
                           episodes=episodes, eval_seed=eval_seed)


def validate_analysis(*, run_dir: Path, output_dir: Path, episodes: int = 10,
                      eval_seed: int) -> dict:
    """Verify completed artifacts and rederive metrics without loading JAX."""
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("episodes") != episodes \
            or manifest.get("eval_seed") != eval_seed:
        raise ValueError("Analysis is incomplete or uses a different evaluation protocol")
    if manifest.get("analysis_source_sha256") != _sha256(Path(__file__)):
        raise ValueError("Analysis source changed since evaluation")
    for name in INPUT_FILES:
        if _sha256(run_dir / name) != manifest["input_sha256"].get(name):
            raise ValueError(f"Analysis input changed: {name}")
    for name in OUTPUT_FILES:
        if _sha256(output_dir / name) != manifest["output_sha256"].get(name):
            raise ValueError(f"Analysis output changed: {name}")
    summary = _summarize_paths(run_dir, output_dir, episodes, eval_seed)
    if summary != json.loads((output_dir / "summary.json").read_text()):
        raise ValueError("Saved analysis summary does not match raw evaluation")
    return summary


def run_analysis(*, run_dir: Path, output_dir: Path, eval_seed: int,
                 upstream: Path = DEFAULT_UPSTREAM, python: str = str(DEFAULT_PYTHON),
                 episodes: int = 10, timeout: int = 600) -> dict:
    """Run unchanged upstream checkpoint evaluation into a fresh directory."""
    run_dir, output_dir, upstream = (Path(path).resolve()
                                    for path in (run_dir, output_dir, upstream))
    revision = verify_upstream(upstream)
    training_manifest = json.loads((run_dir / "manifest.json").read_text())
    if training_manifest.get("status") != "complete":
        raise ValueError("Training must be complete before checkpoint analysis")
    if type(episodes) is not int or episodes <= 0 or type(eval_seed) is not int \
            or eval_seed < 0 or eval_seed == training_manifest["seed"]:
        raise ValueError("Use positive episodes and an independent nonnegative evaluation seed")
    if training_manifest["upstream_commit"] != revision:
        raise ValueError("Training and evaluation source revisions differ")
    interpreter = shutil.which(str(python))
    if interpreter is None:
        raise FileNotFoundError(f"Upstream Python not found: {python}")
    python = str(Path(interpreter).absolute())
    input_hashes = {name: _sha256(run_dir / name) for name in INPUT_FILES}
    if input_hashes["training_metrics.json"] != training_manifest["metrics_sha256"]:
        raise ValueError("Training metrics hash does not match completed training manifest")
    output_dir.mkdir(parents=True, exist_ok=False)
    copied = output_dir / "evaluation-input"
    command = [python, str(upstream / UPSTREAM_EVALUATOR), "--root", str(copied),
               "--episodes", str(episodes), "--seed", str(eval_seed)]
    manifest = {"status": "running", "upstream_commit": revision,
                "episodes": episodes, "eval_seed": eval_seed, "command": command,
                "training_run": str(run_dir), "input_sha256": input_hashes,
                "analysis_source_sha256": _sha256(Path(__file__)),
                "upstream_source_sha256": {name: _sha256(upstream / name)
                                           for name in (UPSTREAM_EVALUATOR, UPSTREAM_INTEGRATOR)},
                "jax_platforms": os.environ.get("JAX_PLATFORMS"),
                "python_version": subprocess.check_output([python, "--version"], text=True).strip()}
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        copied.mkdir()
        for name in ("results.json", "checkpoints.npz"):
            shutil.copyfile(run_dir / name, copied / name)
        # NumPy is supplied by the pinned training environment, not required by
        # the lightweight reporting package. Never load object/pickle arrays.
        probe = """
import json, sys
import numpy as np
with np.load(sys.argv[1], allow_pickle=False) as data:
    sources = {name: list(data[name].shape) for name in data.files if name != 'noise_vectors'}
    result = {'sources': sources, 'noise_vectors': data['noise_vectors'].tolist(),
              'finite': all(bool(np.isfinite(data[name]).all()) for name in data.files)}
print(json.dumps(result, allow_nan=False))
"""
        metadata = json.loads(subprocess.check_output(
            [python, "-c", probe, str(copied / "checkpoints.npz")], text=True, timeout=60))
        _write_json(output_dir / "checkpoint-metadata.json", metadata)
        with (output_dir / "process.log").open("w") as log:
            subprocess.run(command, cwd=upstream, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=timeout,
                           env={**os.environ, "PYTHONUNBUFFERED": "1"})
        shutil.copyfile(copied / "evaluation.json", output_dir / "evaluation.json")
        summary = _summarize_paths(run_dir, output_dir, episodes, eval_seed)
        _write_json(output_dir / "summary.json", summary)
        if any(_sha256(run_dir / name) != digest for name, digest in input_hashes.items()):
            raise ValueError("Training inputs changed during checkpoint evaluation")
        manifest.update(status="complete", output_sha256={
            name: _sha256(output_dir / name) for name in OUTPUT_FILES})
        return summary
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest["wall_seconds"] = time.monotonic() - started
        _write_json(manifest_path, manifest)
