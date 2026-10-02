"""Frozen active/previous-task objective for the adaptive CartPole extension.

The caller verifies artifact, program, source, and runtime receipts. This module
binds the score to ordered training curves and fresh centroid episode returns,
and independently checks the corresponding analysis summary. It does not read
files, evaluate policies, authorize seed partitions, or change reference metrics.
"""

from __future__ import annotations

import math

from shinka_crl.analysis import integrate_curve
from shinka_crl.experiment import nominal_training_steps, score_curve

OBJECTIVE_VERSION = "adaptive-active-previous-v1"
EPISODE_CAP = 500


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _identity(actual, expected, label: str) -> None:
    _require(type(actual) is type(expected) and actual == expected,
             f"Adaptive objective identity mismatch: {label}")


def _number(value, label: str, *, low: float | None = None,
            high: float | None = None) -> float:
    _require(type(value) in (int, float) and math.isfinite(value),
             f"Expected finite numeric {label}")
    _require((low is None or value >= low) and (high is None or value <= high),
             f"Out-of-bounds {label}")
    return float(value)


def _matches(actual, expected: float, label: str) -> None:
    actual = _number(actual, label)
    _require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10),
             f"Analysis differs from raw evidence: {label}")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _episode_mean(entry: dict, field: str, episodes: int) -> float:
    values = entry.get(field)
    _require(isinstance(values, list) and len(values) == episodes,
             f"Missing or incomplete {field} episode vector")
    return _mean([_number(value, field, low=0, high=EPISODE_CAP) for value in values])


def score_adaptive_trial(*, profile: dict, records: list[dict], analysis: dict,
                         evaluation: dict, method: str, seed: int, trial: int,
                         eval_seed: int, episodes: int = 10) -> dict:
    """Return 0.5 active score + 0.5 fresh previous-task score and reference metrics.

    ``method`` remains the true native ``ga`` or ``ga_focus`` identity. The
    caller separately retains ``algorithm_variant`` and candidate provenance.
    Previously trained task returns refer to each immediately preceding phase,
    including both directions of the repeated A/B task sequence. Low previous
    acquisition cannot improve this term merely by reducing signed forgetting.
    """
    _require(method in {"ga", "ga_focus"}, "Adaptive objective supports GA controls only")
    _require(type(seed) is int and 0 <= seed < 2**32, "Invalid training seed")
    _require(type(trial) is int and trial > 0, "Invalid task trial")
    _require(type(eval_seed) is int and 0 <= eval_seed < 2**32 and eval_seed != seed,
             "Evaluation seed must be a distinct nonnegative 32-bit integer")
    _require(type(episodes) is int and episodes > 0, "Evaluation episodes must be positive")
    active = score_curve(records, profile=profile, method=method)
    _require(profile["env"] == "CartPole-v1_sigma0.5" and profile["num_tasks"] == 2
             and profile["num_phases"] >= 2 and profile["episode_length"] == EPISODE_CAP,
             "Adaptive objective requires switching CartPole with episode cap 500")
    _require(seed in profile["seeds"], "Training seed is outside the supplied profile")
    _require(isinstance(analysis, dict) and isinstance(evaluation, dict),
             "Analysis and evaluation must be objects")
    phases, interval = profile["num_phases"], profile["ne"]["task_interval"]
    sequence = [i % 2 for i in range(phases)]
    steps = profile["ne"]["pop_size"] * profile["ne"]["num_evals"] * EPISODE_CAP
    identity = {
        "analysis_version": 1, "method": method, "profile": profile["name"],
        "condition": "switching", "seed": seed, "trial": trial,
        "agent_source": "centroid", "episodes": episodes, "eval_seed": eval_seed,
        "num_phases": phases, "phase_sequence": sequence,
        "nominal_training_steps": nominal_training_steps(profile, method),
        "steps_per_update": steps,
        "transfer_interpretation": "consecutive_switches_including_both_directions",
    }
    for name, expected in identity.items():
        _identity(analysis.get(name), expected, name)
    for name, expected in {
        "method": method, "env": "CartPole-v1", "trial": trial,
        "pop_size": profile["ne"]["pop_size"], "episodes": episodes,
        "eval_seed": eval_seed, "num_tasks": phases,
    }.items():
        _identity(evaluation.get(name), expected, f"evaluation.{name}")

    sources = evaluation.get("agent_sources")
    _require(isinstance(sources, list) and sources and all(type(s) is str for s in sources),
             "Evaluation sources must be a nonempty list")
    _require(len(sources) == len(set(sources)) and "centroid" in sources
             and set(sources) <= {"finalgen", "incumbent", "centroid"},
             "Evaluation must declare the centroid and distinct native sources")
    entries = evaluation.get("per_task")
    _require(isinstance(entries, list) and all(isinstance(entry, dict) for entry in entries),
             "Missing evaluation entries")
    expected_order = [(source, phase) for source in sources for phase in range(phases)]
    _require([(entry.get("source"), entry.get("task_idx")) for entry in entries] == expected_order
             and all(type(entry.get("task_idx")) is int for entry in entries),
             "Missing, duplicated, reordered, or incorrect evaluation source/phase")
    centroid = []
    for entry in entries:
        phase = entry["task_idx"]
        fields = {"returns"}
        if phase > 0:
            fields.add("prev_returns")
        if phase < phases - 1:
            fields.add("zero_shot_next_returns")
        _require(set(entry) == {"source", "task_idx", *fields},
                 "Unexpected evaluation return-vector fields")
        values = {field: _episode_mean(entry, field, episodes) for field in fields}
        if entry["source"] == "centroid":
            centroid.append({"phase": phase, "task": sequence[phase],
                             "checkpoint_completed_steps": (phase + 1) * interval * steps,
                             "own_mean": values["returns"],
                             "previous_mean": values.get("prev_returns"),
                             "next_mean": values.get("zero_shot_next_returns")})

    reported_phases = analysis.get("phase_returns")
    _require(isinstance(reported_phases, list) and len(reported_phases) == phases,
             "Missing analysis phase returns")
    for expected, reported in zip(centroid, reported_phases, strict=True):
        _require(isinstance(reported, dict) and set(reported) == set(expected),
                 "Unexpected analysis phase-return fields")
        for field in ("phase", "task", "checkpoint_completed_steps"):
            _identity(reported[field], expected[field], f"phase_returns.{field}")
        for field in ("own_mean", "previous_mean", "next_mean"):
            if expected[field] is None:
                _identity(reported[field], None, f"phase_returns.{field}")
            else:
                _matches(reported[field], expected[field], f"phase_returns.{field}")

    active_returns = [row[f"centroid_task{row['task']}"] for row in records]
    phase_active = [_mean(active_returns[start:start + interval])
                    for start in range(0, len(records), interval)]
    reported_active = analysis.get("phase_training_returns")
    _require(isinstance(reported_active, list) and len(reported_active) == phases,
             "Missing analysis phase training returns")
    for actual, expected in zip(reported_active, phase_active, strict=True):
        _matches(actual, expected, "phase_training_returns")
    switches = [{"from_phase": i, "to_phase": i + 1, "task": centroid[i]["task"],
                 "before": centroid[i]["own_mean"], "after": centroid[i + 1]["previous_mean"],
                 "forgetting": centroid[i]["own_mean"] - centroid[i + 1]["previous_mean"]}
                for i in range(phases - 1)]
    learning = _mean([row["own_mean"] for row in centroid])
    forgetting = _mean([row["forgetting"] for row in switches])
    integral = integrate_curve(active_returns, steps_per_update=steps,
                               reference_steps_per_generation=steps)
    metrics = {
        "learning_accuracy": learning, "forgetting": forgetting,
        "zero_shot_transfer": _mean([row["next_mean"] for row in centroid[:-1]]),
        "learning_minus_forgetting": learning - forgetting,
        **{key: value for key, value in integral.items() if key != "integration_grid_points"},
        "normalized_curve_average": integral["cumulative_reward_steps"]
        / nominal_training_steps(profile, method) / EPISODE_CAP,
    }
    reported_metrics = analysis.get("metrics")
    _require(isinstance(reported_metrics, dict) and set(reported_metrics) == set(metrics),
             "Unexpected reference metric fields")
    for field, expected in metrics.items():
        _matches(reported_metrics[field], expected, field)
    previous_return = _mean([row["after"] for row in switches])
    previous = previous_return / EPISODE_CAP
    combined = 0.5 * active["normalized_score"] + 0.5 * previous
    _number(combined, "combined score", low=0, high=1)
    return {
        "objective_version": OBJECTIVE_VERSION,
        "combined_score": combined, "active_score": active["normalized_score"],
        "previous_score": previous, "active_mean_return": active["mean_return"],
        "previous_mean_return": previous_return, "active_checkpoints": len(records),
        "previous_checkpoints": phases - 1, "switches": switches, "reference_metrics": metrics,
    }
