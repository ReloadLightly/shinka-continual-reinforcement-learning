"""Recorded optimizer and architecture contracts for the pinned CartPole baselines.

Budget, task, and seed checks belong to the experiment protocol. This module
checks only the architecture and active baseline settings serialized by upstream
commit 821570eb6a22db0f7aa77111b2ea541fe8fa795b. PPO's GAE lambda, value-loss
coefficient, and gradient clipping are not serialized: their pinned values
(0.95, 0.5, and 0.0) are protected by source verification and the exact command,
not inferred from these artifacts. Unused GA optimizer fields and PPO sigma
placeholders are deliberately outside this contract.
"""

from __future__ import annotations

FOCUS_SIGMA = 0.5
FOCUS_SEARCHER_KWARGS = {
    "elite_ratio": 0.5,
    "init_around_mean": False,
    "cross_over_rate": 0.0,
    "focus_rate": 0.3,
    "sigma_rate": 0.1,
    "track_target": 0.9,
    "sigma_min": 0.00001,
    "explore_fraction": 0.25,
}


def _expect(actual: object, expected: object, label: str) -> None:
    """Compare JSON values without treating booleans as numbers."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"Baseline config mismatch: {label}")
        for key, value in expected.items():
            _expect(actual[key], value, f"{label}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise ValueError(f"Baseline config mismatch: {label}")
        for index, value in enumerate(expected):
            _expect(actual[index], value, f"{label}[{index}]")
    elif (type(actual) is bool) != (type(expected) is bool) or actual != expected:
        raise ValueError(f"Baseline config mismatch: {label}; expected {expected!r}")


def validate_baseline_config(config: dict, method: str) -> None:
    """Reject drift from fixed baseline settings; not a candidate validator.

    ``ga_focus`` is the explicit CartPole transfer of the paper's FocusGA,
    including CartPole's independent archive initialization. Its centroid
    occupies one slot inside the population's fixed evaluation budget.
    """
    if method not in {"ga", "ga_focus", "es", "ppo"}:
        raise ValueError(f"Unsupported baseline method: {method}")
    if not isinstance(config, dict):
        raise ValueError("Baseline config must be an object")
    expected = {
        "env": "CartPole-v1", "method": method, "hidden_dims": [16, 16],
        "num_params": 386, "first_task_clean": True, "task_warmup": 0,
    }
    if method in {"ga", "ga_focus", "es"}:
        expected.update(objective="mean", obs_norm=False)
    if method in {"ga", "ga_focus"}:
        population = config.get("pop_size")
        if type(population) is not int or population < 2 or population % 2:
            raise ValueError("Baseline GA pop_size must be a positive even integer")
        expected.update(
            sigma=0.5,
            searcher_kwargs={"elite_ratio": 0.5, "init_around_mean": False},
            searcher_resolved={"refresh": True, "num_elites": population // 2,
                               "num_offspring": population // 2, "variation": "gaussian",
                               "sigma": 0.5, "cross_over_rate": 0.0},
        )
        if method == "ga_focus":
            if population < 4:
                raise ValueError("FocusGA needs room for a centroid and at least one offspring")
            expected["sigma"] = FOCUS_SIGMA
            expected["searcher_kwargs"] = FOCUS_SEARCHER_KWARGS
            expected["searcher_resolved"]["num_offspring"] -= 1
    elif method == "es":
        expected.update(sigma=0.1, learning_rate=0.05, optimizer="sgd", shaping="zscore",
                        sigma_lr=0.0, searcher_kwargs={}, searcher_resolved={})
    else:
        expected.update(
            learning_rate=0.0003, optimizer="adam", adam_b1=0.9, num_epochs=10,
            gamma=0.95, ent_coef=0.01, clip_eps=0.2, head="categorical",
            normalize_obs=False, reward_scale=1.0, eval_interval=1,
            value_hidden_dims=[128, 128, 128],
        )
    for key, value in expected.items():
        if key not in config:
            raise ValueError(f"Missing baseline config field: {key}")
        _expect(config[key], value, key)
