"""Guard active optimizer settings while ignoring irrelevant artifact fields."""

import pytest

from shinka_crl.baseline_contract import validate_baseline_config


def config_for(method):
    config = {"env": "CartPole-v1", "method": method, "hidden_dims": [16, 16],
              "num_params": 386, "first_task_clean": True, "task_warmup": 0}
    if method in {"ga", "es"}:
        config.update(objective="mean", obs_norm=False)
    if method == "ga":
        config.update(pop_size=64, sigma=0.5,
                      searcher_kwargs={"elite_ratio": 0.5, "init_around_mean": False},
                      searcher_resolved={"refresh": True, "num_elites": 32,
                                         "num_offspring": 32, "variation": "gaussian",
                                         "sigma": 0.5, "cross_over_rate": 0.0})
    elif method == "es":
        config.update(sigma=0.1, learning_rate=0.05, optimizer="sgd", shaping="zscore",
                      sigma_lr=0.0, searcher_kwargs={}, searcher_resolved={})
    else:
        config.update(learning_rate=0.0003, optimizer="adam", adam_b1=0.9,
                      num_epochs=10, gamma=0.95, ent_coef=0.01, clip_eps=0.2,
                      head="categorical", normalize_obs=False, reward_scale=1.0,
                      eval_interval=1, value_hidden_dims=[128, 128, 128])
    return config


@pytest.mark.parametrize("method", ["ga", "es", "ppo"])
def test_fixed_defaults_and_unrelated_fields_are_accepted(method):
    config = config_for(method)
    config.update(seed=1001, trial=1002, schedule="switch", num_generations=80)
    if method == "ga":
        config.update(optimizer="unused", learning_rate=999)
    elif method == "ppo":
        config.update(sigma=float("nan"), shaping="n/a")
    validate_baseline_config(config, method)


@pytest.mark.parametrize("method,key,value", [
    ("ga", "sigma", 0.1),
    ("ga", "objective", "worstk"),
    ("ga", "obs_norm", True),
    ("es", "learning_rate", 0.1),
    ("es", "optimizer", "adam"),
    ("es", "shaping", "centered_rank"),
    ("es", "sigma_lr", 0.01),
    ("ppo", "gamma", 0.99),
    ("ppo", "num_epochs", 4),
    ("ppo", "learning_rate", 0.001),
    ("ppo", "normalize_obs", True),
    ("ppo", "value_hidden_dims", [16, 16]),
    ("ppo", "hidden_dims", [64, 64]),
    ("ppo", "eval_interval", True),
    ("ga", "first_task_clean", 1),
])
def test_optimizer_and_architecture_drift_is_rejected(method, key, value):
    config = config_for(method)
    config[key] = value
    with pytest.raises(ValueError, match=key):
        validate_baseline_config(config, method)


@pytest.mark.parametrize("key,value", [
    ("refresh", False), ("num_elites", 16), ("num_offspring", 48),
    ("variation", "iso_line"), ("sigma", 0.1), ("cross_over_rate", 0.5),
])
def test_ga_resolved_survivor_and_mutation_contract(key, value):
    config = config_for("ga")
    config["searcher_resolved"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_baseline_config(config, "ga")


def test_hidden_searcher_overrides_and_missing_fields_fail():
    config = config_for("ga")
    config["searcher_kwargs"]["refresh"] = False
    with pytest.raises(ValueError, match="searcher_kwargs"):
        validate_baseline_config(config, "ga")
    config = config_for("ppo")
    del config["ent_coef"]
    with pytest.raises(ValueError, match="Missing.*ent_coef"):
        validate_baseline_config(config, "ppo")


def test_ga_elite_counts_follow_population_size():
    config = config_for("ga")
    config["pop_size"] = 512
    config["searcher_resolved"].update(num_elites=256, num_offspring=256)
    validate_baseline_config(config, "ga")
