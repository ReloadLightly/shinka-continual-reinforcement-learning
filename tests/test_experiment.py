import math
from pathlib import Path

import pytest

from shinka_crl.experiment import (
    PROFILE_NAMES, build_command, load_profile, nominal_training_steps, score_curve,
    validate_ga_settings, validate_profile,
)


@pytest.mark.parametrize("name", PROFILE_NAMES)
def test_packaged_profiles_have_valid_contracts(name):
    profile = load_profile(name)
    validate_profile(profile)
    assert profile["name"] == name


@pytest.mark.parametrize("name,tasks,env", [
    ("pilot-stationary", 1, "CartPole-v1"),
    ("pilot-switching", 2, "CartPole-v1_sigma0.5"),
])
def test_pilot_profiles_match_total_and_per_phase_budgets(name, tasks, env):
    profile = load_profile(name)
    assert profile["env"] == env
    assert profile["num_tasks"] == tasks
    assert profile["num_phases"] == 4
    assert profile["eval_episodes"] == 10
    assert profile["seeds"] == [1001, 1002, 1003]
    for method in ("ga", "es", "ppo"):
        assert nominal_training_steps(profile, method) == 7_680_000
        assert nominal_training_steps(profile, method) // profile["num_phases"] == 1_920_000
        command = build_command(profile=profile, method=method, seed=1001, trial=1002,
                                output_dir=Path("results/pilot"))
        assert command[command.index("--num_tasks") + 1] == str(tasks)
        assert command[command.index("--eval_episodes") + 1] == "10"
        assert "--oracle" not in command
        if method == "ppo":
            assert command[command.index("--num_updates") + 1] == "600"
            assert command[command.index("--task_interval") + 1] == "150"
            assert command[command.index("--ppo_override") + 1:] == [
                "num_envs=256", "num_steps=50", "num_minibatches=8",
            ]
        else:
            assert command[command.index("--num_generations") + 1] == "80"
            assert command[command.index("--task_interval") + 1] == "20"
            assert "--ne_override" not in command


@pytest.mark.parametrize("method", ["ga", "es", "ppo"])
def test_stationary_score_accepts_only_task_zero(method):
    profile = load_profile("pilot-stationary")
    count = profile["ppo"]["num_updates"] if method == "ppo" else 80
    records = [{"generation": i, "task": 0, "centroid_task0": 100} for i in range(count)]
    assert score_curve(records, profile=profile, method=method)["mean_return"] == 100
    records[-1]["task"] = 1
    with pytest.raises(ValueError, match="task schedule"):
        score_curve(records, profile=profile, method=method)


@pytest.mark.parametrize("field,value,message", [
    ("num_tasks", 2, "environment"),
    ("num_phases", 3, "phase grid"),
    ("eval_episodes", False, "positive integer"),
    ("episode_length", 501, "exceed 500"),
    ("seeds", [1001, 1001], "distinct"),
    ("seeds", [True], "integers"),
    ("ppo", None, "omit a PPO budget"),
])
def test_inconsistent_pilot_profiles_fail_before_launch(field, value, message):
    profile = load_profile("pilot-stationary")
    profile[field] = value
    with pytest.raises(ValueError, match=message):
        build_command(profile=profile, method="ga", seed=1001, output_dir=Path("unused"))


def test_comparison_rejects_unmatched_budgets_and_uneven_minibatches():
    profile = load_profile("pilot-switching")
    profile["ppo"]["num_envs"] = 128
    with pytest.raises(ValueError, match="matched nominal"):
        validate_profile(profile)
    profile = load_profile("pilot-switching")
    profile["ppo"]["num_minibatches"] = 7
    with pytest.raises(ValueError, match="minibatches"):
        validate_profile(profile)


def test_unknown_budget_flags_cannot_enter_upstream_command():
    profile = load_profile("pilot-switching")
    profile["ne"]["oracle"] = 1
    with pytest.raises(ValueError, match="budget fields"):
        build_command(profile=profile, method="ga", seed=1001, output_dir=Path("unused"))


@pytest.mark.parametrize("seed,trial", [(True, 1), (-1, 1), (2**32, 1), (1001, False), (1001, 0)])
def test_seed_and_trial_arguments_are_validated(seed, trial):
    with pytest.raises(ValueError):
        build_command(profile=load_profile("smoke"), method="ga", seed=seed, trial=trial,
                      output_dir=Path("unused"))


def test_paper_budgets_are_matched_and_twenty_phases():
    p = load_profile("paper-cartpole")
    ne, rl = p["ne"], p["ppo"]
    assert p["num_tasks"] == 2
    assert ne["num_generations"] // ne["task_interval"] == p["num_phases"] == 20
    assert rl["num_updates"] // rl["task_interval"] == 20
    assert (ne["num_generations"] * ne["pop_size"] * ne["num_evals"] * p["episode_length"]
            == rl["num_updates"] * rl["num_envs"] * rl["num_steps"] == 3_072_000_000)
    development_seeds = set(load_profile("search")["seeds"]) | set(load_profile("smoke")["seeds"])
    assert not development_seeds.intersection(p["seeds"])
    assert not {s + 1 for s in development_seeds}.intersection(range(1, 11))


def test_baseline_has_no_oracle_or_hyperparameter_overrides():
    command = build_command(profile=load_profile("paper-cartpole"), method="ppo", seed=42,
                            output_dir=Path("results/test"))
    assert "--oracle" not in command
    assert "--ppo_override" not in command
    assert "--ne_override" not in command
    assert command[command.index("--num_phases") + 1] == "20"
    assert command[command.index("--num_tasks") + 1] == "2"


def test_score_uses_active_centroid_not_training_best():
    p = load_profile("smoke")
    records = [{"generation": i, "task": i // 2, "centroid_task0": 8,
                "centroid_task1": 24, "train_fitness_max": 32} for i in range(4)]
    assert score_curve(records, profile=p, method="ga")["normalized_score"] == 0.5
    with pytest.raises(ValueError, match="Expected 4"):
        score_curve(records[:-1], profile=p, method="ga")
    records[3]["centroid_task1"] = math.nan
    with pytest.raises(ValueError, match="non-finite"):
        score_curve(records, profile=p, method="ga")


def test_incorrect_schedule_and_duplicate_rows_rejected():
    p = load_profile("smoke")
    records = [{"generation": i, "task": 0, "centroid_task0": 12} for i in range(4)]
    with pytest.raises(ValueError, match="task schedule"):
        score_curve(records, profile=p, method="es")
    records = [{"generation": 0, "task": i // 2, f"centroid_task{i // 2}": 12}
               for i in range(4)]
    with pytest.raises(ValueError, match="generation"):
        score_curve(records, profile=p, method="es")


@pytest.mark.parametrize("value", [False, -1, 0, 3, float("inf"), float("nan"), "0.5"])
def test_invalid_ga_sigma_rejected(value):
    with pytest.raises(ValueError):
        validate_ga_settings({"sigma": value, "elite_ratio": 0.5})
