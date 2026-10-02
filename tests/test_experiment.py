import math
from pathlib import Path

import pytest

from shinka_crl.experiment import build_command, load_profile, score_curve, validate_ga_settings


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
