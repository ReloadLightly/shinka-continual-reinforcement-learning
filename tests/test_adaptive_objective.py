"""Known switching traces distinguish fresh prior-task performance from forgetting."""

from copy import deepcopy

import pytest

from shinka_crl.adaptive_objective import OBJECTIVE_VERSION, score_adaptive_trial
from shinka_crl.analysis import summarize_trial
from shinka_crl.experiment import load_profile


@pytest.fixture
def trial_data():
    profile = load_profile("adaptive-gate-switching")
    profile.update(num_phases=4, seeds=[4001])
    profile["ne"].update(num_generations=8, task_interval=2)
    profile["eval_episodes"] = 3
    config = {
        "method": "ga", "seed": 4001, "trial": 4002, "env": "CartPole-v1",
        "episode_length": 500, "num_tasks": 2, "task_sequence": [0, 1, 0, 1],
        "eval_episodes": 3, "num_generations": 8, "task_interval": 2,
        "pop_size": 16, "num_evals": 3, "num_params": 7,
    }
    entries = []
    for source in ("finalgen", "incumbent", "centroid"):
        for phase, own in enumerate((100, 200, 300, 400)):
            entry = {"task_idx": phase, "source": source,
                     "returns": [own if source == "centroid" else 500] * 10}
            if phase < 3:
                entry["zero_shot_next_returns"] = [(phase + 1) * 30] * 10
            if phase > 0:
                entry["prev_returns"] = [[80, 250, 200][phase - 1]
                                         if source == "centroid" else 500] * 10
            entries.append(entry)
    return {
        "manifest": {"status": "complete", "method": "ga", "seed": 4001,
                     "trial": 4002, "profile": profile},
        "results": {"config": config, "noise_vectors": [[0, 0], [.2, .3]],
                    "env_steps": 8 * 16 * 3 * 500},
        "records": [{"generation": i, "task": (i // 2) % 2,
                     f"centroid_task{(i // 2) % 2}": (i // 2 + 1) * 100,
                     f"incumbent_task{(i // 2) % 2}": 500} for i in range(8)],
        "evaluation": {"method": "ga", "env": "CartPole-v1", "trial": 4002,
                       "pop_size": 16, "episodes": 10, "eval_seed": 904001,
                       "num_tasks": 4, "agent_sources": ["finalgen", "incumbent", "centroid"],
                       "per_task": entries},
        "checkpoint_metadata": {
            "sources": {source: [4, 7] for source in ("finalgen", "incumbent", "centroid")},
            "noise_vectors": [[0, 0], [.2, .3], [0, 0], [.2, .3]], "finite": True,
        },
        "episodes": 10, "eval_seed": 904001,
    }


def arguments(trial_data):
    manifest = trial_data["manifest"]
    return {
        "profile": manifest["profile"], "records": trial_data["records"],
        "analysis": summarize_trial(**trial_data), "evaluation": trial_data["evaluation"],
        "method": manifest["method"], "seed": manifest["seed"], "trial": manifest["trial"],
        "eval_seed": trial_data["eval_seed"], "episodes": trial_data["episodes"],
    }


def test_objective_uses_centroid_active_and_fresh_previous_means(trial_data):
    args = arguments(trial_data)
    original = deepcopy(args)
    result = score_adaptive_trial(**args)
    assert result["objective_version"] == OBJECTIVE_VERSION
    assert result["active_score"] == 0.5
    assert result["previous_score"] == pytest.approx((80 + 250 + 200) / 3 / 500)
    assert result["combined_score"] == pytest.approx(0.4266666666666667)
    assert result["active_checkpoints"] == 8 and result["previous_checkpoints"] == 3
    assert result["reference_metrics"]["learning_accuracy"] == 250
    assert result["reference_metrics"]["forgetting"] == pytest.approx(70 / 3)
    assert result["reference_metrics"]["zero_shot_transfer"] == 60
    assert [row["forgetting"] for row in result["switches"]] == [20, -50, 100]
    assert result["reference_metrics"] == args["analysis"]["metrics"]
    assert args == original


def test_lower_initial_acquisition_does_not_reward_the_objective(trial_data):
    original = score_adaptive_trial(**arguments(trial_data))
    for entry in trial_data["evaluation"]["per_task"]:
        if entry["source"] == "centroid":
            entry["returns"] = [(entry["task_idx"] + 1) * 10] * 10
    poorer = score_adaptive_trial(**arguments(trial_data))
    assert poorer["combined_score"] == original["combined_score"]
    assert poorer["previous_score"] == original["previous_score"]
    assert poorer["reference_metrics"]["forgetting"] < 0
    assert poorer["reference_metrics"]["learning_accuracy"] < original["reference_metrics"]["learning_accuracy"]


@pytest.mark.parametrize("reward,expected", [(0, 0), (500, 1)])
def test_objective_inclusive_bounds(trial_data, reward, expected):
    for record in trial_data["records"]:
        record[f"centroid_task{record['task']}"] = reward
    for entry in trial_data["evaluation"]["per_task"]:
        if "prev_returns" in entry:
            entry["prev_returns"] = [reward] * 10
    assert score_adaptive_trial(**arguments(trial_data))["combined_score"] == expected


@pytest.mark.parametrize("field", ["seed", "trial", "eval_seed", "episodes"])
def test_boolean_identity_arguments_are_rejected(trial_data, field):
    args = arguments(trial_data)
    args[field] = True
    with pytest.raises(ValueError):
        score_adaptive_trial(**args)


@pytest.mark.parametrize("change", ["missing", "duplicate", "reordered", "wrong_source", "boolean_phase"])
def test_invalid_raw_phase_or_source_identities_fail(trial_data, change):
    args = arguments(trial_data)
    entries = args["evaluation"]["per_task"]
    if change == "missing":
        entries.pop()
    elif change == "duplicate":
        entries[-1] = deepcopy(entries[-2])
    elif change == "reordered":
        entries[-1], entries[-2] = entries[-2], entries[-1]
    elif change == "wrong_source":
        entries[-1]["source"] = "incumbent"
    else:
        entries[-3]["task_idx"] = True
    with pytest.raises(ValueError, match="source/phase"):
        score_adaptive_trial(**args)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 501, True, "100"])
def test_invalid_previous_episode_values_fail(trial_data, value):
    args = arguments(trial_data)
    args["evaluation"]["per_task"][-1]["prev_returns"][0] = value
    with pytest.raises(ValueError, match="prev_returns"):
        score_adaptive_trial(**args)


@pytest.mark.parametrize("change", ["short", "missing", "wrong_boundary"])
def test_previous_return_vectors_follow_fresh_evaluation_protocol(trial_data, change):
    args = arguments(trial_data)
    entries = args["evaluation"]["per_task"]
    if change == "short":
        entries[-1]["prev_returns"].pop()
    elif change == "missing":
        del entries[-1]["prev_returns"]
    else:
        entries[-4]["prev_returns"] = [100] * 10
    with pytest.raises(ValueError, match="vector"):
        score_adaptive_trial(**args)


@pytest.mark.parametrize("change", ["generation", "task", "reordered", "missing", "nonfinite", "bounds"])
def test_active_training_trace_must_be_complete_and_ordered(trial_data, change):
    args = arguments(trial_data)
    rows = args["records"]
    if change == "generation":
        rows[-1]["generation"] = 6
    elif change == "task":
        rows[-1]["task"] = 0
    elif change == "reordered":
        rows[-1], rows[-2] = rows[-2], rows[-1]
    elif change == "missing":
        rows.pop()
    else:
        rows[-1]["centroid_task1"] = float("nan") if change == "nonfinite" else 501
    with pytest.raises(ValueError):
        score_adaptive_trial(**args)


@pytest.mark.parametrize("change", ["wrong_source", "wrong_method", "wrong_eval_seed", "phase_order",
                                    "previous_mean", "forgetting", "cumulative", "phase_training"])
def test_analysis_must_match_raw_evidence(trial_data, change):
    args = arguments(trial_data)
    summary = args["analysis"]
    if change == "wrong_source":
        summary["agent_source"] = "incumbent"
    elif change == "wrong_method":
        summary["method"] = "ga_focus"
    elif change == "wrong_eval_seed":
        summary["eval_seed"] = 4001
    elif change == "phase_order":
        summary["phase_returns"][-1], summary["phase_returns"][-2] = (
            summary["phase_returns"][-2], summary["phase_returns"][-1])
    elif change == "previous_mean":
        summary["phase_returns"][-1]["previous_mean"] = 500
    elif change == "forgetting":
        summary["metrics"]["forgetting"] = 0
    elif change == "cumulative":
        summary["metrics"]["normalized_curve_average"] = 1
    else:
        summary["phase_training_returns"][0] = 500
    with pytest.raises(ValueError, match="identity|raw evidence"):
        score_adaptive_trial(**args)


def test_raw_evaluation_and_declared_centroid_cannot_be_swapped(trial_data):
    args = arguments(trial_data)
    args["evaluation"]["method"] = "ga_focus"
    with pytest.raises(ValueError, match="evaluation.method"):
        score_adaptive_trial(**args)
    args["evaluation"]["method"] = "ga"
    args["evaluation"]["agent_sources"] = ["finalgen", "incumbent"]
    with pytest.raises(ValueError, match="centroid"):
        score_adaptive_trial(**args)


def test_native_focus_method_remains_explicit(trial_data):
    args = arguments(trial_data)
    args["method"] = args["analysis"]["method"] = args["evaluation"]["method"] = "ga_focus"
    result = score_adaptive_trial(**args)
    assert result["combined_score"] == pytest.approx(0.4266666666666667)
    args["method"] = "es"
    with pytest.raises(ValueError, match="GA controls"):
        score_adaptive_trial(**args)


def test_statistical_objective_requires_switching_cap_500_and_profile_seed(trial_data):
    args = arguments(trial_data)
    args["seed"] = 42
    with pytest.raises(ValueError, match="outside"):
        score_adaptive_trial(**args)
    args = arguments(trial_data)
    args["profile"]["episode_length"] = 499
    with pytest.raises(ValueError, match="cap 500"):
        score_adaptive_trial(**args)
