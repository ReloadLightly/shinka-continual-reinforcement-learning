"""Known-return traces and artifact-integrity checks for checkpoint reporting."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from shinka_crl import analysis


@pytest.fixture
def trial_data():
    profile = {"name": "smoke", "purpose": "unit test", "seeds": [1001],
               "env": "CartPole-v1_sigma0.5", "num_phases": 4,
               "num_tasks": 2, "episode_length": 500, "eval_episodes": 10,
               "ne": {"num_generations": 4, "task_interval": 1,
                      "pop_size": 2, "num_evals": 1},
               "ppo": {"num_updates": 8, "task_interval": 2, "num_envs": 20,
                       "num_steps": 25, "num_minibatches": 4}}
    cfg = {"method": "ga", "seed": 1001, "trial": 1002, "env": "CartPole-v1",
           "episode_length": 500, "num_tasks": 2, "task_sequence": [0, 1, 0, 1],
           "eval_episodes": 10, "num_generations": 4, "task_interval": 1,
           "pop_size": 2, "num_evals": 1, "num_params": 3}
    vectors = [[0, 0], [0.2, 0.3]]
    # Own [100,200,300,400], next [30,60,90], and following previous
    # [80,250,200] give LA=250, F=(20-50+100)/3, ZT=60.
    entries = []
    for phase, own in enumerate((100, 200, 300, 400)):
        entry = {"task_idx": phase, "source": "centroid", "returns": [own] * 10}
        if phase < 3:
            entry["zero_shot_next_returns"] = [(phase + 1) * 30] * 10
        if phase > 0:
            entry["prev_returns"] = [[80, 250, 200][phase - 1]] * 10
        entries.append(entry)
    return {"manifest": {"status": "complete", "method": "ga", "seed": 1001,
                         "trial": 1002, "profile": profile},
            "results": {"config": cfg, "noise_vectors": vectors, "env_steps": 4000},
            "records": [{"generation": phase, "task": phase % 2,
                         f"centroid_task{phase % 2}": own}
                        for phase, own in enumerate((10, 20, 30, 40))],
            "evaluation": {"method": "ga", "env": "CartPole-v1", "trial": 1002,
                           "pop_size": 2, "episodes": 10, "eval_seed": 901001,
                           "num_tasks": 4, "agent_sources": ["centroid"],
                           "per_task": entries},
            "checkpoint_metadata": {"sources": {"centroid": [4, 3]},
                                    "noise_vectors": vectors * 2, "finite": True},
            "episodes": 10, "eval_seed": 901001}


def test_known_switching_metrics_include_both_directions(trial_data):
    summary = analysis.summarize_trial(**trial_data)
    metrics = summary["metrics"]
    assert metrics["learning_accuracy"] == 250
    assert metrics["forgetting"] == pytest.approx(70 / 3)
    assert metrics["zero_shot_transfer"] == 60
    assert metrics["learning_minus_forgetting"] == pytest.approx(250 - 70 / 3)
    assert metrics["cumulative_reward_steps"] == 85000
    assert summary["phase_training_returns"] == [10, 20, 30, 40]
    assert [row["checkpoint_completed_steps"] for row in summary["phase_returns"]] == [
        1000, 2000, 3000, 4000]
    assert summary["clock"]["record_timing"] == "after_update"
    assert summary["clock"]["initial_untrained_policy_evaluated"] is False


def test_negative_forgetting_is_not_clipped(trial_data):
    for entry in trial_data["evaluation"]["per_task"][1:]:
        entry["prev_returns"] = [500] * 10
    assert analysis.summarize_trial(**trial_data)["metrics"]["forgetting"] == -300


def test_stationary_control_does_not_claim_task_transfer(trial_data):
    profile = trial_data["manifest"]["profile"]
    profile.update(num_tasks=1, env="CartPole-v1")
    trial_data["results"]["config"].update(num_tasks=1, task_sequence=[0] * 4)
    trial_data["results"]["noise_vectors"] = [[0, 0]]
    trial_data["checkpoint_metadata"]["noise_vectors"] = [[0, 0]] * 4
    for record in trial_data["records"]:
        value = record[f"centroid_task{record['task']}"]
        record.update(task=0, centroid_task0=value)
    summary = analysis.summarize_trial(**trial_data)
    assert summary["condition"] == "stationary"
    assert summary["metrics"]["learning_accuracy"] == 250
    for metric in ("forgetting", "zero_shot_transfer", "learning_minus_forgetting"):
        assert summary["metrics"][metric] is None


def test_ppo_uses_final_policy_and_completed_transition_clock(trial_data):
    trial_data["manifest"]["method"] = "ppo"
    trial_data["results"]["config"].update(method="ppo", num_generations=8,
                                          task_interval=2, pop_size=20, num_envs=20,
                                          num_steps=25, num_minibatches=4, num_timesteps=4000)
    trial_data["records"] = [{"generation": step, "task": (step // 2) % 2,
                              f"centroid_task{(step // 2) % 2}": step * 10}
                             for step in range(8)]
    trial_data["evaluation"].update(method="ppo", pop_size=20, agent_sources=["final"])
    trial_data["checkpoint_metadata"]["sources"] = {"final": [4, 3]}
    for entry in trial_data["evaluation"]["per_task"]:
        entry["source"] = "final"
    summary = analysis.summarize_trial(**trial_data)
    assert summary["agent_source"] == "final"
    assert summary["steps_per_update"] == 500
    assert summary["phase_training_returns"] == [5, 25, 45, 65]
    assert summary["metrics"]["learning_accuracy"] == 250


def test_pinned_generation_grid_and_exact_dense_integral_are_distinct():
    # PPO has a return spike between NE-grid points; preserve the published
    # sampling convention and disclose the dense-curve answer separately.
    result = analysis.integrate_curve([0, 0, 100, 0], steps_per_update=5,
                                      reference_steps_per_generation=10)
    assert result["cumulative_reward_steps"] == 0
    assert result["cumulative_reward_steps_exact"] == 500
    assert result["integration_grid_points"] == 3


def test_initial_tail_uses_first_trained_value_not_zero():
    result = analysis.integrate_curve([20, 40], steps_per_update=10,
                                      reference_steps_per_generation=10)
    assert result["cumulative_reward_steps"] == 500
    assert result["cumulative_reward_steps_exact"] == 500


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["manifest"].update(status="failed"), "complete"),
    (lambda d: d.update(eval_seed=1001), "distinct"),
    (lambda d: d["results"]["config"].update(task_sequence=[0, 1]), "task_sequence"),
    (lambda d: d["results"].update(env_steps=999), "budget"),
    (lambda d: d["records"][1].update(task=0), "schedule"),
    (lambda d: d["evaluation"].update(episodes=100), "episodes"),
    (lambda d: d["evaluation"].update(eval_seed=0), "eval_seed"),
    (lambda d: d["evaluation"].update(num_tasks=2), "num_tasks"),
    (lambda d: d["evaluation"].update(agent_sources=["incumbent"]), "sources"),
    (lambda d: d["evaluation"]["per_task"][0].update(returns=[100]), "episodes"),
    (lambda d: d["evaluation"]["per_task"][0].update(returns=[float("nan")] * 10), "Invalid"),
    (lambda d: d["evaluation"]["per_task"][0].update(returns=[501] * 10), "Invalid"),
    (lambda d: d["evaluation"]["per_task"][1].pop("prev_returns"), "episodes"),
    (lambda d: d["evaluation"]["per_task"][1].update(task_idx=0), "Duplicate"),
    (lambda d: d["evaluation"]["per_task"].pop(), "Missing"),
    (lambda d: d["checkpoint_metadata"].update(sources={"incumbent": [4, 3]}), "source"),
    (lambda d: d["checkpoint_metadata"]["sources"].update(centroid=[2, 3]), "shape"),
    (lambda d: d["checkpoint_metadata"]["noise_vectors"].reverse(), "phase identities"),
    (lambda d: d["checkpoint_metadata"].update(finite=False), "Non-finite"),
])
def test_corrupt_or_mismatched_evidence_is_rejected(trial_data, mutation, match):
    mutation(trial_data)
    with pytest.raises(ValueError, match=match):
        analysis.summarize_trial(**trial_data)


def _write_fixture_artifacts(tmp_path, trial_data):
    run_dir, output_dir = tmp_path / "training", tmp_path / "analysis"
    run_dir.mkdir()
    output_dir.mkdir()
    (output_dir / "evaluation-input").mkdir()
    for name, data in (("manifest.json", trial_data["manifest"]),
                       ("results.json", trial_data["results"]),
                       ("config.json", trial_data["results"]["config"]),
                       ("training_metrics.json", trial_data["records"])):
        (run_dir / name).write_text(json.dumps(data))
    (run_dir / "checkpoints.npz").write_bytes(b"fixture-hashed-checkpoint")
    summary = analysis.summarize_trial(**trial_data)
    for name, data in (("evaluation.json", trial_data["evaluation"]),
                       ("checkpoint-metadata.json", trial_data["checkpoint_metadata"]),
                       ("summary.json", summary)):
        (output_dir / name).write_text(json.dumps(data))
    (output_dir / "process.log").write_text("complete\n")
    for name in ("results.json", "checkpoints.npz"):
        (output_dir / "evaluation-input" / name).write_bytes((run_dir / name).read_bytes())
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"status": "complete", "episodes": 10, "eval_seed": 901001,
                "analysis_source_sha256": digest(Path(analysis.__file__)),
                "input_sha256": {name: digest(run_dir / name) for name in analysis.INPUT_FILES},
                "output_sha256": {name: digest(output_dir / name) for name in analysis.OUTPUT_FILES}}
    (output_dir / "manifest.json").write_text(json.dumps(manifest))
    return {"run_dir": run_dir, "output_dir": output_dir, "eval_seed": 901001}, summary


def test_resume_rederives_summary_from_frozen_inputs(tmp_path, trial_data):
    paths, expected = _write_fixture_artifacts(tmp_path, trial_data)
    assert analysis.validate_analysis(**paths) == expected


@pytest.mark.parametrize("kind,name", [("run_dir", "checkpoints.npz"),
                                      ("run_dir", "results.json"),
                                      ("output_dir", "evaluation.json"),
                                      ("output_dir", "summary.json")])
def test_resume_rejects_modified_input_or_output(tmp_path, trial_data, kind, name):
    paths, _ = _write_fixture_artifacts(tmp_path, trial_data)
    (paths[kind] / name).write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        analysis.validate_analysis(**paths)


def test_summary_is_rederived_even_if_its_hash_is_updated(tmp_path, trial_data):
    paths, summary = _write_fixture_artifacts(tmp_path, trial_data)
    changed = deepcopy(summary)
    changed["metrics"]["learning_accuracy"] = 499
    summary_path = paths["output_dir"] / "summary.json"
    summary_path.write_text(json.dumps(changed))
    manifest_path = paths["output_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["output_sha256"]["summary.json"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="does not match raw"):
        analysis.validate_analysis(**paths)
