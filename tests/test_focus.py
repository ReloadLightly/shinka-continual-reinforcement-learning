"""Keep native FocusGA identity, paper settings, and population cost explicit."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from shinka_crl import experiment
from shinka_crl.analysis import summarize_trial
from shinka_crl.baseline_contract import validate_baseline_config
from shinka_crl.experiment import build_command, load_profile, nominal_training_steps, score_curve


def focus_config(population=16):
    return {
        "env": "CartPole-v1", "method": "ga_focus", "hidden_dims": [16, 16],
        "num_params": 386, "first_task_clean": True, "task_warmup": 0,
        "objective": "mean", "obs_norm": False, "pop_size": population, "sigma": 0.5,
        "searcher_kwargs": {
            "elite_ratio": 0.5, "init_around_mean": False, "cross_over_rate": 0.0,
            "focus_rate": 0.3, "sigma_rate": 0.1, "track_target": 0.9,
            "sigma_min": 1e-5, "explore_fraction": 0.25,
        },
        "searcher_resolved": {
            "refresh": True, "num_elites": population // 2,
            "num_offspring": population // 2 - 1, "variation": "gaussian",
            "sigma": 0.5, "cross_over_rate": 0.0,
        },
    }


def test_focus_command_selects_native_method_and_explicit_paper_settings(tmp_path):
    profile = load_profile("adaptive-gate-switching")
    command = build_command(profile=profile, method="ga_focus", seed=3001, trial=3002,
                            output_dir=tmp_path / "run")
    assert command[command.index("--method") + 1] == "ga"
    overrides = command[command.index("--ne_override") + 1:]
    assert dict(item.split("=", 1) for item in overrides) == {
        "method": "ga_focus", "sigma": "0.5", "searcher_kwargs.elite_ratio": "0.5",
        "searcher_kwargs.cross_over_rate": "0.0",
        "searcher_kwargs.focus_rate": "0.3", "searcher_kwargs.sigma_rate": "0.1",
        "searcher_kwargs.track_target": "0.9", "searcher_kwargs.sigma_min": "1e-05",
        "searcher_kwargs.explore_fraction": "0.25",
    }
    assert "--oracle" not in command
    assert nominal_training_steps(profile, "ga_focus") == nominal_training_steps(profile, "ga")
    with pytest.raises(ValueError, match="overrides"):
        build_command(profile=profile, method="ga_focus", seed=3001, output_dir=tmp_path,
                      ga_settings={"sigma": 0.2, "elite_ratio": 0.5})


def test_native_parser_preserves_boolean_archive_initialization(tmp_path):
    upstream, python = experiment.DEFAULT_UPSTREAM, experiment.DEFAULT_PYTHON
    if not python.exists() or not (upstream / "source/run.py").exists():
        pytest.skip("Pinned upstream environment is not installed")
    command = build_command(profile=load_profile("adaptive-gate-switching"), method="ga_focus",
                            seed=3001, trial=3002, output_dir=tmp_path / "run")
    script = (
        "import json,sys;from source.run import load_config,build_parser,build_run;"
        "c=load_config(sys.argv[1:]);a=build_parser(c).parse_args(sys.argv[1:]);"
        "f,p,common,k=build_run(c,a);"
        "print(json.dumps({'method':k['method'],'sigma':k['sigma'],"
        "'searcher_kwargs':k['searcher_kwargs']}))"
    )
    result = subprocess.run([str(python), "-c", script, *command[2:]], cwd=upstream,
                            capture_output=True, text=True, check=True, timeout=30)
    parsed = json.loads(result.stdout)
    assert parsed["method"] == "ga_focus" and parsed["sigma"] == 0.5
    assert parsed["searcher_kwargs"] == focus_config()["searcher_kwargs"]
    assert parsed["searcher_kwargs"]["init_around_mean"] is False


@pytest.mark.parametrize("population", [4, 16, 64, 512])
def test_focus_centroid_is_inside_fixed_population_budget(population):
    config = focus_config(population)
    validate_baseline_config(config, "ga_focus")
    resolved = config["searcher_resolved"]
    assert resolved["num_elites"] + resolved["num_offspring"] + 1 == population


@pytest.mark.parametrize("field,value", [
    ("init_around_mean", True), ("elite_ratio", 0.25), ("cross_over_rate", 0.5),
    ("focus_rate", 0.1), ("sigma_rate", 0.3), ("track_target", 0.5),
    ("sigma_min", 0.001), ("explore_fraction", 0.0),
])
def test_focus_class_defaults_cannot_replace_declared_settings(field, value):
    config = focus_config()
    config["searcher_kwargs"][field] = value
    with pytest.raises(ValueError, match=field):
        validate_baseline_config(config, "ga_focus")


def test_focus_rejects_missing_centroid_slot_and_wrong_method():
    config = focus_config()
    config["searcher_resolved"]["num_offspring"] = 8
    with pytest.raises(ValueError, match="num_offspring"):
        validate_baseline_config(config, "ga_focus")
    config = focus_config()
    config["method"] = "ga"
    with pytest.raises(ValueError, match="method"):
        validate_baseline_config(config, "ga_focus")
    profile = load_profile("smoke")
    profile["ne"]["pop_size"] = 2
    with pytest.raises(ValueError, match="centroid"):
        build_command(profile=profile, method="ga_focus", seed=1001, output_dir=Path("unused"))


def focus_trial():
    profile = load_profile("adaptive-gate-switching")
    config = focus_config(profile["ne"]["pop_size"])
    config.update(seed=3001, trial=3002, episode_length=500, num_tasks=2,
                  task_sequence=[0, 1], eval_episodes=10, num_generations=6,
                  task_interval=3, num_evals=3)
    return {
        "manifest": {"status": "complete", "method": "ga_focus", "seed": 3001,
                     "trial": 3002, "profile": profile},
        "results": {"config": config, "env_steps": 144000,
                    "noise_vectors": [[0, 0], [0.1, 0.2]]},
        "records": [{"generation": i, "task": i // 3,
                     f"centroid_task{i // 3}": 100 if i < 3 else 300} for i in range(6)],
        "evaluation": {"method": "ga_focus", "env": "CartPole-v1", "trial": 3002,
                       "pop_size": 16, "episodes": 10, "eval_seed": 903001, "num_tasks": 2,
                       "agent_sources": ["centroid"], "per_task": [
                           {"task_idx": 0, "source": "centroid", "returns": [200] * 10,
                            "zero_shot_next_returns": [50] * 10},
                           {"task_idx": 1, "source": "centroid", "returns": [400] * 10,
                            "prev_returns": [150] * 10}]},
        "checkpoint_metadata": {"sources": {"centroid": [2, 386]}, "finite": True,
                                "noise_vectors": [[0, 0], [0.1, 0.2]]},
        "episodes": 10, "eval_seed": 903001,
    }


def test_focus_scoring_and_posthoc_keep_native_identity_and_centroid():
    data = focus_trial()
    assert score_curve(data["records"], profile=data["manifest"]["profile"],
                       method="ga_focus")["normalized_score"] == 0.4
    summary = summarize_trial(**data)
    assert summary["method"] == "ga_focus"
    assert summary["agent_source"] == "centroid"
    assert summary["nominal_training_steps"] == 144000
    assert summary["metrics"]["learning_accuracy"] == 300
    assert summary["metrics"]["forgetting"] == 50
    assert summary["metrics"]["zero_shot_transfer"] == 50
    tampered = deepcopy(data)
    tampered["evaluation"]["method"] = "ga"
    with pytest.raises(ValueError, match="identity.*method"):
        summarize_trial(**tampered)


def test_focus_run_manifest_keeps_true_method_and_checks_resolved_config(tmp_path, monkeypatch):
    data = focus_trial()
    monkeypatch.setattr(experiment, "verify_upstream", lambda _: experiment.UPSTREAM_COMMIT)
    monkeypatch.setattr(experiment.subprocess, "check_output", lambda *a, **k: "Python 3.11\n")

    def train(command, **kwargs):
        output = Path(command[command.index("--output_dir") + 1])
        for name, value in (("training_metrics.json", data["records"]),
                            ("results.json", data["results"])):
            (output / name).write_text(json.dumps(value))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(experiment.subprocess, "run", train)
    output = tmp_path / "complete"
    result = experiment.run_experiment(profile=data["manifest"]["profile"], method="ga_focus",
                                       seed=3001, trial=3002, python=sys.executable,
                                       upstream=tmp_path, output_dir=output)
    assert result["method"] == "ga_focus"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["method"] == "ga_focus" and manifest["status"] == "complete"
    data["results"]["config"]["method"] = "ga"
    output = tmp_path / "invalid"
    with pytest.raises(ValueError, match="method"):
        experiment.run_experiment(profile=data["manifest"]["profile"], method="ga_focus",
                                  seed=3001, trial=3002, python=sys.executable,
                                  upstream=tmp_path, output_dir=output)
    assert json.loads((output / "manifest.json").read_text())["status"] == "failed"
