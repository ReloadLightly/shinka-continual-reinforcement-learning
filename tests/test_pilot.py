"""Check stage resume, provenance rejection, and immutable failed attempts."""

import json
from pathlib import Path

import pytest

from shinka_crl import experiment, pilot


@pytest.fixture
def fake_runtime(monkeypatch):
    from shinka_crl import analysis

    calls = {"training": [], "analysis": [], "fail_next": False}
    monkeypatch.setattr(pilot, "verify_upstream", lambda path: experiment.UPSTREAM_COMMIT)
    monkeypatch.setattr(pilot, "source_hashes", lambda: {"test-source": "fixed-hash"})
    monkeypatch.setattr(pilot.os, "sched_getaffinity", lambda pid: {0, 1})
    monkeypatch.setattr(pilot.os, "sched_setaffinity", lambda pid, cpus: None)
    monkeypatch.setattr(pilot.shutil, "which", lambda path: "/usr/bin/python3")
    monkeypatch.setattr(pilot.subprocess, "check_output", lambda command, **kwargs:
                        json.dumps({"python": "3.11", "jax_backend": "cpu",
                                    "jax_devices": ["cpu"], "packages": {"jax": "0.5.3"}})
                        if "-c" in command else "source-revision")

    def training(**kwargs):
        path, profile = kwargs["output_dir"], kwargs["profile"]
        method, seed, trial = kwargs["method"], kwargs["seed"], kwargs["trial"]
        calls["training"].append(path)
        path.mkdir(parents=True)
        if calls["fail_next"]:
            calls["fail_next"] = False
            pilot.write_json(path / "manifest.json", {"status": "failed"})
            raise RuntimeError("injected trainer failure")
        budget = profile["ppo"] if method == "ppo" else profile["ne"]
        count = budget["num_updates" if method == "ppo" else "num_generations"]
        records = []
        for i in range(count):
            task = i // budget["task_interval"] % profile["num_tasks"]
            records.append({"generation": i, "task": task, f"centroid_task{task}": 100.0})
        pilot.write_json(path / "training_metrics.json", records)
        summary = {**experiment.score_curve(records, profile=profile, method=method),
                   "profile": profile["name"], "method": method, "seed": seed,
                   "trial": trial, "upstream_commit": experiment.UPSTREAM_COMMIT}
        pilot.write_json(path / "summary.json", summary)
        switching = "_sigma" in profile["env"]
        # Use checked-in real resolved baseline settings; override only the pilot protocol.
        config = pilot.read_json(experiment.REPO_ROOT / "reports" / "smoke-20261002"
                                 / method / "results.json")["config"]
        config.update({"env": "CartPole-v1", "method": method, "seed": seed, "trial": trial,
                  "schedule": "switch" if switching else "task0",
                  "num_tasks": profile["num_tasks"], "noise_range": 0.5 if switching else 0.0,
                  "num_generations": count, "task_interval": budget["task_interval"],
                  "task_sequence": [i % profile["num_tasks"] for i in range(4)],
                  "episode_length": 500, "eval_episodes": 10, "hidden_dims": [16, 16]})
        keys = ("num_envs", "num_steps", "num_minibatches") if method == "ppo" else (
            "pop_size", "num_evals")
        config.update({k: budget[k] for k in keys})
        if method == "ga":
            config["searcher_resolved"].update(num_elites=budget["pop_size"] // 2,
                                               num_offspring=budget["pop_size"] // 2)
        vectors = [[0, 0, 0, 0]] + ([[0.1, 0.2, 0.3, 0.4]] if switching else [])
        # Upstream raw PPO artifacts contain NaN placeholders for unused NE fields.
        (path / "config.json").write_text(json.dumps(config))
        (path / "results.json").write_text(json.dumps({
            "config": config, "noise_vectors": vectors,
            "env_steps": pilot.training_steps(profile, method)}))
        (path / "checkpoints.npz").write_bytes(b"mock-checkpoint-validated-by-mock-analysis")
        for name in ("train.log", "process.log"):
            (path / name).write_text("complete\n")
        pilot.write_json(path / "manifest.json", {
            "status": "complete", "profile": profile, "method": method, "seed": seed,
            "trial": trial, "upstream_commit": experiment.UPSTREAM_COMMIT,
            "ga_settings": None, "wall_seconds": 1.0,
            "command": experiment.build_command(
                profile=profile, method=method, seed=seed, trial=trial, output_dir=path,
                python=kwargs["python"], upstream=kwargs["upstream"]),
            "metrics_sha256": pilot.sha256(path / "training_metrics.json")})
        return summary

    def run_analysis(**kwargs):
        path = kwargs["output_dir"]
        calls["analysis"].append(path)
        path.mkdir(parents=True)
        pilot.write_json(path / "manifest.json", {"status": "complete", "wall_seconds": 2.0})
        pilot.write_json(path / "summary.json", {"eval_seed": kwargs["eval_seed"]})

    def validate_analysis(run_dir, output_dir, **kwargs):
        return pilot.read_json(output_dir / "summary.json")

    monkeypatch.setattr(pilot, "run_experiment", training)
    monkeypatch.setattr(analysis, "run_analysis", run_analysis)
    monkeypatch.setattr(analysis, "validate_analysis", validate_analysis)
    return calls


def test_staged_resume_skips_validated_trials(tmp_path, fake_runtime):
    output = tmp_path / "pilot"
    first = pilot.run_pilot(output=output, max_trials=2)
    assert first["status"] == "partial" and first["completed_trials"] == 2
    saved = (output / first["rows"][0]["training_path"] / "receipt.json").read_bytes()
    second = pilot.run_pilot(output=output, resume=True, max_trials=1)
    assert second["completed_trials"] == 3
    assert len(fake_runtime["training"]) == len(fake_runtime["analysis"]) == 3
    assert (output / first["rows"][0]["training_path"] / "receipt.json").read_bytes() == saved
    final = pilot.run_pilot(output=output, resume=True)
    assert final["status"] == "complete" and final["completed_trials"] == 18
    assert len(fake_runtime["training"]) == 18
    assert [s["new_completed_trials"] for s in final["sessions"]] == [2, 1, 15]


def test_resume_rejects_changed_checkpoint(tmp_path, fake_runtime):
    output = tmp_path / "pilot"
    first = pilot.run_pilot(output=output, max_trials=1)
    checkpoint = output / first["rows"][0]["training_path"] / "checkpoints.npz"
    checkpoint.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact changed"):
        pilot.run_pilot(output=output, resume=True)
    assert len(fake_runtime["training"]) == 1


def test_failed_attempt_is_preserved_and_retried_in_new_directory(tmp_path, fake_runtime):
    output = tmp_path / "pilot"
    fake_runtime["fail_next"] = True
    with pytest.raises(RuntimeError, match="injected"):
        pilot.run_pilot(output=output)
    original = fake_runtime["training"][0] / "manifest.json"
    saved = original.read_bytes()
    result = pilot.run_pilot(output=output, resume=True, max_trials=1)
    assert result["completed_trials"] == 1
    assert original.read_bytes() == saved
    assert Path(result["rows"][0]["training_path"]).name == "attempt_002"


def test_requires_explicit_resume_and_identical_plan(tmp_path, fake_runtime, monkeypatch):
    output = tmp_path / "pilot"
    pilot.run_pilot(output=output, max_trials=1)
    with pytest.raises(ValueError, match="explicitly use --resume"):
        pilot.run_pilot(output=output)
    monkeypatch.setattr(pilot, "source_hashes", lambda: {"test-source": "changed"})
    with pytest.raises(ValueError, match="Frozen pilot plan"):
        pilot.run_pilot(output=output, resume=True)
    assert len(fake_runtime["training"]) == 1
