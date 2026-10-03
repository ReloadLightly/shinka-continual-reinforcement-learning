"""The reporting protocol, native process observer, and independent stage resume."""

import os
from pathlib import Path
import sys

import pytest

from shinka_crl import reference_comparison as comparison
from shinka_crl.experiment import load_profile


def test_reporting_is_exact_declared_thirty_trial_protocol(tmp_path):
    plan = comparison.make_plan(mode="reporting", cpus=1)
    profile = plan["profiles"]["paper-cartpole"]
    assert profile == load_profile("paper-cartpole")
    assert len(plan["jobs"]) == 30
    assert plan["nominal_training_steps_per_method"] == dict.fromkeys(comparison.METHODS, 3072000000)
    assert [(j["seed"], j["trial"]) for j in plan["jobs"][::3]] == list(zip(range(42, 52), range(1, 11)))
    for job in plan["jobs"]:
        command = comparison.command_for(plan, profile, job, tmp_path)
        assert "--ppo_override" not in command and "--ne_override" not in command
        assert command[-2:] == ["--checkpoint_every", "1500" if job["method"] == "ppo" else "200"]


def test_amended_reporting_keeps_twenty_full_budget_ga_es_trials(tmp_path):
    plan = comparison.make_plan(mode="reporting", methods=["ga", "es"], cpus=1)
    original = comparison.make_plan(mode="reporting", cpus=1)
    profile = plan["profiles"]["paper-cartpole"]
    assert profile == load_profile("paper-cartpole")
    assert plan["methods"] == ["ga", "es"]
    assert len(plan["jobs"]) == 20
    assert plan["jobs"] == [job for job in original["jobs"] if job["method"] != "ppo"]
    assert plan["nominal_training_steps_per_method"] == dict.fromkeys(["ga", "es"], 3072000000)
    for job in plan["jobs"]:
        command = comparison.command_for(plan, profile, job, tmp_path)
        assert command == comparison.command_for(original, profile, job, tmp_path)
        assert "--ppo_override" not in command and "--ne_override" not in command
        assert command[-2:] == ["--checkpoint_every", "200"]


def test_development_uses_full_budget_and_separate_identity(tmp_path):
    plan = comparison.make_plan(mode="development", cpus=1)
    assert plan["methods"] == ["es", "ppo"]
    assert {(j["seed"], j["trial"], j["eval_seed"]) for j in plan["jobs"]} == {(1001, 1002, 901001)}
    profile = plan["profiles"]["paper-cartpole"]
    assert profile["ne"] == load_profile("paper-cartpole")["ne"]
    assert profile["ppo"] == load_profile("paper-cartpole")["ppo"]
    assert "--ppo_override" not in comparison.command_for(plan, profile, plan["jobs"][1], tmp_path)


@pytest.mark.parametrize("mode,methods", [
    ("reporting", ["ga"]), ("reporting", ["es"]), ("reporting", ["ppo"]),
    ("reporting", ["ga", "ppo"]), ("reporting", ["es", "ppo"]),
    ("reporting", ["es", "ga"]), ("reporting", ["ppo", "ga", "es"]),
    ("diagnostic", ["es"]), ("diagnostic", ["ga", "es"]),
    ("development", ["ga", "ga"]),
])
def test_methods_cannot_silently_shrink_reporting(mode, methods):
    with pytest.raises(ValueError):
        comparison.make_plan(mode=mode, methods=methods, cpus=1)


def test_diagnostic_keeps_reporting_and_calibration_separate():
    plan = comparison.make_plan(mode="diagnostic", cpus=1)
    assert len(plan["jobs"]) == 3
    assert {j["seed"] for j in plan["jobs"]} == {3001}
    assert {j["trial"] for j in plan["jobs"]} == {3002}
    assert plan["profiles"]["smoke"]["num_phases"] == 2


@pytest.mark.parametrize("method,unit,last", [("es", "generation", "gen"), ("ppo", "update", "update")])
def test_monitor_measures_native_ne_and_ppo_child(tmp_path, method, unit, last):
    command = [sys.executable, "-c", f"print('checkpoint: {unit} 2 -> state',flush=True);"
               f"print('{last} 3 task=1',flush=True)"]
    active = tmp_path / "active.json"
    result = comparison.monitor_training(command=command, cwd=tmp_path, path=tmp_path,
                                         profile=load_profile("smoke"), method=method,
                                         timeout=5, active_path=active)
    assert result["status"] == "complete" and result["returncode"] == 0
    assert result["maximum_trainer_rss_kib"] > 0
    assert [e["completed_updates"] for e in result["phase_events"]] == [2, 4]
    assert not active.exists()


def test_monitor_timeout_preserves_progress_and_reaps_child(tmp_path):
    command = [sys.executable, "-c", "import time; print('checkpoint: update 2 -> state',flush=True);time.sleep(20)"]
    result = comparison.monitor_training(command=command, cwd=tmp_path, path=tmp_path,
                                         profile=load_profile("smoke"), method="ppo",
                                         timeout=1, active_path=tmp_path / "active.json")
    assert result["status"] == "timed_out" and result["returncode"] < 0
    assert result["phase_events"][0]["completed_updates"] == 2
    assert result["wall_seconds"] < 5


def test_duplicate_controller_and_live_orphan_are_rejected(tmp_path):
    with comparison.suite_lock(tmp_path):
        with pytest.raises(ValueError, match="controller"):
            with comparison.suite_lock(tmp_path):
                pass
    comparison.write_json(tmp_path / "active-process.json", {
        "pid": os.getpid(), "start_ticks": comparison._pid_identity(os.getpid())})
    with pytest.raises(ValueError, match="still active"):
        with comparison.suite_lock(tmp_path):
            pass


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = {"training": [], "analysis": [], "fail_analysis": False}
    monkeypatch.setattr(comparison, "verify_upstream", lambda path: None)
    monkeypatch.setattr(comparison, "source_hashes", lambda: {"test-source": "unchanged"})
    monkeypatch.setattr(comparison, "runtime_probe", lambda python, env: {
        "jax_backend": "cpu", "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": comparison.THREAD_ENV, "packages": {"synthetic": "1"}})

    def train(**kwargs):
        path, job, profile, plan = (kwargs[k] for k in ("path", "job", "profile", "plan"))
        calls["training"].append(path)
        path.mkdir(parents=True)
        comparison.write_json(path / "manifest.json", {"status": "complete", "method": job["method"]})
        comparison.write_json(path / "test-validation.json", {
            "wall_seconds": 2.0, "maximum_trainer_rss_kib": 100,
            "noise_vectors": [[0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4]],
            "nominal_training_steps": plan["nominal_training_steps_per_method"][job["method"]]})
        comparison.write_json(path / "receipt.json", comparison.artifact_hashes(path))

    def validate(path, **kwargs):
        assert comparison.read_json(path / "receipt.json") == comparison.artifact_hashes(path)
        return comparison.read_json(path / "test-validation.json")

    def analyze(**kwargs):
        path = kwargs["output_dir"]
        calls["analysis"].append(path)
        path.mkdir(parents=True)
        comparison.write_json(path / "manifest.json", {
            "status": "failed" if calls["fail_analysis"] else "complete", "wall_seconds": 1.0})
        if calls["fail_analysis"]:
            calls["fail_analysis"] = False
            raise RuntimeError("injected analysis failure")
        comparison.write_json(path / "summary.json", {"eval_seed": kwargs["eval_seed"]})

    monkeypatch.setattr(comparison, "run_training", train)
    monkeypatch.setattr(comparison, "validate_training", validate)
    monkeypatch.setattr(comparison, "run_analysis", analyze)
    monkeypatch.setattr(comparison, "validate_analysis", lambda **kwargs:
                        comparison.read_json(kwargs["output_dir"] / "summary.json"))
    return calls


def run_mock(output, **kwargs):
    return comparison.run_comparison(output=output, mode="diagnostic", cpus=1,
                                     python=sys.executable, **kwargs)


def test_resume_reuses_verified_completed_trial_without_training(tmp_path, fake_runtime):
    output = tmp_path / "suite"
    first = run_mock(output, max_trials=1)
    assert first["completed_trials"] == 1 and first["status"] == "partial"
    result = run_mock(output, resume=True)
    assert result["status"] == "complete" and result["completed_trials"] == 3
    assert len(fake_runtime["training"]) == len(fake_runtime["analysis"]) == 3


def test_analysis_failure_preserves_training_and_retries_only_analysis(tmp_path, fake_runtime):
    output = tmp_path / "suite"
    fake_runtime["fail_analysis"] = True
    with pytest.raises(RuntimeError, match="injected"):
        run_mock(output)
    failed = fake_runtime["analysis"][0] / "manifest.json"
    original = failed.read_bytes()
    result = run_mock(output, resume=True, max_trials=1)
    assert result["completed_trials"] == 1
    assert len(fake_runtime["training"]) == 1 and len(fake_runtime["analysis"]) == 2
    assert failed.read_bytes() == original
    assert Path(result["rows"][0]["analysis_path"]).name == "attempt_002"


def test_resume_requires_unchanged_source_and_artifacts(tmp_path, fake_runtime, monkeypatch):
    output = tmp_path / "suite"
    run_mock(output, max_trials=1)
    monkeypatch.setattr(comparison, "source_hashes", lambda: {"test-source": "changed"})
    with pytest.raises(ValueError, match="Frozen plan"):
        run_mock(output, resume=True)
    assert len(fake_runtime["training"]) == 1


def test_two_interruptions_before_next_boundary_keep_native_checkpoint_resumable(tmp_path, monkeypatch):
    plan = comparison.make_plan(mode="diagnostic", cpus=1)
    profile, job = plan["profiles"]["smoke"], plan["jobs"][2]
    first = tmp_path / "attempt_001"
    first.mkdir()
    (first / "resume.pkl").write_bytes(b"trusted native state, never unpickled by harness")
    comparison.write_json(first / "manifest.json", {
        "status": "failed", "profile": profile, "method": job["method"],
        "seed": job["seed"], "trial": job["trial"], "upstream_commit": comparison.UPSTREAM_COMMIT,
        "ga_settings": None, "command": comparison.command_for(plan, profile, job, first)})
    comparison.write_json(first / "process-measurement.json", {
        "start_update": 0, "phase_events": [{"kind": "native_checkpoint_written", "completed_updates": 2}]})
    comparison.write_json(first / "receipt.json", comparison.artifact_hashes(first))
    seen = []

    def interrupt(**kwargs):
        seen.append((kwargs["start_update"], (kwargs["path"] / "resume.pkl").read_bytes()))
        return {"status": "timed_out", "wall_seconds": 1.0, "phase_events": [], "start_update": 2}

    monkeypatch.setattr(comparison, "monitor_training", interrupt)
    previous = first
    for number in (2, 3):
        current = tmp_path / f"attempt_{number:03d}"
        with pytest.raises(ValueError, match="timed_out"):
            comparison.run_training(path=current, plan=plan, profile=profile, job=job,
                                    output=tmp_path, resume_source=previous)
        assert comparison.read_json(current / "receipt.json") == comparison.artifact_hashes(current)
        previous = current
    assert seen[0] == seen[1] == (2, (first / "resume.pkl").read_bytes())
