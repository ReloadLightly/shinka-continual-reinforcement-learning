"""Reference timing boundaries, child accounting, immutable partial evidence."""

import json
import os
import sys

import pytest

from shinka_crl import experiment, reference_timing as timing


def test_timing_profile_matches_paper_budget_with_development_identity():
    profile = timing.read_json(experiment.REPO_ROOT / "src/shinka_crl/profiles"
                               / "paper-cartpole-timing.json")
    paper = experiment.load_profile("paper-cartpole")
    for key in ("env", "num_phases", "num_tasks", "episode_length", "eval_episodes", "ne"):
        assert profile[key] == paper[key]
    assert profile["seeds"] == [1001]
    assert profile["ppo"] is None
    assert experiment.nominal_training_steps(profile, "ga") == 3_072_000_000


@pytest.mark.skipif(timing.PROFILE not in experiment.PROFILE_NAMES,
                    reason="Profile support patch is deferred until static search export")
def test_preview_full_reference_does_not_touch_reporting_seeds(tmp_path):
    output = tmp_path / "new"
    plan = timing.make_plan(output=output, cpus=1)
    assert plan["profile"]["seeds"] == [1001]
    assert plan["trial"] == 1002
    assert plan["command"][-2:] == ["--checkpoint_every", "200"]
    assert "--max_gens_this_run" not in plan["command"]
    assert not output.exists()


def test_phase_events_use_checkpoint_completions_and_final_generation():
    profile = experiment.load_profile("smoke")
    assert timing.boundary_event(" gen 0 task=0", elapsed=2.0, profile=profile) is None
    first = timing.boundary_event(" checkpoint: generation 2 -> /tmp/resume.pkl",
                                   elapsed=4.0, profile=profile)
    last = timing.boundary_event(" gen 3 task=1 train=100.0", elapsed=7.0, profile=profile)
    phases = timing.phase_intervals([first, last], profile)
    assert [p["observed_wall_seconds"] for p in phases] == [4.0, 3.0]
    assert [p["task"] for p in phases] == [0, 1]
    with pytest.raises(ValueError, match="contiguous"):
        timing.phase_intervals([last], profile)
    with pytest.raises(ValueError, match="contiguous"):
        timing.phase_intervals([first, first], profile)
    with pytest.raises(ValueError, match="timestamp"):
        timing.phase_intervals([first, {**last, "elapsed_seconds": 3.0}], profile)


@pytest.mark.parametrize("line", ["checkpoint: generation 1 -> x",
                                  "checkpoint: generation 4 -> x"])
def test_nonphase_checkpoint_events_are_rejected(line):
    with pytest.raises(ValueError, match="boundary"):
        timing.boundary_event(line, elapsed=1.0, profile=experiment.load_profile("smoke"))


def test_monitor_accounts_for_exact_child_and_retains_logs(tmp_path):
    script = (
        "import time; print('checkpoint: generation 2 -> /tmp/resume.pkl',flush=True);"
        "time.sleep(0.05); print('gen 3 task=1 train=100.0',flush=True)"
    )
    result = timing.monitor_training(command=[sys.executable, "-c", script], cwd=tmp_path,
                                     output=tmp_path, profile=experiment.load_profile("smoke"),
                                     timeout=5, environment=dict(os.environ))
    assert result["status"] == "complete"
    assert result["returncode"] == 0
    assert result["maximum_trainer_rss_kib"] > 0
    assert len(result["phase_timings"]) == 2
    assert result["wall_seconds"] >= result["phase_events"][-1]["elapsed_seconds"]
    assert "gen 3" in (tmp_path / "process.log").read_text()


def test_timeout_retains_completed_boundaries_and_reaps_child(tmp_path):
    script = (
        "import time; print('checkpoint: generation 2 -> /tmp/resume.pkl',flush=True);"
        "time.sleep(20)"
    )
    result = timing.monitor_training(command=[sys.executable, "-c", script], cwd=tmp_path,
                                     output=tmp_path, profile=experiment.load_profile("smoke"),
                                     timeout=1, environment=dict(os.environ))
    assert result["status"] == "timed_out"
    assert result["returncode"] < 0
    assert result["wall_seconds"] < 5
    assert len(result["phase_timings"]) == 1
    assert result["phase_timings"][0]["completed_generations"] == 2


@pytest.fixture
def fake_reference_runtime(monkeypatch):
    monkeypatch.setattr(timing, "verify_upstream", lambda path: experiment.UPSTREAM_COMMIT)
    monkeypatch.setattr(timing, "source_hashes", lambda: {"source": "frozen"})
    monkeypatch.setattr(timing, "runtime_probe", lambda python, env: {
        "jax_backend": "cpu", "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "thread_environment": timing.THREAD_ENV})

    def monitor(**kwargs):
        root, profile = kwargs["output"], kwargs["profile"]
        original = experiment.REPO_ROOT / "reports/smoke-20261002/ga"
        results = timing.read_json(original / "results.json")
        results["config"].update(seed=1001, trial=1002, checkpoint_every=2)
        timing.write_json(root / "results.json", results)
        (root / "training_metrics.json").write_bytes((original / "training_metrics.json").read_bytes())
        (root / "process.log").write_text("native training output\n")
        (root / "resume.pkl").write_bytes(b"must-never-be-loaded-or-published")
        events = [{"completed_generations": 2, "elapsed_seconds": 1.0,
                   "kind": "native_checkpoint_written"},
                  {"completed_generations": 4, "elapsed_seconds": 2.0,
                   "kind": "last_generation_logged"}]
        (root / "phase-events.jsonl").write_text(
            "".join(json.dumps(event) + "\n" for event in events))
        return {"status": "complete", "returncode": 0, "wall_seconds": 2.5,
                "trainer_cpu_affinity": sorted(os.sched_getaffinity(0)),
                "maximum_trainer_rss_kib": 100, "phase_events": events,
                "phase_timings": timing.phase_intervals(events, profile)}

    monkeypatch.setattr(timing, "monitor_training", monitor)
    monkeypatch.setattr(timing, "run_analysis", lambda **kwargs: {"metrics": {"LA": 100.0}})
    monkeypatch.setattr(timing, "validate_analysis", lambda **kwargs: {"metrics": {"LA": 100.0}})


def test_complete_run_export_preserves_receipts_and_omits_binary(tmp_path, fake_reference_runtime):
    output, report = tmp_path / "run", tmp_path / "report"
    before_affinity = os.sched_getaffinity(0)
    before_threads = {k: os.environ.get(k) for k in timing.THREAD_ENV}
    result = timing.run_reference_timing(output=output, python=sys.executable,
                                         profile_name="smoke", cpus=1)
    assert result["status"] == "complete", result.get("error")
    assert result["completed_phases_observed"] == 2
    assert result["training_steps_nominal_observed_lower_bound"] == 1024
    assert result["training"]["finalization_wall_seconds"] == 0.5
    assert os.sched_getaffinity(0) == before_affinity
    assert {k: os.environ.get(k) for k in timing.THREAD_ENV} == before_threads
    timing.export_reference_timing(run_dir=output, report_dir=report)
    assert not list(report.rglob("*.pkl"))
    assert "training/resume.pkl" in timing.read_json(report / "checksums.json")[
        "original_artifact_sha256"]
    with pytest.raises(ValueError, match="immutable"):
        timing.run_reference_timing(output=output, python=sys.executable, profile_name="smoke")
    (output / "training/process.log").write_text("changed")
    with pytest.raises(ValueError, match="changed after receipt"):
        timing.export_reference_timing(run_dir=output, report_dir=tmp_path / "tampered")


def test_failed_runtime_has_exportable_evidence(tmp_path, fake_reference_runtime, monkeypatch):
    monkeypatch.setattr(timing, "runtime_probe", lambda *args: {"jax_backend": "gpu"})
    output = tmp_path / "failed"
    result = timing.run_reference_timing(output=output, python=sys.executable,
                                         profile_name="smoke", cpus=1)
    assert result["status"] == "failed"
    assert result["completed_phases_observed"] == 0
    assert result["training_steps_nominal_observed_lower_bound"] == 0
    assert "requires CPU JAX" in result["error"]
    assert timing.export_reference_timing(run_dir=output, report_dir=tmp_path / "report") == result


@pytest.mark.parametrize("field,value,expected", [
    ("active_score", {"normalized_score": 1.0}, "Active score"),
    ("completed_phases_observed", 1, "Observed phase count"),
    ("training_steps_nominal_observed_lower_bound", 0, "Observed phase count"),
])
def test_export_rederives_scientific_summary_even_with_refreshed_receipt(
        tmp_path, fake_reference_runtime, field, value, expected):
    output = tmp_path / "run"
    timing.run_reference_timing(output=output, python=sys.executable, profile_name="smoke", cpus=1)
    summary = timing.read_json(output / "summary.json")
    summary[field] = value
    timing.write_json(output / "summary.json", summary)
    timing.write_json(output / "receipt.json", timing.artifact_hashes(output))
    with pytest.raises(ValueError, match=expected):
        timing.export_reference_timing(run_dir=output, report_dir=tmp_path / "report")


def test_artifact_inventory_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    output = tmp_path / "run"
    output.mkdir()
    (output / "linked.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        timing.artifact_hashes(output)


def test_supervisor_error_recovers_flushed_phase_evidence(tmp_path, fake_reference_runtime,
                                                         monkeypatch):
    def interrupted_monitor(**kwargs):
        event = {"completed_generations": 2, "elapsed_seconds": 1.0,
                 "kind": "native_checkpoint_written"}
        (kwargs["output"] / "phase-events.jsonl").write_text(json.dumps(event) + "\n")
        raise KeyboardInterrupt()

    monkeypatch.setattr(timing, "monitor_training", interrupted_monitor)
    result = timing.run_reference_timing(output=tmp_path / "interrupted", python=sys.executable,
                                         profile_name="smoke", cpus=1)
    assert result["status"] == "interrupted"
    assert result["completed_phases_observed"] == 1
    assert result["training_steps_nominal_observed_lower_bound"] == 512
    assert result["training"]["maximum_trainer_rss_kib"] is None
