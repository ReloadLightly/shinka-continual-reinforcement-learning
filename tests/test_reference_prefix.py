"""A stopped development prefix must preserve its identity, policies and history."""

from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import time

import pytest

from shinka_crl import reference_prefix as prefix
from shinka_crl.analysis import run_analysis, validate_analysis
from shinka_crl.experiment import DEFAULT_PYTHON, DEFAULT_UPSTREAM, build_command, load_profile
from shinka_crl.pilot import read_json, sha256, write_json
from shinka_crl.reference_comparison import (
    THREAD_ENV, UPSTREAM_COMMIT, _pid_identity, artifact_hashes, command_for,
    make_plan, monitor_training,
)


def test_prefix_keeps_per_phase_budget_and_complete_cycles():
    original = load_profile("paper-cartpole")
    saved = deepcopy(original)
    result = prefix.prefix_profile(original, 6000)
    assert original == saved
    assert result["num_phases"] == 4
    assert result["ppo"] == {**original["ppo"], "num_updates": 6000}
    assert result["ne"] == {**original["ne"], "num_generations": 800}
    assert prefix.nominal_training_steps(result, "ppo") == 614400000
    assert prefix.nominal_training_steps(result, "ga") == 614400000
    assert "not the full-budget" in result["purpose"]


@pytest.mark.parametrize("endpoint", [True, 0, 1500, 7500, 5999, 30000, 33000])
def test_prefix_rejects_incomplete_cycles_and_non_prefixes(endpoint):
    with pytest.raises(ValueError):
        prefix.prefix_profile(load_profile("paper-cartpole"), endpoint)


@pytest.fixture
def source_suite(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    (root / "controller.lock").touch()
    plan = make_plan(mode="development", methods=["ppo"], cpus=1)
    write_json(root / "plan.json", plan)
    write_json(root / "suite.json", {"status": "failed"})
    attempt = root / "trials/ppo/seed_1001/training/attempt_001"
    attempt.mkdir(parents=True)
    profile = plan["profiles"]["paper-cartpole"]
    events = [{"kind": "native_checkpoint_written", "completed_updates": step,
               "elapsed_seconds": step * 3} for step in (1500, 3000, 4500, 6000)]
    write_json(attempt / "manifest.json", {
        "status": "failed", "profile": profile, "method": "ppo", "seed": 1001, "trial": 1002,
        "ga_settings": None, "upstream_commit": UPSTREAM_COMMIT, "wall_seconds": 22001,
        "command": command_for(plan, profile, plan["jobs"][0], attempt)})
    write_json(attempt / "process-measurement.json", {
        "status": "failed", "returncode": -15, "trainer_cpu_affinity": plan["cpu_affinity"],
        "wall_seconds": 22001, "phase_events": events, "start_update": 0,
        "maximum_trainer_rss_kib": 600000})
    (attempt / "phase-events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
    (attempt / "resume.pkl").write_bytes(b"a sealed test placeholder; never unpickle this")
    write_json(attempt / "receipt.json", artifact_hashes(attempt))
    monkeypatch.setattr(prefix, "verify_upstream", lambda path: UPSTREAM_COMMIT)
    return root, attempt, plan


def test_source_requires_sealed_checkpoint_provenance(source_suite):
    root, attempt, _ = source_suite
    checked = prefix.inspect_source(root, 6000)
    assert checked["source"] == attempt
    assert checked["checkpoint_sha256"] == sha256(attempt / "resume.pkl")
    assert checked["attempt_costs"][0]["wall_seconds"] == 22001
    (attempt / "resume.pkl").write_bytes(b"changed")
    with pytest.raises(ValueError, match="receipt"):
        prefix.inspect_source(root, 6000)


def test_source_rejects_changed_endpoint_or_reporting_identity(source_suite):
    root, _, plan = source_suite
    with pytest.raises(ValueError, match="last recorded native checkpoint"):
        prefix.inspect_source(root, 3000)
    plan["mode"] = "reporting"
    write_json(root / "plan.json", plan)
    with pytest.raises(ValueError, match="development"):
        prefix.inspect_source(root, 6000)


def test_lock_rejects_active_controller_and_live_native_process(source_suite):
    root, _, _ = source_suite
    with (root / "controller.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="controller is still active"):
            with prefix.terminal_source_lock(root):
                pass
    write_json(root / "active-process.json", {
        "pid": os.getpid(), "start_ticks": _pid_identity(os.getpid())})
    with pytest.raises(ValueError, match="trainer is still active"):
        with prefix.terminal_source_lock(root):
            pass


def test_endpoint_verification_prevents_accidental_additional_training(tmp_path, monkeypatch):
    checkpoint = tmp_path / "resume.pkl"
    checkpoint.write_bytes(b"sealed checkpoint")
    monkeypatch.setattr(prefix, "probe", lambda *args: {"step": 1500, "record_count": 1500})
    monkeypatch.setattr(prefix, "monitor_training", lambda **kwargs: pytest.fail("must not train"))
    with pytest.raises(ValueError, match="additional training is forbidden"):
        prefix.finalize_checkpoint(checkpoint=checkpoint, checkpoint_sha256=sha256(checkpoint),
                                   profile=prefix.prefix_profile(load_profile("paper-cartpole"), 6000),
                                   plan={}, job={}, output=tmp_path / "derived")
    assert checkpoint.read_bytes() == b"sealed checkpoint"


def test_resource_cap_and_existing_destination_rejected(tmp_path):
    with pytest.raises(ValueError, match="eight hours"):
        prefix.finalize_prefix(suite=tmp_path, output=tmp_path / "out", allocation_seconds=28801)
    with pytest.raises(ValueError, match="fresh output"):
        prefix.finalize_prefix(suite=tmp_path, output=tmp_path)


@pytest.mark.parametrize("seconds", [-1, True, float("inf"), float("nan")])
def test_resource_cap_rejects_invalid_prior_auxiliary_compute(tmp_path, seconds):
    with pytest.raises(ValueError, match="Prior auxiliary compute"):
        prefix.finalize_prefix(suite=tmp_path, output=tmp_path / "out", prior_auxiliary_seconds=seconds)


@pytest.mark.skipif(os.environ.get("SHINKA_RUN_NATIVE_PREFIX_TEST") != "1",
                    reason="Explicit native diagnostic; uses pinned PPO runtime")
def test_native_zero_update_finalizer_preserves_real_trained_agents(tmp_path):
    """A real segmented PPO run must produce identical prefix history and agents."""
    started = time.monotonic()
    profile = load_profile("smoke")
    profile["num_phases"] = 4
    profile["ppo"]["num_updates"] = 8
    profile["ne"]["num_generations"] = 8
    profile["seeds"] = [3001]
    short = prefix.prefix_profile(profile, 4)
    job = {"method": "ppo", "seed": 3001, "trial": 3002, "eval_seed": 903001}
    plan = {"python": str(DEFAULT_PYTHON), "upstream": str(DEFAULT_UPSTREAM)}
    reused = os.environ.get("SHINKA_PREFIX_TEST_CHECKPOINT")
    source = Path(reused).resolve().parent if reused else tmp_path / "native-segment"
    if not reused:
        source.mkdir()
    command = build_command(profile=profile, method="ppo", seed=job["seed"], trial=job["trial"],
                            output_dir=source) + ["--checkpoint_every", "2", "--max_gens_this_run", "4"]
    previous_affinity = os.sched_getaffinity(0)
    previous_env = {key: os.environ.get(key) for key in THREAD_ENV}
    try:
        # The root coordinates this diagnostic; a separate core avoids changing
        # the live calibration's recorded CPU allocation.
        os.sched_setaffinity(0, sorted(previous_affinity)[-2:])
        os.environ.update(THREAD_ENV)
        if reused:
            measured = read_json(source / "process-measurement.json")
        else:
            measured = monitor_training(command=command, cwd=DEFAULT_UPSTREAM, path=source,
                                        profile=profile, method="ppo", timeout=60,
                                        active_path=source / "active.json")
            write_json(source / "process-measurement.json", measured)
        assert measured["status"] == "complete"
        assert measured["phase_events"][-1]["completed_updates"] == 4
        assert not (source / "results.json").exists()
        checkpoint = source / "resume.pkl"
        digest = sha256(checkpoint)
        derived = tmp_path / "derived"
        finalized = prefix.finalize_checkpoint(checkpoint=checkpoint, checkpoint_sha256=digest,
                                               profile=short, plan=plan, job=job, output=derived,
                                               timeout=30)
        assert finalized["phase_events"] == []
        assert sha256(checkpoint) == digest
        assert not (derived / "resume.pkl").exists()
        assert read_json(derived / "manifest.json")["derivation"]["additional_training_updates"] == 0
        assert len(read_json(derived / "training_metrics.json")) == 4
        run_analysis(run_dir=derived, output_dir=tmp_path / "analysis", eval_seed=job["eval_seed"],
                     episodes=2, timeout=30)
        analysis = validate_analysis(run_dir=derived, output_dir=tmp_path / "analysis",
                                     episodes=2, eval_seed=job["eval_seed"])
        assert analysis["num_phases"] == 2
        assert analysis["nominal_training_steps"] == 512
        write_json(tmp_path / "diagnostic-summary.json", {
            "training_measurement": measured, "finalization_measurement": finalized,
            "source_checkpoint_sha256": digest, "analysis": analysis,
            "reused_training_checkpoint": str(checkpoint) if reused else None,
            "source_checkpoint_preserved": True, "additional_training_updates": 0})
    finally:
        write_json(tmp_path / "diagnostic-cost.json", {
            "total_measured_wall_seconds": time.monotonic() - started,
            "timing_basis": "monotonic native diagnostic, including all subprocesses and analysis"})
        os.sched_setaffinity(0, previous_affinity)
        for key, value in previous_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
