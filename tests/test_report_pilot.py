"""Pilot evidence rejects incomplete, stale, unpaired, or overwritten reports."""

import importlib.util
import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("report_pilot", REPO_ROOT / "scripts/report_pilot.py")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def update_json(path, update):
    value = json.loads(path.read_text())
    update(value)
    write_json(path, value)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    """Synthetic artifacts isolate exporter checks; training/analysis test their own validators."""
    import shinka_crl.analysis
    import shinka_crl.pilot

    repo = tmp_path / "repo"
    root = repo / "results/pilot"
    root.mkdir(parents=True)
    (repo / "source.py").write_text("# synthetic provenance\n")
    write_json(repo / "upstream.lock.json", {"continual_neuroevolution": {"commit": "pinned"}})
    profile = {"name": "pilot-stationary", "num_tasks": 1, "num_phases": 4,
               "episode_length": 500}
    rows, jobs, training, analyses = [], [], {}, {}
    for method in ("ga", "es", "ppo"):
        for seed in (1001, 1002, 1003):
            job = {"profile": profile["name"], "method": method, "seed": seed,
                   "trial": seed + 1, "eval_seed": 900000 + seed}
            jobs.append(job)
            path = root / f"trials/{method}/seed_{seed}"
            train_path = path / "training/attempt_001"
            analysis_path = path / "analysis/attempt_001"
            train_path.mkdir(parents=True)
            analysis_path.mkdir(parents=True)
            (train_path / "process.log").write_text(f"Synthetic run at {repo}\n")
            (train_path / "checkpoints.npz").write_bytes(b"synthetic checkpoint placeholder")
            write_json(analysis_path / "evaluation.json", {"synthetic": True})
            write_json(analysis_path / "manifest.json", {"wall_seconds": 5.0})
            summary = {**job, "condition": "stationary", "nominal_training_steps": 7680000,
                       "metrics": {"learning_accuracy": float(seed - 900),
                                   "forgetting": None, "zero_shot_transfer": None,
                                   "learning_minus_forgetting": None,
                                   "cumulative_reward_steps": 768000000.0,
                                   "cumulative_reward_steps_exact": 768000000.0,
                                   "normalized_curve_average": 0.2},
                       "phase_training_returns": [100.0, 110.0, 140.0, 160.0],
                       "phase_returns": []}
            analyses[analysis_path.resolve()] = summary
            training[train_path.resolve()] = {"noise_vectors": [[0, 0, 0, 0]],
                                             "nominal_training_steps": 7680000,
                                             "wall_seconds": 10.0}
            rows.append({**job, "training_path": str(train_path.relative_to(root)),
                         "analysis_path": str(analysis_path.relative_to(root)),
                         "training_wall_seconds": 10.0, "analysis_wall_seconds": 5.0,
                         "analysis": summary})
    plan = {"profiles": {profile["name"]: profile}, "methods": ["ga", "es", "ppo"],
            "jobs": jobs, "eval_episodes": 10, "upstream_commit": "pinned",
            "cpu_affinity": [0, 1], "thread_environment": {"OMP_NUM_THREADS": "1"},
            "source_sha256": {"source.py": reporter.sha256((repo / "source.py").read_bytes())}}
    write_json(root / "plan.json", plan)
    write_json(root / "environment.json", {
        "upstream_commit": "pinned", "jax_backend": "cpu", "python": "synthetic Python",
        "packages": {"jax": "synthetic"}, "jax_devices": ["synthetic CPU"],
        "source_sha256": plan["source_sha256"], "cpu_affinity": plan["cpu_affinity"],
        "thread_environment": plan["thread_environment"],
    })
    write_json(root / "suite.json", {"status": "complete", "rows": rows,
                                      "completed_trials": 9, "planned_trials": 9,
                                      "wall_seconds": 135.0})
    monkeypatch.setattr(shinka_crl.pilot, "validate_training",
                        lambda path, **kwargs: training[path])
    monkeypatch.setattr(shinka_crl.analysis, "validate_analysis",
                        lambda **kwargs: analyses[kwargs["output_dir"]])
    return {"repo": repo, "root": root, "output": repo / "reports/pilot", "plan": plan,
            "rows": rows, "training": training, "analyses": analyses}


def export(evidence, **kwargs):
    return reporter.export_pilot(evidence["root"], evidence["output"], evidence["repo"], **kwargs)


def test_export_records_seed_uncertainty_redacts_paths_and_keeps_binary_hashes(evidence):
    report = export(evidence)
    assert report["completed_trials"] == 9
    assert report["training_env_steps_nominal"] == 9 * 7680000
    assert report["adequacy_gate"]["status"] == "pass"
    assert report["groups"][0]["metrics"]["learning_accuracy"] == {
        "mean": 102.0, "sample_sd": 1.0, "n": 3, "values": [101.0, 102.0, 103.0]}
    checksums = json.loads((evidence["output"] / "checksums.json").read_text())
    binary = [value for key, value in checksums.items() if key.endswith(".npz")]
    assert len(binary) == 9 and not any(value["exported"] for value in binary)
    assert not list(evidence["output"].rglob("*.npz"))
    for path in evidence["output"].rglob("*.log"):
        assert str(evidence["repo"]) not in path.read_text()
        assert "<repo>" in path.read_text()


def test_partial_suite_requires_explicit_label(evidence):
    def partial(suite):
        suite.update(status="partial", rows=suite["rows"][:1], completed_trials=1)
    update_json(evidence["root"] / "suite.json", partial)
    with pytest.raises(ValueError, match="Incomplete pilot"):
        export(evidence)
    report = export(evidence, allow_partial=True)
    assert report["status"] == "partial"
    assert report["adequacy_gate"]["status"] == "not_assessed"
    assert report["validation"]["all_planned_trials"] is False


def test_reject_stale_source_before_creating_report(evidence):
    (evidence["repo"] / "source.py").write_text("# changed source\n")
    with pytest.raises(ValueError, match="Stale source hash"):
        export(evidence)
    assert not evidence["output"].exists()


def test_reject_environment_that_disagrees_with_frozen_plan(evidence):
    update_json(evidence["root"] / "environment.json", lambda env: env.update(cpu_affinity=[3]))
    with pytest.raises(ValueError, match="cpu_affinity differ"):
        export(evidence)


def test_reject_different_task_draws_between_methods(evidence):
    path = evidence["root"] / evidence["rows"][3]["training_path"]
    evidence["training"][path]["noise_vectors"] = [[1, 0, 0, 0]]
    with pytest.raises(ValueError, match="different task vectors"):
        export(evidence)


def test_reject_unmatched_nominal_budgets(evidence):
    row = evidence["rows"][0]
    evidence["training"][evidence["root"] / row["training_path"]]["nominal_training_steps"] = 12
    row["analysis"]["nominal_training_steps"] = 12
    update_json(evidence["root"] / "suite.json", lambda suite: suite["rows"].__setitem__(0, row))
    with pytest.raises(ValueError, match="Unmatched training budgets"):
        export(evidence)


def test_reject_analysis_embedded_in_suite_when_raw_changes(evidence):
    def alter(suite):
        suite["rows"][0]["analysis"]["metrics"]["learning_accuracy"] = 999.0
    update_json(evidence["root"] / "suite.json", alter)
    with pytest.raises(ValueError, match="Stale suite analysis"):
        export(evidence)


@pytest.mark.parametrize("mutation,error", [
    (lambda suite: suite["rows"].append(suite["rows"][0]), "Duplicate completed"),
    (lambda suite: suite["rows"][0].update(training_path="../../outside"), "escapes"),
    (lambda suite: suite.update(completed_trials=1), "completed-trial count"),
    (lambda suite: suite.update(status="running"), "running"),
])
def test_reject_inconsistent_or_unsafe_suite(evidence, mutation, error):
    update_json(evidence["root"] / "suite.json", mutation)
    with pytest.raises(ValueError, match=error):
        export(evidence)


def test_report_refuses_overwrite(evidence):
    evidence["output"].mkdir(parents=True)
    marker = evidence["output"] / "keep.txt"
    marker.write_text("retain existing evidence")
    with pytest.raises(ValueError, match="overwrite"):
        export(evidence)
    assert marker.read_text() == "retain existing evidence"


def test_gate_does_not_mix_the_two_thresholds_across_seeds(evidence):
    # One improving seed plus a different already-high seed does not satisfy either
    # preregistered two-of-three branch independently.
    rows = []
    for seed, means in zip((1001, 1002, 1003),
                           ([100, 100, 100, 160], [410, 410, 410, 420], [10, 10, 10, 10])):
        rows.append({"profile": "pilot-stationary", "method": "ga", "seed": seed,
                     "phase_training_returns": means})
    gate = reporter.adequacy_gate(rows, evidence["plan"])
    assert gate["assessments"][0]["status"] == "fail"
    assert gate["status"] == "fail"


def test_failed_attempts_remain_exported_with_their_original_identity(evidence):
    failed = evidence["root"] / "trials/failed/attempt_001"
    failed.mkdir(parents=True)
    write_json(failed / "manifest.json", {"status": "failed", "error": "synthetic failure"})
    export(evidence)
    exported = evidence["output"] / "raw/trials/failed/attempt_001/manifest.json"
    assert json.loads(exported.read_text())["status"] == "failed"
