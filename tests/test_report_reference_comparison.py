"""Scientific exports reject protocol drift and keep completed and unsuccessful work distinct."""

import importlib.util
import json
from pathlib import Path

import pytest

from shinka_crl import reference_comparison as runner

SPEC = importlib.util.spec_from_file_location(
    "report_reference_comparison", Path(__file__).resolve().parents[1] / "scripts/report_reference_comparison.py")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def alter(path, change):
    value = json.loads(path.read_text())
    change(value)
    write(path, value)


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    """Mock only native validators; exporter identity, accounting and publication run normally."""
    training, analyses, calls = {}, {}, []
    monkeypatch.setattr(runner, "verify_upstream", lambda path: runner.UPSTREAM_COMMIT)

    def validate(path, **kwargs):
        calls.append((path, kwargs["job"]))
        return training[path]

    monkeypatch.setattr(runner, "validate_training", validate)
    monkeypatch.setattr(reporter, "validate_analysis", lambda **kwargs: analyses[kwargs["output_dir"]])

    def create(mode="diagnostic", methods=None):
        root = tmp_path / mode
        root.mkdir()
        plan = runner.make_plan(mode=mode, methods=methods, cpus=1)
        rows = []
        for job in plan["jobs"]:
            profile = plan["profiles"][job["profile"]]
            method, seed = job["method"], job["seed"]
            budget = profile["ppo"] if method == "ppo" else profile["ne"]
            count = budget["num_updates" if method == "ppo" else "num_generations"]
            steps = plan["nominal_training_steps_per_method"][method]
            base = root / "trials" / method / f"seed_{seed}"
            train, analysis = base / "training/attempt_0001", base / "analysis/attempt_0001"
            write(train / "manifest.json", {**job, "status": "complete", "wall_seconds": 10.0})
            write(train / "process-measurement.json", {
                "start_update": 0, "maximum_trainer_rss_kib": 1000,
                "phase_events": [{"completed_updates": count}]})
            (train / "checkpoints.npz").write_bytes(b"synthetic binary; never executed")
            (train / "process.log").write_text(f"Synthetic evidence at {root}\n")
            write(train / "training_metrics.json", [{"synthetic": True}])
            write(analysis / "manifest.json", {"status": "complete", "wall_seconds": 2.0})
            write(analysis / "evaluation.json", {"per_task": [{"returns": [10.0] * 10}]})
            write(analysis / "evaluation-input/evaluation.json", {"per_task": [{"returns": [10.0] * 10}]})
            noise = [[0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4]]
            training[train] = {"nominal_training_steps": steps, "noise_vectors": noise,
                               "wall_seconds": 10.0, "maximum_trainer_rss_kib": 1000}
            metric = {key: 100.0 for key in reporter.METRICS}
            metric.update(learning_accuracy=float(seed), forgetting=-10.0)
            measured = {**job, "nominal_training_steps": steps, "metrics": metric,
                        "phase_returns": [], "phase_training_returns": [100.0] * profile["num_phases"]}
            analyses[analysis] = measured
            rows.append({**job, "training_path": str(train.relative_to(root)),
                         "analysis_path": str(analysis.relative_to(root)),
                         "noise_vectors": noise, "analysis": measured,
                         "training_wall_seconds": 10.0, "analysis_wall_seconds": 2.0,
                         "maximum_trainer_rss_kib": 1000})
        write(root / "plan.json", plan)
        write(root / "environment.json", {
            **{key: plan[key] for key in ("upstream_commit", "source_sha256", "cpu_affinity", "thread_environment")},
            "jax_backend": "cpu", "jax_devices": ["synthetic CPU"], "python": "synthetic",
            "packages": {"jax": "synthetic"}})
        write(root / "suite.json", {"mode": mode, "status": "complete", "rows": rows,
                                   "completed_trials": len(rows), "planned_trials": len(rows),
                                   "wall_seconds": 12.0 * len(rows),
                                   "sessions": [{"wall_seconds": 12.0 * len(rows)}]})
        return {"root": root, "output": tmp_path / f"report-{mode}", "plan": plan,
                "rows": rows, "training": training, "analyses": analyses, "calls": calls}
    return create


def export(data, **kwargs):
    return reporter.export_reference_comparison(data["root"], data["output"], **kwargs)


@pytest.mark.parametrize("mode,count", [("diagnostic", 3), ("development", 2), ("reporting", 30)])
def test_modes_counts_and_raw_evidence_are_kept_distinct(evidence, mode, count):
    data = evidence(mode)
    report = export(data)
    assert report["mode"] == mode and report["completed_trials"] == count
    assert report["declared_methods"] == data["plan"]["methods"]
    assert report["original_three_method_target_complete"] == (mode == "reporting")
    assert len(data["calls"]) == count
    assert report["compute"]["scored_fresh_evaluation_episodes"] == count * 10
    assert report["compute"]["saved_fresh_evaluation_episodes_all_attempts"] == count * 10
    assert report["compute"]["training_wall_seconds_all_attempts"] == count * 10
    assert report["groups"][0]["metrics"]["forgetting"]["mean"] == -10
    if mode == "reporting":
        assert report["compute"]["completed_trial_training_env_steps_nominal"] == 92160000000
        assert report["groups"][0]["metrics"]["learning_accuracy"]["mean"] == 46.5
        assert report["groups"][0]["metrics"]["learning_accuracy"]["sample_sd"] > 0
    else:
        assert report["groups"][0]["metrics"]["learning_accuracy"]["sample_sd"] is None
    checksums = json.loads((data["output"] / "checksums.json").read_text())
    binaries = [entry for name, entry in checksums["original_artifacts"].items() if name.endswith(".npz")]
    assert len(binaries) == count and all(not entry["exported"] for entry in binaries)
    assert not list(data["output"].rglob("*.npz"))
    assert len(list(data["output"].rglob("training_metrics.json"))) == count
    for path in data["output"].rglob("*.log"):
        assert str(data["root"]) not in path.read_text()
    for path, expected in checksums["published_sha256"].items():
        assert reporter.digest((data["output"] / path).read_bytes()) == expected


def test_amended_reporting_is_complete_only_for_ga_es(evidence):
    data = evidence("reporting", methods=["ga", "es"])
    report = export(data)
    assert report["mode"] == "reporting" and report["status"] == "complete"
    assert report["declared_methods"] == ["ga", "es"]
    assert report["planned_trials"] == report["completed_trials"] == 20
    assert report["validation"]["all_planned_trials"]
    assert not report["original_three_method_target_complete"]
    assert "PPO reporting is deferred" in report["caption"]
    assert "original three-method target incomplete" in report["caption"]
    assert [group["method"] for group in report["groups"]] == ["ga", "es"]
    for group in report["groups"]:
        assert group["n"] == 10 and group["seeds"] == list(range(42, 52))
        assert group["metrics"]["learning_accuracy"]["n"] == 10
        assert group["metrics"]["learning_accuracy"]["sample_sd"] > 0
    assert report["compute"]["planned_training_env_steps_nominal"] == 61440000000
    assert report["compute"]["completed_trial_training_env_steps_nominal"] == 61440000000
    assert len(data["calls"]) == 20


@pytest.mark.parametrize("methods,count", [(None, 30), (["ga", "es"], 20)])
def test_partial_reporting_is_explicit_and_never_complete(evidence, methods, count):
    data = evidence("reporting", methods=methods)
    alter(data["root"] / "suite.json", lambda suite: suite.update(
        status="partial", rows=suite["rows"][:1], completed_trials=1))
    with pytest.raises(ValueError, match="Incomplete comparison"):
        export(data)
    report = export(data, allow_partial=True)
    assert report["status"] == "partial"
    assert report["planned_trials"] == count and report["completed_trials"] == 1
    assert not report["validation"]["all_planned_trials"]
    assert not report["original_three_method_target_complete"]
    # Trained agents not yet registered as completed analyses still incur compute.
    assert report["compute"]["training_wall_seconds_all_attempts"] == count * 10.0


@pytest.mark.parametrize("change", [
    lambda p: p["jobs"].pop(),
    lambda p: p["jobs"][0].update(seed=1001),
    lambda p: p["profiles"]["paper-cartpole"]["ne"].update(pop_size=64),
    lambda p: p["source_sha256"].update({"unexpected": "sha"}),
])
@pytest.mark.parametrize("methods", [None, ["ga", "es"]])
def test_reporting_rejects_changed_jobs_budget_or_sources(evidence, change, methods):
    data = evidence("reporting", methods=methods)
    alter(data["root"] / "plan.json", change)
    with pytest.raises(ValueError, match="Frozen reference plan"):
        export(data, allow_partial=True)
    assert not data["output"].exists()


@pytest.mark.parametrize("change,message", [
    (lambda s: s.update(status="running"), "running"),
    (lambda s: s.update(mode="reporting"), "mode"),
    (lambda s: s["rows"].append(s["rows"][0]), "Duplicate"),
    (lambda s: s.update(completed_trials=99), "trial counts"),
    (lambda s: s["rows"][0].update(training_path="../../outside"), "escapes"),
    (lambda s: s["rows"][0]["analysis"]["metrics"].update(forgetting=999), "raw evidence"),
    (lambda s: s["rows"][0].update(training_wall_seconds=1), "wall_seconds"),
    (lambda s: s.update(wall_seconds=1), "timing"),
])
def test_rejects_inconsistent_suite(evidence, change, message):
    data = evidence()
    alter(data["root"] / "suite.json", change)
    with pytest.raises(ValueError, match=message):
        export(data)


def test_failed_checkpoint_work_is_retained_and_resumption_is_not_double_counted(evidence):
    data = evidence()
    base = data["root"] / "trials/ga/seed_3001/training"
    previous = base / "attempt_0000"
    write(previous / "manifest.json", {"status": "failed", "wall_seconds": 3.0})
    write(previous / "process-measurement.json", {"start_update": 0, "maximum_trainer_rss_kib": 1500,
        "phase_events": [{"completed_updates": 2}]})
    (previous / "process.log").write_text("gen     2  task=1\n")
    (previous / "resume.pkl").write_bytes(b"retained; never loaded")
    alter(base / "attempt_0001/process-measurement.json", lambda m: m.update(start_update=2))
    report = export(data)
    compute = report["compute"]
    assert compute["unsuccessful_attempts"] == 1
    assert compute["training_wall_seconds_all_attempts"] == 33.0
    assert compute["maximum_trainer_rss_kib"] == 1500
    # First attempt performs 3 updates, resume repeats update 3: charge one extra update.
    assert compute["attempted_training_env_steps_nominal_observed_lower_bound"] == (
        compute["completed_trial_training_env_steps_nominal"] + 256)
    assert (data["output"] / "raw/trials/ga/seed_3001/training/attempt_0000/manifest.json").exists()
    assert not list(data["output"].rglob("*.pkl"))


def test_rejects_artifact_symlinks_and_overwrite(evidence, tmp_path):
    data = evidence()
    (data["root"] / "foreign.json").symlink_to(tmp_path / "elsewhere.json")
    with pytest.raises(ValueError, match="symlinks"):
        export(data)
    (data["root"] / "foreign.json").unlink()
    data["output"].mkdir()
    marker = data["output"] / "keep.txt"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="overwrite"):
        export(data)
    assert marker.read_text() == "preserve"


def test_different_task_draws_fail_even_with_valid_individual_trials(evidence):
    data = evidence()
    row = data["rows"][1]
    changed = [[0, 0, 0, 0], [9, 0, 0, 0]]
    data["training"][data["root"] / row["training_path"]]["noise_vectors"] = changed
    alter(data["root"] / "suite.json", lambda s: s["rows"][1].update(noise_vectors=changed))
    with pytest.raises(ValueError, match="different task vectors"):
        export(data)
