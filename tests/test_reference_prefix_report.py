"""Prefix publication preserves partial scope, original failures, and raw evidence."""

from contextlib import nullcontext
from copy import deepcopy

import pytest

from shinka_crl import reference_prefix_report as reporter
from shinka_crl.experiment import DEFAULT_PYTHON, DEFAULT_UPSTREAM, REPO_ROOT
from shinka_crl.pilot import read_json, sha256, write_json
from shinka_crl.reference_comparison import artifact_hashes

DIAGNOSTIC = (REPO_ROOT / "results/reference-prefix-diagnostic-20261003-v3"
              / "test_native_zero_update_finali0")


def diagnostic_kwargs():
    training, analysis = DIAGNOSTIC / "derived", DIAGNOSTIC / "analysis"
    manifest = read_json(training / "manifest.json")
    return {"training": training, "analysis": analysis, "profile": manifest["profile"],
            "plan": {"python": str(DEFAULT_PYTHON), "upstream": str(DEFAULT_UPSTREAM), "eval_episodes": 2},
            "job": {"method": "ppo", "seed": 3001, "trial": 3002, "eval_seed": 903001},
            "checkpoint_sha256": manifest["derivation"]["source_checkpoint_sha256"]}


@pytest.mark.skipif(not DIAGNOSTIC.exists(), reason="Retained native diagnostic is local evidence")
def test_read_only_validation_of_actual_native_diagnostic():
    before = artifact_hashes(DIAGNOSTIC)
    checked = reporter.validate_finalized_artifacts(**diagnostic_kwargs())
    assert checked["analysis"]["num_phases"] == 2
    assert checked["analysis"]["nominal_training_steps"] == 512
    assert checked["fresh_evaluation_episodes"] == 8
    assert checked["finalizer"]["start_update"] == 4
    assert checked["finalizer"]["phase_events"] == []
    assert artifact_hashes(DIAGNOSTIC) == before


@pytest.mark.skipif(not DIAGNOSTIC.exists(), reason="Retained native diagnostic is local evidence")
def test_policy_change_is_rejected_without_retraining(monkeypatch):
    # A modified probe represents a corrupted policy array; original evidence
    # remains untouched, and the exporter must reject the disagreement.
    before = read_json(DIAGNOSTIC / "derived/checkpoint-probe.json")
    changed = {key: deepcopy(before[key]) for key in ("record_count", "records_sha256", "phase_agents")}
    changed["phase_agents"][0]["sha256"] = "0" * 64
    monkeypatch.setattr(reporter.prefix, "probe", lambda *args: changed)
    with pytest.raises(ValueError, match="records or phase policies differ"):
        reporter.validate_finalized_artifacts(**diagnostic_kwargs())


def test_auxiliary_verification_cost_is_grounded_in_published_evidence():
    root = reporter.DEFAULT_BUDGET_REPORT
    protocol, verification = (read_json(root / name) for name in ("protocol.json", "verification.json"))
    result = reporter.validate_auxiliary_evidence(
        root, plan_hash=protocol["source_plan_sha256"], endpoint=6000,
        seconds=verification["prior_auxiliary_seconds_to_charge"])
    assert result["allocation_seconds"] == 28800
    with pytest.raises(ValueError, match="Auxiliary compute"):
        reporter.validate_auxiliary_evidence(root, plan_hash=protocol["source_plan_sha256"],
                                              endpoint=6000, seconds=0)


@pytest.fixture
def synthetic_publication(tmp_path, monkeypatch):
    """Synthetic copy/checksum fixture only; never scientific trial evidence."""
    root, source, budget = (tmp_path / name for name in ("prefix", "original", "budget"))
    for path in (root, source, budget):
        path.mkdir()
    (root / "training").mkdir()
    write_json(root / "provenance.json", {"status": "complete", "original_suite": str(source),
                                          "profile": {"ppo": {"num_updates": 4}}})
    write_json(root / "training/training_metrics.json", [{"synthetic_fixture": True}])
    (root / "training/checkpoints.npz").write_bytes(b"test binary, never loaded")
    (root / "training/process.log").write_text(f"synthetic log from {root}\n")
    write_json(root / "receipt.json", artifact_hashes(root))
    (source / "attempt_001").mkdir()
    (source / "attempt_001/resume.pkl").write_bytes(b"test binary, never unpickled")
    write_json(source / "attempt_001/manifest.json", {"status": "failed", "synthetic_fixture": True})
    original_receipt = artifact_hashes(source / "attempt_001")
    write_json(budget / "protocol.json", {"synthetic_fixture": True})
    write_json(budget / "verification.json", {"synthetic_fixture": True})
    source_info = {"attempt_receipts": {"attempt_001": original_receipt}}
    monkeypatch.setattr(reporter.prefix, "terminal_source_lock", lambda path: nullcontext())
    monkeypatch.setattr(reporter.prefix, "inspect_source", lambda *args: source_info)
    # The real native reader and cost validator are tested separately above.
    # This fixture tests only atomic publication, redaction and immutability.
    report = {"mode": "synthetic_fixture", "status": "complete", "completed_reporting_trials": 0}
    monkeypatch.setattr(reporter, "validate_prefix", lambda *args: (report, {"synthetic_fixture": True}))
    return root, source, budget, tmp_path / "published", report


def test_export_preserves_sources_and_publishes_binary_hashes_only(synthetic_publication):
    root, source, budget, output, expected = synthetic_publication
    before = {path: artifact_hashes(path) for path in (root, source, budget)}
    report = reporter.export_reference_prefix(root, output, budget_report=budget)
    assert all(report[key] == value for key, value in expected.items())
    assert len(report["publication_provenance"]["repository_revision"]) == 40
    assert reporter.EXPORT_SOURCES[0] in report["publication_provenance"]["source_sha256"]
    checksums = read_json(output / "checksums.json")
    assert checksums["original_artifacts"]["raw/training/checkpoints.npz"]["exported"] is False
    assert checksums["original_artifacts"]["source-raw/attempt_001/resume.pkl"]["exported"] is False
    assert not list(output.rglob("*.npz")) and not list(output.rglob("*.pkl"))
    assert "<prefix>" in (output / "raw/training/process.log").read_text()
    assert read_json(output / "source-raw/attempt_001/manifest.json")["status"] == "failed"
    assert (output / "raw/training/training_metrics.json").exists()
    for relative, digest in checksums["published_sha256"].items():
        assert sha256(output / relative) == digest
    assert all(artifact_hashes(path) == before[path] for path in before)


def test_export_never_overwrites_or_writes_into_source(synthetic_publication):
    root, source, budget, output, _ = synthetic_publication
    output.mkdir()
    with pytest.raises(ValueError, match="fresh report"):
        reporter.export_reference_prefix(root, output, budget_report=budget)
    with pytest.raises(ValueError, match="outside its source"):
        reporter.export_reference_prefix(root, source / "report", budget_report=budget)


def test_export_rejects_symlink_and_mid_export_mutation(synthetic_publication, monkeypatch):
    root, _, budget, output, _ = synthetic_publication
    link = root / "linked.json"
    link.symlink_to(budget / "protocol.json")
    with pytest.raises(ValueError, match="symlinks"):
        reporter.export_reference_prefix(root, output, budget_report=budget)
    assert not output.exists()
    link.unlink()
    real_hashes = reporter.artifact_hashes
    monkeypatch.setattr(reporter, "artifact_hashes", lambda path: {**real_hashes(path), "changed": "sha"})
    with pytest.raises(ValueError, match="changed during publication"):
        reporter.export_reference_prefix(root, output, budget_report=budget)
    assert not output.exists()


def test_rejects_unfinished_prefix_before_creating_output(tmp_path):
    root = tmp_path / "prefix"
    root.mkdir()
    write_json(root / "provenance.json", {"status": "running"})
    with pytest.raises(ValueError, match="unfinished"):
        reporter.export_reference_prefix(root, tmp_path / "report")
    assert not (tmp_path / "report").exists()


def test_rejects_modified_prefix_receipt_before_inspecting_metrics(tmp_path):
    write_json(tmp_path / "receipt.json", {"not-real": "digest"})
    with pytest.raises(ValueError, match="receipt"):
        reporter.validate_prefix(tmp_path, {}, tmp_path)


def test_nonfinite_compute_cannot_be_published():
    for value in (True, float("nan"), float("inf"), "1"):
        with pytest.raises(ValueError, match="Invalid"):
            reporter.finite(value, "compute")
    with pytest.raises(ValueError, match="Negative"):
        reporter.nonnegative(-1, "compute")
