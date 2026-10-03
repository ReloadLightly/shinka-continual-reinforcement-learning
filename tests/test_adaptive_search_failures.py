"""Preserve search failures and training already spent before scoring failed."""

from pathlib import Path

import pytest

from shinka_crl import adaptive_search as search
from shinka_crl.experiment import REPO_ROOT, load_profile, nominal_training_steps
from shinka_crl.pilot import read_json, sha256, write_json
from shinka_crl.reference_timing import artifact_hashes


@pytest.fixture
def failed_search(tmp_path, monkeypatch):
    output, study = tmp_path / "search", tmp_path / "study"
    output.mkdir()
    study.mkdir()
    evaluation = {"profile": load_profile("adaptive-search"), "objective_version": "test-objective"}
    write_json(study / "plan.json", evaluation)
    write_json(study / "plan-receipt.json", {"plan_sha256": sha256(study / "plan.json")})
    plan = {"source_sha256": {name: sha256(REPO_ROOT / name) for name in search.SOURCE_FILES},
            "evaluation_study": str(study), "evaluation_context_sha256": search.digest(evaluation),
            "protocol": "adaptive-shinka-v1", "controls": {}}
    write_json(output / "plan.json", plan)
    write_json(output / "state.json", {"status": "failed", "sessions": [{"wall_seconds": 12.}],
                                        "programs": [], "artifact_sha256": {}})
    for target, source in search.TASK_FILES.items():
        path = output / "task" / target
        path.parent.mkdir(exist_ok=True)
        path.write_bytes((REPO_ROOT / source).read_bytes())
    request_dir = output / "shinka/gen_1/results/evaluation"
    request_dir.mkdir(parents=True)
    origin = "cache/new-candidate/attempt_0001"
    attempt = study / origin
    attempt.mkdir(parents=True)
    write_json(attempt / "summary.json", {"status": "failed", "context_sha256": search.digest(evaluation),
                                          "trials": [], "error": "post-hoc scoring failed"})
    for seed, status in ((4001, "complete"), (4002, "failed")):
        training = attempt / f"seed_{seed}/training"
        training.mkdir(parents=True)
        write_json(training / "manifest.json", {"status": status, "seed": seed, "wall_seconds": 3.})
    (attempt / "checkpoint.npz").write_bytes(b"retained binary evidence")
    write_json(attempt / "receipt.json", artifact_hashes(attempt))
    request = {"request_id": "evaluation", "context_sha256": search.digest(evaluation),
               "status": "failed", "slot_consumed": True, "evaluator_model_calls": 0,
               "cache_hit": False, "cache_origin": origin, "wall_seconds": 9.,
               "error": "post-hoc scoring failed"}
    write_json(request_dir / "request.json", request)
    write_json(request_dir / "correct.json", {"correct": False, "error": request["error"]})
    write_json(request_dir / "metrics.json", {"combined_score": 0.})
    write_json(request_dir / "receipt.json", artifact_hashes(request_dir))
    monkeypatch.setattr(search, "read_plan", lambda _: evaluation)

    def verified(*args, allow_failed=False):
        assert allow_failed is True
        return []

    monkeypatch.setattr(search, "verified_rows", verified)
    return output, study, evaluation, request_dir, attempt


def test_posthoc_failure_counts_completed_and_incomplete_training(failed_search):
    output, _, evaluation, _, _ = failed_search
    result = search.summarize(output)
    steps = nominal_training_steps(evaluation["profile"], "ga")
    assert result["status"] == "failed"
    assert result["slots_consumed"] == result["proposal_slots_consumed"] == 1
    assert result["programs_evaluated"] == result["new_scored_training_trials"] == 0
    assert result["new_training_trials"] == result["incomplete_training_attempts"] == 1
    assert result["new_nominal_training_steps"] == steps
    assert result["new_training_trials_allocated"] == 2
    assert result["new_nominal_training_steps_allocated"] == 2 * steps
    assert result["training_wall_seconds"] == 6.
    assert result["evaluation_wall_seconds"] == 9.
    assert result["failed_requests"][0]["receipt_verified"] is True
    assert result["cache_attempts"][0]["receipt_verified"] is True


@pytest.mark.parametrize("part", ["request", "training"])
def test_failed_evidence_tampering_is_rejected(failed_search, part):
    output, _, _, request_dir, attempt = failed_search
    target = request_dir / "metrics.json" if part == "request" else attempt / "seed_4001/training/manifest.json"
    target.write_text("{}")
    with pytest.raises(ValueError, match="evidence changed"):
        search.summarize(output)


def test_failed_export_includes_partial_attempt_and_hashes_binary(failed_search, tmp_path):
    output, _, _, _, attempt = failed_search
    report = tmp_path / "report"
    result = search.export_search(output=output, report=report)
    assert result["status"] == "failed"
    raw = report / "raw/evaluation/cache/new-candidate/attempt_0001"
    assert read_json(raw / "summary.json")["status"] == "failed"
    assert read_json(raw / "seed_4001/training/manifest.json")["status"] == "complete"
    assert not list(report.rglob("*.npz"))
    checksums = read_json(report / "checksums.json")
    assert checksums["original_sha256"][str(Path("raw/evaluation/cache/new-candidate/attempt_0001/checkpoint.npz"))] == sha256(attempt / "checkpoint.npz")
    assert all(sha256(report / name) == value for name, value in checksums["published_sha256"].items())


def test_interrupted_unsealed_evidence_is_explicit_and_exportable(failed_search, tmp_path):
    output, _, _, request_dir, attempt = failed_search
    (request_dir / "receipt.json").unlink()
    (attempt / "receipt.json").unlink()
    saved = read_json(request_dir / "request.json")
    saved["status"] = "running"
    write_json(request_dir / "request.json", saved)
    result = search.export_search(output=output, report=tmp_path / "interrupted-report")
    assert result["status"] == "failed"
    assert result["failed_requests"][0]["status"] == "running"
    assert result["failed_requests"][0]["receipt_verified"] is False
    assert result["cache_attempts"][0]["receipt_verified"] is False
    assert result["new_training_trials"] == 1
