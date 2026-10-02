"""Exercise export against real pilot curves with synthetic artifact identities.

No trainer, held-out seed, or NumPy checkpoint loader is invoked. Real checked-in
development evaluations supply the returns; binary payloads are explicit test
placeholders and all affected receipts are rebuilt around the temporary paths.
Only finalist-handoff validation is stubbed, so trial and analysis validators run.
"""

import copy
from pathlib import Path
import shutil

import pytest

from shinka_crl import analysis, experiment, validation
from shinka_crl.pilot import TRAINING_FILES, read_json, sha256, write_json


@pytest.fixture
def export_fixture(tmp_path, monkeypatch):
    def make(*, complete=True):
        root = tmp_path / "runs"
        root.mkdir()
        frozen_dir = root / "finalists"
        frozen_dir.mkdir()
        profile = experiment.load_profile("pilot-switching")
        profile["seeds"] = [1001, 1002]
        settings = {"sigma": 0.5, "elite_ratio": 0.5}
        candidate = {
            "id": "candidate_001", "settings": settings, "program_sha256": "fixture-program",
            "memberships": [{"arm": arm, "development_rank": 1, "settings": settings,
                             "program_sha256": "fixture-program",
                             "program_path": "programs/fixture.py"}
                            for arm in ("shinka", "random")],
        }
        frozen = {"profile": profile, "candidates": [candidate]}
        write_json(frozen_dir / "manifest.json", frozen)
        monkeypatch.setattr(validation, "finalists", lambda path: read_json(path / "manifest.json"))
        plan = {"profile": profile, "candidate_ids": [candidate["id"]],
                "finalist_manifest_sha256": sha256(frozen_dir / "manifest.json"),
                "source_sha256": {"src/shinka_crl/validation.py": sha256(
                    experiment.REPO_ROOT / "src/shinka_crl/validation.py")}}
        write_json(root / "plan.json", plan)
        failed = root / "trials/candidate_001/seed_1001/training/attempt_001"
        failed.mkdir(parents=True)
        write_json(failed / "manifest.json", {"status": "failed", "wall_seconds": 14.5,
                                             "error": "synthetic earlier timeout"})
        (failed / "process.log").write_text(f"Earlier attempt retained at {root}\n")
        rows = []
        for seed in profile["seeds"] if complete else profile["seeds"][:1]:
            base = root / "trials" / candidate["id"] / f"seed_{seed}"
            training = base / "training" / ("attempt_002" if seed == 1001 else "attempt_001")
            evaluated = base / "analysis/attempt_001"
            training.mkdir(parents=True)
            evaluated.mkdir(parents=True)
            original = (experiment.REPO_ROOT / "reports/pilot-20261002/raw/trials"
                        / "pilot-switching/ga" / f"seed_{seed}")
            for name in TRAINING_FILES:
                if name == "checkpoints.npz":
                    (training / name).write_bytes(b"synthetic hashed checkpoint; never loaded")
                else:
                    shutil.copyfile(original / "training/attempt_001" / name, training / name)
            manifest = read_json(training / "manifest.json")
            manifest.update(profile=copy.deepcopy(profile), ga_settings=settings,
                            command=experiment.build_command(
                                profile=profile, method="ga", seed=seed, trial=seed + 1,
                                output_dir=training, ga_settings=settings))
            write_json(training / "manifest.json", manifest)
            write_json(training / "receipt.json", {name: sha256(training / name)
                                                     for name in TRAINING_FILES})
            for name in ("evaluation.json", "checkpoint-metadata.json", "process.log"):
                shutil.copyfile(original / "analysis/attempt_001" / name, evaluated / name)
            (evaluated / "evaluation-input").mkdir()
            for name in ("results.json", "checkpoints.npz"):
                shutil.copyfile(training / name, evaluated / "evaluation-input" / name)
            analyzed = analysis.summarize_trial(
                manifest=manifest, results=read_json(training / "results.json"),
                records=read_json(training / "training_metrics.json"),
                evaluation=read_json(evaluated / "evaluation.json"),
                checkpoint_metadata=read_json(evaluated / "checkpoint-metadata.json"),
                episodes=10, eval_seed=900000 + seed)
            write_json(evaluated / "summary.json", analyzed)
            analysis_manifest = read_json(original / "analysis/attempt_001/manifest.json")
            analysis_manifest.update(
                analysis_source_sha256=sha256(Path(analysis.__file__)),
                input_sha256={name: sha256(training / name) for name in analysis.INPUT_FILES},
                output_sha256={name: sha256(evaluated / name) for name in analysis.OUTPUT_FILES})
            write_json(evaluated / "manifest.json", analysis_manifest)
            phases = analyzed["phase_returns"]
            switches = [{"from_phase": i, "to_phase": i + 1, "task": phases[i]["task"],
                         "before": phases[i]["own_mean"],
                         "after": phases[i + 1]["previous_mean"],
                         "forgetting": phases[i]["own_mean"] - phases[i + 1]["previous_mean"]}
                        for i in range(3)]
            rows.append({"candidate_id": candidate["id"], "seed": seed, "trial": seed + 1,
                         "eval_seed": 900000 + seed,
                         "normalized_score": read_json(training / "summary.json")["normalized_score"],
                         "training_wall_seconds": manifest["wall_seconds"],
                         "analysis_wall_seconds": analysis_manifest["wall_seconds"],
                         "training_path": str(training.relative_to(root)),
                         "analysis_path": str(evaluated.relative_to(root)),
                         "analysis": analyzed, "switches": switches})
        suite = {"status": "complete" if complete else "partial", "planned_trials": 2,
                 "completed_trials": len(rows), "rows": rows,
                 "sessions": [{"wall_seconds": 100.0}], "wall_seconds": 100.0,
                 **validation.aggregate(rows, frozen)}
        write_json(root / "suite.json", suite)
        return root, tmp_path / "report", suite

    return make


def test_complete_export_rederives_metrics_and_preserves_failed_work(export_fixture):
    root, output, suite = export_fixture()
    report = validation.export_validation(root, output)
    assert report["selection_complete"]
    assert set(report["selected_winners"]) == {"shinka", "random"}
    assert report["rows"] == suite["rows"]
    # This real pilot seed improves early but loses performance after later switches.
    assert report["rows"][1]["analysis"]["metrics"]["forgetting"] == -21.5
    assert [round(s["forgetting"], 1) for s in report["rows"][1]["switches"]] == [
        -341.4, 135.4, 141.5]
    assert report["incomplete_training_attempts"] == 1
    assert report["nominal_steps_completed"] == 2 * 7_680_000
    assert report["nominal_steps_allocated"] == 3 * 7_680_000
    failure = next(attempt for attempt in report["training_attempts"] if attempt["status"] == "failed")
    assert failure["nominal_steps_completed"] is None
    assert failure["wall_seconds"] == 14.5
    raw_failed = output / "raw/trials/candidate_001/seed_1001/training/attempt_001"
    assert read_json(raw_failed / "manifest.json")["status"] == "failed"
    assert "<runs>" in (raw_failed / "process.log").read_text()
    checksums = read_json(output / "checksums.json")
    binaries = [entry for name, entry in checksums.items() if name.endswith(".npz")]
    assert binaries and all(not entry["exported"] and entry["original_sha256"] for entry in binaries)
    assert not list(output.rglob("*.npz"))
    assert read_json(output / "summary.json") == report


def test_partial_export_requires_explicit_flag_and_selects_no_winner(export_fixture):
    root, output, _ = export_fixture(complete=False)
    with pytest.raises(ValueError, match="incomplete"):
        validation.export_validation(root, output)
    assert not output.exists()
    report = validation.export_validation(root, output, allow_partial=True)
    assert report["completed_trials"] == 1 and report["planned_trials"] == 2
    assert not report["selection_complete"]
    assert report["selected_winners"] == {}
    assert report["nominal_steps_completed"] == 7_680_000
    assert report["nominal_steps_allocated"] == 2 * 7_680_000


def test_failed_stage_preserves_last_completed_trial_and_failed_seed_cost(export_fixture):
    root, output, suite = export_fixture(complete=False)
    failed = root / "trials/candidate_001/seed_1002/training/attempt_001"
    failed.mkdir(parents=True)
    write_json(failed / "manifest.json", {"status": "failed", "wall_seconds": 9.0,
                                         "error": "synthetic process failure"})
    suite.update(status="failed", error="synthetic process failure")
    write_json(root / "suite.json", suite)
    report = validation.export_validation(root, output, allow_partial=True)
    assert report["status"] == "failed" and report["completed_trials"] == 1
    assert report["selected_winners"] == {}
    assert report["incomplete_training_attempts"] == 2
    assert report["nominal_steps_completed"] == 7_680_000
    assert report["nominal_steps_allocated"] == 3 * 7_680_000


@pytest.mark.parametrize("change,message", [
    (lambda suite: suite["rows"][0].update(normalized_score=1.0), "Stale trial summary"),
    (lambda suite: suite["rows"][0]["switches"][0].update(forgetting=0.0),
     "Switch differences changed"),
    (lambda suite: suite["selected_winners"]["shinka"].update(validation_score=1.0),
     "Finalist aggregation changed"),
])
def test_export_rejects_stale_scores_switches_or_winner(export_fixture, change, message):
    root, output, suite = export_fixture()
    change(suite)
    write_json(root / "suite.json", suite)
    with pytest.raises(ValueError, match=message):
        validation.export_validation(root, output)
    assert not output.exists()


def test_export_rejects_changed_source_receipt(export_fixture):
    root, output, _ = export_fixture()
    plan = read_json(root / "plan.json")
    plan["source_sha256"]["src/shinka_crl/validation.py"] = "0" * 64
    write_json(root / "plan.json", plan)
    with pytest.raises(ValueError, match="Validation source changed"):
        validation.export_validation(root, output)
    assert not output.exists()


def test_export_rejects_changed_training_evidence(export_fixture):
    root, output, suite = export_fixture()
    path = root / suite["rows"][0]["training_path"] / "training_metrics.json"
    records = read_json(path)
    records[0]["centroid_task0"] = 499.0
    write_json(path, records)
    with pytest.raises(ValueError, match="Training artifact changed"):
        validation.export_validation(root, output)
    assert not output.exists()
