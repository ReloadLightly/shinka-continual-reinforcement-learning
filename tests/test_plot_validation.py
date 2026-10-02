"""Synthetic-only checks for paired finalist plots and signed switch losses."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

import pytest

pytest.importorskip("matplotlib", reason="Plotting dependencies are optional in the core harness")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plot_validation", ROOT / "scripts/plot_validation.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def synthetic_report(root: Path) -> Path:
    """All scores and episodes below are invented fixture data, not agent training."""
    root.mkdir(parents=True)
    original = ROOT / "reports/finalists-static-20261002"
    frozen = json.loads((original / "manifest.json").read_text())
    profile, candidates = frozen["profile"], frozen["candidates"]
    checksums = {}

    def artifact(relative, value, *, raw=False):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if raw else json.dumps(value).encode())
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksums[relative] = {"source": relative.removeprefix("raw/"), "exported": True,
                               "original_sha256": digest, "exported_sha256": digest}
        return digest

    frozen_hash = artifact("raw/finalists/manifest.json", frozen)
    artifact("raw/plan.json", {"profile": profile, "candidate_ids": [c["id"] for c in candidates],
                               "finalist_manifest_sha256": frozen_hash})
    for candidate in candidates:
        artifact("raw/finalists/" + candidate["program_path"],
                 (original / candidate["program_path"]).read_bytes(), raw=True)
    rows = []
    for seed_index, seed in enumerate(profile["seeds"]):
        for candidate_index, candidate in enumerate(candidates):
            training = f"trials/{candidate['id']}/seed_{seed}/training/attempt_001"
            analysis = f"trials/{candidate['id']}/seed_{seed}/analysis/attempt_001"
            reward = 140 + 50 * candidate_index + (seed_index - 2) * (18 - 8 * (candidate_index % 3))
            records = [{"generation": i, "task": i // 80 % 2,
                        f"centroid_task{i // 80 % 2}": reward} for i in range(320)]
            artifact("raw/" + training + "/training_metrics.json", records)
            artifact("raw/" + training + "/manifest.json", {
                "profile": profile, "method": "ga", "seed": seed, "trial": seed + 1,
                "ga_settings": candidate["settings"]})
            own = [100 + 70 * candidate_index + 5 * seed_index,
                   440 - 11 * candidate_index + 3 * seed_index,
                   400 - 7 * candidate_index + 2 * seed_index,
                   450 - 12 * candidate_index + seed_index]
            losses = [-230 + 100 * candidate_index, 80 + 7 * seed_index, 60 - 10 * candidate_index]
            previous = [None] + [own[i] - losses[i] for i in range(3)]
            entries = []
            for phase in range(4):
                entry = {"source": "centroid", "task_idx": phase, "returns": [own[phase]] * 10}
                if phase:
                    entry["prev_returns"] = [previous[phase]] * 10
                entries.append(entry)
            artifact("raw/" + analysis + "/evaluation.json", {
                "method": "ga", "env": "CartPole-v1", "pop_size": 64, "trial": seed + 1,
                "num_tasks": 4, "episodes": 10, "eval_seed": 900000 + seed, "per_task": entries})
            phases = [{"phase": i, "task": i % 2, "own_mean": own[i],
                       "previous_mean": previous[i]} for i in range(4)]
            switches = [{"from_phase": i, "to_phase": i + 1, "task": i % 2,
                         "before": own[i], "after": previous[i + 1],
                         "forgetting": losses[i]} for i in range(3)]
            rows.append({"candidate_id": candidate["id"], "seed": seed, "trial": seed + 1,
                         "eval_seed": 900000 + seed, "training_path": training,
                         "analysis_path": analysis, "normalized_score": reward / 500,
                         "analysis": {"phase_returns": phases,
                                      "metrics": {"forgetting": sum(losses) / 3}},
                         "switches": switches})
    groups = []
    for candidate in candidates:
        trials = [row for row in rows if row["candidate_id"] == candidate["id"]]
        def describe(values):
            return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "n": 5}
        groups.append({"candidate_id": candidate["id"], "settings": candidate["settings"],
                       "program_sha256": candidate["program_sha256"],
                       "memberships": candidate["memberships"], "completed_seeds": 5,
                       "score": describe([r["normalized_score"] for r in trials]),
                       "metrics": {"forgetting": describe([r["analysis"]["metrics"]["forgetting"]
                                                           for r in trials])}})
    write(root / "summary.json", {"status": "complete", "selection_complete": True,
                                  "synthetic": True, "profile": profile,
                                  "finalist_manifest_sha256": frozen_hash,
                                  "rows": rows, "groups": groups,
                                  "completed_trials": 25, "planned_trials": 25,
                                  "validation": {key: True for key in (
                                      "scores_rederived", "posthoc_metrics_rederived",
                                      "switch_differences_rederived", "artifact_receipts_match",
                                      "task_vectors_match")}})
    write(root / "checksums.json", checksums)
    return root


@pytest.fixture
def report(tmp_path):
    return synthetic_report(tmp_path / "synthetic-validation-fixture")


def test_reconstructs_paired_scores_and_preserves_signed_individual_losses(report):
    data, inputs = plot.load_plot_data(report)
    assert data["synthetic"] is True
    assert data["seeds"] == list(range(2001, 2006))
    assert [c["label"] for c in data["candidates"]] == ["Default", "Shinka 12", "Shinka 11",
                                                      "Random 7", "Random 24"]
    assert len(inputs) == 84
    for trial in data["candidates"][0]["trials"]:
        assert trial["mean_forgetting"] < 0
        assert [switch["forgetting"] > 0 for switch in trial["switches"]] == [False, True, True]


def test_emits_immutable_vector_artifacts_with_exact_provenance(report, tmp_path):
    output = tmp_path / "synthetic-validation.svg"
    sidecar = plot.plot_validation(report, output)
    assert sidecar["synthetic"] is True
    assert "SYNTHETIC TEST FIXTURE" in output.read_text()
    assert len(sidecar["candidates"]) == 5
    for name, value in sidecar["input_sha256"].items():
        assert plot.digest(report / name) == value
    for name, value in sidecar["outputs_sha256"].items():
        assert plot.digest(tmp_path / name) == value
    assert json.loads(output.with_suffix(".json").read_text()) == sidecar
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        plot.plot_validation(report, output)


@pytest.mark.parametrize("mutation,match", [
    ("partial", "complete reserved"), ("missing", "Missing, duplicate"),
    ("reordered", "Missing, duplicate"), ("score", "Active score differs"),
    ("switch", "Switch differences differ"), ("aggregate", "aggregation differs"),
    ("path", "Artifact escapes"),
])
def test_rejects_incomplete_or_altered_scientific_claims(report, mutation, match):
    summary = json.loads((report / "summary.json").read_text())
    if mutation == "partial":
        summary["status"] = "partial"
    elif mutation == "missing":
        summary["rows"].pop()
    elif mutation == "reordered":
        summary["rows"][0], summary["rows"][1] = summary["rows"][1], summary["rows"][0]
    elif mutation == "score":
        summary["rows"][0]["normalized_score"] = .999
    elif mutation == "switch":
        summary["rows"][0]["switches"][0]["forgetting"] = 0
    elif mutation == "aggregate":
        summary["groups"][0]["score"]["mean"] = .999
    else:
        summary["rows"][0]["training_path"] = "../../outside"
    write(report / "summary.json", summary)
    with pytest.raises(ValueError, match=match):
        plot.load_plot_data(report)


def test_rejects_changed_raw_episode_returns(report):
    path = next((report / "raw/trials").rglob("evaluation.json"))
    data = json.loads(path.read_text())
    data["per_task"][0]["returns"][0] = 500
    write(path, data)
    with pytest.raises(ValueError, match="Exported artifact changed"):
        plot.load_plot_data(report)


def test_shared_membership_labels_retain_both_original_programs():
    candidate = {"memberships": [{"arm": "shinka", "index": 12}, {"arm": "random", "index": 7}]}
    assert plot.candidate_style(candidate) == ("Shinka 12 / Random 7", plot.COLORS["shared"])
    candidate = copy.deepcopy(candidate)
    candidate["memberships"].append({"arm": "default", "index": 0})
    assert plot.candidate_style(candidate) == ("Default", plot.COLORS["default"])
