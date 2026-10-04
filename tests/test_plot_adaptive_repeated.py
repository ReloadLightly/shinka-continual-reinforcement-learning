"""Check scientific aggregation and refusal of incomplete/tampered evidence."""

import copy
import importlib.util
import json
from pathlib import Path
import statistics

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plot_adaptive_repeated", ROOT / "scripts/plot_adaptive_repeated.py")
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)


def paired_conditions():
    conditions = {}
    for outer, difference in zip(PLOT.OUTER_SEEDS, (.1, .5), strict=True):
        for arm in ("evolutionary", "independent"):
            conditions[f"{arm}_{outer}"] = {"trials": [
                {"seed": seed, "score": {"combined_score": .2 + (difference if arm == "evolutionary" else 0)}}
                for seed in PLOT.SEEDS]}
    return conditions


def test_primary_uses_two_outer_means_not_ten_task_draws():
    repetitions, aggregate = PLOT.primary_comparison(paired_conditions())
    assert len(repetitions) == 2
    assert all(row["aggregate"]["n"] == 5 for row in repetitions)
    assert aggregate["n"] == 2
    assert aggregate["mean"] == pytest.approx(.3)
    assert aggregate["sample_sd"] == pytest.approx(statistics.stdev([.1, .5]))


def test_missing_task_draw_is_not_zero_imputed():
    conditions = paired_conditions()
    conditions[f"independent_{PLOT.OUTER_SEEDS[0]}"]["trials"].pop()
    with pytest.raises(ValueError, match="all five paired"):
        PLOT.primary_comparison(conditions)


def test_cannot_drop_an_outer_search():
    with pytest.raises(ValueError, match="both prespecified"):
        PLOT.primary_comparison(paired_conditions(), outer_seeds=PLOT.OUTER_SEEDS[:1])


def test_shared_recipe_can_retain_distinct_search_memberships():
    conditions = paired_conditions()
    for outer in PLOT.OUTER_SEEDS:
        conditions[f"independent_{outer}"] = conditions[f"evolutionary_{outer}"]
    repetitions, aggregate = PLOT.primary_comparison(conditions)
    assert aggregate == {"mean": 0., "sample_sd": 0., "n": 2}
    assert all(row["complete_five_seed_comparison"] for row in repetitions)


def published_old_trial():
    report = ROOT / "reports/adaptive-validation-complete-20261003"
    plan = json.loads((report / "frozen/plan.json").read_text())
    row = plan["trial_order"][0]
    prefix = "raw/trials/000/attempt_0001"

    def read(relative):
        return json.loads((report / relative).read_text())

    return plan, row, prefix, read


def test_metric_reconstruction_matches_actual_published_trial():
    plan, row, prefix, read = published_old_trial()
    score, analysis, curve = PLOT.recompute_trial(read, prefix, plan, row)
    saved = read(prefix + "/summary.json")["result"]
    assert score == saved["score"]
    assert analysis["phase_returns"] == saved["phase_returns"]
    assert len(curve) == 320
    assert score["reference_metrics"]["learning_accuracy"] == 500
    assert score["reference_metrics"]["forgetting"] > 0


def test_metric_reconstruction_rejects_changed_fresh_episode():
    plan, row, prefix, read = published_old_trial()

    def changed(relative):
        value = copy.deepcopy(read(relative))
        if relative.endswith("/evaluation.json"):
            entry = next(entry for entry in value["per_task"]
                         if entry["source"] == "centroid" and entry["task_idx"] == 1)
            entry["prev_returns"][0] += 1
        return value

    with pytest.raises(ValueError, match="reference metrics differ"):
        PLOT.recompute_trial(changed, prefix, plan, row)


def test_partial_export_cannot_generate_final_artifacts(tmp_path):
    report = tmp_path / "report"
    (report / "frozen").mkdir(parents=True)
    identity = {"kind": "repeated_reserved", "protocol_version": "adaptive-repeated-validation-v1"}
    (report / "summary.json").write_text(json.dumps({**identity, "status": "partial"}))
    (report / "frozen/plan.json").write_text(json.dumps(identity))
    hashes = {name: PLOT.digest(report / name) for name in ("summary.json", "frozen/plan.json")}
    (report / "checksums.json").write_text(json.dumps({"published_sha256": hashes}))
    output = tmp_path / "figure.svg"
    with pytest.raises(ValueError, match="Partial comparison"):
        PLOT.plot_adaptive_repeated(report, output)
    assert not any(output.with_suffix(suffix).exists() for suffix in (".svg", ".pdf", ".json"))


def test_published_artifact_hashes_are_checked_before_plotting(tmp_path):
    report = tmp_path / "report"
    report.mkdir()
    path = report / "summary.json"
    path.write_text('{}')
    (report / "checksums.json").write_text(json.dumps({"published_sha256": {"summary.json": "0" * 64}}))
    with pytest.raises(ValueError, match="Published artifact changed"):
        PLOT.load_data(report)
