"""Cumulative search figures require completed declared stages and verified evidence."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib", reason="Plotting dependencies are optional in the core harness")

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "plot_adaptive_search", ROOT / "scripts/plot_adaptive_search.py")
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)


def stage_summary(target):
    """Synthetic completion metadata; no proposals or training are run."""
    return {"programs_evaluated": target, "slots_consumed": target, "status": "complete",
            "programs": [{"generation": generation} for generation in range(target)],
            "sessions": [{"status": "complete", "target": target}]}


@pytest.mark.parametrize("target", [5, 13, 25])
def test_accepts_each_frozen_cumulative_stage(target):
    assert plot.completed_stage(stage_summary(target), {"stages": [5, 13, 25]}) == target


@pytest.mark.parametrize("mutation,match", [
    ("undeclared", "declared"), ("unsupported", "declared"),
    ("incomplete", "complete search stage"), ("consumed", "complete search stage"),
    ("running", "Completed search session"), ("earlier", "Completed search session"),
    ("no_session", "Completed search session"), ("missing", "Native generation sequence"),
    ("duplicate", "Native generation sequence"), ("reordered", "Native generation sequence"),
])
def test_rejects_partial_or_inconsistent_stage_metadata(mutation, match):
    summary, plan = stage_summary(13), {"stages": [5, 13, 25]}
    if mutation == "undeclared":
        plan["stages"] = [5, 25]
    elif mutation == "unsupported":
        summary = stage_summary(7)
        plan["stages"].append(7)
    elif mutation == "incomplete":
        summary["status"] = "running"
    elif mutation == "consumed":
        summary["slots_consumed"] = 14
    elif mutation == "running":
        summary["sessions"][-1]["status"] = "running"
    elif mutation == "earlier":
        summary["sessions"][-1]["target"] = 5
    elif mutation == "no_session":
        summary["sessions"] = []
    elif mutation == "missing":
        summary["programs"].pop()
    elif mutation == "duplicate":
        summary["programs"][-1] = deepcopy(summary["programs"][-2])
    else:
        summary["programs"].reverse()
    with pytest.raises(ValueError, match=match):
        plot.completed_stage(summary, plan)


def test_five_slot_regression_preserves_checked_scores_and_widths(tmp_path):
    report = ROOT / "reports/adaptive-shinka-stage5-20261003"
    controls = ROOT / "reports/adaptive-controls-20261003"
    historical = ROOT / "figures/adaptive-shinka-stage5-20261003.json"
    previous = json.loads(historical.read_text())
    preserved = {path: plot.sha256(path) for path in
                 (historical, historical.with_suffix(".svg"), historical.with_suffix(".pdf"))}
    output = tmp_path / "stage5-regression.svg"
    provenance = plot.plot_search(report, controls, output)
    written = json.loads(output.with_suffix(".json").read_text())
    assert provenance["stage_target"] == 5
    for key in ("selected_generation", "selected_program_sha256", "selected_seed_widths",
                "displayed_scores", "focus_control", "sigma_timing", "objective"):
        assert written[key] == previous[key]
    assert "raw/search/plan.json" in written["inputs"][report.name]
    assert "Five evaluated programs" in output.read_text()
    for name, digest in written["outputs_sha256"].items():
        assert plot.sha256(tmp_path / name) == digest
    assert {path: plot.sha256(path) for path in preserved} == preserved
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        plot.plot_search(report, controls, output)
