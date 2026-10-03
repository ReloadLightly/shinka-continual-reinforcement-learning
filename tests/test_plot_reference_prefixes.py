"""Check prefix evidence boundaries without rendering unobserved PPO results."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from shinka_crl.experiment import load_profile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plot_reference_prefixes", ROOT / "scripts/plot_reference_prefixes.py")
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


def prefix_profile():
    profile = load_profile("paper-cartpole")
    profile.update(num_phases=4, seeds=[1001])
    profile["ne"]["num_generations"] = 800
    profile["ppo"]["num_updates"] = 6000
    return profile


def test_exact_shortened_horizon_retains_native_phase_duration():
    plotter.validate_prefix_profile(prefix_profile())
    with pytest.raises(ValueError, match="exact four-phase"):
        plotter.validate_prefix_profile(load_profile("paper-cartpole"))
    profile = prefix_profile()
    profile["ppo"]["task_interval"] = 300
    with pytest.raises(ValueError, match="exact four-phase"):
        plotter.validate_prefix_profile(profile)


def test_reporting_seed_cannot_be_presented_as_this_development_trial():
    profile = prefix_profile()
    profile["seeds"] = [42]
    with pytest.raises(ValueError, match="development prefix protocol"):
        plotter.validate_prefix_profile(profile)


def test_real_ga_es_data_reproduce_exact_prefix_clock_and_excluded_future():
    grouped, vectors, hashes = plotter.load_ne_data()
    assert set(grouped) == {"ga", "es"}
    assert len(vectors) == 2
    assert len(hashes) == 47
    for data in grouped.values():
        assert data["x"].tolist() == [768000 * i for i in range(1, 801)]
        assert data["analysis"]["phase_returns"][-1]["next_mean"] is None
    assert grouped["ga"]["analysis"]["metrics"]["forgetting"] == pytest.approx(32.06666666666666)
    assert grouped["es"]["analysis"]["metrics"]["forgetting"] == 0.


@pytest.mark.parametrize("change,message", [
    (lambda rows: rows[0].update(generation=1), "contiguous"),
    (lambda rows: rows[200].update(task=0), "task schedule"),
])
def test_native_curve_validation_detects_shifted_clock_or_switch(change, message):
    report = plotter.NE_REPORT
    row = plotter.read(report / "summary.json")["rows"][0]
    records = deepcopy(plotter.read(report / row["evidence"]["training_records"])[:800])
    change(records)
    with pytest.raises(ValueError, match=message):
        plotter.curve(records, row["analysis"])


def test_incomplete_or_diagnostic_ppo_is_rejected_before_any_plot(tmp_path):
    (tmp_path / "summary.json").write_text(json.dumps({"status": "running", "mode": "development"}))
    (tmp_path / "checksums.json").write_text(json.dumps({"published_sha256": {
        "summary.json": plotter.digest(tmp_path / "summary.json")}}))
    with pytest.raises(ValueError, match="completed PPO development prefix"):
        plotter.load_data(tmp_path)


def test_published_tampering_or_unlisted_file_is_rejected(tmp_path):
    data = tmp_path / "data.json"
    data.write_text("{}")
    (tmp_path / "checksums.json").write_text(json.dumps({"published_sha256": {"data.json": plotter.digest(data)}}))
    plotter.published_hashes(tmp_path)
    data.write_text("[]")
    with pytest.raises(ValueError, match="Changed published evidence"):
        plotter.published_hashes(tmp_path)
    (tmp_path / "extra.json").write_text("{}")
    with pytest.raises(ValueError, match="file set"):
        plotter.published_hashes(tmp_path)
