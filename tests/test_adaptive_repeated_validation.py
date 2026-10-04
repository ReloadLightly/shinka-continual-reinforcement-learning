"""Fresh finalist selection and outer-repetition inference without training."""
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import runpy
import statistics
import sys

import pytest

from shinka_crl import adaptive_search as search
from shinka_crl import adaptive_validation as av
from shinka_crl.adaptive import load_program
from shinka_crl.experiment import load_profile
from shinka_crl.pilot import read_json, sha256, write_json
from shinka_crl.reference_timing import artifact_hashes


@pytest.fixture
def repeated(tmp_path, monkeypatch):
    study = tmp_path / "study"
    study.mkdir()
    allocation = av.REPO_ROOT / "docs/adaptive-repeated-seed-allocation-20261004.json"
    (study / "seed-allocation.json").write_bytes(allocation.read_bytes())
    development = {
        "protocol_version": av.ae.PARTITION_VERSION,
        "profile": {**load_profile("adaptive-search"), "seeds": [6001, 6002, 6003]},
        "reserved_validation_profile": {
            **load_profile("adaptive-validation"), "seeds": [7001, 7002, 7003, 7004, 7005]},
        "seed_allocation": {"source_path": str(allocation), "frozen_path": "seed-allocation.json",
                            "sha256": sha256(study / "seed-allocation.json")},
        "source_sha256": av.ae.source_hashes(), "controls": av.ae.controls(),
        "upstream": "/fake-upstream", "python": "/fake-python", "cpu_affinity": [0, 1],
        "runtime": {"backend": "fake"},
    }
    write_json(study / "plan.json", development)
    monkeypatch.setattr(av.ae, "read_plan", lambda path: deepcopy(development))
    monkeypatch.setattr(av, "verify_upstream", lambda path: None)
    monkeypatch.setattr(av, "native_sources", lambda path: {"native.py": "pinned"})
    monkeypatch.setattr(av.ae, "runtime_scope", lambda affinity: nullcontext())
    monkeypatch.setattr(av.ae, "runtime_fingerprint", lambda python: development["runtime"])

    def git_output(command, **kwargs):
        if command[:3] == ["git", "rev-parse", "HEAD"]:
            return "a" * 40 + "\n"
        if command[:2] == ["git", "show"]:
            return (av.REPO_ROOT / command[2].split(":", 1)[1]).read_bytes()
        pytest.fail(f"Unexpected metadata command: {command}")

    monkeypatch.setattr(av.subprocess, "check_output", git_output)
    archives, summaries = [], {}
    for seed in (20261004, 20261005):
        for arm in ("evolutionary", "independent"):
            archive = tmp_path / f"{arm}_{seed}"
            archive.mkdir()
            plan = {"evaluation_context_sha256": av.ae.digest(development),
                    "outer_random_seed": seed, "arm": arm}
            write_json(archive / "plan.json", plan)
            write_json(archive / "state.json", {"status": "complete"})
            programs = []
            for generation in range(5):
                source = archive / "shinka" / f"gen_{generation}" / "main.py"
                source.parent.mkdir(parents=True)
                source.write_text((av.REPO_ROOT / "tasks/cartpole_adaptive/initial.py").read_text()
                                  if generation == 0 else
                                  f"def update_sigma(sigma, stats, memory):\n"
                                  f"    return sigma * {generation / 2}, memory\n")
                score = .500000000000001 if seed == 20261004 and arm == "evolutionary" and generation == 1 else .5
                programs.append({"generation": generation, "request": {
                    "aggregate": {"scores": {"combined_score": {"mean": score}}},
                    "candidate": {"source_sha256": sha256(source),
                                  "program": load_program(source).metadata()}}})
            summaries[archive] = {"slots_consumed": 5, "programs_evaluated": 5,
                                  "programs": programs, "failed_requests": [],
                                  "terminal_proposal_failures": []}
            archives.append(archive)
    monkeypatch.setattr(search, "summarize", lambda archive: deepcopy(summaries[archive]))
    protocol = tmp_path / "protocol.md"
    protocol.write_text("# Frozen repeated search test protocol\n")
    return {"study": study, "archives": archives, "summaries": summaries,
            "development": development, "protocol": protocol, "output": tmp_path / "frozen"}


def freeze(repeated):
    return av.freeze_repeated(study=repeated["study"], archives=repeated["archives"],
                              output=repeated["output"], protocol_path=repeated["protocol"])


def resign(repeated, plan):
    frozen = repeated["output"]
    write_json(frozen / "plan.json", plan)
    write_json(frozen / "receipt.json", artifact_hashes(frozen))


def test_freeze_exact_selection_identity_ties_and_shared_recipe_cost(repeated):
    plan = freeze(repeated)
    assert av.read_plan(repeated["output"]) == plan
    selections = plan["provenance"]["selected_searches"]
    assert [row["generation"] for row in selections] == [1, 0, 0, 0]
    assert selections[0]["development_score"] == .500000000000001
    assert len(plan["conditions"]) == 7
    assert len(plan["unique_recipes"]) == 4
    assert plan["planned_trials"] == 20
    assert plan["planned_nominal_training_steps"] == 20 * 30_720_000
    identity = next(row for row in plan["unique_recipes"] if "identity" in row["memberships"])
    assert identity["memberships"] == ["independent_20261004", "evolutionary_20261005",
                                        "independent_20261005", "identity"]
    assert plan["profile"]["seeds"] == [7001, 7002, 7003, 7004, 7005]
    assert sha256(repeated["output"] / "seed-allocation.json") == plan["provenance"]["seed_allocation"]["sha256"]
    assert "not independent search repetitions" in plan["reporting"]["inference"]


@pytest.mark.parametrize("change", ["missing_arm", "duplicate_arm", "one_pair", "partial",
                                    "failed", "missing_identity", "source", "legacy"])
def test_common_freeze_rejects_incomplete_or_unbound_selections(repeated, change):
    first = repeated["archives"][0]
    if change == "missing_arm":
        repeated["archives"].pop()
    elif change == "duplicate_arm":
        repeated["archives"][-1] = first
    elif change == "one_pair":
        repeated["archives"] = repeated["archives"][:2]
    elif change == "partial":
        repeated["summaries"][first]["slots_consumed"] = 4
    elif change == "failed":
        repeated["summaries"][first]["failed_requests"] = [{"status": "failed"}]
    elif change == "missing_identity":
        repeated["summaries"][first]["programs"][0]["generation"] = 5
    elif change == "source":
        with (first / "shinka/gen_1/main.py").open("a") as output:
            output.write("# Changed after summarized development request\n")
    else:
        repeated["development"]["protocol_version"] = av.ae.VERSION
    with pytest.raises(ValueError):
        freeze(repeated)
    assert not repeated["output"].exists()


@pytest.mark.parametrize("change", ["reported_score", "ranking", "missing_identity", "static_finalist",
                                    "focus_variant", "identity_settings", "development_budget", "allocation",
                                    "objective_version", "trial_offset", "eval_seed_offset"])
def test_replay_rejects_resigned_selection_control_or_context_drift(repeated, change):
    plan = freeze(repeated)
    selected = plan["provenance"]["selected_searches"][0]
    if change == "reported_score":
        selected["development_score"] = .5
    elif change == "ranking":
        selected["generation"] = 0
    elif change == "missing_identity":
        selected["ranked_candidates"] = [row for row in selected["ranked_candidates"] if row["generation"] != 0]
    elif change == "static_finalist":
        plan["conditions"][0]["variant"] = "ga_static"
    elif change == "focus_variant":
        plan["conditions"][-1]["variant"] = "ga"
    elif change == "identity_settings":
        plan["conditions"][-3]["settings"] = {"sigma": .1, "elite_ratio": .5}
    elif change == "development_budget":
        development = plan["provenance"]["development_plan"]
        development["profile"]["ne"]["pop_size"] = 128
        plan["provenance"]["development_plan_sha256"] = av.ae.digest(development)
    elif change in {"objective_version", "trial_offset", "eval_seed_offset"}:
        plan[change] = "different" if change == "objective_version" else 2
    else:
        path = repeated["output"] / "seed-allocation.json"
        path.write_text(path.read_text() + "\n")
    resign(repeated, plan)
    with pytest.raises(ValueError):
        av.read_plan(repeated["output"])


def test_replay_independently_rejects_historical_validation_partition(repeated):
    plan = freeze(repeated)
    development = plan["provenance"]["development_plan"]
    reused = [5001, 5002, 5003, 5004, 5005]
    plan["profile"]["seeds"] = reused
    development["reserved_validation_profile"]["seeds"] = reused
    path = repeated["output"] / "seed-allocation.json"
    audit = read_json(path)
    audit["proposed"]["adaptive_validation"] = {
        "seeds": reused, "trials": [seed + 1 for seed in reused],
        "eval_seeds": [seed + 900000 for seed in reused]}
    write_json(path, audit)
    development["seed_allocation"]["sha256"] = sha256(path)
    plan["provenance"]["seed_allocation"]["sha256"] = sha256(path)
    plan["provenance"]["development_plan_sha256"] = av.ae.digest(development)
    resign(repeated, plan)
    with pytest.raises(ValueError, match="overlap"):
        av.read_plan(repeated["output"])


def fake_results(plan, *, partial=False):
    # Separate unit-test recipes let each condition have a distinct synthetic score.
    rows = {}
    for condition in plan["conditions"]:
        for index, seed in enumerate(plan["profile"]["seeds"]):
            name = condition["id"]
            if partial and name == "independent_20261005" and index == 4:
                continue
            value = .3
            if name == "evolutionary_20261004":
                value += [.1, .2, .3, .4, .5][index]
            elif name == "evolutionary_20261005":
                value += [-.2, -.1, 0., .1, .2][index]
            rows[len(rows)] = {
                "seed": seed, "memberships": [name], "fresh_evaluation_episodes": 300,
                "score": {"combined_score": value, "active_score": value, "previous_score": value,
                          "reference_metrics": {key: value for key in (
                              "learning_accuracy", "forgetting", "learning_minus_forgetting",
                              "zero_shot_transfer", "normalized_curve_average")}}}
    return rows


@pytest.mark.parametrize("partial", [False, True])
def test_primary_statistics_use_outer_repetitions_not_ten_evaluation_pairs(repeated, tmp_path,
                                                                        monkeypatch, partial):
    plan = freeze(repeated)
    completed = fake_results(plan, partial=partial)
    monkeypatch.setattr(av, "_state", lambda output, plan: (
        {"sessions": [], "attempts": []}, completed, []))
    summary = av.summarize(frozen=repeated["output"], output=tmp_path / "results")
    repetitions = summary["repetition_comparisons"]
    assert repetitions[0]["aggregate"]["n"] == 5
    assert repetitions[0]["aggregate"]["mean"] == pytest.approx(.3)
    assert repetitions[1]["aggregate"]["n"] == (4 if partial else 5)
    if partial:
        assert summary["across_search_repetitions"] is None
    else:
        result = summary["across_search_repetitions"]
        assert result["n"] == 2
        assert result["mean"] == pytest.approx(.15)
        assert result["sample_sd"] == pytest.approx(statistics.stdev([.3, 0.]))


def test_cli_forwards_repeated_freeze_without_training(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr(av, "freeze_repeated", lambda **kwargs: calls.append(kwargs) or {"frozen": True})
    monkeypatch.setattr(av, "run", lambda **kwargs: pytest.fail("No execution was requested"))
    monkeypatch.setattr(sys, "argv", ["run_adaptive_validation.py", "--frozen", str(tmp_path / "freeze"),
                                      "--study-dir", str(tmp_path / "study"), "--protocol", "protocol.md",
                                      "--repeated-searches", "evolutionary", "independent"])
    runpy.run_path(str(av.REPO_ROOT / "scripts/run_adaptive_validation.py"), run_name="__main__")
    assert calls == [{"study": tmp_path / "study", "archives": [Path("evolutionary"), Path("independent")],
                      "output": tmp_path / "freeze", "protocol_path": Path("protocol.md")}]
    assert '"frozen": true' in capsys.readouterr().out
