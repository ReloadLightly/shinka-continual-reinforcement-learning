import copy
import json
from pathlib import Path

import pytest

from shinka_crl.experiment import REPO_ROOT, UPSTREAM_COMMIT, build_command, score_curve
from shinka_crl.pilot import TRAINING_FILES, sha256, write_json
from shinka_crl.validation import aggregate, validate_trial


def candidate(identifier, score_rank, program_hash, memberships=None):
    settings = {"sigma": 0.08, "elite_ratio": 0.075}
    return {"id": identifier, "settings": settings,
            "program_sha256": program_hash,
            "memberships": memberships or [
                {"arm": arm, "development_rank": score_rank, "settings": settings,
                 "program_sha256": program_hash, "program_path": f"programs/{program_hash}.py"}
                for arm in ("shinka", "random")]}


def row(identifier, seed, score, forgetting):
    return {"candidate_id": identifier, "seed": seed, "normalized_score": score,
            "analysis": {"metrics": {
                "learning_accuracy": 200.0, "forgetting": forgetting,
                "learning_minus_forgetting": 200.0 - forgetting,
                "zero_shot_transfer": 100.0, "normalized_curve_average": score}}}


def test_selection_uses_active_return_not_forgetting():
    candidates = [candidate("a", 1, "b"), candidate("b", 2, "a")]
    manifest = {"candidates": candidates, "profile": {"seeds": [1, 2]}}
    rows = [row(c, seed, score, forgetting) for seed in [1, 2]
            for c, score, forgetting in [("a", 0.7, 200.0), ("b", 0.6, -100.0)]]
    result = aggregate(rows, manifest)
    assert result["selected_winners"]["shinka"]["candidate_id"] == "a"
    assert result["selected_winners"]["random"]["candidate_id"] == "a"


def test_exact_tie_uses_development_rank_then_source_hash():
    candidates = [candidate("a", 2, "a"), candidate("b", 1, "z"), candidate("c", 1, "b")]
    manifest = {"candidates": candidates, "profile": {"seeds": [1, 2]}}
    rows = [row(c["id"], seed, 0.8, 0.0) for seed in [1, 2] for c in candidates]
    result = aggregate(rows, manifest)
    assert result["selected_winners"]["shinka"]["candidate_id"] == "c"


def test_partial_validation_does_not_select_a_winner():
    manifest = {"candidates": [candidate("a", 1, "x")], "profile": {"seeds": [1, 2]}}
    result = aggregate([row("a", 1, 0.9, 0.0)], manifest)
    assert result["selected_winners"] == {}
    assert not result["selection_complete"]


def test_winner_preserves_arm_source_when_configs_only_match_at_small_population():
    item = candidate("shared", 1, "shinka-source")
    random_member = item["memberships"][1]
    random_member.update(program_sha256="random-source", program_path="programs/random-source.py",
                         settings={"sigma": 0.08, "elite_ratio": 0.074})
    manifest = {"candidates": [item], "profile": {"seeds": [1, 2]}}
    result = aggregate([row("shared", seed, 0.8, 0.0) for seed in [1, 2]], manifest)
    shinka, random = (result["selected_winners"][arm] for arm in ("shinka", "random"))
    assert shinka["settings"]["elite_ratio"] == 0.075
    assert random["settings"]["elite_ratio"] == 0.074
    assert random["program_sha256"] == "random-source"
    assert random["validation_representative_program_sha256"] == "shinka-source"


def training_fixture(path: Path):
    profile = json.loads((REPO_ROOT / "src/shinka_crl/profiles/search.json").read_text())
    profile.update(eval_episodes=10, seeds=[1001])
    profile["ne"].update(num_generations=320, task_interval=80)
    item = candidate("a", 1, "x")
    source = (REPO_ROOT / "reports/search-stage13-20261002/raw/shinka/gen_12/results"
              "/seed_1001/results.json")
    result = json.loads(source.read_text())
    result["config"].update(num_generations=320, task_interval=80, eval_episodes=10)
    result["env_steps"] = 30_720_000
    path.mkdir()
    command = build_command(profile=profile, method="ga", seed=1001, trial=1002,
                            output_dir=path, ga_settings=item["settings"])
    records = [{"generation": i, "task": i // 80 % 2,
                f"centroid_task{i // 80 % 2}": 100.0} for i in range(320)]
    write_json(path / "training_metrics.json", records)
    summary = {**score_curve(records, profile=profile, method="ga"),
               "profile": profile["name"], "method": "ga", "seed": 1001,
               "trial": 1002, "upstream_commit": UPSTREAM_COMMIT}
    write_json(path / "summary.json", summary)
    write_json(path / "results.json", result)
    write_json(path / "config.json", result["config"])
    write_json(path / "manifest.json", {
        "status": "complete", "profile": profile, "method": "ga", "seed": 1001, "trial": 1002,
        "upstream_commit": UPSTREAM_COMMIT, "ga_settings": item["settings"], "command": command,
        "metrics_sha256": sha256(path / "training_metrics.json"), "wall_seconds": 1.0})
    for name in ("checkpoints.npz", "process.log", "train.log"):
        (path / name).write_bytes(b"synthetic fixture; not a trained checkpoint")
    write_json(path / "receipt.json", {name: sha256(path / name) for name in TRAINING_FILES})
    return profile, item


def test_tuned_training_contract_preserves_other_ga_settings(tmp_path):
    path = tmp_path / "trial"
    profile, item = training_fixture(path)
    assert validate_trial(path, profile=profile, candidate=item, seed=1001)["normalized_score"] == 0.2
    result = json.loads((path / "results.json").read_text())
    result["config"]["searcher_resolved"]["cross_over_rate"] = 0.1
    write_json(path / "results.json", result)
    with pytest.raises(ValueError, match="Tuned GA"):
        validate_trial(path, profile=profile, candidate=item, seed=1001, receipt=False)


def test_candidate_validation_rejects_architecture_drift(tmp_path):
    path = tmp_path / "trial"
    profile, item = training_fixture(path)
    result = json.loads((path / "results.json").read_text())
    result["config"]["hidden_dims"] = [32, 32]
    write_json(path / "results.json", result)
    with pytest.raises(ValueError, match="Baseline config"):
        validate_trial(path, profile=profile, candidate=item, seed=1001, receipt=False)


def test_candidate_validation_rejects_changed_artifact(tmp_path):
    path = tmp_path / "trial"
    profile, item = training_fixture(path)
    (path / "process.log").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_trial(path, profile=profile, candidate=item, seed=1001)


def test_candidate_validation_rejects_other_candidate_settings(tmp_path):
    path = tmp_path / "trial"
    profile, item = training_fixture(path)
    other = copy.deepcopy(item)
    other["settings"]["sigma"] = 0.5
    with pytest.raises(ValueError, match="identity mismatch"):
        validate_trial(path, profile=profile, candidate=other, seed=1001)


@pytest.fixture
def mocked_runner(tmp_path, monkeypatch):
    """Exercise orchestration failures without starting training or touching held-out data."""
    import shinka_crl.validation as module
    frozen, output = tmp_path / "frozen", tmp_path / "run"
    frozen.mkdir()
    write_json(frozen / "manifest.json", {"fixture": True})
    profile = json.loads((REPO_ROOT / "src/shinka_crl/profiles/cartpole-validation.json").read_text())
    item = candidate("candidate_001", 1, "x")
    manifest = {"profile": profile, "candidates": [item]}
    monkeypatch.setattr(module, "finalists", lambda _: manifest)
    monkeypatch.setattr(module, "SOURCES", ())
    monkeypatch.setattr(module, "verify_upstream", lambda _: UPSTREAM_COMMIT)
    affinity = {"value": {0, 1, 2, 3}}
    monkeypatch.setattr(module.os, "sched_getaffinity", lambda _: affinity["value"].copy())
    monkeypatch.setattr(module.os, "sched_setaffinity", lambda _, cpus:
                        affinity.update(value=set(cpus)))
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **k: json.dumps({
        "python": "fixture", "jax_backend": "cpu", "jax_devices": ["fixture"],
        "cpu_affinity": sorted(affinity["value"]), "packages": {}}))
    counts = {"training": 0, "analysis": 0, "fail": None}

    def train(**kwargs):
        counts["training"] += 1
        path = kwargs["output_dir"]
        path.mkdir()
        write_json(path / "manifest.json", {"status": "complete", "wall_seconds": 1.0})
        if counts["fail"] == "training":
            counts["fail"] = None
            raise RuntimeError("simulated crash after completed trainer")

    def checked(path, *, receipt=True, **kwargs):
        hashes = {"manifest.json": sha256(path / "manifest.json")}
        if receipt:
            assert json.loads((path / "receipt.json").read_text()) == hashes
        return {"normalized_score": 0.8, "artifact_sha256": hashes,
                "noise_vectors": [[0, 0, 0, 0], [0.1, 0.1, 0.1, 0.1]], "wall_seconds": 1.0}

    def analyze(**kwargs):
        counts["analysis"] += 1
        path = kwargs["output_dir"]
        path.mkdir()
        failed = counts["fail"] == "analysis"
        write_json(path / "manifest.json", {"status": "failed" if failed else "complete",
                                           "wall_seconds": 1.0})
        if failed:
            counts["fail"] = None
            raise RuntimeError("simulated analysis failure")

    analysis = row("candidate_001", 2001, 0.8, 0.0)["analysis"]
    analysis["phase_returns"] = [
        {"task": i % 2, "own_mean": 100.0, "previous_mean": None if i == 0 else 100.0}
        for i in range(4)]
    monkeypatch.setattr(module, "run_experiment", train)
    monkeypatch.setattr(module, "validate_trial", checked)
    monkeypatch.setattr(module, "run_analysis", analyze)
    monkeypatch.setattr(module, "validate_analysis", lambda **kwargs: analysis)
    return module, frozen, output, counts, affinity


@pytest.mark.parametrize("failure", ["training", "analysis"])
def test_resume_reuses_training_after_crash_or_failed_analysis(mocked_runner, failure):
    module, frozen, output, counts, affinity = mocked_runner
    before = {k: module.os.environ.get(k) for k in module.SEARCH_THREAD_ENV}
    counts["fail"] = failure
    with pytest.raises(RuntimeError, match="simulated"):
        module.run_validation(frozen=frozen, output=output, max_trials=1)
    assert affinity["value"] == {0, 1, 2, 3}
    assert {k: module.os.environ.get(k) for k in before} == before
    result = module.run_validation(frozen=frozen, output=output, resume=True, max_trials=1)
    assert result["status"] == "partial" and result["completed_trials"] == 1
    assert result["selected_winners"] == {}
    assert counts["training"] == 1
    assert counts["analysis"] == (2 if failure == "analysis" else 1)
    assert len(result["sessions"]) == 2
    assert result["sessions"][0]["status"] == "failed"
    assert affinity["value"] == {0, 1, 2, 3}


def test_failed_runtime_probe_leaves_new_output_uncreated(mocked_runner, monkeypatch):
    module, frozen, output, _, affinity = mocked_runner
    def fail(*args, **kwargs):
        raise RuntimeError("runtime unavailable")
    monkeypatch.setattr(module.subprocess, "check_output", fail)
    with pytest.raises(RuntimeError, match="runtime unavailable"):
        module.run_validation(frozen=frozen, output=output, max_trials=1)
    assert not output.exists()
    assert affinity["value"] == {0, 1, 2, 3}
