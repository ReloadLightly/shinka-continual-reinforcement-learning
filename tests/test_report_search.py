"""Independent search score reconstruction and matched-control evidence checks."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3

import pytest

from shinka_crl.experiment import UPSTREAM_COMMIT, build_command, load_profile, score_curve
from shinka_crl.search import artifact_receipts, database_snapshot


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("report_search", ROOT / "scripts/report_search.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def make_candidate(tmp_path):
    program = tmp_path / "main.py"
    program.parent.mkdir(parents=True, exist_ok=True)
    program.write_bytes((ROOT / "tasks/cartpole_ga/initial.py").read_bytes())
    output = tmp_path / "results"
    seed_dir = output / "seed_1001"
    seed_dir.mkdir(parents=True)
    profile = load_profile("smoke")
    settings = {"sigma": 0.5, "elite_ratio": 0.5}
    curve = [{"generation": i, "task": i // 2,
              "centroid_task0": 10.0 + i, "centroid_task1": 10.0 + i} for i in range(4)]
    computed = score_curve(curve, profile=profile, method="ga")
    identity = {"profile": "smoke", "method": "ga", "seed": 1001, "trial": 1002,
                "upstream_commit": UPSTREAM_COMMIT}
    write_json(seed_dir / "training_metrics.json", curve)
    write_json(seed_dir / "summary.json", {**computed, **identity})
    config = {"sigma": 0.5,
              "searcher_kwargs": {"elite_ratio": 0.5, "init_around_mean": False},
              "searcher_resolved": {"refresh": True, "num_elites": 4, "num_offspring": 4,
                                    "variation": "gaussian", "sigma": 0.5,
                                    "cross_over_rate": 0.0}}
    write_json(seed_dir / "config.json", config)
    write_json(seed_dir / "results.json", {"env_steps": 1024, "config": config,
                                          "noise_vectors": [[0.0] * 4, [0.1] * 4]})
    write_json(seed_dir / "manifest.json", {
        **identity, "profile": profile, "status": "complete", "ga_settings": settings,
        "wall_seconds": 1.0,
        "command": build_command(profile=profile, method="ga", seed=1001, trial=1002,
                                 output_dir=seed_dir, upstream=tmp_path / "upstream",
                                 python="/test/python", ga_settings=settings),
        "metrics_sha256": report.sha256(seed_dir / "training_metrics.json"),
    })
    for name in ("checkpoints.npz", "train.log", "process.log"):
        (seed_dir / name).write_bytes(b"test evidence")
    receipt = {path.name: report.sha256(path) for path in seed_dir.iterdir()}
    evaluator_hash = report.sha256(ROOT / "tasks/cartpole_ga/evaluate.py")
    seed_result = {"seed": 1001, "trial": 1002,
                   "normalized_score": computed["normalized_score"],
                   "mean_return": computed["mean_return"]}
    metrics = {"combined_score": computed["normalized_score"],
               "public": {"profile": "smoke", "method": "ga", **settings,
                          "mean_return": computed["mean_return"], "seeds_completed": 1},
               "private": {"seed_results": [seed_result], "evaluation_wall_seconds": 1.1,
                           "seed_attempts": [{"seed": 1001, "trial": 1002, "status": "complete",
                                              "wall_seconds": 1.0, "artifact_sha256": receipt}],
                           "provenance": {"profile": profile,
                                          "cpu_affinity": [0, 1],
                                          "thread_environment": {"OMP_NUM_THREADS": "1"},
                                          "profile_sha256": report.canonical_hash(profile),
                                          "program_sha256": report.sha256(program),
                                          "evaluator_sha256": evaluator_hash}}}
    write_json(output / "metrics.json", metrics)
    write_json(output / "correct.json", {"correct": True, "error": None})
    return program, output, profile, evaluator_hash


def test_rederives_complete_candidate(tmp_path):
    program, output, profile, evaluator_hash = make_candidate(tmp_path)
    actual = report.validate_candidate(program, output, profile=profile,
                                       evaluator_sha256=evaluator_hash)
    assert actual["combined_score"] == 11.5 / 32
    assert actual["training_steps_nominal_completed"] == 1024
    assert actual["effective_key"] == [0.5, 4]


def test_rejects_unplanned_actual_thread_settings(tmp_path):
    program, output, profile, evaluator_hash = make_candidate(tmp_path)
    with pytest.raises(ValueError, match="thread environment differs"):
        report.validate_candidate(program, output, profile=profile,
                                  evaluator_sha256=evaluator_hash,
                                  expected_thread_environment={"OMP_NUM_THREADS": "8"})
    with pytest.raises(ValueError, match="CPU affinity differs"):
        report.validate_candidate(program, output, profile=profile,
                                  evaluator_sha256=evaluator_hash, expected_affinity=[0])


@pytest.mark.parametrize("mutation,match", [
    ("curve", "Seed artifact changed"),
    ("score", "Candidate score differs"),
    ("seed_score", "Candidate seed score differs"),
    ("source", "Candidate source hash mismatch"),
    ("profile", "profile provenance mismatch"),
])
def test_rejects_tampered_evidence(tmp_path, mutation, match):
    program, output, profile, evaluator_hash = make_candidate(tmp_path)
    metrics = json.loads((output / "metrics.json").read_text())
    if mutation == "curve":
        (output / "seed_1001/training_metrics.json").write_text("[]")
    elif mutation == "score":
        metrics["combined_score"] = 0.9
    elif mutation == "seed_score":
        metrics["private"]["seed_results"][0]["normalized_score"] = 0.9
    elif mutation == "source":
        program.write_text(program.read_text().replace("0.5", "0.4"))
    elif mutation == "profile":
        metrics["private"]["provenance"]["profile"]["seeds"] = [42]
    write_json(output / "metrics.json", metrics)
    with pytest.raises(ValueError, match=match):
        report.validate_candidate(program, output, profile=profile,
                                  evaluator_sha256=evaluator_hash)


def test_partial_failure_keeps_work_but_has_no_candidate_score(tmp_path):
    program, output, profile, evaluator_hash = make_candidate(tmp_path)
    profile["seeds"] += [1002, 1003]
    manifest_path = output / "seed_1001/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["profile"] = profile
    write_json(manifest_path, manifest)
    metrics = json.loads((output / "metrics.json").read_text())
    metrics["combined_score"] = 0.0
    metrics["private"]["provenance"].update(profile=profile,
                                          profile_sha256=report.canonical_hash(profile))
    metrics["private"]["seed_attempts"][0]["artifact_sha256"]["manifest.json"] = report.sha256(
        manifest_path)
    metrics["private"]["seed_attempts"].append({"seed": 1002, "trial": 1003,
                                             "status": "failed", "wall_seconds": 2.0,
                                             "error": "timeout", "artifact_sha256": {}})
    write_json(output / "metrics.json", metrics)
    write_json(output / "correct.json", {"correct": False, "error": "timeout"})
    actual = report.validate_candidate(program, output, profile=profile,
                                       evaluator_sha256=evaluator_hash)
    assert actual["combined_score"] == 0.0
    assert actual["training_steps_nominal_completed"] == 1024
    assert actual["training_steps_nominal_allocated"] == 2048
    assert actual["incomplete_seeds"][0]["actual_training_steps"] is None


def test_random_prefix_matches_distinct_configs_and_shares_default():
    def row(arm, index, sigma, elites, score):
        return {"arm": arm, "index": index, "correct": True,
                "effective_key": [sigma, elites], "combined_score": score}
    rows = [row("shinka", 0, 0.5, 32, 0.4), row("shinka", 1, 0.5, 32, 0.4),
            row("shinka", 2, 0.2, 16, 0.6), row("shinka", 3, 0.2, 16, 0.6),
            row("random", 1, 0.1, 10, 0.5), row("random", 2, 0.8, 50, 0.9)]
    actual = report.compare_arms(rows)
    assert actual["shinka_completed_distinct_mutants"] == 1
    assert actual["shinka_duplicate_completed_evaluations"] == 2
    assert actual["matched_random_prefix_count"] == 1
    assert actual["random_best_matched_including_default"] == 0.5
    assert actual["random_best_all_completed_including_default"] == 0.9


def test_usage_counts_launches_without_inventing_backend_requests(tmp_path):
    ledger = tmp_path / "requests.jsonl"
    events = [{"request_id": "first", "event": "started"},
              {"request_id": "first", "event": "codex_exec"},
              {"request_id": "first", "event": "finished", "returncode": 0,
               "outcome": "success"},
              {"request_id": "second", "event": "started"},
              {"request_id": "second", "event": "finished", "returncode": 1,
               "outcome": "preflight_failure"}]
    ledger.write_text("\n".join(json.dumps(event) for event in events))
    archive = {"programs": [{"metadata": {"llm_result": {
        "input_tokens": 500, "output_tokens": 120, "thinking_tokens": 90, "num_tool_calls": 0,
    }}}]}
    actual = report.summarize_usage(archive, ledger)
    assert actual["adapter_attempts"] == 2
    assert actual["codex_exec_launches"] == 1
    assert actual["failed_adapter_attempts"] == 1
    assert actual["backend_request_count"] is None
    assert actual["remaining_subscription_quota"] is None
    assert actual["reported_tokens_for_recorded_responses"]["output_tokens"] == 120


def test_export_checks_native_ancestry_and_keeps_binary_hashes(tmp_path):
    runs = tmp_path / "runs"
    program, results, profile, evaluator_hash = make_candidate(runs / "shinka/gen_0")
    metrics = json.loads((results / "metrics.json").read_text())
    database = runs / "shinka/programs.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE programs (id, generation, parent_id, code, combined_score, correct, "
                   "public_metrics, private_metrics, metadata, archive_inspiration_ids, "
                   "top_k_inspiration_ids)")
        db.execute("INSERT INTO programs VALUES (?,?,?,?,?,?,?,?,?,?,?)", (
            "initial", 0, None, program.read_text(), metrics["combined_score"], 1,
            json.dumps(metrics["public"]), json.dumps(metrics["private"]), "{}", "[]", "[]"))
        db.execute("CREATE TABLE archive (program_id)")
        db.execute("CREATE TABLE attempt_log (id)")
        db.execute("CREATE TABLE generation_event_log (id)")
    pool = {"entries": []}
    plan = {"purpose": "test fixture only", "profile": profile, "model": "fixture",
            "auth": "none", "upstream_commit": UPSTREAM_COMMIT, "random_pool": pool,
            "cpu_affinity": [0, 1], "thread_environment": {"OMP_NUM_THREADS": "1"},
            "source_sha256": {"tasks/cartpole_ga/evaluate.py": evaluator_hash,
                              "tasks/cartpole_ga/initial.py": report.sha256(program)}}
    state = {"status": "complete", "programs": database_snapshot(database), "sessions": [],
             "random_completed": 0, "artifact_sha256": artifact_receipts(runs)}
    write_json(runs / "plan.json", plan)
    write_json(runs / "state.json", state)
    write_json(runs / "random_pool.json", pool)
    output = tmp_path / "export"
    actual = report.export_search(runs, output)
    assert actual["completed_seed_trials"] == 1
    checksums = json.loads((output / "checksums.json").read_text())
    assert checksums["raw/shinka/gen_0/results/seed_1001/checkpoints.npz"]["exported"] is False
    for relative, record in checksums.items():
        if record["exported"]:
            assert hashlib.sha256((output / relative).read_bytes()).hexdigest() == record[
                "exported_sha256"]
    archive = json.loads((output / "native_archive.json").read_text())
    assert archive["programs"][0]["id"] == "initial"
    assert str(ROOT) not in (output / "summary.json").read_text()
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        report.export_search(runs, output)
    with sqlite3.connect(database) as db:
        db.execute("UPDATE programs SET parent_id='missing'")
    state["programs"] = database_snapshot(database)
    write_json(runs / "state.json", state)
    with pytest.raises(ValueError, match="Invalid candidate parentage"):
        report.export_search(runs, tmp_path / "bad-export")
