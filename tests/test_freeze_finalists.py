"""Finalist selection must close the full search before any held-out evaluation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sqlite3

import pytest

from shinka_crl.experiment import UPSTREAM_COMMIT, build_command, load_profile, score_curve
from shinka_crl.search import (
    RANDOM_SEED, SEARCH_THREAD_ENV, artifact_receipts, database_snapshot, effective_key,
    program_source, random_pool,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("freeze_finalists", ROOT / "scripts/freeze_finalists.py")
freeze = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(freeze)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def row(arm, index, sigma, ratio, score, tag=""):
    settings = {"sigma": sigma, "elite_ratio": ratio}
    source = program_source(settings) + (f"# {tag}\n" if tag else "")
    return {"arm": arm, "index": index, "settings": settings,
            "effective_key": effective_key(settings), "combined_score": score,
            "program_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "program_path": f"{arm}/{index}.py", "correct": True}


def test_shared_default_is_eligible_in_both_top_two_pools():
    rows = [row("shinka", 0, .5, .5, .9), row("shinka", 1, .1, .1, .8),
            row("shinka", 2, .2, .1, .7), row("random", 1, .3, .1, .8),
            row("random", 2, .4, .1, .7)]
    chosen = freeze.select_finalists(rows)
    assert len(chosen) == 3
    assert [(m["arm"], m["development_rank"]) for m in chosen[0]["memberships"]] == [
        ("default", None), ("shinka", 1), ("random", 1)]
    assert [m["development_rank"] for c in chosen[1:] for m in c["memberships"]] == [2, 2]


def test_float32_and_elite_count_dedupe_preserve_all_memberships():
    rows = [row("shinka", 0, .5, .5, .4), row("shinka", 1, .08, .075, .9),
            row("shinka", 2, .08, .076, .9), row("shinka", 3, .1, .1, .8),
            row("random", 1, .0800000001, .074, .9), row("random", 2, .2, .2, .8)]
    chosen = freeze.select_finalists(rows)
    assert len(chosen) == 4
    assert [(m["arm"], m["index"]) for m in chosen[1]["memberships"]] == [
        ("shinka", 1), ("random", 1)]
    assert len({tuple(c["effective_key"]) for c in chosen}) == len(chosen)


def test_score_ties_use_lower_index_then_source_hash():
    rows = [row("shinka", 0, .5, .5, .1), row("shinka", 2, .2, .1, .9),
            row("shinka", 1, .1, .1, .9), row("random", 2, .3, .1, .9),
            row("random", 1, .4, .1, .9)]
    chosen = freeze.select_finalists(list(reversed(rows)))
    assert [m["index"] for c in chosen for m in c["memberships"] if m["arm"] == "shinka"] == [1, 2]
    assert [m["index"] for c in chosen for m in c["memberships"] if m["arm"] == "random"] == [1, 2]


@pytest.fixture(scope="module")
def endpoint(tmp_path_factory):
    """Synthetic complete endpoint; all training curves are constructed test evidence."""
    root = tmp_path_factory.mktemp("finalist-endpoint")
    runs, profile, pool = root / "runs", load_profile("search"), random_pool()
    evaluator = freeze.sha256(ROOT / "tasks/cartpole_ga/evaluate.py")
    native_rows = []
    jobs = []
    for generation in range(25):
        effective_generation = min(generation, 22)  # Two legitimate repeated slots.
        settings = ({"sigma": .5, "elite_ratio": .5} if generation == 0 else
                    {"sigma": .01 * effective_generation, "elite_ratio": .075})
        program = runs / f"shinka/gen_{generation}/main.py"
        program.parent.mkdir(parents=True, exist_ok=True)
        if generation == 0:
            program.write_bytes((ROOT / "tasks/cartpole_ga/initial.py").read_bytes())
        else:
            program.write_text(program_source(settings))
        jobs.append(("shinka", generation, settings, program, program.parent / "results",
                     100.0 + effective_generation * 10))
    for entry in pool["entries"]:
        program = runs / entry["program_path"]
        program.parent.mkdir(parents=True, exist_ok=True)
        program.write_text(program_source(entry["settings"]))
        jobs.append(("random", entry["index"], entry["settings"], program,
                     runs / f"random/candidate_{entry['index']:03d}", 400.0 - entry["index"]))
    for arm, index, settings, program, output, reward in jobs:
        attempts, seed_results = [], []
        for seed in profile["seeds"]:
            directory = output / f"seed_{seed}"
            directory.mkdir(parents=True)
            curve = [{"generation": i, "task": i // 20 % 2,
                      "centroid_task0": reward, "centroid_task1": reward} for i in range(80)]
            score = score_curve(curve, profile=profile, method="ga")
            identity = {"profile": "search", "method": "ga", "seed": seed,
                        "trial": seed + 1, "upstream_commit": UPSTREAM_COMMIT}
            write(directory / "training_metrics.json", curve)
            write(directory / "summary.json", {**score, **identity})
            elites = effective_key(settings)[1]
            config = {"sigma": settings["sigma"],
                      "searcher_kwargs": {"elite_ratio": settings["elite_ratio"],
                                          "init_around_mean": False},
                      "searcher_resolved": {"refresh": True, "num_elites": elites,
                                            "num_offspring": 64 - elites, "variation": "gaussian",
                                            "sigma": settings["sigma"], "cross_over_rate": 0.0}}
            write(directory / "config.json", config)
            write(directory / "results.json", {"env_steps": 7_680_000, "config": config,
                                                "noise_vectors": [[0.0] * 4, [seed / 10000] * 4]})
            write(directory / "manifest.json", {
                **identity, "profile": profile, "status": "complete", "ga_settings": settings,
                "wall_seconds": 1.0,
                "command": build_command(profile=profile, method="ga", seed=seed, trial=seed + 1,
                                         output_dir=directory, upstream=root / "upstream",
                                         python="/test/python", ga_settings=settings),
                "metrics_sha256": freeze.sha256(directory / "training_metrics.json")})
            for name in ("checkpoints.npz", "train.log", "process.log"):
                (directory / name).write_bytes(b"synthetic test evidence")
            attempts.append({"seed": seed, "trial": seed + 1, "status": "complete",
                             "wall_seconds": 1.0, "artifact_sha256": {
                                 p.name: freeze.sha256(p) for p in directory.iterdir()}})
            seed_results.append({"seed": seed, "trial": seed + 1,
                                 "normalized_score": score["normalized_score"],
                                 "mean_return": score["mean_return"]})
        metrics = {"combined_score": score["normalized_score"],
                   "public": {"profile": "search", "method": "ga", **settings,
                              "mean_return": reward, "seeds_completed": 3},
                   "private": {"seed_results": seed_results, "seed_attempts": attempts,
                               "evaluation_wall_seconds": 3.1,
                               "provenance": {"profile": profile,
                                              "profile_sha256": freeze.REPORT.canonical_hash(profile),
                                              "program_sha256": freeze.sha256(program),
                                              "evaluator_sha256": evaluator,
                                              "cpu_affinity": [0, 1],
                                              "thread_environment": SEARCH_THREAD_ENV}}}
        write(output / "metrics.json", metrics)
        write(output / "correct.json", {"correct": True, "error": None})
        if arm == "shinka":
            native_rows.append((f"program-{index}", index, f"program-{index-1}" if index else None,
                                program.read_text(), metrics["combined_score"], 1,
                                json.dumps(metrics["public"]), json.dumps(metrics["private"]),
                                "{}", "[]", "[]"))
    database = runs / "shinka/programs.sqlite"
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE programs (id, generation, parent_id, code, combined_score, correct, "
                   "public_metrics, private_metrics, metadata, archive_inspiration_ids, "
                   "top_k_inspiration_ids)")
        db.executemany("INSERT INTO programs VALUES (?,?,?,?,?,?,?,?,?,?,?)", native_rows)
        db.execute("CREATE TABLE archive (program_id)")
        db.execute("CREATE TABLE attempt_log (id)")
        db.execute("CREATE TABLE generation_event_log (id)")
    plan = {"purpose": "synthetic endpoint fixture", "profile": profile, "model": "fixture",
            "auth": "none", "upstream_commit": UPSTREAM_COMMIT, "random_pool": pool,
            "random_pool_seed": RANDOM_SEED, "proposal_slots": 24,
            "cpu_affinity": [0, 1], "thread_environment": SEARCH_THREAD_ENV,
            "source_sha256": {"tasks/cartpole_ga/evaluate.py": evaluator,
                              "tasks/cartpole_ga/initial.py": freeze.sha256(
                                  ROOT / "tasks/cartpole_ga/initial.py")}}
    events = [{"request_id": str(i), "event": event, "returncode": 0, "outcome": "success"}
              for i in range(24) for event in ("started", "codex_exec", "finished")]
    (runs / "model_requests.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    write(runs / "shinka/rng_state.json", {"fixture": True})
    sessions = [{"index": i + 1, "arm": arm, "endpoint": n, "status": "complete",
                 "returncode": 0, "stop_reason": None, "wall_seconds": 1.0}
                for i, (arm, n) in enumerate([("shinka", n) for n in (2, 5, 13, 25)] +
                                             [("random", n) for n in range(1, 25)])]
    state = {"status": "complete", "programs": database_snapshot(database), "sessions": sessions,
             "random_completed": 24, "artifact_sha256": artifact_receipts(runs),
             "model_requests_sha256": freeze.sha256(runs / "model_requests.jsonl"),
             "rng_sha256": freeze.sha256(runs / "shinka/rng_state.json")}
    write(runs / "plan.json", plan)
    write(runs / "state.json", state)
    write(runs / "random_pool.json", pool)
    report = root / "report"
    freeze.REPORT.export_search(runs, report)
    return report


def test_complete_endpoint_accepts_charged_duplicates_and_freezes(endpoint, tmp_path):
    plan, rows, evidence = freeze.validate_endpoint(endpoint)
    assert len(rows) == 49 and len(evidence.inputs) > 100
    assert plan["profile"]["seeds"] == [1001, 1002, 1003]
    destination = tmp_path / "finalists"
    manifest = freeze.freeze_finalists(endpoint, destination)
    assert len(manifest["candidates"]) == 5
    assert manifest["profile"] == json.loads((ROOT / "src/shinka_crl/profiles/cartpole-validation.json").read_text())
    assert manifest["evaluation"]["trials"] == [{"seed": s, "trial": s + 1} for s in range(2001, 2006)]
    assert manifest["search_status"] == "closed_for_selection"
    assert freeze.validate_finalists(destination) == manifest
    for candidate in manifest["candidates"]:
        for member in candidate["memberships"]:
            assert (destination / member["program_path"]).read_bytes() == (
                endpoint / "raw" / member["report_program_path"]).read_bytes()
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        freeze.freeze_finalists(endpoint, destination)


def test_stage13_is_refused_without_creating_output(tmp_path):
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="complete 25-program"):
        freeze.freeze_finalists(ROOT / "reports/search-stage13-20261002", destination)
    assert not destination.exists()


@pytest.mark.parametrize("mutation,match", [
    ("source", "Changed exported evidence"), ("curve", "Changed exported evidence"),
    ("score", "Candidate score differs"), ("missing", "Missing, duplicated"),
    ("replacement", "Candidate assigned to a different slot"),
    ("failed", "Failed proposal evidence"),
])
def test_endpoint_rejects_changed_or_missing_selection_inputs(endpoint, tmp_path, mutation, match):
    copied = tmp_path / "report"
    shutil.copytree(endpoint, copied)
    summary = json.loads((copied / "summary.json").read_text())
    if mutation == "source":
        (copied / "raw/shinka/gen_1/main.py").write_text("def get_ga_config(): return {}")
    elif mutation == "curve":
        (copied / "raw/shinka/gen_1/results/seed_1001/training_metrics.json").write_text("[]")
    elif mutation == "score":
        summary["rows"][1]["combined_score"] = .999
    elif mutation == "missing":
        summary["rows"].pop()
    elif mutation == "replacement":
        summary["rows"][1]["program_path"] = "shinka/gen_2/main.py"
    else:
        summary["failed_proposals"] = [{"generation": 24, "error": "fixture failure"}]
    write(copied / "summary.json", summary)
    with pytest.raises(ValueError, match=match):
        freeze.freeze_finalists(copied, tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


@pytest.mark.parametrize("mutation", ["program", "profile", "manifest", "extra"])
def test_frozen_handoff_detects_mutation(endpoint, tmp_path, mutation):
    destination = tmp_path / "finalists"
    manifest = freeze.freeze_finalists(endpoint, destination)
    if mutation == "program":
        (destination / manifest["candidates"][0]["program_path"]).write_text("changed")
    elif mutation == "profile":
        (destination / "profile.json").write_text("{}")
    elif mutation == "manifest":
        (destination / "manifest.json").write_text("{}")
    else:
        (destination / "extra.py").write_text("unexpected")
    with pytest.raises(ValueError, match="Frozen artifact"):
        freeze.validate_finalists(destination)


def test_frozen_selector_version_is_pinned(endpoint, tmp_path, monkeypatch):
    destination = tmp_path / "finalists"
    freeze.freeze_finalists(endpoint, destination)
    original = freeze.sha256
    monkeypatch.setattr(freeze, "sha256", lambda path: "changed-selector" if
                        Path(path) == Path(freeze.__file__) else original(path))
    with pytest.raises(ValueError, match="selector implementation changed"):
        freeze.validate_finalists(destination)


def test_changed_native_slot_is_rejected(endpoint, tmp_path):
    copied = tmp_path / "report"
    shutil.copytree(endpoint, copied)
    archive = json.loads((copied / "native_archive.json").read_text())
    archive["programs"][24]["code"] = archive["programs"][0]["code"]
    write(copied / "native_archive.json", archive)
    with pytest.raises(ValueError, match="Saved and exported native archive differ"):
        freeze.validate_endpoint(copied)
