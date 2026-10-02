"""Validate and export compact evidence from a frozen Shinka search stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import runpy
import shutil
import sqlite3
import statistics
import tempfile

from shinka_crl.experiment import (
    REPO_ROOT, UPSTREAM_COMMIT, build_command, nominal_training_steps, score_curve,
)
from shinka_crl.pilot import read_json, require, sha256
from shinka_crl.search import effective_key, validate_saved_state


def contained(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and not Path(relative).is_absolute(),
            "Artifact paths must be relative")
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f"Artifact escapes root: {relative}")
    return path


def finite(value, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid {label}")
    return float(value)


def canonical_hash(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def validate_candidate(program: Path, output: Path, *, profile: dict,
                       evaluator_sha256: str, expected_affinity: list[int] | None = None,
                       expected_thread_environment: dict | None = None) -> dict:
    """Derive scores from raw training curves, checking source and all seed receipts."""
    metrics, correctness = read_json(output / "metrics.json"), read_json(output / "correct.json")
    require(type(correctness.get("correct")) is bool, "Missing candidate correctness")
    correct = correctness["correct"]
    private = metrics["private"]
    provenance = private["provenance"]
    program_hash = sha256(program)
    require(provenance.get("program_sha256") == program_hash, "Candidate source hash mismatch")
    require(provenance.get("evaluator_sha256") == evaluator_sha256,
            "Candidate evaluator source hash mismatch")
    if expected_affinity is not None:
        require(provenance.get("cpu_affinity") == expected_affinity,
                "Actual evaluator CPU affinity differs from frozen plan")
    if expected_thread_environment is not None:
        require(provenance.get("thread_environment") == expected_thread_environment,
                "Actual evaluator thread environment differs from frozen plan")
    parse = runpy.run_path(str(REPO_ROOT / "tasks/cartpole_ga/evaluate.py"))["parse_ga_config"]
    try:
        settings = parse(program)
    except (ValueError, SyntaxError):
        require(not correct and not private["seed_results"] and not private["seed_attempts"],
                "Invalid source has successful training evidence")
        settings = None
    if settings is not None:
        require(provenance.get("profile") == profile
                and provenance.get("profile_sha256") == canonical_hash(profile),
                "Candidate profile provenance mismatch")
        require(all(metrics["public"].get(key) == value for key, value in settings.items()),
                "Candidate public settings mismatch")
    require(metrics["public"].get("profile") == profile["name"]
            and metrics["public"].get("method") == "ga", "Candidate protocol mismatch")
    seed_results, attempts = private["seed_results"], private["seed_attempts"]
    require([row["seed"] for row in attempts] == profile["seeds"][:len(attempts)]
            and len(attempts) <= len(profile["seeds"]), "Unexpected attempted-seed sequence")
    require([row["seed"] for row in seed_results] == profile["seeds"][:len(seed_results)],
            "Unexpected completed-seed sequence")
    require(metrics["public"].get("seeds_completed") == len(seed_results),
            "Completed-seed count mismatch")
    steps = nominal_training_steps(profile, "ga")
    rows, failed = [], []
    for attempt in attempts:
        seed = attempt["seed"]
        require(attempt.get("trial") == seed + 1, "Development task trial mismatch")
        seed_dir = output / f"seed_{seed}"
        hashes = attempt.get("artifact_sha256", {})
        for relative, digest in hashes.items():
            require(sha256(contained(seed_dir, relative)) == digest,
                    f"Seed artifact changed: {seed}/{relative}")
        if attempt["status"] != "complete":
            require(not correct, "Successful candidate has an incomplete seed")
            failed.append({"seed": seed, "trial": seed + 1, "status": attempt["status"],
                           "wall_seconds": attempt.get("wall_seconds"),
                           "error": attempt.get("error"),
                           "nominal_steps_allocated": steps,
                           "actual_training_steps": None})
            continue
        required = {"manifest.json", "summary.json", "results.json", "config.json",
                    "training_metrics.json", "checkpoints.npz", "process.log", "train.log"}
        require(required <= hashes.keys(), "Incomplete successful-seed receipt")
        manifest = read_json(seed_dir / "manifest.json")
        expected = {"status": "complete", "profile": profile, "method": "ga", "seed": seed,
                    "trial": seed + 1, "upstream_commit": UPSTREAM_COMMIT,
                    "ga_settings": settings}
        require(all(manifest.get(key) == value for key, value in expected.items()),
                "Seed manifest identity mismatch")
        require(manifest.get("metrics_sha256") == hashes["training_metrics.json"],
                "Training curve hash mismatch")
        command = manifest["command"]
        require(command == build_command(
            profile=profile, method="ga", seed=seed, trial=seed + 1, output_dir=seed_dir.resolve(),
            upstream=Path(command[1]).parent.parent, python=command[0], ga_settings=settings),
            "Seed command differs from fixed protocol")
        curve = read_json(seed_dir / "training_metrics.json")
        computed = score_curve(curve, profile=profile, method="ga")
        expected_summary = {**computed, "profile": profile["name"], "method": "ga", "seed": seed,
                            "trial": seed + 1, "upstream_commit": UPSTREAM_COMMIT}
        require(read_json(seed_dir / "summary.json") == expected_summary,
                "Seed summary differs from raw curve")
        result = read_json(seed_dir / "results.json")
        require(result.get("env_steps") == steps, "Seed training budget mismatch")
        config = result["config"]
        population = profile["ne"]["pop_size"]
        elites = effective_key(settings, population)[1]
        resolved = {"refresh": True, "num_elites": elites,
                    "num_offspring": population - elites, "variation": "gaussian",
                    "sigma": settings["sigma"], "cross_over_rate": 0.0}
        require(config.get("sigma") == settings["sigma"]
                and config.get("searcher_kwargs") == {
                    "elite_ratio": settings["elite_ratio"], "init_around_mean": False}
                and config.get("searcher_resolved") == resolved,
                "Resolved candidate settings mismatch")
        expected_seed = {"seed": seed, "trial": seed + 1,
                         "normalized_score": computed["normalized_score"],
                         "mean_return": computed["mean_return"]}
        require(expected_seed in seed_results, "Candidate seed score differs from raw curve")
        wall = finite(manifest.get("wall_seconds"), "seed duration")
        require(wall >= 0, "Negative seed duration")
        rows.append({**expected_seed, "training_wall_seconds": wall,
                     "training_steps_nominal": steps, "noise_vectors": result["noise_vectors"]})
    require(len(rows) == len(seed_results), "Completed attempt and score counts differ")
    combined = statistics.mean(row["normalized_score"] for row in rows) if correct else 0.0
    if correct:
        require(len(rows) == len(profile["seeds"]), "Candidate lacks a planned seed")
        require(metrics["public"].get("mean_return") == statistics.mean(
            row["mean_return"] for row in rows), "Candidate public return mismatch")
    require(finite(metrics["combined_score"], "candidate score") == combined,
            "Candidate score differs from raw seed curves")
    return {"program_sha256": program_hash, "correct": correct,
            "error": correctness.get("error"), "settings": settings,
            "effective_key": effective_key(settings, profile["ne"]["pop_size"])
            if settings else None, "combined_score": combined,
            "completed_seeds": rows, "incomplete_seeds": failed,
            "training_steps_nominal_completed": steps * len(rows),
            "training_steps_nominal_allocated": steps * len(attempts),
            "training_wall_seconds": sum(row["training_wall_seconds"] for row in rows),
            "evaluation_wall_seconds": finite(private["evaluation_wall_seconds"],
                                               "evaluation duration")}


def read_archive(path: Path) -> dict:
    if not path.exists():
        return {"programs": [], "archive": [], "attempts": [], "events": []}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        connection.row_factory = sqlite3.Row
        def table(name, order):
            return [dict(row) for row in connection.execute(f"SELECT * FROM {name} ORDER BY {order}")]
        programs = table("programs", "generation,id")
        for program in programs:
            for field in ("public_metrics", "private_metrics", "metadata",
                          "archive_inspiration_ids", "top_k_inspiration_ids"):
                program[field] = json.loads(program[field]) if program[field] else None
        return {"programs": programs, "archive": table("archive", "program_id"),
                "attempts": table("attempt_log", "id"),
                "events": table("generation_event_log", "id")}


def compare_arms(rows: list[dict]) -> dict:
    """Match random evaluations to distinct completed Shinka mutants, sharing the default."""
    valid = [row for row in rows if row["correct"]]
    initial = [row for row in valid if row["arm"] == "shinka" and row["index"] == 0]
    require(len(initial) <= 1, "Multiple initial candidates")
    seen = {tuple(row["effective_key"]) for row in initial}
    distinct = []
    for row in sorted([row for row in valid if row["arm"] == "shinka"],
                      key=lambda row: row["index"]):
        key = tuple(row["effective_key"])
        if key not in seen:
            distinct.append(row)
            seen.add(key)
    random_rows = sorted([row for row in valid if row["arm"] == "random"],
                         key=lambda row: row["index"])
    matched = len(random_rows) >= len(distinct) and bool(initial)
    prefix = random_rows[:len(distinct)]
    def best(candidates):
        return max((row["combined_score"] for row in candidates), default=None)
    return {"shared_default_score": best(initial),
            "shinka_completed_distinct_mutants": len(distinct),
            "shinka_duplicate_completed_evaluations": sum(
                row["arm"] == "shinka" for row in valid) - len(distinct) - len(initial),
            "random_completed_candidates": len(random_rows),
            "matched_random_prefix_available": matched,
            "matched_random_prefix_count": len(distinct) if matched else None,
            "shinka_best_including_default": best(initial + distinct),
            "random_best_matched_including_default": best(initial + prefix) if matched else None,
            "random_best_all_completed_including_default": best(initial + random_rows),
            "interpretation": "Development configuration search only. Repeated duplicate and "
                              "partial failed evaluations remain charged as actual work. "
                              "No held-out performance or search-method superiority is established."}


def summarize_usage(archive: dict, ledger: Path) -> dict:
    """Keep observed CLI launches distinct from unobservable backend requests and quota."""
    events = [json.loads(line) for line in ledger.read_text().splitlines()
              if line.strip()] if ledger.exists() else []
    requests = {}
    for event in events:
        request = requests.setdefault(event["request_id"], {})
        require(event["event"] in {"started", "codex_exec", "finished"},
                "Unknown subscription ledger event")
        require(event["event"] not in request, "Duplicate subscription ledger event")
        request[event["event"]] = event
    for request in requests.values():
        require("started" in request, "Subscription event lacks its start")
    responses = [(row.get("metadata") or {}).get("llm_result") for row in archive["programs"]]
    responses = [row for row in responses if row]
    tokens = {}
    for key in ("input_tokens", "output_tokens", "thinking_tokens", "num_tool_calls"):
        values = [row[key] for row in responses if row.get(key) is not None]
        require(all(type(value) is int and value >= 0 for value in values),
                f"Invalid native usage field: {key}")
        tokens[key] = sum(values) if len(values) == len(responses) and responses else None
    return {"ledger_available": ledger.exists(), "adapter_attempts": len(requests),
            "codex_exec_launches": sum("codex_exec" in row for row in requests.values()),
            "successful_adapter_responses": sum(row.get("finished", {}).get("outcome")
                                                == "success" for row in requests.values()),
            "failed_adapter_attempts": sum(row.get("finished", {}).get("returncode", 0) != 0
                                           for row in requests.values()),
            "incomplete_adapter_attempts": sum("finished" not in row for row in requests.values()),
            "recorded_native_llm_responses": len(responses),
            "reported_tokens_for_recorded_responses": tokens,
            "backend_request_count": None, "remaining_subscription_quota": None,
            "interpretation": "CLI launches and stored response usage are observed. Internal "
                              "provider requests, quota remaining, and account billing are not "
                              "observable here. Native price estimates are not subscription charges."}


def export_search(runs_root: Path, output: Path) -> dict:
    runs_root, output = runs_root.resolve(), output.resolve()
    require(not output.exists(), f"Refusing to overwrite report: {output}")
    require(not output.is_relative_to(runs_root), "Report must be outside the raw run")
    plan, state = read_json(runs_root / "plan.json"), read_json(runs_root / "state.json")
    require(state["status"] != "running", "Stop the stage before exporting its evidence")
    require(plan["upstream_commit"] == UPSTREAM_COMMIT, "Upstream source pin mismatch")
    for relative, digest in plan["source_sha256"].items():
        require(sha256(contained(REPO_ROOT, relative)) == digest, f"Source changed: {relative}")
    require(read_json(runs_root / "random_pool.json") == plan["random_pool"],
            "Frozen random pool mismatch")
    validate_saved_state(runs_root, state)
    archive = read_archive(runs_root / "shinka/programs.sqlite")
    native = {row["generation"]: row for row in archive["programs"]}
    require(len(native) == len(archive["programs"]), "Duplicate native generation rows")
    ids = {row["id"]: row for row in archive["programs"]}
    rows, failures = [], []
    profile = plan["profile"]
    evaluator_hash = plan["source_sha256"]["tasks/cartpole_ga/evaluate.py"]
    for directory in sorted((runs_root / "shinka").glob("gen_*"),
                            key=lambda path: int(path.name.split("_")[-1])):
        generation = int(directory.name.split("_")[-1])
        if (directory / "failure.json").exists():
            failures.append(read_json(directory / "failure.json"))
        candidate = directory / "main.py"
        if not (directory / "results/metrics.json").exists():
            require(generation not in native, "Archive program has no evaluation metrics")
            continue
        row = validate_candidate(candidate, directory / "results", profile=profile,
                                 evaluator_sha256=evaluator_hash,
                                 expected_affinity=plan["cpu_affinity"],
                                 expected_thread_environment=plan["thread_environment"])
        if generation == 0:
            require(row["program_sha256"] == plan["source_sha256"]["tasks/cartpole_ga/initial.py"],
                    "Initial candidate differs from frozen baseline")
        require(generation in native or not row["correct"],
                "Successful candidate absent from native archive")
        program = native.get(generation)
        if program:
            require(program["code"] == candidate.read_text(), "Archive and evaluated source differ")
            require(bool(program["correct"]) == row["correct"]
                    and program["combined_score"] == row["combined_score"],
                    "Archive and evaluator score differ")
            parent = program["parent_id"]
            require((generation == 0 and parent is None) or
                    (parent in ids and ids[parent]["generation"] < generation),
                    "Invalid candidate parentage")
            metadata = program["metadata"] or {}
            row.update(program_id=program["id"], parent_id=parent,
                       native_metadata={key: metadata[key] for key in (
                           "patch_type", "patch_name", "patch_description", "model_name",
                           "sampling_seconds", "evaluation_seconds", "pipeline_seconds")
                           if key in metadata})
        rows.append({"arm": "shinka", "index": generation,
                     "program_path": str(candidate.relative_to(runs_root)), **row})
    for entry in plan["random_pool"]["entries"]:
        candidate = contained(runs_root, entry["program_path"])
        require(sha256(candidate) == entry["program_sha256"], "Random candidate changed")
        directory = runs_root / "random" / f"candidate_{entry['index']:03d}"
        if not (directory / "metrics.json").exists():
            continue
        row = validate_candidate(candidate, directory, profile=profile,
                                 evaluator_sha256=evaluator_hash,
                                 expected_affinity=plan["cpu_affinity"],
                                 expected_thread_environment=plan["thread_environment"])
        require(row["settings"] == entry["settings"] and row["effective_key"] == entry["effective_key"],
                "Evaluated random candidate differs from frozen pool")
        rows.append({"arm": "random", "index": entry["index"],
                     "program_path": entry["program_path"], **row})
    require(set(native) <= {row["index"] for row in rows if row["arm"] == "shinka"},
            "Native archive contains a missing candidate directory")
    random_indices = [row["index"] for row in rows if row["arm"] == "random" and row["correct"]]
    require(random_indices == list(range(1, state["random_completed"] + 1)),
            "Completed random evidence differs from recorded prefix")
    vectors = {}
    for row in rows:
        for seed in row["completed_seeds"]:
            key = seed["seed"]
            require(key not in vectors or vectors[key] == seed["noise_vectors"],
                    "Candidates used different development task offsets")
            vectors[key] = seed["noise_vectors"]
    report = {
        "schema_version": 1, "status": state["status"], "purpose": plan["purpose"],
        "profile": profile, "upstream_commit": plan["upstream_commit"],
        "model": plan["model"], "auth": plan["auth"], "rows": rows,
        "failed_proposals": failures, "comparison": compare_arms(rows),
        "model_usage": summarize_usage(archive, runs_root / "model_requests.jsonl"),
        "completed_seed_trials": sum(len(row["completed_seeds"]) for row in rows),
        "incomplete_seed_trials": sum(len(row["incomplete_seeds"]) for row in rows),
        "training_steps_nominal_completed": sum(
            row["training_steps_nominal_completed"] for row in rows),
        "training_steps_nominal_allocated": sum(
            row["training_steps_nominal_allocated"] for row in rows),
        "sessions": state["sessions"],
        "session_wall_seconds": sum(finite(row["wall_seconds"], "session duration")
                                    for row in state["sessions"] if "wall_seconds" in row),
        "validation": {"source_hashes_match": True, "artifact_receipts_match": True,
                       "scores_rederived_from_raw_curves": True, "parentage_checked": True,
                       "task_vectors_match": True},
        "artifact_policy": "Raw text and a readable SQLite snapshot are exported with local "
                           "root paths redacted. Binary checkpoints and native SQLite remain "
                           "local with hashes. Model-equivalent prices are not subscription charges.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    checksums = {}
    def redact(data: bytes) -> bytes:
        for original, replacement in ((str(runs_root), "<runs>"), (str(REPO_ROOT), "<repo>"),
                                      (str(Path.home()), "<home>")):
            data = data.replace(original.encode(), replacement.encode())
        return data
    try:
        for source in sorted(runs_root.rglob("*")):
            if not source.is_file() or "__pycache__" in source.parts:
                continue
            require(source.resolve().is_relative_to(runs_root), "Artifact symlink escapes run root")
            relative, data = source.relative_to(runs_root), source.read_bytes()
            entry = {"source": str(relative), "original_sha256": hashlib.sha256(data).hexdigest(),
                     "original_bytes": len(data)}
            text = source.suffix in {".json", ".jsonl", ".log", ".txt", ".py", ".md",
                                     ".diff", ".yaml", ".yml"}
            if text:
                require(b"\0" not in data, f"NUL bytes in text artifact: {relative}")
                exported = redact(data)
                destination = staging / "raw" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(exported)
                entry.update(exported=True, redacted=exported != data,
                             exported_sha256=hashlib.sha256(exported).hexdigest(),
                             exported_bytes=len(exported))
            else:
                entry.update(exported=False, reason="Binary/native artifact retained locally")
            checksums[str(Path("raw") / relative)] = entry
        for filename, payload in (("summary.json", report), ("native_archive.json", archive),
                                  ("checksums.json", checksums)):
            data = json.dumps(payload, indent=2, allow_nan=False).encode() + b"\n"
            (staging / filename).write_bytes(redact(data))
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export_search(args.results_dir, args.output)
    print(json.dumps({"status": report["status"], "comparison": report["comparison"],
                      "completed_seed_trials": report["completed_seed_trials"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
