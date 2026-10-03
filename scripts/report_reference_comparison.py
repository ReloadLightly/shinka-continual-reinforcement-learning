"""Validate and export compact evidence for the declared CartPole reference comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import tempfile

from shinka_crl.analysis import validate_analysis
from shinka_crl.experiment import REPO_ROOT, nominal_training_steps
from shinka_crl.pilot import read_json, require
from shinka_crl import reference_comparison as runner

METRICS = (
    "learning_accuracy", "forgetting", "learning_minus_forgetting", "zero_shot_transfer",
    "cumulative_reward_steps", "cumulative_reward_generation_equivalents",
    "cumulative_reward_steps_exact", "normalized_curve_average",
)
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def finite(value, label):
    require(type(value) in (float, int) and math.isfinite(value), f"Invalid {label}")
    return value


def contained(root, relative):
    require(isinstance(relative, str) and not Path(relative).is_absolute(),
            "Artifact paths must be relative")
    path = root / relative
    require(path.resolve().is_relative_to(root), "Artifact path escapes run root")
    require(not any(p.is_symlink() for p in (path, *path.parents) if p != root.parent),
            "Artifact symlinks are forbidden")
    return path


def identity(row):
    return tuple(row[key] for key in ("profile", "method", "seed", "trial", "eval_seed"))


def describe(values):
    return {"n": len(values), "mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "values": values}


def aggregate(rows):
    groups = []
    for method in runner.METHODS:
        trials = sorted((row for row in rows if row["method"] == method), key=lambda r: r["seed"])
        if not trials:
            continue
        groups.append({"method": method, "n": len(trials), "seeds": [r["seed"] for r in trials],
                       "metrics": {key: describe([r["metrics"][key] for r in trials]) for key in METRICS},
                       **{key: describe([r[key] for r in trials]) for key in (
                           "training_wall_seconds", "analysis_wall_seconds", "maximum_trainer_rss_kib")}})
    return groups


def evaluation_episodes(path):
    """Count every saved native evaluation source, once per analysis attempt."""
    candidates = [path / "evaluation.json", path / "evaluation-input/evaluation.json"]
    source = next((p for p in candidates if p.is_file()), None)
    if source is None:
        return 0
    total = 0
    for row in read_json(source).get("per_task", []):
        for field in ("returns", "prev_returns", "zero_shot_next_returns"):
            values = row.get(field, [])
            require(isinstance(values, list), "Invalid saved evaluation episode vector")
            for value in values:
                finite(value, "saved episode return")
            total += len(values)
    return total


def attempt_records(root, plan, validated):
    """Account for completed work and retained unsuccessful attempts without loading pickles."""
    attempts = []
    for job in plan["jobs"]:
        base = root / "trials" / job["method"] / f"seed_{job['seed']}"
        profile = plan["profiles"][job["profile"]]
        budget = profile["ppo"] if job["method"] == "ppo" else profile["ne"]
        count = budget["num_updates" if job["method"] == "ppo" else "num_generations"]
        step_size = nominal_training_steps(profile, job["method"]) // count
        for stage in ("training", "analysis"):
            for path in sorted((base / stage).glob("attempt_*")):
                require(path.is_dir() and not path.is_symlink(), "Invalid attempt directory")
                manifest = read_json(path / "manifest.json")
                wall = finite(manifest.get("wall_seconds", 0), "attempt wall time")
                require(wall >= 0, "Negative attempt wall time")
                row = {**job, "stage": stage, "path": str(path.relative_to(root)),
                       "status": manifest["status"], "wall_seconds": wall,
                       "wall_measurement_available": "wall_seconds" in manifest}
                if stage == "analysis":
                    row["saved_fresh_evaluation_episodes"] = evaluation_episodes(path)
                else:
                    if manifest["status"] == "complete" and path not in validated:
                        validated[path] = runner.validate_training(path, plan=plan, profile=profile, job=job)
                    measurement_path = path / "process-measurement.json"
                    measurement = read_json(measurement_path) if measurement_path.exists() else {}
                    start = measurement.get("start_update", (manifest.get("resume_from") or {}).get("start_update", 0))
                    observed = [start] + [event["completed_updates"] for event in measurement.get("phase_events", [])]
                    process_log = path / "process.log"
                    if process_log.exists():
                        observed += [int(m[1]) + 1 for m in re.finditer(
                            r"^\s*(?:gen|update)\s+(\d+)\s+task=", process_log.read_text(), re.M)]
                    end = count if manifest["status"] == "complete" else max(observed)
                    require(type(start) is int and type(end) is int and 0 <= start <= end <= count,
                            "Invalid attempted training progress")
                    peak = measurement.get("maximum_trainer_rss_kib")
                    if peak is not None:
                        require(finite(peak, "trainer RSS") >= 0, "Negative trainer RSS")
                    row.update(start_update=start, observed_completed_updates=end,
                               nominal_training_steps_observed_lower_bound=(end - start) * step_size,
                               maximum_trainer_rss_kib=peak)
                attempts.append(row)
    return attempts


def _validate(root, allow_partial):
    plan, suite = read_json(root / "plan.json"), read_json(root / "suite.json")
    expected = runner.make_plan(mode=plan["mode"], cpus=len(plan["cpu_affinity"]),
                                timeout=plan["timeout_seconds"], analysis_timeout=plan["analysis_timeout_seconds"],
                                methods=plan["methods"], python=plan["python"], upstream=Path(plan["upstream"]))
    require(plan == expected, "Frozen reference plan, source, or protocol changed")
    runner.verify_upstream(Path(plan["upstream"]))
    environment = read_json(root / "environment.json")
    for key in ("upstream_commit", "source_sha256", "cpu_affinity", "thread_environment"):
        require(environment.get(key) == plan[key], f"Runtime {key} differs from frozen plan")
    require(environment.get("jax_backend") == "cpu" and environment.get("packages")
            and environment.get("python") and environment.get("jax_devices"), "Incomplete CPU runtime provenance")
    require(suite.get("mode") == plan["mode"], "Suite mode changed")
    require(suite.get("status") in {"complete", "partial", "failed", "interrupted"},
            "Cannot export a running or unknown suite")
    planned = {identity(job): job for job in plan["jobs"]}
    raw_rows = suite["rows"]
    keys = [identity(row) for row in raw_rows]
    require(len(planned) == len(plan["jobs"]) and len(keys) == len(set(keys)), "Duplicate trial identity")
    require(set(keys) <= planned.keys(), "Completed trial absent from declared plan")
    require(suite["completed_trials"] == len(keys) and suite["planned_trials"] == len(planned),
            "Inconsistent trial counts")
    complete = set(keys) == planned.keys()
    require(suite["status"] != "complete" or complete, "Incomplete suite marked complete")
    require(allow_partial or (complete and suite["status"] == "complete"),
            "Incomplete comparison requires explicit --allow-partial")
    require(plan["mode"] != "reporting" or len(planned) == 30, "Reporting requires exactly 30 declared trials")
    wall = finite(suite["wall_seconds"], "suite wall time")
    require(wall >= 0 and wall == sum(finite(s["wall_seconds"], "session wall time") for s in suite["sessions"]),
            "Inconsistent suite timing")
    rows, validated, vectors = [], {}, {}
    for raw in raw_rows:
        job = planned[identity(raw)]
        profile = plan["profiles"][job["profile"]]
        train, analysis = (contained(root, raw[key]) for key in ("training_path", "analysis_path"))
        base = root / "trials" / job["method"] / f"seed_{job['seed']}"
        require(train.parent == base / "training" and analysis.parent == base / "analysis"
                and train.name.startswith("attempt_") and analysis.name.startswith("attempt_"),
                "Trial artifact path differs from declared job")
        training = runner.validate_training(train, plan=plan, profile=profile, job=job)
        validated[train] = training
        measured = validate_analysis(run_dir=train, output_dir=analysis,
                                     episodes=plan["eval_episodes"], eval_seed=job["eval_seed"])
        require(measured == raw["analysis"], "Suite analysis differs from verified raw evidence")
        require(identity(measured) == identity(job), "Analysis trial identity changed")
        require(measured["nominal_training_steps"] == training["nominal_training_steps"], "Analysis budget changed")
        analysis_wall = finite(read_json(analysis / "manifest.json")["wall_seconds"], "analysis wall time")
        for key, actual in (("training_wall_seconds", training["wall_seconds"]),
                            ("analysis_wall_seconds", analysis_wall),
                            ("maximum_trainer_rss_kib", training["maximum_trainer_rss_kib"])):
            require(raw[key] == actual and finite(actual, key) >= 0, f"Suite {key} differs from evidence")
        require(raw["noise_vectors"] == training["noise_vectors"], "Suite task vectors changed")
        require(job["seed"] not in vectors or vectors[job["seed"]] == training["noise_vectors"],
                "Methods have different task vectors")
        vectors[job["seed"]] = training["noise_vectors"]
        rows.append({**job, "metrics": {key: finite(measured["metrics"][key], key) for key in METRICS},
                     "phase_returns": measured["phase_returns"],
                     "phase_training_returns": measured["phase_training_returns"],
                     "training_env_steps_nominal": training["nominal_training_steps"],
                     "training_wall_seconds": training["wall_seconds"], "analysis_wall_seconds": analysis_wall,
                     "maximum_trainer_rss_kib": training["maximum_trainer_rss_kib"],
                     "fresh_evaluation_episodes": evaluation_episodes(analysis),
                     "training_path": raw["training_path"], "analysis_path": raw["analysis_path"]})
    attempts = attempt_records(root, plan, validated)
    return {"schema_version": 1, "mode": plan["mode"], "status": suite["status"],
            "caption": "Only reporting mode is the declared final comparison. Development and diagnostic "
                       "results remain separate. Dispersion is sample SD across trials, not a confidence interval. "
                       "Training steps are nominal episode-cap budgets; evaluation is separate. "
                       "Raw reward units and /500 normalization differ from paper reference-based rescaling.",
            "upstream_commit": plan["upstream_commit"], "profiles": plan["profiles"],
            "completed_trials": len(rows), "planned_trials": len(planned), "rows": rows,
            "groups": aggregate(rows), "attempts": attempts,
            "compute": {"suite_wall_seconds": wall,
                        "planned_training_env_steps_nominal": sum(plan["nominal_training_steps_per_method"][j["method"]] for j in plan["jobs"]),
                        "completed_trial_training_env_steps_nominal": sum(r["training_env_steps_nominal"] for r in rows),
                        "attempted_training_env_steps_nominal_observed_lower_bound": sum(r.get("nominal_training_steps_observed_lower_bound", 0) for r in attempts),
                        "training_wall_seconds_all_attempts": sum(r["wall_seconds"] for r in attempts if r["stage"] == "training"),
                        "analysis_wall_seconds_all_attempts": sum(r["wall_seconds"] for r in attempts if r["stage"] == "analysis"),
                        "maximum_trainer_rss_kib": max((r["maximum_trainer_rss_kib"] for r in attempts if r.get("maximum_trainer_rss_kib") is not None), default=None),
                        "saved_fresh_evaluation_episodes_all_attempts": sum(r.get("saved_fresh_evaluation_episodes", 0) for r in attempts),
                        "scored_fresh_evaluation_episodes": sum(r["fresh_evaluation_episodes"] for r in rows),
                        "unsuccessful_attempts": sum(r["status"] != "complete" for r in attempts),
                        "note": "Failed/resumed attempt work is counted separately. Unsaved progress and unsaved "
                                "evaluation episodes are unknown, not zero. Peak RSS covers native trainers only."},
            "validation": {"frozen_protocol_and_sources_match": True, "training_receipts_verified": True,
                           "posthoc_metrics_recomputed": True, "matched_task_vectors": True,
                           "all_planned_trials": complete}}


def export_reference_comparison(runs_root: Path, output: Path, *, allow_partial=False):
    root, output = Path(runs_root).resolve(), Path(output).resolve()
    require(not output.exists(), f"Refusing to overwrite report: {output}")
    require(not output.is_relative_to(root), "Report must be outside raw runs")
    with runner.suite_lock(root):
        report = _validate(root, allow_partial)
        files = [p for p in sorted(root.rglob("*")) if p.name != "controller.lock"]
        require(not any(p.is_symlink() for p in files), "Artifact symlinks are forbidden")
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        originals, published = {}, {}
        try:
            for source in files:
                if not source.is_file():
                    continue
                relative, raw = source.relative_to(root), source.read_bytes()
                originals[str(relative)] = {"sha256": digest(raw), "bytes": len(raw),
                                            "exported": source.suffix in TEXT_SUFFIXES}
                if source.suffix not in TEXT_SUFFIXES:
                    continue
                require(b"\0" not in raw, f"NUL byte in text artifact: {relative}")
                exported = raw
                for path, replacement in sorted(((str(root), "<runs>"), (str(REPO_ROOT), "<repo>")), key=lambda item: -len(item[0])):
                    exported = exported.replace(path.encode(), replacement.encode())
                destination = staging / "raw" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(exported)
                published[str(Path("raw") / relative)] = digest(exported)
            summary = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode()
            (staging / "summary.json").write_bytes(summary)
            published["summary.json"] = digest(summary)
            (staging / "checksums.json").write_text(json.dumps({"original_artifacts": originals,
                "published_sha256": published, "exporter_sha256": digest(Path(__file__).read_bytes()),
                "artifact_policy": "All original artifacts are hashed; text is copied with local root paths "
                                   "redacted. Binary checkpoints remain outside Git. Every attempt is retained."},
                indent=2, allow_nan=False) + "\n")
            require(not output.exists(), f"Refusing to overwrite report: {output}")
            staging.rename(output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    try:
        report = export_reference_comparison(args.runs_root, args.output, allow_partial=args.allow_partial)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Reference report rejected: {exc}\n")
    print(json.dumps({"output": str(args.output), "mode": report["mode"],
                      "status": report["status"], "completed_trials": report["completed_trials"]}, indent=2))


if __name__ == "__main__":
    main()
