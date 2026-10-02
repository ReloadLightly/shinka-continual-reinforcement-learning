"""Validate and export compact evidence from a matched CartPole pilot."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile

from shinka_crl.experiment import REPO_ROOT


METRICS = (
    "learning_accuracy", "forgetting", "zero_shot_transfer", "learning_minus_forgetting",
    "cumulative_reward_steps", "cumulative_reward_steps_exact", "normalized_curve_average",
)
CAPTION = (
    "Development pilot, not a full paper reproduction. Every seed is reported. "
    "Uncertainty columns are sample standard deviations across independent training trials, "
    "not confidence intervals or episode-level errors. Training steps are nominal; evaluation "
    "work is excluded. The stationary learning gate is an engineering criterion."
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def finite(value, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"Non-finite {label}")
    return float(value)


def contained(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and not Path(relative).is_absolute(),
            "Artifact paths must be relative")
    path = (root / relative).resolve()
    require(path.is_relative_to(root), f"Path escapes evidence root: {relative}")
    return path


def identity(row: dict) -> tuple:
    return tuple(row[key] for key in ("profile", "method", "seed", "trial", "eval_seed"))


def describe(values: list[float]) -> dict:
    require(bool(values), "Cannot aggregate an empty group")
    return {"mean": statistics.mean(values),
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values), "values": values}


def aggregate_rows(rows: list[dict]) -> list[dict]:
    """Keep protocol groups separate and expose every value entering each mean."""
    groups = defaultdict(list)
    for row in rows:
        groups[(row["profile"], row["method"])].append(row)
    output = []
    for (profile, method), trials in sorted(groups.items()):
        trials.sort(key=lambda row: row["seed"])
        metrics = {}
        for metric in METRICS:
            values = [row["metrics"][metric] for row in trials]
            require(all(value is None for value in values)
                    or all(value is not None for value in values),
                    f"Partially undefined metric {profile}/{method}/{metric}")
            metrics[metric] = None if values[0] is None else describe(
                [finite(value, metric) for value in values])
        output.append({"profile": profile, "condition": trials[0]["condition"],
                       "method": method, "seeds": [row["seed"] for row in trials],
                       "n": len(trials), "metrics": metrics,
                       "training_wall_seconds": describe(
                           [row["training_wall_seconds"] for row in trials]),
                       "analysis_wall_seconds": describe(
                           [row["analysis_wall_seconds"] for row in trials])})
    return output


def adequacy_gate(rows: list[dict], plan: dict) -> dict:
    """The frozen 50-point / 400-point rule; no inference from missing trials."""
    assessments = []
    for name, profile in sorted(plan["profiles"].items()):
        if profile["num_tasks"] != 1:
            continue
        for method in plan["methods"]:
            jobs = [job for job in plan["jobs"]
                    if job["profile"] == name and job["method"] == method]
            trials = sorted([row for row in rows
                             if row["profile"] == name and row["method"] == method],
                            key=lambda row: row["seed"])
            eligible = (len(jobs) == len(trials) == 3 and profile["episode_length"] == 500
                        and profile["num_phases"] == 4)
            per_seed = []
            for row in trials:
                phase_means = row["phase_training_returns"]
                first, last = phase_means[0], phase_means[-1]
                per_seed.append({"seed": row["seed"], "first_phase_mean": first,
                                 "last_phase_mean": last, "gain": last - first,
                                 "gain_at_least_50": last - first >= 50,
                                 "last_at_least_400": last >= 400})
            gain_count = sum(row["gain_at_least_50"] for row in per_seed)
            ceiling_count = sum(row["last_at_least_400"] for row in per_seed)
            passed = gain_count >= 2 or ceiling_count >= 2
            assessments.append({"profile": name, "method": method,
                                "status": ("pass" if passed else "fail")
                                if eligible else "not_assessed",
                                "gain_seed_count": gain_count,
                                "ceiling_seed_count": ceiling_count, "trials": per_seed})
    statuses = [item["status"] for item in assessments]
    status = ("pass" if statuses and all(value == "pass" for value in statuses)
              else "fail" if "fail" in statuses else "not_assessed")
    return {"status": status, "assessments": assessments,
            "rule": "At least two of three seeds gain >=50 between first/last phase means, "
                    "or at least two finish with last-phase mean >=400; all methods must pass.",
            "interpretation": "Budget adequacy only, not significance or paper solved threshold. "
                              "The first phase is trained performance, not an untrained reference."}


def export_pilot(runs_root: Path, output: Path, repo_root: Path = REPO_ROOT,
                 *, allow_partial: bool = False) -> dict:
    # Lazy imports let the aggregation helpers run without the optional training runtime.
    from shinka_crl.analysis import validate_analysis
    from shinka_crl.pilot import validate_training

    runs_root, output, repo_root = runs_root.resolve(), output.resolve(), repo_root.resolve()
    require(not output.exists(), f"Refusing to overwrite report: {output}")
    require(not output.is_relative_to(runs_root), "Report output must be outside raw runs")
    plan = read_json(runs_root / "plan.json")
    suite = read_json(runs_root / "suite.json")
    revision = read_json(repo_root / "upstream.lock.json")["continual_neuroevolution"]["commit"]
    require(plan.get("upstream_commit") == revision, "Mismatched upstream revision")
    sources = plan.get("source_sha256")
    require(isinstance(sources, dict) and bool(sources), "Missing source hashes")
    for source, digest in sources.items():
        path = contained(repo_root, source)
        require(path.is_file() and sha256(path.read_bytes()) == digest,
                f"Stale source hash: {source}")
    environment = read_json(runs_root / "environment.json")
    require(environment.get("upstream_commit") == revision,
            "Environment upstream revision mismatch")
    require(environment.get("jax_backend") == "cpu", "Expected CPU pilot runtime")
    require(bool(environment.get("python")) and bool(environment.get("packages"))
            and bool(environment.get("jax_devices")), "Incomplete runtime provenance")
    require(environment.get("source_sha256") == sources,
            "Environment and frozen plan source hashes differ")
    for name in ("cpu_affinity", "thread_environment"):
        require(bool(plan.get(name)) and environment.get(name) == plan[name],
                f"Environment and frozen plan {name} differ")
    planned = {identity(job): job for job in plan["jobs"]}
    require(len(planned) == len(plan["jobs"]), "Duplicate planned trials")
    raw_rows = suite.get("rows")
    require(isinstance(raw_rows, list) and bool(raw_rows), "No completed trials to export")
    keys = [identity(row) for row in raw_rows]
    require(len(set(keys)) == len(keys), "Duplicate completed trials")
    require(set(keys) <= planned.keys(), "Completed trial absent from frozen plan")
    require(suite.get("completed_trials") == len(raw_rows), "Inconsistent completed-trial count")
    require(suite.get("planned_trials") == len(planned), "Inconsistent planned-trial count")
    complete = set(keys) == planned.keys()
    require(suite.get("status") in ("complete", "partial", "failed"),
            "Cannot export a running or unknown suite")
    require(suite["status"] != "complete" or complete, "Incomplete suite marked complete")
    require(allow_partial or (complete and suite["status"] == "complete"),
            "Incomplete pilot; pass --allow-partial only for explicitly partial evidence")
    wall_seconds = finite(suite.get("wall_seconds"), "suite wall time")
    require(wall_seconds >= 0, "Negative suite wall time")

    rows, vectors, budgets = [], {}, defaultdict(set)
    for raw in raw_rows:
        name, method, seed, trial, eval_seed = identity(raw)
        profile = plan["profiles"][name]
        training_path = contained(runs_root, raw["training_path"])
        analysis_path = contained(runs_root, raw["analysis_path"])
        training = validate_training(training_path, profile=profile, method=method,
                                     seed=seed, trial=trial)
        analysis = validate_analysis(run_dir=training_path, output_dir=analysis_path,
                                     episodes=plan["eval_episodes"], eval_seed=eval_seed)
        require(analysis == raw["analysis"], f"Stale suite analysis: {name}/{method}/{seed}")
        for key, value in (("profile", name), ("method", method), ("seed", seed),
                           ("trial", trial), ("eval_seed", eval_seed)):
            require(analysis.get(key) == value, f"Mismatched analysis {key}")
        condition = "stationary" if profile["num_tasks"] == 1 else "switching"
        require(analysis.get("condition") == condition, "Mismatched condition")
        metrics = {key: analysis["metrics"][key] for key in METRICS}
        for key, value in metrics.items():
            if value is not None:
                finite(value, f"{name}/{method}/{seed}/{key}")
        phase_means = [finite(value, "phase training mean")
                       for value in analysis["phase_training_returns"]]
        require(len(phase_means) == profile["num_phases"], "Mismatched phase means")
        training_wall = finite(raw["training_wall_seconds"], "training wall time")
        analysis_wall = finite(raw["analysis_wall_seconds"], "analysis wall time")
        require(training_wall >= 0 and analysis_wall >= 0, "Negative trial wall time")
        require(math.isclose(training_wall, training["wall_seconds"], rel_tol=1e-9),
                "Mismatched training wall time")
        recorded_analysis_wall = finite(read_json(analysis_path / "manifest.json").get(
            "wall_seconds"), "analysis manifest wall time")
        require(math.isclose(analysis_wall, recorded_analysis_wall, rel_tol=1e-9),
                "Mismatched analysis wall time")
        steps = training["nominal_training_steps"]
        require(analysis["nominal_training_steps"] == steps, "Mismatched analysis budget")
        budgets[name].add(steps)
        key = (name, seed, trial)
        if key in vectors:
            require(vectors[key] == training["noise_vectors"],
                    f"Methods have different task vectors: {name}/{seed}")
        vectors[key] = training["noise_vectors"]
        rows.append({"profile": name, "condition": condition, "method": method,
                     "seed": seed, "trial": trial, "eval_seed": eval_seed,
                     "metrics": metrics, "phase_training_returns": phase_means,
                     "phase_returns": analysis["phase_returns"],
                     "training_env_steps_nominal": steps,
                     "training_wall_seconds": training_wall,
                     "analysis_wall_seconds": analysis_wall,
                     "training_path": raw["training_path"], "analysis_path": raw["analysis_path"]})
    require(all(len(values) == 1 for values in budgets.values()), "Unmatched training budgets")
    rows.sort(key=lambda row: (row["profile"], row["method"], row["seed"]))
    report = {
        "schema_version": 1, "status": suite["status"], "caption": CAPTION,
        "upstream_commit": revision, "completed_trials": len(rows),
        "planned_trials": len(planned), "suite_wall_seconds": wall_seconds,
        "training_env_steps_nominal": sum(row["training_env_steps_nominal"] for row in rows),
        "rows": rows, "groups": aggregate_rows(rows), "adequacy_gate": adequacy_gate(rows, plan),
        "validation": {"source_hashes_match": True, "artifact_receipts_match": True,
                       "posthoc_metrics_recomputed": True, "matched_task_vectors": True,
                       "matched_nominal_training_budgets": True, "all_planned_trials": complete},
        "artifact_policy": "Raw text retained with local root paths redacted. Original and "
                           "exported hashes identify redactions. Binary checkpoints remain "
                           "local and are represented by hashes. Failed attempts are retained. "
                           "Raw upstream JSON may contain NaN placeholders; this summary is "
                           "strict finite JSON.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    checksums = {}
    try:
        for source in sorted(runs_root.rglob("*")):
            if not source.is_file():
                continue
            resolved = source.resolve()
            require(resolved.is_relative_to(runs_root), "Artifact symlink escapes run root")
            if source.suffix not in (".json", ".log", ".txt", ".npz", ".npy", ".pkl"):
                continue
            relative = source.relative_to(runs_root)
            raw = source.read_bytes()
            entry = {"source": str(relative), "original_sha256": sha256(raw),
                     "original_bytes": len(raw)}
            if source.suffix in (".npz", ".npy", ".pkl"):
                entry.update(exported=False, reason="Binary artifact retained locally")
            else:
                require(b"\0" not in raw, f"NUL bytes in text artifact: {relative}")
                exported = raw
                for path, replacement in sorted(((str(runs_root), "<runs>"),
                                                  (str(repo_root), "<repo>")),
                                                 key=lambda pair: -len(pair[0])):
                    exported = exported.replace(path.encode(), replacement.encode())
                destination = staging / "raw" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(exported)
                entry.update(exported=True, redacted=exported != raw,
                             exported_sha256=sha256(exported), exported_bytes=len(exported))
            checksums[str(Path("raw") / relative)] = entry
        for name, value in (("summary.json", report), ("checksums.json", checksums)):
            (staging / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        require(not output.exists(), f"Refusing to overwrite report: {output}")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    try:
        report = export_pilot(args.runs_root, args.output, args.repo_root,
                              allow_partial=args.allow_partial)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Pilot report rejected: {exc}\n")
    print(json.dumps({"output": str(args.output), "completed_trials": report["completed_trials"],
                      "adequacy_gate": report["adequacy_gate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
