"""Validate and export a complete CartPole smoke suite using only the standard library."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ("training_metrics.json", "results.json", "summary.json", "manifest.json",
             "train.log", "process.log")
CAPTION = (
    "Single-seed pipeline smoke validation only; training budgets are not matched. "
    "Returns are active-task centroid evaluation means, not reproduction or comparative "
    "performance evidence. Nominal training steps exclude evaluation and may include "
    "post-termination rollout padding. Wall time includes process startup and compilation."
)


def read_json(path: Path):
    # Preserve upstream's NaN placeholders (e.g. PPO sigma) in the raw copy.
    # Every value used in derived evidence is separately checked for finiteness.
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read artifact {path}: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite(value, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"Non-finite {label}")
    return float(value)


def equal_number(actual, expected: float, label: str) -> None:
    require(math.isclose(finite(actual, label), expected, rel_tol=1e-9, abs_tol=1e-9),
            f"Mismatched {label}: {actual} != {expected}")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def training_budget(profile: dict, method: str) -> int:
    """Nominal training transitions, with evaluation work excluded."""
    if method == "ppo":
        budget = profile["ppo"]
        return budget["num_updates"] * budget["num_envs"] * budget["num_steps"]
    budget = profile["ne"]
    return (budget["num_generations"] * budget["pop_size"] * budget["num_evals"]
            * profile["episode_length"])


def validate_run(path: Path, label: str, profile: dict, revision: str) -> dict:
    """Cross-check runner provenance, upstream configuration and recorded scores."""
    for name in ARTIFACTS:
        require((path / name).is_file(), f"Missing {label}/{name}")
        if name.endswith(".log"):
            require(b"\0" not in (path / name).read_bytes(), f"Corrupt {label}/{name}: NUL bytes")
    manifest = read_json(path / "manifest.json")
    summary = read_json(path / "summary.json")
    results = read_json(path / "results.json")
    records = read_json(path / "training_metrics.json")
    method = "ga" if label == "shinka" else label
    seed, = profile["seeds"]
    trial = seed + 1
    require(manifest.get("status") == "complete", f"Incomplete {label} manifest")
    require(manifest.get("profile") == profile, f"Mismatched {label} profile")
    for name, expected in (("upstream_commit", revision), ("method", method),
                           ("seed", seed), ("trial", trial)):
        require(manifest.get(name) == expected, f"Mismatched {label} manifest {name}")
        require(summary.get(name) == expected, f"Mismatched {label} summary {name}")
    require(summary.get("profile") == "smoke", f"Mismatched {label} summary profile")
    require(manifest.get("metrics_sha256") == sha256((path / ARTIFACTS[0]).read_bytes()),
            f"Stale or corrupt {label} metrics hash")
    command = manifest.get("command")
    require(isinstance(command, list) and command and all(isinstance(s, str) for s in command),
            f"Missing {label} command")
    for flag, expected in (("--method", method), ("--seed", str(seed)),
                           ("--trial", str(trial)), ("--output_dir", str(path.resolve()))):
        require(command.count(flag) == 1 and command.index(flag) + 1 < len(command)
                and command[command.index(flag) + 1] == expected,
                f"Mismatched {label} command {flag}")
    require(bool(manifest.get("python_version")), f"Missing {label} Python version")
    wall_seconds = finite(manifest.get("wall_seconds"), f"{label} wall_seconds")
    require(wall_seconds >= 0, f"Negative {label} wall_seconds")
    expected_settings = {"sigma": 0.5, "elite_ratio": 0.5} if label == "shinka" else None
    require("ga_settings" in manifest and manifest["ga_settings"] == expected_settings,
            f"Mismatched {label} initial GA settings")

    budget = profile["ppo"] if method == "ppo" else profile["ne"]
    count = budget["num_updates" if method == "ppo" else "num_generations"]
    tasks = [(i // budget["task_interval"]) % profile["num_tasks"] for i in range(count)]
    require(tasks == [0, 0, 1, 1], "Expected the four-row, two-task smoke profile")
    require(isinstance(records, list) and len(records) == count,
            f"Expected {count} {label} metric rows")
    active_returns = []
    for index, (record, task) in enumerate(zip(records, tasks)):
        require(record.get("generation") == index and record.get("task") == task,
                f"Mismatched {label} task schedule at row {index}")
        value = finite(record.get(f"centroid_task{task}"), f"{label} centroid row {index}")
        require(0 <= value <= profile["episode_length"], f"Out-of-range {label} centroid")
        active_returns.append(value)
    mean_return = sum(active_returns) / count
    normalized_score = mean_return / profile["episode_length"]
    equal_number(summary.get("mean_return"), mean_return, f"{label} summary mean_return")
    equal_number(summary.get("normalized_score"), normalized_score,
                 f"{label} summary normalized_score")
    require(summary.get("metric_rows") == count, f"Mismatched {label} summary metric_rows")

    config = results.get("config", {})
    env, noise = profile["env"].split("_sigma")
    expected_config = {
        "env": env, "method": method, "seed": seed, "trial": trial,
        "schedule": "switch", "num_tasks": profile["num_tasks"],
        "num_generations": count, "task_interval": budget["task_interval"],
        "episode_length": profile["episode_length"], "eval_episodes": profile["eval_episodes"],
        "task_sequence": list(range(profile["num_phases"])), "noise_range": float(noise),
        "first_task_clean": True,
    }
    keys = ("num_envs", "num_steps", "num_minibatches") if method == "ppo" else (
        "pop_size", "num_evals")
    expected_config.update({key: budget[key] for key in keys})
    for name, expected in expected_config.items():
        require(config.get(name) == expected, f"Mismatched {label} results config {name}")
    nominal_steps = training_budget(profile, method)
    require(results.get("env_steps") == nominal_steps, f"Mismatched {label} env_steps")
    vectors = results.get("noise_vectors")
    require(isinstance(vectors, list) and len(vectors) == profile["num_tasks"]
            and all(isinstance(v, list) and len(v) == 4 for v in vectors),
            f"Missing or invalid {label} CartPole noise_vectors")
    for vector in vectors:
        for value in vector:
            finite(value, f"{label} noise vector")
    require(vectors[0] == [0, 0, 0, 0], f"Expected clean first {label} task")
    return {
        "path": path, "manifest": manifest, "records": records, "noise_vectors": vectors,
        "row": {"method": label, "seed": seed, "trial": trial, "metric_rows": count,
                "task_schedule": tasks, "active_centroid_returns": active_returns,
                "mean_return": mean_return, "normalized_score": normalized_score,
                "training_env_steps_nominal": nominal_steps, "wall_seconds": wall_seconds},
    }


def export_smoke(runs_root: Path, output: Path, repo_root: Path = REPO_ROOT) -> dict:
    """Validate the entire suite before publishing any report files; refuse overwrite."""
    runs_root, output, repo_root = runs_root.resolve(), output.resolve(), repo_root.resolve()
    require(not output.exists(), f"Refusing to overwrite report: {output}")
    require(not output.is_relative_to(runs_root), "Report output must be outside raw runs")
    profile = read_json(repo_root / "src/shinka_crl/profiles/smoke.json")
    revision = read_json(repo_root / "upstream.lock.json")["continual_neuroevolution"]["commit"]
    suite = read_json(runs_root / "suite.json")
    require(suite.get("status") == "complete"
            and suite.get("completed") == ["ga", "es", "ppo", "shinka_initial"],
            "Incomplete smoke suite")
    require(finite(suite.get("wall_seconds"), "suite wall_seconds") >= 0,
            "Negative suite wall_seconds")
    environment = read_json(runs_root / "environment.json")
    require(environment.get("upstream_commit") == revision,
            "Mismatched environment upstream revision")
    require(environment.get("jax_backend") == "cpu", "Expected CPU smoke environment")
    require(bool(environment.get("python")) and bool(environment.get("packages"))
            and bool(environment.get("jax_devices")), "Incomplete environment provenance")
    sources = environment.get("source_sha256")
    require(isinstance(sources, dict) and bool(sources), "Missing environment source hashes")
    for source, expected_hash in sources.items():
        source_path = (repo_root / source).resolve()
        require(source_path.is_relative_to(repo_root), "Source hash path escapes repository")
        require(source_path.is_file() and sha256(source_path.read_bytes()) == expected_hash,
                f"Stale environment source hash: {source}")
    seed, = profile["seeds"]
    paths = {method: runs_root / "baselines" / "smoke" / method / f"trial_{seed + 1}"
             for method in ("ga", "es", "ppo")}
    paths["shinka"] = runs_root / "shinka" / f"seed_{seed}"
    runs = {label: validate_run(path, label, profile, revision) for label, path in paths.items()}
    for label, run in runs.items():
        require(run["noise_vectors"] == runs["ga"]["noise_vectors"],
                f"Mismatched {label} noise/task vectors")
    ga, shinka = runs["ga"], runs["shinka"]
    for index, (baseline, evaluated) in enumerate(zip(ga["records"], shinka["records"])):
        for task in range(profile["num_tasks"]):
            column = f"centroid_task{task}"
            equal_number(evaluated.get(column), finite(baseline.get(column), column),
                         f"Shinka/GA parity row {index} {column}")
    correctness = read_json(runs_root / "shinka/correct.json")
    metrics = read_json(runs_root / "shinka/metrics.json")
    require(correctness.get("correct") is True and correctness.get("error") is None,
            "Shinka evaluator did not report correct=true")
    equal_number(metrics.get("combined_score"), shinka["row"]["normalized_score"],
                 "Shinka combined_score")
    public = metrics.get("public", {})
    expected_public = {"profile": "smoke", "method": "ga", "seeds_completed": 1,
                       "smoke_validation_only": True, "sigma": 0.5, "elite_ratio": 0.5}
    for key, value in expected_public.items():
        require(public.get(key) == value, f"Mismatched Shinka public {key}")
    equal_number(public.get("mean_return"), shinka["row"]["mean_return"],
                 "Shinka public mean_return")
    expected_seed = {key: shinka["row"][key]
                     for key in ("seed", "trial", "mean_return", "normalized_score")}
    require(metrics.get("private", {}).get("seed_results") == [expected_seed],
            "Mismatched Shinka private seed_results")

    report = {
        "schema_version": 1, "profile": profile, "upstream_commit": revision,
        "suite_wall_seconds": suite["wall_seconds"],
        "caption": CAPTION, "rows": [run["row"] for run in runs.values()],
        "validation": {"complete_manifests": True, "metric_hashes_match": True,
                       "task_vectors_match": True, "shinka_correct": True,
                       "shinka_initial_matches_ga": True},
        "noise_vectors": ga["noise_vectors"],
        "artifact_policy": "Text artifacts retained; local repository paths replaced by <repo>. "
                           "Original and exported SHA-256 hashes identify any redacted copies. "
                           "Binary NPZ checkpoints remain local; only their hashes are exported. "
                           "Raw upstream JSON may contain NaN placeholders; derived summary is "
                           "strict finite JSON.",
    }
    checksums = {}
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        def export_file(source: Path, destination: Path) -> None:
            raw = source.read_bytes()
            entry = {"source": str(source.relative_to(runs_root)),
                     "original_sha256": sha256(raw), "original_bytes": len(raw)}
            if source.suffix == ".npz":
                entry.update(exported=False, reason="Binary checkpoint retained locally")
            else:
                exported = raw.replace(str(repo_root).encode(), b"<repo>")
                destination.write_bytes(exported)
                entry.update(exported=True, redacted=exported != raw,
                             exported_sha256=sha256(exported), exported_bytes=len(exported))
            checksums[str(destination.relative_to(staging))] = entry

        for name in ("suite.json", "environment.json"):
            export_file(runs_root / name, staging / name)
        for label, run in runs.items():
            destination = staging / label
            destination.mkdir()
            source_files = [run["path"] / name for name in ARTIFACTS]
            source_files.extend(sorted(run["path"].glob("*.npz")))
            if label == "shinka":
                source_files.extend(runs_root / "shinka" / name
                                    for name in ("metrics.json", "correct.json"))
            for source in source_files:
                export_file(source, destination / source.name)
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
    parser.add_argument("--runs-root", type=Path, default=Path("results/smoke-20261002"))
    parser.add_argument("--output", type=Path, default=Path("reports/smoke-20261002"))
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()
    try:
        report = export_smoke(args.runs_root, args.output, args.repo_root)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Smoke report rejected: {exc}\n")
    print(json.dumps({"output": str(args.output), "rows": report["rows"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
