"""Publish verified evidence for the resource-limited PPO development prefix.

This exporter deliberately does not relax the full-budget comparison exporter.
It preserves the shorter horizon, original unfinished attempt, and all incurred
costs as a separate development experiment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from shinka_crl.analysis import validate_analysis
from shinka_crl.baseline_contract import validate_baseline_config
from shinka_crl.experiment import REPO_ROOT, nominal_training_steps, score_curve
from shinka_crl.pilot import read_json, require, sha256
from shinka_crl.reference_comparison import artifact_hashes, command_for
from shinka_crl import reference_prefix as prefix

METRICS = ("learning_accuracy", "forgetting", "learning_minus_forgetting", "zero_shot_transfer",
           "cumulative_reward_steps", "cumulative_reward_generation_equivalents",
           "cumulative_reward_steps_exact", "normalized_curve_average")
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".txt"}
DEFAULT_BUDGET_REPORT = REPO_ROOT / "reports/reference-ppo-budget-20261003"
EXPORT_SOURCES = ("src/shinka_crl/reference_prefix_report.py", "scripts/report_reference_prefix.py",
                  "src/shinka_crl/reference_prefix.py", "src/shinka_crl/analysis.py",
                  "src/shinka_crl/baseline_contract.py", "src/shinka_crl/experiment.py")


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"Invalid {label}")
    return value


def nonnegative(value, label):
    require(finite(value, label) >= 0, f"Negative {label}")
    return value


def describe(value):
    return {"n": 1, "mean": value, "sample_sd": None, "values": [value]}


def verified_files(root: Path) -> list[Path]:
    paths = sorted(root.rglob("*"))
    require(not any(path.is_symlink() for path in paths), "Artifact symlinks are forbidden")
    return [path for path in paths if path.is_file() and path.name != "controller.lock"]


def validate_auxiliary_evidence(root: Path, *, plan_hash: str, endpoint: int, seconds: float) -> dict:
    protocol, verification = (read_json(root / name) for name in ("protocol.json", "verification.json"))
    require(protocol["source_plan_sha256"] == plan_hash and protocol["partition"] == "development"
            and protocol["amended_target"]["updates"] == endpoint
            and protocol["amended_target"]["phases"] == endpoint // 1500,
            "Prefix differs from the declared resource amendment")
    require(verification["prior_auxiliary_seconds_to_charge"] == seconds == sum(
        nonnegative(row["charged_wall_seconds"], "auxiliary attempt time")
        for row in verification["attempts"]), "Auxiliary compute differs from recorded verification")
    for relative, expected in verification["published_sha256"].items():
        path = root / relative
        require(not Path(relative).is_absolute() and path.resolve().is_relative_to(root)
                and not any(item.is_symlink() for item in (path, *path.parents)),
                "Auxiliary artifact path escapes evidence root")
        require(sha256(path) == expected, "Auxiliary verification artifact changed")
    return {"report": os.path.relpath(root, REPO_ROOT), "protocol_sha256": sha256(root / "protocol.json"),
            "verification_sha256": sha256(root / "verification.json"),
            "prior_auxiliary_seconds": seconds, "allocation_seconds": protocol["total_active_seconds_limit"]}


def validate_finalized_artifacts(*, training: Path, analysis: Path, profile: dict,
                                 plan: dict, job: dict, checkpoint_sha256: str) -> dict:
    """Recompute metrics and verify zero-update native finalization without training."""
    require(read_json(training / "receipt.json") == artifact_hashes(training),
            "Finalized training receipt mismatch")
    manifest = read_json(training / "manifest.json")
    endpoint = profile["ppo"]["num_updates"]
    require(manifest["status"] == "complete" and manifest["profile"] == profile
            and manifest["method"] == "ppo" and manifest["seed"] == job["seed"]
            and manifest["trial"] == job["trial"] and manifest["ga_settings"] is None
            and manifest["upstream_commit"] == prefix.UPSTREAM_COMMIT
            and manifest["command"] == command_for(plan, profile, job, training),
            "Finalization identity or exact native command differs")
    require(manifest["derivation"] == {"kind": "native_checkpoint_completed_prefix",
                                      "source_checkpoint_sha256": checkpoint_sha256,
                                      "completed_updates": endpoint, "additional_training_updates": 0},
            "Finalization must preserve the native checkpoint with zero additional updates")
    measured = read_json(training / "process-measurement.json")
    require(measured["status"] == "complete" and measured["returncode"] == 0
            and measured["start_update"] == endpoint and measured["phase_events"] == []
            and measured["wall_seconds"] == manifest["wall_seconds"],
            "Native finalization did not complete at the saved endpoint")
    if "cpu_affinity" in plan:
        require(measured["trainer_cpu_affinity"] == plan["cpu_affinity"], "Finalizer CPU allocation changed")
    nonnegative(measured["wall_seconds"], "finalization time")
    nonnegative(measured["maximum_trainer_rss_kib"], "finalizer peak RSS")
    log = (training / "process.log").read_text()
    require(f"at update {endpoint} " in log and not re.search(r"^\s*update\s+\d+\s+task=", log, re.M),
            "Finalizer log does not establish a zero-update resume")
    records = read_json(training / "training_metrics.json")
    require(manifest["metrics_sha256"] == sha256(training / "training_metrics.json")
            and read_json(training / "summary.json") == score_curve(records, profile=profile, method="ppo"),
            "Finalized curve summary differs from raw records")
    results = read_json(training / "results.json")
    validate_baseline_config(results["config"], "ppo")
    before = read_json(training / "checkpoint-probe.json")
    # This branch of the existing probe imports NumPy only and never opens a
    # pickle or initializes the training algorithm.
    after = prefix.probe(plan, "artifacts", training)
    require(before["step"] == before["record_count"] == after["record_count"] == endpoint
            and before["records_sha256"] == after["records_sha256"]
            and before["phase_agents"] == after["phase_agents"]
            and before["phase_tasks"] == [phase % 2 for phase in range(profile["num_phases"])],
            "Finalized records or phase policies differ from the checked checkpoint")
    expected_vectors = [results["noise_vectors"][task] for task in before["phase_tasks"]]
    require(after["noise_vectors"] == expected_vectors, "Phase task vectors differ")
    analyzed = validate_analysis(run_dir=training, output_dir=analysis,
                                 episodes=plan["eval_episodes"], eval_seed=job["eval_seed"])
    require(analyzed["nominal_training_steps"] == nominal_training_steps(profile, "ppo"),
            "Post-hoc analysis budget differs")
    episodes = sum(len(entry.get(field, [])) for entry in read_json(analysis / "evaluation.json")["per_task"]
                   for field in ("returns", "prev_returns", "zero_shot_next_returns"))
    require(episodes == (3 * profile["num_phases"] - 2) * plan["eval_episodes"],
            "Fresh evaluation episode count differs")
    return {"analysis": analyzed, "finalizer": measured, "artifacts_probe": after,
            "analysis_wall_seconds": nonnegative(read_json(analysis / "manifest.json")["wall_seconds"],
                                                 "analysis time"),
            "fresh_evaluation_episodes": episodes, "noise_vectors": results["noise_vectors"]}


def validate_prefix(root: Path, source: dict, budget_report: Path) -> tuple[dict, dict]:
    require(read_json(root / "receipt.json") == artifact_hashes(root), "Prefix receipt mismatch")
    provenance, saved = (read_json(root / name) for name in ("provenance.json", "summary.json"))
    plan, profile, job = (source[key] for key in ("plan", "profile", "job"))
    endpoint = profile["ppo"]["num_updates"]
    require(provenance["status"] == saved["status"] == "complete"
            and provenance["kind"] == saved["kind"] == "resource_limited_development_prefix"
            and provenance["profile"] == profile and provenance["job"] == job
            and provenance["original_plan_sha256"] == sha256(source["suite"] / "plan.json")
            and provenance["original_attempt"] == str(source["source"].relative_to(source["suite"]))
            and provenance["original_attempt_receipts"] == source["attempt_receipts"]
            and provenance["original_attempt_costs"] == source["attempt_costs"]
            and provenance["source_checkpoint_sha256"] == source["checkpoint_sha256"]
            and provenance["upstream_commit"] == prefix.UPSTREAM_COMMIT
            and provenance["source_sha256"] == plan["source_sha256"]
            and provenance["adapter_sha256"] == sha256(Path(prefix.__file__)),
            "Prefix provenance differs from its sealed original development attempt")
    for key, value in {"reporting_trials": 0, "completed_prefix_phases": profile["num_phases"],
                       "original_planned_phases": 20, "completed_prefix_updates": endpoint,
                       "original_planned_updates": 30000, "additional_training_updates": 0,
                       "nominal_training_steps": nominal_training_steps(profile, "ppo")}.items():
        require(saved[key] == value, f"Prefix summary differs: {key}")
    require(read_json(root / "environment.json") == read_json(source["suite"] / "environment.json"),
            "Finalization runtime differs from original training")
    checked = validate_finalized_artifacts(training=root / "training", analysis=root / "analysis",
                                           profile=profile, plan=plan, job=job,
                                           checkpoint_sha256=source["checkpoint_sha256"])
    require(saved["analysis"] == checked["analysis"], "Prefix metrics differ from fresh raw episodes")
    auxiliary = validate_auxiliary_evidence(budget_report, plan_hash=provenance["original_plan_sha256"],
                                            endpoint=endpoint, seconds=provenance["prior_auxiliary_seconds"])
    require(auxiliary["allocation_seconds"] == provenance["allocation_seconds"] <= 28800,
            "PPO allocation differs from amendment")
    training_seconds = sum(nonnegative(row["wall_seconds"], "original attempt time")
                           for row in source["attempt_costs"])
    peak = max(row["maximum_trainer_rss_kib"] for row in source["attempt_costs"])
    costs = saved["compute"]
    expected_costs = {"original_attempt_total_wall_seconds": training_seconds,
                      "prior_auxiliary_seconds": provenance["prior_auxiliary_seconds"],
                      "finalizer_wall_seconds": checked["finalizer"]["wall_seconds"],
                      "finalizer_maximum_rss_kib": checked["finalizer"]["maximum_trainer_rss_kib"],
                      "analysis_wall_seconds": checked["analysis_wall_seconds"],
                      "original_training_maximum_rss_kib": peak}
    require(all(costs[key] == value for key, value in expected_costs.items())
            and provenance["original_attempt_total_wall_seconds"] == training_seconds,
            "Saved compute differs from measured stages")
    adapter_seconds = nonnegative(provenance["adapter_wall_seconds"], "adapter duration")
    total = training_seconds + auxiliary["prior_auxiliary_seconds"] + adapter_seconds
    require(checked["finalizer"]["wall_seconds"] + checked["analysis_wall_seconds"] <= adapter_seconds
            and training_seconds + auxiliary["prior_auxiliary_seconds"]
                <= finite(costs["total_accounted_wall_seconds"], "accounted wall time") <= total
            and total <= provenance["allocation_seconds"], "Inconsistent or exceeded compute allocation")
    analyzed = checked["analysis"]
    row = {**job, "metrics": {key: finite(analyzed["metrics"][key], key) for key in METRICS},
           "phase_returns": analyzed["phase_returns"], "phase_training_returns": analyzed["phase_training_returns"],
           "training_env_steps_nominal": analyzed["nominal_training_steps"],
           "training_wall_seconds": training_seconds, "analysis_wall_seconds": checked["analysis_wall_seconds"],
           "maximum_trainer_rss_kib": peak, "fresh_evaluation_episodes": checked["fresh_evaluation_episodes"],
           "training_path": "training", "analysis_path": "analysis", "noise_vectors": checked["noise_vectors"]}
    group = {"method": "ppo", "n": 1, "seeds": [job["seed"]],
             "metrics": {key: describe(row["metrics"][key]) for key in METRICS},
             **{key: describe(row[key]) for key in (
                 "training_wall_seconds", "analysis_wall_seconds", "maximum_trainer_rss_kib")}}
    report = {"schema_version": 1, "mode": "development", "status": "complete",
              "experiment_kind": provenance["kind"], "completed_trials": 1, "planned_trials": 1,
              "completed_reporting_trials": 0, "completed_full_budget_trials": 0,
              "upstream_commit": prefix.UPSTREAM_COMMIT, "profiles": {profile["name"]: profile},
              "rows": [row], "groups": [group],
              "original_target": {"phases": 20, "ppo_updates": 30000, "reporting_trials_per_method": 10},
              "caption": f"One development trajectory, truncated to {profile['num_phases']} complete phases "
              f"({endpoint} PPO updates) with unchanged per-phase budgets and baseline settings. "
              "This is not the full-budget or reporting comparison. No estimate of between-trial variation. "
              "Signed forgetting and transfer cover the retained consecutive switches. "
              "Cumulative return uses the pinned NE-generation integration grid and episode-cap normalization.",
              "compute": {**expected_costs, "adapter_wall_seconds": adapter_seconds,
                          "total_accounted_wall_seconds": total,
                          "allocation_seconds": provenance["allocation_seconds"],
                          "retained_training_steps_nominal": analyzed["nominal_training_steps"],
                          "additional_training_updates": 0,
                          "scored_fresh_evaluation_episodes": checked["fresh_evaluation_episodes"],
                          "original_attempts": source["attempt_costs"],
                          "prefix_checkpoint_elapsed_seconds_in_last_attempt": source["attempt_costs"][-1]["phase_events"][-1]["elapsed_seconds"],
                          "timing_basis": "Measured monotonic durations; entire original attempts are charged, "
                          "including discarded beyond-checkpoint work. Adapter includes finalizer, evaluation, "
                          "probes and validation. Previously incurred verification is charged separately."},
              "auxiliary_evidence": auxiliary,
              "validation": {"original_attempt_receipts_verified": True, "frozen_sources_match": True,
                             "zero_additional_training_updates": True, "policies_and_history_preserved": True,
                             "posthoc_metrics_recomputed": True, "eight_hour_allocation_verified": True,
                             "all_full_budget_reporting_trials": False}}
    return report, checked["artifacts_probe"]


def export_reference_prefix(runs_root: Path, output: Path, *, budget_report=DEFAULT_BUDGET_REPORT) -> dict:
    root, output, budget_report = (Path(path).resolve() for path in (runs_root, output, budget_report))
    require(not output.exists() and not output.is_relative_to(root), "Use a fresh report outside the raw prefix")
    provenance = read_json(root / "provenance.json")
    require(provenance["status"] == "complete", "Cannot export unfinished prefix")
    source_root = Path(provenance["original_suite"]).resolve()
    require(not output.is_relative_to(source_root) and not output.is_relative_to(budget_report),
            "Report must be outside its source evidence")
    with prefix.terminal_source_lock(source_root):
        source = prefix.inspect_source(source_root, provenance["profile"]["ppo"]["num_updates"])
        report, artifacts_probe = validate_prefix(root, source, budget_report)
        publication = {"repository_revision": subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip(),
            "source_sha256": {name: sha256(REPO_ROOT / name) for name in EXPORT_SOURCES},
            "verification_note": "The raw original plan records execution source hashes. This revision "
            "and the exporter/helper hashes identify publication. Published artifacts remain verifiable "
            "by their own checksums after later protocol changes; current full-budget validators are "
            "not substituted for this archived development protocol."}
        report = {**report, "publication_provenance": publication}
        files = [("raw", root, path) for path in verified_files(root)]
        files += [("source-raw", source_root, path) for path in verified_files(source_root)]
        files += [("support", budget_report, budget_report / name)
                  for name in ("protocol.json", "verification.json")]
        output.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
        originals, published = {}, {}
        replacements = sorted(((str(root), "<prefix>"), (str(source_root), "<original-run>"),
                               (str(REPO_ROOT), "<repo>")), key=lambda item: -len(item[0]))
        try:
            for section, parent, path in files:
                relative = Path(section) / path.relative_to(parent)
                raw = path.read_bytes()
                originals[str(relative)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                                            "exported": path.suffix in TEXT_SUFFIXES}
                if path.suffix not in TEXT_SUFFIXES:
                    continue
                require(b"\0" not in raw, f"NUL in text artifact: {relative}")
                for old, new in replacements:
                    raw = raw.replace(old.encode(), new.encode())
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                published[str(relative)] = sha256(target)
            for name, value in (("summary.json", report), ("validation/artifacts-probe.json", artifacts_probe)):
                target = staging / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
                published[name] = sha256(target)
            checksums = {"original_artifacts": originals, "published_sha256": published,
                         "exporter_sha256": sha256(Path(__file__)),
                         "publication_provenance": publication,
                         "artifact_policy": "Original and derived evidence retained. All files hashed; text "
                         "published with local paths redacted, binary checkpoints retained outside Git. "
                         "Support files describe already-published execution verification, not scientific results."}
            (staging / "checksums.json").write_text(json.dumps(checksums, indent=2, allow_nan=False) + "\n")
            require(read_json(root / "receipt.json") == artifact_hashes(root), "Prefix changed during publication")
            for relative, receipt in source["attempt_receipts"].items():
                require(artifact_hashes(source_root / relative) == receipt, "Original attempt changed during publication")
            require(not output.exists(), "Refusing to overwrite report")
            staging.rename(output)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    return report
