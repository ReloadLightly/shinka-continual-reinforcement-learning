#!/usr/bin/env python3
"""Recompute development prefixes from published evidence; never train/evaluate.

Run with --check to verify summary.json, or without arguments to emit the JSON.
Only in-memory analysis views are shortened. Original manifests, configurations,
records, metadata, and episode returns remain unchanged; no native run is created.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shinka_crl.analysis import summarize_trial  # noqa: E402
from shinka_crl.baseline_contract import validate_baseline_config  # noqa: E402
from shinka_crl.experiment import DEFAULT_UPSTREAM, verify_upstream  # noqa: E402


def read(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_sources(report, specification, protocol):
    checksums = report / "checksums.json"
    require(sha256(checksums) == specification["checksums_sha256"], "Changed source receipt")
    published = read(checksums)["published_sha256"]
    for name, expected in published.items():
        path = report / name
        require(path.resolve().is_relative_to(report) and not path.is_symlink(), name)
        require(sha256(path) == expected, f"Published artifact changed: {path}")
    actual = {str(path.relative_to(report)) for path in report.rglob("*") if path.is_file()}
    require(actual == set(published) | {"checksums.json"}, "Unreceipted published files")
    plan, environment = read(report / "raw/plan.json"), read(report / "raw/environment.json")
    require(plan["upstream_commit"] == protocol["upstream_commit"], "Upstream identity changed")
    commit = specification["source_archive_commit"]
    require(len(commit) == 40 and all(character in "0123456789abcdef" for character in commit),
            "Source archive must name an exact Git commit")
    if specification["archive_is_recorded_runtime_commit"]:
        require(environment.get("harness_commit") == commit
                and environment.get("harness_dirty") is False,
                "Archive differs from the recorded clean runtime commit")
    for name, expected in plan["source_sha256"].items():
        try:
            recorded = subprocess.check_output(["git", "show", f"{commit}:{name}"],
                                               cwd=ROOT, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            raise ValueError(f"Archived source unavailable: {name}") from exc
        require(hashlib.sha256(recorded).hexdigest() == expected, f"Archived source differs: {name}")
    archive = {"commit": commit, "basis": specification["source_archive_basis"],
               "is_recorded_runtime_commit": specification["archive_is_recorded_runtime_commit"],
               "source_sha256": plan["source_sha256"]}
    return published, plan, environment, archive


def derive_method(method, specification, protocol):
    report = (HERE / specification["report"]).resolve()
    published, plan, environment, archive = verify_sources(report, specification, protocol)
    training, analysis = report / specification["training"], report / specification["analysis"]
    manifest, results = read(training / "manifest.json"), read(training / "results.json")
    records, evaluation = read(training / "training_metrics.json"), read(analysis / "evaluation.json")
    metadata, analysis_manifest = read(analysis / "checkpoint-metadata.json"), read(analysis / "manifest.json")
    profile = manifest["profile"]
    require(manifest["method"] == method and manifest["seed"] == protocol["seed"]
            and manifest["trial"] == protocol["task_trial"] and manifest["ga_settings"] is None,
            "Source trial identity changed")
    require(manifest["upstream_commit"] == analysis_manifest["upstream_commit"]
            == protocol["upstream_commit"] and analysis_manifest["status"] == "complete",
            "Training/evaluation source identity changed")
    planned_profile = plan.get("profile") or plan["profiles"][profile["name"]]
    require(profile == planned_profile and profile["num_phases"] == 20,
            "Source profile differs from its frozen full-budget plan")
    require(profile["ne"] == {"num_generations": 4000, "task_interval": 200,
                              "pop_size": 512, "num_evals": 3}, "Source NE budget changed")
    validate_baseline_config(results["config"], method)
    require(read(training / "config.json") == results["config"], "Resolved configuration differs")
    require(manifest["metrics_sha256"] == sha256(training / "training_metrics.json"),
            "Source training metric hash differs")
    for name in ("results.json", "config.json", "training_metrics.json"):
        require(analysis_manifest["input_sha256"][name] == sha256(training / name),
                f"Source evaluation input changed: {name}")
    require(analysis_manifest["analysis_source_sha256"] == plan["source_sha256"]["src/shinka_crl/analysis.py"],
            "Source analysis implementation differs from its plan")
    for name, expected in analysis_manifest["upstream_source_sha256"].items():
        require(sha256(DEFAULT_UPSTREAM / name) == expected, f"Native analysis source changed: {name}")
    full = summarize_trial(manifest=manifest, results=results, records=records,
                           evaluation=evaluation, checkpoint_metadata=metadata,
                           episodes=protocol["episodes_per_target"], eval_seed=protocol["evaluation_seed"])
    require(full == read(analysis / "summary.json"), "Full source metrics do not reproduce")

    # Adapt only transient metric inputs to the selected horizon. These are
    # analysis views of completed source trials, not new training receipts.
    phases, count = protocol["phase_count"], protocol["selected_training_records"]
    view_manifest, view_results = deepcopy(manifest), deepcopy(results)
    view_profile = view_manifest["profile"]
    view_profile["num_phases"] = phases
    view_profile["ne"]["num_generations"] = count
    if view_profile["ppo"] is not None:
        view_profile["ppo"]["num_updates"] = phases * view_profile["ppo"]["task_interval"]
    view_results["env_steps"] = protocol["nominal_training_steps_per_prefix"]
    view_results["config"].update(num_generations=count, task_sequence=protocol["phase_sequence"])
    selected = [deepcopy(entry) for entry in evaluation["per_task"]
                if entry["source"] == protocol["agent_source"] and entry["task_idx"] < phases]
    for entry in selected:
        if entry["task_idx"] == phases - 1:
            del entry["zero_shot_next_returns"]
    view_evaluation = {**evaluation, "num_tasks": phases, "agent_sources": ["centroid"],
                       "per_task": selected}
    view_metadata = {**metadata, "sources": {"centroid": [phases, results["config"]["num_params"]]},
                     "noise_vectors": metadata["noise_vectors"][:phases]}
    derived = summarize_trial(manifest=view_manifest, results=view_results, records=records[:count],
                              evaluation=view_evaluation, checkpoint_metadata=view_metadata,
                              episodes=protocol["episodes_per_target"], eval_seed=protocol["evaluation_seed"])
    # Do not present the source profile name as a new registered run protocol.
    derived["source_profile"] = derived.pop("profile")
    events = [json.loads(line) for line in (training / "phase-events.jsonl").read_text().splitlines()]
    event_key = "completed_generations" if method == "ga" else "completed_updates"
    require([event[event_key] for event in events] == list(range(200, 4001, 200)),
            "Source phase boundaries are not contiguous")
    boundary = events[phases - 1]
    require(boundary[event_key] == count and boundary["kind"] == "native_checkpoint_written",
            "Selected horizon has no observed checkpoint boundary")
    source_summary = read(report / "summary.json")
    if method == "ga":
        measurement = source_summary["training"]
        suite_seconds = source_summary["total_wall_seconds"]
    else:
        measurement = read(training / "process-measurement.json")
        suite_seconds = source_summary["compute"]["suite_wall_seconds"]
    require(measurement["phase_events"] == events and measurement["status"] == "complete"
            and measurement["returncode"] == 0, "Source timing record differs")
    require(measurement["trainer_cpu_affinity"] == plan["cpu_affinity"]
            == environment["cpu_affinity"], "Source CPU allocation differs")
    selected_episode_count = sum(len(entry.get(field, [])) for entry in selected
                                 for field in ("returns", "prev_returns", "zero_shot_next_returns"))
    all_episode_count = sum(len(entry.get(field, [])) for entry in evaluation["per_task"]
                           for field in ("returns", "prev_returns", "zero_shot_next_returns"))
    require(selected_episode_count == 100 and all_episode_count == 1740, "Unexpected evaluation count")
    return {
        "method": method, "kind": "derived_development_prefix", "new_trial": False,
        "analysis": derived, "noise_vectors": results["noise_vectors"],
        "selected_existing_centroid_evaluation_episodes": selected_episode_count,
        "compute": {
            "observed_prefix_checkpoint_elapsed_seconds": boundary["elapsed_seconds"],
            "observed_prefix_phase_events": events[:phases],
            "prefix_peak_rss_kib": None,
            "full_source_training_seconds_already_incurred": measurement["wall_seconds"],
            "full_source_analysis_seconds_already_incurred": analysis_manifest["wall_seconds"],
            "full_source_suite_seconds_already_incurred": suite_seconds,
            "full_source_training_steps_already_incurred": full["nominal_training_steps"],
            "full_source_peak_rss_kib": measurement["maximum_trainer_rss_kib"],
            "full_source_evaluation_episodes_already_incurred": all_episode_count,
            "new_training_steps": 0, "new_evaluation_episodes": 0,
        },
        "evidence": {
            "report": specification["report"],
            "checksums": {"path": specification["report"] + "/checksums.json",
                          "sha256": specification["checksums_sha256"]},
            "training_records": specification["report"] + "/" + specification["training"] + "/training_metrics.json",
            "evaluation": specification["report"] + "/" + specification["analysis"] + "/evaluation.json",
            "original_plan": specification["report"] + "/raw/plan.json",
            "verified_published_file_count": len(published),
            "verified_archived_source_count": len(plan["source_sha256"]),
            "source_archive": archive,
            "source_checkpoints_sha256": analysis_manifest["input_sha256"]["checkpoints.npz"],
            "checkpoint_note": "Recorded native checkpoint hash; binary was not reloaded for this arithmetic derivation",
        },
    }


def derive():
    protocol = read(HERE / "protocol.json")
    require(verify_upstream(DEFAULT_UPSTREAM) == protocol["upstream_commit"], "Pinned source differs")
    rows = [derive_method(method, specification, protocol)
            for method, specification in protocol["sources"].items()]
    require(rows[0]["noise_vectors"] == rows[1]["noise_vectors"], "Development task draws differ")
    return {
        "schema_version": 1, "status": "derived_analysis_complete", "kind": protocol["experiment_kind"],
        "protocol": {"path": "protocol.json", "sha256": sha256(HERE / "protocol.json")},
        "derivation": {"path": "derivation.py", "sha256": sha256(Path(__file__)),
                       "reused_analysis_sha256": sha256(ROOT / "src/shinka_crl/analysis.py")},
        "scope": "Matched four-phase development prefixes; no new trials or PPO results",
        "n_existing_trials_per_method": 1, "sample_standard_deviation": None,
        "matched_task_vectors": True, "rows": rows,
        "interpretation": "Describe initial acquisition and three switches only. Reused development evidence cannot establish a twenty-phase three-method ranking or reporting-trial variation.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Recompute and compare existing summary.json")
    args = parser.parse_args()
    summary = derive()
    if args.check:
        require(summary == read(HERE / "summary.json"), "Derived summary differs")
        print("Verified both published archives, source identities, and matched four-phase metrics.")
    else:
        print(json.dumps(summary, indent=2, allow_nan=False))
