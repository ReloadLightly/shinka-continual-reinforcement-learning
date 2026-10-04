"""Read-only integrity and deduplicated compute audit of the 2026-10-04 study."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sqlite3
import time

from shinka_crl import adaptive_evaluation as ae
from shinka_crl import adaptive_search
from shinka_crl import adaptive_validation
from shinka_crl.reference_timing import artifact_hashes
from shinka.prompts import DIFF_ITER_MSG, FULL_ITER_MSG, perf_str, format_text_feedback_section


ROOT = Path.cwd()
PREREG = ROOT / "reports/adaptive-repeated-preregistration-20261004"
STUDY = ROOT / "results/adaptive-repeated-controls-20261004"
CAMPAIGN = ROOT / "results/adaptive-repeated-20261004"
NATIVE = Path(importlib.metadata.distribution("shinka-evolve").locate_file("shinka")).resolve()
INPUTS = {}
STORAGE_RECORD = Path(__file__).with_name("adaptive-repeated-sqlite-storage-audit-20261004.json")


def portable(path):
    path = path.resolve()
    if path.is_relative_to(NATIVE):
        return "$SHINKA_PACKAGE/" + str(path.relative_to(NATIVE))
    if path.is_relative_to(ROOT):
        return "$REPO_ROOT/" + str(path.relative_to(ROOT))
    if path.is_relative_to(Path(__file__).parent):
        return "$AUDIT_BUNDLE/" + str(path.relative_to(Path(__file__).parent))
    raise ValueError(f"Unexpected input root: {path}")


def read(path):
    contents = path.read_bytes()
    INPUTS[portable(path)] = hashlib.sha256(contents).hexdigest()
    return json.loads(contents)


def sha(path):
    value = hashlib.sha256(path.read_bytes()).hexdigest()
    INPUTS[portable(path)] = value
    return value


def verify_map(root, hashes):
    for name, digest in hashes.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root.resolve()), name
        assert sha(path) == digest, str(path)
    return len(hashes)


def published(directory, search_root=None):
    checks = read(directory / "checksums.json")
    counts = {"published": verify_map(directory, checks["published_sha256"])}
    if "original_artifact_sha256" in checks:
        counts["originals"] = verify_map(STUDY, checks["original_artifact_sha256"])
    if "original_sha256" in checks:
        counts["originals"] = 0
        changes = {row["path"]: row for archive in read(STORAGE_RECORD)["archives"]
                   for row in archive["sqlite_storage_changes"]} if STORAGE_RECORD.exists() else {}
        deviations = []
        for name, digest in checks["original_sha256"].items():
            prefix = "raw/search/" if name.startswith("raw/search/") else "raw/evaluation/"
            root = search_root if prefix == "raw/search/" else STUDY
            assert root is not None
            actual_path = (root / name.removeprefix(prefix)).resolve()
            actual = sha(actual_path)
            if actual != digest:
                relative = str(actual_path.relative_to(ROOT))
                accepted = changes[relative]
                assert accepted["expected_sha256"] == digest and accepted["current_sha256"] == actual
                deviations.append(accepted)
            counts["originals"] += 1
        if deviations:
            counts["sqlite_checkpoint_storage_deviations"] = deviations
        # The original database may be serialized from WAL to its main file.
        # Its scientific rows must still equal the immutable exported snapshot.
        state = read(directory / "raw/search/state.json")
        rows = adaptive_search.database_snapshot(search_root / "shinka/programs.sqlite")
        text = json.dumps(rows)
        for original, replacement in ((str(search_root), "$SEARCH"),
                                      (str(STUDY), "$EVALUATION_STUDY"), (str(ROOT), "$REPO_ROOT")):
            text = text.replace(original, replacement)
        assert json.loads(text) == state["programs"]
        counts["scientific_rows_equal_immutable_exported_state"] = True
    return counts


def cache_accounting(plan):
    attempts, trials = [], []
    control_origins = {read(path)["cache_origin"]
                       for path in (STUDY / "requests").glob("*/request.json")}
    for attempt in sorted((STUDY / "cache").glob("*/attempt_*")):
        receipt = attempt / "receipt.json"
        sealed = receipt.is_file()
        if sealed:
            hashes = artifact_hashes(attempt)
            assert hashes == read(receipt), str(attempt)
            INPUTS.update({portable(attempt / name): value for name, value in hashes.items()})
        saved = read(attempt / "summary.json") if (attempt / "summary.json").exists() else {}
        if saved:
            assert saved["context_sha256"] == ae.digest(plan), str(attempt)
        relative = str(attempt.relative_to(STUDY))
        attempts.append({"path": relative, "sealed": sealed,
                         "status": saved.get("status", "unsealed"),
                         "role": "fixed_controls" if relative in control_origins else "search_proposals"})
        for path in sorted(attempt.glob("seed_*/training/manifest.json")):
            row = read(path)
            assert row["profile"] == plan["profile"]
            assert row["seed"] in [6001, 6002, 6003] and row["trial"] == row["seed"] + 1
            profile = row["profile"]
            steps = (profile["ne"]["num_generations"] * profile["ne"]["pop_size"]
                     * profile["ne"]["num_evals"] * profile["episode_length"])
            result = {"path": str(path.parent.parent.relative_to(STUDY)), "seed": row["seed"],
                      "role": attempts[-1]["role"], "training_status": row["status"],
                      "training_wall_seconds": row.get("wall_seconds"),
                      "allocated_nominal_steps": steps,
                      "completed_nominal_steps": steps if row["status"] == "complete" else 0,
                      "cache_attempt_sealed": sealed, "analysis_status": None,
                      "analysis_wall_seconds": None, "fresh_episodes_all_sources": 0,
                      "fresh_episodes_centroid": 0}
            evidence_candidates = [ROOT / "reports/adaptive-repeated-controls-20261004/raw" / path.relative_to(STUDY)]
            evidence_candidates.extend(report / "raw/evaluation" / path.relative_to(STUDY)
                                       for report in sorted((ROOT / "reports").glob("adaptive-repeated-*")))
            evidence = next(item for item in evidence_candidates if item.is_file())
            published_manifest = read(evidence)
            assert all(published_manifest[key] == row[key]
                       for key in ("status", "profile", "seed", "trial", "wall_seconds"))
            result["published_training_evidence"] = str(evidence.relative_to(ROOT))
            analysis = path.parent.parent / "analysis"
            if (analysis / "manifest.json").exists():
                meta = read(analysis / "manifest.json")
                result.update(analysis_status=meta["status"], analysis_wall_seconds=meta.get("wall_seconds"))
                if meta["status"] == "complete":
                    evaluation = read(analysis / "evaluation.json")
                    assert evaluation["eval_seed"] == row["seed"] + 900000
                    for item in evaluation["per_task"]:
                        count = sum(len(item.get(key, [])) for key in
                                    ("returns", "prev_returns", "zero_shot_next_returns"))
                        result["fresh_episodes_all_sources"] += count
                        if item["source"] == "centroid":
                            result["fresh_episodes_centroid"] += count
            trials.append(result)
    assert len({r["path"] for r in trials}) == len(trials)
    totals = {}
    for role in ("all", "fixed_controls", "search_proposals"):
        rows = [row for row in trials if role == "all" or row["role"] == role]
        totals[role] = {"allocated_training_trials": len(rows),
                        "completed_training_trials": sum(r["training_status"] == "complete" for r in rows),
                        "completed_analysis_trials": sum(r["analysis_status"] == "complete" for r in rows),
                        "recorded_training_wall_seconds": sum(r["training_wall_seconds"] or 0. for r in rows),
                        "recorded_analysis_wall_seconds": sum(r["analysis_wall_seconds"] or 0. for r in rows),
                        **{name: sum(r[name] for r in rows) for name in
                           ("allocated_nominal_steps", "completed_nominal_steps",
                            "fresh_episodes_all_sources", "fresh_episodes_centroid")}}
    return {"attempts": attempts, "unique_physical_trials": trials, "totals": totals,
            "policy": "Count each physical cache attempt/seed once, including failures. Reused identity, "
                      "preflight, and canonical matches add zero training trials. Running manifests "
                      "without wall_seconds are pending, not zero-cost completed work."}


def audit():
    prereg = read(PREREG / "plan.json")
    sources = verify_map(ROOT, prereg["source_sha256"])
    assert sha(STUDY / "plan.json") == prereg["development_plan_sha256"]
    evaluation = ae.read_plan(STUDY)
    controls = {name: ae.validate_request(STUDY, STUDY / "requests" / name, evaluation)
                for name in ae.CONTROL_IDS}
    publications = {str(PREREG.relative_to(ROOT)): published(PREREG)}
    controls_report = ROOT / "reports/adaptive-repeated-controls-20261004"
    publications[str(controls_report.relative_to(ROOT))] = published(controls_report)
    archives = []
    shinka_root = Path(importlib.metadata.distribution("shinka-evolve").locate_file("shinka"))
    for expected in prereg["search_order"]:
        path = ROOT / expected["path"]
        if not (path / "plan.json").exists():
            archives.append({**expected, "status": "not_started"})
            continue
        plan, state = read(path / "plan.json"), read(path / "state.json")
        assert plan["arm"] == expected["arm"] and plan["outer_random_seed"] == expected["outer_seed"]
        assert plan["evaluation_context_sha256"] == ae.digest(evaluation)
        assert plan["model"] == prereg["model"] and plan["reasoning_effort"] == "medium"
        assert plan["auth"] == "chatgpt" and plan["runtime"]["subscription_preflight"]["auth"] == "chatgpt"
        assert plan["runtime"]["subscription_preflight"]["model"] == prereg["model"]
        verify_map(ROOT, plan["source_sha256"])
        native_sources = verify_map(shinka_root, plan["runtime"]["harness"]["shinka_source_sha256"])
        events = []
        if (path / "model_requests.jsonl").exists():
            events = [json.loads(line) for line in (path / "model_requests.jsonl").read_text().splitlines()]
        for event in events:
            if event["event"] == "codex_exec":
                assert event["model"] == prereg["model"]
        summary = {**expected, "status": state["status"], "verified_pinned_native_sources": native_sources,
                   "guarded_requests": len({e["request_id"] for e in events}),
                   "codex_cli_launches": sum(e["event"] == "codex_exec" for e in events),
                   "successful_requests": sum(e["event"] == "finished" and e.get("outcome") == "success" for e in events),
                   "recorded_proposal_wall_seconds": sum(e.get("wall_seconds", 0.) for e in events),
                   "native_rows": [], "paid_api_calls": 0}
        db = path / "shinka/programs.sqlite"
        if db.exists():
            with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
                for generation, parent, identity, score, archive, top_k in connection.execute(
                        "SELECT generation,parent_id,id,combined_score,archive_inspiration_ids,"
                        "top_k_inspiration_ids FROM programs ORDER BY generation"):
                    summary["native_rows"].append({"generation": generation, "parent_id": parent,
                                                   "id": identity, "score": score})
                    if expected["arm"] == "independent":
                        assert json.loads(archive) == json.loads(top_k) == []
                        assert parent is None if generation == 0 else parent == summary["native_rows"][0]["id"]
        summary["slots_recorded"] = len(summary["native_rows"])
        assert summary["slots_recorded"] <= 5 and summary["guarded_requests"] <= 4
        if state["status"] == "complete":
            derived = adaptive_search.summarize(path)
            assert derived == read(path / "summary.json")
            summary["scores_and_receipts_rederived"] = True
            if expected["arm"] == "independent":
                with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
                    code, score, metrics, feedback = connection.execute(
                        "SELECT code,combined_score,public_metrics,text_feedback FROM programs "
                        "WHERE generation=0").fetchone()
                values = dict(language="python", code_content=code,
                              performance_metrics=perf_str(score, json.loads(metrics)),
                              text_feedback_section="\n" + format_text_feedback_section(feedback))
                expected_messages = {template.format(**values).strip()
                                     for template in (DIFF_ITER_MSG, FULL_ITER_MSG)}
                prompts = sorted((path / "shinka/headless_prompts").glob("*.md"))
                assert len(prompts) == 4
                for prompt in prompts:
                    sha(prompt)
                    history, message = prompt.read_text().split("# Previous Messages\n\n", 1)[1].split(
                        "# User Request\n\n", 1)
                    assert json.loads(history.strip()) == []
                    assert message.strip() in expected_messages
                summary["fixed_identity_prompts_and_empty_history_verified"] = len(prompts)
        archives.append(summary)
        for report in sorted((ROOT / "reports").glob("adaptive-repeated-*")):
            if (report / "raw/search/plan.json").is_file():
                report_plan = read(report / "raw/search/plan.json")
                if (report_plan.get("arm"), report_plan.get("outer_random_seed")) == (expected["arm"], expected["outer_seed"]):
                    publications[str(report.relative_to(ROOT))] = published(report, path)
    budget = read(CAMPAIGN / "budget.json")
    assert budget["limit_seconds"] == prereg["elapsed_limit_seconds"]
    assert budget["deadline_utc_seconds"] == budget["started"]["utc_seconds"] + budget["limit_seconds"]
    assert budget["deadline_boottime_seconds"] == budget["started"]["boottime_seconds"] + budget["limit_seconds"]
    action_names = [a["label"] for a in budget["actions"]]
    assert len(set(action_names)) == len(action_names)
    search_names = [Path(row["path"]).name for row in prereg["search_order"]]
    assert [n for n in action_names if n in search_names] == search_names[:len([n for n in action_names if n in search_names])]
    assert sum(a["status"] == "running" for a in budget["actions"]) <= 1
    for action in budget["actions"]:
        if action["label"] in search_names:
            spent = max(action["started"]["utc_seconds"] - budget["started"]["utc_seconds"],
                        action["started"]["boottime_seconds"] - budget["started"]["boottime_seconds"])
            assert spent < prereg["new_archive_launch_cutoff_seconds"], action["label"]
    now = time.time()
    return {"snapshot_utc": datetime.now(timezone.utc).isoformat(), "status": "partial_live_audit",
            "verified_preregistered_sources": sources, "publications": publications,
            "controls_verified": list(controls), "archives": archives,
            "development_compute": cache_accounting(evaluation),
            "budget": {"started_utc": budget["started"]["utc"],
                       "deadline_utc_seconds": budget["deadline_utc_seconds"],
                       "elapsed_utc_seconds_at_snapshot": now - budget["started"]["utc_seconds"],
                       "remaining_utc_seconds_at_snapshot": max(0., budget["deadline_utc_seconds"] - now),
                       "finished_action_utc_seconds": sum(a.get("utc_wall_seconds", 0.) for a in budget["actions"]),
                       "finished_action_monotonic_seconds": sum(a.get("monotonic_wall_seconds", 0.) for a in budget["actions"]),
                       "actions": [{k: a.get(k) for k in ("label", "status", "returncode", "utc_wall_seconds",
                                                         "monotonic_wall_seconds")} for a in budget["actions"]]},
            "accounting_note": "Budget elapsed, action durations, proposal durations, and learner/analysis "
                               "durations are nested measurements; do not sum them together. Physical cache "
                               "costs are deduplicated across all archive and control memberships. "
                               "Fresh validation is separate and has not been included here.",
            "input_sha256": dict(sorted(INPUTS.items()))}


def validation_accounting(report):
    frozen = ROOT / "reports/adaptive-repeated-finalists-20261004"
    output = ROOT / "results/adaptive-repeated-validation-20261004"
    plan = adaptive_validation.read_plan(frozen)
    verify_map(frozen, read(frozen / "receipt.json"))
    result = adaptive_validation.summarize(frozen=frozen, output=output)
    assert result == read(report / "summary.json")
    checks = read(report / "checksums.json")
    checked = {"published": verify_map(report, checks["published_sha256"]),
               "originals": verify_map(output, checks["original_artifact_sha256"])}
    state = read(output / "state.json")
    attempts = {entry["path"] for entry in state["attempts"]}
    assert attempts == {str(path.relative_to(output)) for path in output.glob("trials/*/attempt_*")}
    rows = []
    for relative in sorted(attempts):
        directory = output / relative
        saved = read(directory / "summary.json")
        row = saved["trial"]
        assert row == plan["trial_order"][row["index"]]
        adaptive_validation.verify_attempt(directory, plan, row)
        trial = {"path": relative, **row, "status": saved["status"],
                 "completed_nominal_steps": 0, "allocated_nominal_steps": 0,
                 "training_wall_seconds": 0., "analysis_wall_seconds": 0.,
                 "fresh_episodes_all_sources": 0, "fresh_episodes_centroid": 0}
        path = directory / "training/manifest.json"
        if path.exists():
            manifest = read(path)
            assert manifest["profile"] == plan["profile"]
            assert manifest["seed"] == row["seed"] and manifest["trial"] == row["trial"]
            trial.update(training_status=manifest["status"],
                         training_wall_seconds=manifest.get("wall_seconds", 0.),
                         allocated_nominal_steps=plan["trial_steps_nominal"],
                         completed_nominal_steps=plan["trial_steps_nominal"] if manifest["status"] == "complete" else 0,
                         published_training_evidence=str((report / "raw" / relative / "training/manifest.json").relative_to(ROOT)))
        path = directory / "analysis/manifest.json"
        if path.exists():
            manifest = read(path)
            assert manifest["eval_seed"] == row["eval_seed"]
            trial.update(analysis_status=manifest["status"], analysis_wall_seconds=manifest.get("wall_seconds", 0.))
        episode_paths = [p for p in (directory / "analysis/evaluation.json",
                                     directory / "analysis/evaluation-input/evaluation.json") if p.exists()]
        if episode_paths:
            assert len({sha(p) for p in episode_paths}) == 1
            evaluation = read(episode_paths[0])
            for item in evaluation["per_task"]:
                count = sum(len(item.get(key, [])) for key in
                            ("returns", "prev_returns", "zero_shot_next_returns"))
                trial["fresh_episodes_all_sources"] += count
                if item["source"] == "centroid":
                    trial["fresh_episodes_centroid"] += count
        rows.append(trial)
    totals = {"allocated_training_trials": sum(r["allocated_nominal_steps"] > 0 for r in rows),
              "completed_training_trials": sum(r["completed_nominal_steps"] > 0 for r in rows),
              "completed_analysis_trials": sum(r.get("analysis_status") == "complete" for r in rows),
              "recorded_training_wall_seconds": sum(r["training_wall_seconds"] for r in rows),
              "recorded_analysis_wall_seconds": sum(r["analysis_wall_seconds"] for r in rows),
              **{name: sum(r[name] for r in rows) for name in
                 ("allocated_nominal_steps", "completed_nominal_steps",
                  "fresh_episodes_all_sources", "fresh_episodes_centroid")}}
    assert totals["completed_training_trials"] == result["completed_training_trials"]
    assert totals["completed_nominal_steps"] == result["completed_nominal_training_steps"]
    assert totals["fresh_episodes_all_sources"] == result["realized_fresh_evaluation_episodes"]
    return {"status": result["status"], "planned_trials": plan["planned_trials"],
            "scored_trials": result["scored_trials"], "unique_physical_trials": rows,
            "totals": totals, "verified_export_hashes": checked,
            "report": str(report.relative_to(ROOT)), "frozen_plan_sha256": sha(frozen / "plan.json"),
            "failed_attempts": result["failed_attempts"]}


def publish(report, destination):
    assert not destination.exists(), "Accounting export must use a fresh path"
    budget = read(CAMPAIGN / "budget.json")
    assert all(action["status"] != "running" and action.get("finished") for action in budget["actions"])
    source = audit()
    validation = validation_accounting(report)
    budget = read(CAMPAIGN / "budget.json")
    assert all(action["status"] != "running" for action in budget["actions"])
    actions = [{key: action.get(key) for key in ("label", "status", "returncode", "stop_reason",
                                                "utc_wall_seconds", "monotonic_wall_seconds")}
               for action in budget["actions"]]
    finished = max((a["finished"] for a in budget["actions"]), key=lambda row: row["utc_seconds"])
    elapsed_utc = finished["utc_seconds"] - budget["started"]["utc_seconds"]
    elapsed_boottime = (finished["boottime_seconds"] - budget["started"]["boottime_seconds"]
                        if finished["boot_id"] == budget["started"]["boot_id"] else None)
    development = source["development_compute"]
    totals = {key: development["totals"]["all"][key] + validation["totals"][key]
              for key in validation["totals"]}
    proposals = {"guarded_requests": sum(row["guarded_requests"] for row in source["archives"]),
                 "codex_cli_launches": sum(row["codex_cli_launches"] for row in source["archives"]),
                 "successful_requests": sum(row["successful_requests"] for row in source["archives"]),
                 "recorded_proposal_wall_seconds": sum(row["recorded_proposal_wall_seconds"] for row in source["archives"]),
                 "paid_api_calls": 0, "auth": "chatgpt_subscription", "model": "gpt-6.1-sol",
                 "note": "Guarded CLI launches are not backend model request counts or a subscription allowance meter"}
    assert proposals["guarded_requests"] == proposals["codex_cli_launches"] == proposals["successful_requests"] == 16
    summary = {"schema_version": 1, "protocol": "adaptive-repeated-accounting-v1",
               "scientific_protocol": "reports/adaptive-repeated-preregistration-20261004/plan.json",
               "status": validation["status"], "audited_at_utc": datetime.now(timezone.utc).isoformat(),
               "totals": totals, "development": development, "fresh_validation": validation,
               "proposals": proposals,
               "budget": {"limit_seconds": budget["limit_seconds"], "started_utc": budget["started"]["utc"],
                          "finished_utc": finished["utc"], "elapsed_utc_seconds_through_final_action": elapsed_utc,
                          "elapsed_boottime_seconds_through_final_action": elapsed_boottime,
                          "recorded_max_clock_elapsed_seconds": budget["observed_elapsed_seconds"],
                          "finished_action_utc_seconds": sum(a["utc_wall_seconds"] for a in actions),
                          "finished_action_monotonic_seconds": sum(a["monotonic_wall_seconds"] for a in actions),
                          "inter_action_utc_gap_seconds": elapsed_utc - sum(a["utc_wall_seconds"] for a in actions),
                          "actions": actions},
               "memory": {"peak_trainer_rss_kib": None,
                          "measurement_status": "Not measured by the existing adaptive manifests; no new instrumentation used"},
               "integrity": {"preregistered_sources_verified": source["verified_preregistered_sources"],
                             "publications": source["publications"], "archives": source["archives"],
                             "sqlite_storage_deviation": "See adaptive-repeated-sqlite-storage-audit-20261004.json; "
                                                         "all scientific rows equal immutable exported state"},
               "accounting_policy": "Count unique physical attempt/seed paths once. The first-block export, "
                                    "identity slots, scheduler preflights, and canonical cache hits add no training. "
                                    "Failed/incomplete attempts retain measured costs. Learner, analysis, proposal, "
                                    "and action durations are nested; never sum them with the total UTC elapsed. "
                                    "Total UTC elapsed ends at final action completion and includes intervening gaps. "
                                    "Subsequent report-writing time is outside this execution measurement."}
    destination.mkdir(parents=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (destination / "input-hashes.json").write_text(json.dumps({"sha256": dict(sorted(INPUTS.items())),
        "path_policy": "$REPO_ROOT is the checkout; $SHINKA_PACKAGE is the pinned installed source; "
                       "$AUDIT_BUNDLE is this evidence directory. Original checkpoints remain local. "
                       "Numeric trial costs also link to their checked-in published training manifests."}, indent=2) + "\n")
    portable_budget = json.loads(json.dumps(budget).replace(str(ROOT), "$REPO_ROOT"))
    (destination / "budget.json").write_text(json.dumps({"original_sha256": sha(CAMPAIGN / "budget.json"),
                                                        "budget": portable_budget}, indent=2) + "\n")
    shutil.copyfile(__file__, destination / "derive.py")
    shutil.copyfile(STORAGE_RECORD, destination / STORAGE_RECORD.name)
    checksums = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted(destination.iterdir()) if path.is_file()}
    (destination / "checksums.json").write_text(json.dumps({"published_sha256": checksums}, indent=2) + "\n")
    print(json.dumps({"output": str(destination.relative_to(ROOT)), "status": summary["status"],
                      "totals": totals, "proposals": proposals, "budget": summary["budget"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    publish(args.validation_report.resolve(), args.output_dir.resolve())
