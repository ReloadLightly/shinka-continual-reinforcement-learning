"""Frozen study and cache integration; training is replaced with small fake artifacts."""

from copy import deepcopy
import os
import sys

import pytest

from shinka_crl import adaptive_evaluation as study_api
from shinka_crl.adaptive import load_program
from shinka_crl.experiment import nominal_training_steps
from shinka_crl.pilot import read_json, sha256, write_json
from shinka_crl.reference_timing import artifact_hashes


@pytest.fixture
def study(tmp_path, monkeypatch):
    state = {"affinity": {0, 1, 2, 3}, "training_calls": [], "fail_seed": None}
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(state["affinity"]))
    monkeypatch.setattr(os, "sched_setaffinity", lambda _, cpus: state.update(affinity=set(cpus)))
    monkeypatch.setattr(study_api, "source_hashes", lambda: {"frozen.py": "a" * 64})
    monkeypatch.setattr(study_api, "verify_upstream", lambda _: study_api.UPSTREAM_COMMIT)
    runtime = {"python": "3.11.15", "jax_backend": "cpu", "jax_devices": ["TFRT_CPU_0"],
               "cpu_affinity": [0, 1], "thread_environment": study_api.SEARCH_THREAD_ENV,
               "packages": {"jax": "0.5.3", "jaxlib": "0.5.3", "numpy": "2.2.5"}}
    monkeypatch.setattr(study_api, "runtime_probe", lambda *args: deepcopy(runtime))

    def fake_train(plan, candidate, seed, training, program):
        training.mkdir(parents=True)
        state["training_calls"].append((candidate["id"], seed))
        manifest = {"status": "complete", "profile": plan["profile"],
                    "method": "ga_focus" if candidate["variant"] == "ga_focus" else "ga",
                    "seed": seed, "trial": seed + 1, "wall_seconds": 2.0,
                    "command": study_api.command_for(plan, candidate, seed, training, program)}
        if state["fail_seed"] == seed:
            manifest["status"] = "failed"
            write_json(training / "manifest.json", manifest)
            raise RuntimeError("injected training failure")
        write_json(training / "manifest.json", manifest)
        records = [{"generation": i, "task": (i // 20) % 2,
                    f"centroid_task{(i // 20) % 2}": 100 + seed - 4001}
                   for i in range(80)]
        write_json(training / "training_metrics.json", records)
        (training / "checkpoints.npz").write_bytes(b"fake binary checkpoint")

    def fake_analysis(**kwargs):
        output = kwargs["output_dir"]
        output.mkdir(parents=True)
        write_json(output / "manifest.json", {"status": "complete", "wall_seconds": 0.5})
        write_json(output / "evaluation.json", {"previous_return": 200.0})

    def fake_score(plan, candidate, seed, training, analysis, program):
        records = read_json(training / "training_metrics.json")
        active = sum(row[f"centroid_task{row['task']}"] for row in records) / len(records) / 500
        previous = read_json(analysis / "evaluation.json")["previous_return"] / 500
        return {"objective_version": "adaptive-active-previous-v1",
                "combined_score": (active + previous) / 2,
                "active_score": active, "previous_score": previous,
                "active_mean_return": active * 500, "previous_mean_return": previous * 500,
                "active_checkpoints": 80, "previous_checkpoints": 3,
                "switches": [{"from_phase": i, "to_phase": i + 1, "task": i % 2,
                              "before": 300.0, "after": 200.0, "forgetting": 100.0}
                             for i in range(3)],
                "reference_metrics": {"learning_accuracy": 300.0, "forgetting": 100.0,
                                      "learning_minus_forgetting": 200.0,
                                      "zero_shot_transfer": 200.0,
                                      "normalized_curve_average": active}}

    monkeypatch.setattr(study_api, "_run_trial", fake_train)
    monkeypatch.setattr(study_api, "run_analysis", fake_analysis)
    monkeypatch.setattr(study_api, "_score_trial", fake_score)
    path = tmp_path / "study"
    plan = study_api.freeze_study(output=path, python=sys.executable)
    state.update(path=path, plan=plan, runtime=runtime)
    return state


def evaluate(state, name="identity", *, request_name=None):
    path = state["path"] / "requests" / (request_name or name)
    return study_api.evaluate_candidate(study=state["path"], request_dir=path, control=name)


def resign(path):
    write_json(path / "receipt.json", artifact_hashes(path))


def test_freeze_and_read_plan_preserve_partitions_controls_and_caller(study):
    plan = study_api.read_plan(study["path"])
    assert plan == study["plan"]
    assert plan["profile"]["seeds"] == [4001, 4002, 4003]
    assert plan["reserved_validation_profile"]["seeds"] == [5001, 5002, 5003, 5004, 5005]
    assert plan["protocol_version"] == study_api.VERSION and "seed_allocation" not in plan
    assert [c["id"] for c in plan["controls"]] == list(study_api.CONTROL_IDS)
    assert study["affinity"] == {0, 1, 2, 3} and not study["training_calls"]
    assert plan["controls"][-2]["settings"] == {"sigma": .065, "elite_ratio": .075}
    for control in plan["controls"]:
        if control["source"]:
            assert sha256(study["path"] / "programs" / f"{control['id']}.py") == control["source_sha256"]
    with pytest.raises(ValueError, match="fresh"):
        study_api.freeze_study(output=study["path"], python=sys.executable)


def allocation_audit(tmp_path, development=None, validation=None, observed=None):
    development = [6001, 6002, 6003] if development is None else development
    validation = [7001, 7002, 7003, 7004, 7005] if validation is None else validation
    evidence = {
        "schema_version": 1, "invalid_json": [], "proposed_existing_hits": [],
        "observed": observed or {"seed": [4001], "seeds": [5001], "trial": [5002],
                                  "eval_seed": [905001]},
        "proposed": {name: {"seeds": seeds, "trials": [seed + 1 for seed in seeds],
                            "eval_seeds": [seed + 900000 for seed in seeds]}
                     for name, seeds in (("adaptive_search", development),
                                         ("adaptive_validation", validation))},
    }
    path = tmp_path / "allocation.json"
    write_json(path, evidence)
    return path, development, validation


def freeze_partitioned(study, tmp_path, *, development=None, validation=None, observed=None):
    audit, development, validation = allocation_audit(tmp_path, development, validation, observed)
    path = tmp_path / "partitioned-study"
    plan = study_api.freeze_study(output=path, python=sys.executable,
                                  development_seeds=development, validation_seeds=validation,
                                  seed_allocation=audit)
    return path, plan, audit


def resign_plan(path):
    receipt = read_json(path / "plan-receipt.json")
    receipt["plan_sha256"] = sha256(path / "plan.json")
    write_json(path / "plan-receipt.json", receipt)


def test_new_partitions_change_only_seeds_and_bind_allocation_copy(study, tmp_path):
    path, plan, audit = freeze_partitioned(study, tmp_path)
    assert study_api.read_plan(path) == plan
    assert plan["protocol_version"] == study_api.PARTITION_VERSION
    for field in ("profile", "reserved_validation_profile"):
        expected = deepcopy(study["plan"][field])
        expected["seeds"] = plan[field]["seeds"]
        assert plan[field] == expected
    assert plan["seed_allocation"] == {
        "source_path": str(audit.resolve()), "frozen_path": "seed-allocation.json",
        "sha256": sha256(audit),
    }
    audit.unlink()
    assert study_api.read_plan(path) == plan
    assert study_api.summarize_study(path)["protocol_version"] == study_api.PARTITION_VERSION
    assert not study["training_calls"]


@pytest.mark.parametrize("development,validation,match", [
    ([6001, 6001, 6003], None, "distinct"),
    ([6001, 6002], None, "distinct"),
    ([True, 6002, 6003], None, "distinct"),
    ([2**32 - 900000, 6002, 6003], None, "distinct"),
    ([6001, 6002, 6003], [6001, 7002, 7003, 7004, 7005], "overlap"),
    ([4001, 6002, 6003], None, "overlap"),
    ([1001, 6002, 6003], None, "overlap"),
    ([42, 6002, 6003], None, "overlap"),
    ([9, 6002, 6003], None, "trials overlap"),
])
def test_partition_freeze_rejects_overlap_invalid_or_reused_ids(study, tmp_path,
                                                               development, validation, match):
    with pytest.raises(ValueError, match=match):
        freeze_partitioned(study, tmp_path, development=development, validation=validation)
    assert not (tmp_path / "partitioned-study").exists()
    assert not study["training_calls"]


@pytest.mark.parametrize("field,value", [("seed", 6001), ("seeds", 6001),
                                         ("trial", 6002), ("eval_seed", 906001)])
def test_partition_freeze_checks_each_audited_identity_namespace(study, tmp_path, field, value):
    observed = {"seed": [], "seeds": [], "trial": [], "eval_seed": []}
    observed[field] = [value]
    with pytest.raises(ValueError, match="overlap"):
        freeze_partitioned(study, tmp_path, observed=observed)


@pytest.mark.parametrize("missing", ["development_seeds", "validation_seeds", "seed_allocation"])
def test_partition_freeze_requires_both_partitions_and_audit(study, tmp_path, missing):
    audit, development, validation = allocation_audit(tmp_path)
    arguments = {"development_seeds": development, "validation_seeds": validation,
                 "seed_allocation": audit}
    arguments.pop(missing)
    with pytest.raises(ValueError, match="require"):
        study_api.freeze_study(output=tmp_path / "partitioned-study", python=sys.executable, **arguments)


@pytest.mark.parametrize("mutation", ["allocation", "budget", "purpose", "seeds", "version"])
def test_partition_replay_rejects_audit_or_resigned_scientific_drift(study, tmp_path, mutation):
    path, _, _ = freeze_partitioned(study, tmp_path)
    plan = read_json(path / "plan.json")
    if mutation == "allocation":
        audit = path / "seed-allocation.json"
        audit.write_text(audit.read_text() + "\n")
    elif mutation == "budget":
        plan["profile"]["ne"]["pop_size"] = 128
    elif mutation == "purpose":
        plan["reserved_validation_profile"]["purpose"] = "Changed scientific interpretation"
    elif mutation == "seeds":
        plan["profile"]["seeds"] = [8001, 8002, 8003]
    else:
        plan["protocol_version"] = "adaptive-cartpole-v3"
    write_json(path / "plan.json", plan)
    resign_plan(path)
    with pytest.raises(ValueError, match="changed|differs"):
        study_api.read_plan(path)


def test_new_seed_cache_is_separate_and_reuses_only_matching_context(study, tmp_path):
    legacy = evaluate(study)
    path, plan, _ = freeze_partitioned(study, tmp_path)
    new = {**study, "path": path, "plan": plan}
    first = evaluate(new)
    second = evaluate(new, request_name="duplicate")
    assert first["cache_key"] != legacy["cache_key"]
    assert first["cache_hit"] is False and second["cache_hit"] is True
    assert [seed for _, seed in study["training_calls"]] == [4001, 4002, 4003, 6001, 6002, 6003]
    assert study_api.validate_request(path, path / "requests/duplicate", plan) == second


@pytest.mark.parametrize("changed", ["plan", "program", "source"])
def test_read_plan_rejects_frozen_input_drift(study, monkeypatch, changed):
    if changed == "plan":
        path = study["path"] / "plan.json"
        plan = read_json(path)
        plan["posthoc_episodes"] = 1
        write_json(path, plan)
    elif changed == "program":
        path = study["path"] / "programs/identity.py"
        path.write_text(path.read_text() + "# changed\n")
    else:
        monkeypatch.setattr(study_api, "source_hashes", lambda: {"frozen.py": "b" * 64})
    with pytest.raises(ValueError, match="changed"):
        study_api.read_plan(study["path"])


def test_cache_key_uses_ast_variant_and_entire_context(study, tmp_path):
    initial = study["path"] / "programs/identity.py"
    other = tmp_path / "formatted.py"
    other.write_text("# comment\ndef update_sigma( sigma, stats, memory ):\n    return (sigma, memory)\n")
    first = {"variant": "ga_adaptive", "settings": None, "program": load_program(initial).metadata()}
    formatted = {**first, "program": load_program(other).metadata()}
    assert first["program"]["source_sha256"] != formatted["program"]["source_sha256"]
    key = study_api.cache_key(study["plan"], first)
    assert key == study_api.cache_key(study["plan"], formatted)
    other.write_text("def update_sigma(sigma, stats, memory):\n    return sigma * .5, memory\n")
    assert key != study_api.cache_key(study["plan"], {**first, "program": load_program(other).metadata()})
    for field in ("profile", "runtime", "source_sha256"):
        changed = deepcopy(study["plan"])
        changed[field]["extra_identity_component"] = "changed"
        assert study_api.cache_key(changed, first) != key
    assert study_api.cache_key(study["plan"], {"variant": "ga_focus", "settings": None}) != key


def test_formatted_duplicate_consumes_slot_and_reuses_verified_training(study, tmp_path):
    first = evaluate(study)
    original = artifact_hashes(study["path"] / first["cache_origin"])
    source = tmp_path / "formatted.py"
    source.write_text("# changed formatting\ndef update_sigma( sigma, stats, memory ):\n    return (sigma, memory)\n")
    output = study["path"] / "requests/duplicate"
    duplicate = study_api.evaluate_candidate(study=study["path"], request_dir=output, program_path=source)
    assert len(study["training_calls"]) == 3
    assert duplicate["slot_consumed"] is True and duplicate["cache_hit"] is True
    assert duplicate["evaluator_model_calls"] == 0 and duplicate["new_training_trials"] == 0
    assert duplicate["new_training_steps_nominal"] == 0
    assert duplicate["avoided_training_steps_nominal"] == 23_040_000
    assert duplicate["candidate"]["source_sha256"] != first["candidate"]["source_sha256"]
    assert duplicate["evaluated_source_sha256"] == first["candidate"]["source_sha256"]
    assert duplicate["cache_origin"] == first["cache_origin"]
    assert duplicate["aggregate"] == first["aggregate"]
    assert artifact_hashes(study["path"] / first["cache_origin"]) == original
    assert study_api.validate_request(study["path"], output, study["plan"]) == duplicate
    assert read_json(output / "correct.json")["correct"] is True
    assert read_json(output / "metrics.json")["combined_score"] == first["aggregate"]["scores"]["combined_score"]["mean"]


@pytest.mark.parametrize("name", ["seed_4001/training/training_metrics.json",
                                 "seed_4001/analysis/evaluation.json",
                                 "seed_4001/training/checkpoints.npz"])
def test_cache_reuse_rejects_tampered_raw_or_binary_artifacts(study, name):
    first = evaluate(study)
    path = study["path"] / first["cache_origin"] / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="receipt|corrupted"):
        evaluate(study, request_name="second")
    assert len(study["training_calls"]) == 3
    request = read_json(study["path"] / "requests/second/request.json")
    assert request["status"] == "failed" and request["slot_consumed"] is True


@pytest.mark.parametrize("change", ["score", "aggregate", "raw_previous", "missing_seed"])
def test_cache_rederives_scores_beyond_receipt_hashes(study, change):
    first = evaluate(study)
    attempt = study["path"] / first["cache_origin"]
    summary = read_json(attempt / "summary.json")
    if change == "score":
        summary["trials"][0]["score"]["combined_score"] = .99
    elif change == "aggregate":
        summary["aggregate"]["scores"]["combined_score"]["mean"] = .99
    elif change == "missing_seed":
        summary["trials"].pop()
    else:
        write_json(attempt / "seed_4001/analysis/evaluation.json", {"previous_return": 500})
    write_json(attempt / "summary.json", summary)
    resign(attempt)
    with pytest.raises(ValueError, match="score|aggregate|seeds"):
        study_api.validate_cache(attempt, study["plan"], first["cache_key"])


def test_failed_attempt_and_request_are_retained_when_new_slot_retries(study):
    study["fail_seed"] = 4002
    with pytest.raises(RuntimeError, match="injected"):
        evaluate(study, request_name="failed_slot")
    failed_request = study["path"] / "requests/failed_slot"
    request_before = artifact_hashes(failed_request)
    cache = next((study["path"] / "cache").glob("*/attempt_0001"))
    cached_before = artifact_hashes(cache)
    assert read_json(cache / "summary.json")["status"] == "failed"
    assert len(read_json(cache / "summary.json")["trials"]) == 1
    assert read_json(failed_request / "correct.json")["correct"] is False
    assert read_json(failed_request / "metrics.json")["combined_score"] == 0
    study["fail_seed"] = None
    result = evaluate(study, request_name="retry_slot")
    assert not result["cache_hit"] and result["cache_origin"].endswith("attempt_0002")
    assert len(study["training_calls"]) == 5
    assert artifact_hashes(failed_request) == request_before
    assert artifact_hashes(cache) == cached_before
    assert study["affinity"] == {0, 1, 2, 3}


def test_invalid_program_records_failed_slot_without_training(study, tmp_path):
    source = tmp_path / "invalid.py"
    source.write_text("import os\n")
    output = study["path"] / "requests/invalid"
    with pytest.raises(ValueError):
        study_api.evaluate_candidate(study=study["path"], request_dir=output, program_path=source)
    request = read_json(output / "request.json")
    assert request["status"] == "failed" and request["slot_consumed"] is True
    assert request["evaluator_model_calls"] == 0 and not study["training_calls"]
    assert read_json(output / "correct.json")["correct"] is False
    assert read_json(output / "metrics.json")["combined_score"] == 0
    assert artifact_hashes(output) == read_json(output / "receipt.json")


def test_runtime_drift_blocks_reuse_and_restores_caller(study, monkeypatch):
    evaluate(study)
    changed = deepcopy(study["runtime"])
    changed["packages"]["jax"] = "different"
    monkeypatch.setattr(study_api, "runtime_probe", lambda *args: changed)
    monkeypatch.setenv("OMP_NUM_THREADS", "caller-value")
    with pytest.raises(ValueError, match="runtime changed"):
        evaluate(study, request_name="runtime_drift")
    assert len(study["training_calls"]) == 3
    assert study["affinity"] == {0, 1, 2, 3}
    assert os.environ["OMP_NUM_THREADS"] == "caller-value"


def test_control_staging_reuses_completed_requests_and_counts_only_fresh_training(study):
    partial = study_api.run_controls(study=study["path"], max_controls=1)
    assert partial["status"] == "partial" and len(study["training_calls"]) == 3
    original = artifact_hashes(study["path"] / "requests/identity")
    complete = study_api.run_controls(study=study["path"])
    assert complete["status"] == "complete" and len(study["training_calls"]) == 15
    assert complete["completed_training_trials"] == 15
    assert complete["nominal_steps_completed"] == 5 * 3 * nominal_training_steps(study["plan"]["profile"], "ga")
    assert complete["cache_hits"] == 1
    assert artifact_hashes(study["path"] / "requests/identity") == original
    study_api.run_controls(study=study["path"])
    assert len(study["training_calls"]) == 15


def test_request_score_tampering_blocks_control_resume(study):
    study_api.run_controls(study=study["path"], max_controls=1)
    path = study["path"] / "requests/identity"
    metric = read_json(path / "metrics.json")
    metric["combined_score"] = .99
    write_json(path / "metrics.json", metric)
    resign(path)
    with pytest.raises(ValueError, match="contract|score"):
        study_api.run_controls(study=study["path"])
    assert len(study["training_calls"]) == 3


def test_failed_control_resume_uses_new_slot_and_keeps_old_attempt(study):
    study["fail_seed"] = 4002
    with pytest.raises(RuntimeError, match="injected"):
        study_api.run_controls(study=study["path"], max_controls=1)
    failed = study["path"] / "requests/identity"
    before = artifact_hashes(failed)
    study["fail_seed"] = None
    state = study_api.run_controls(study=study["path"], max_controls=1)
    assert state["status"] == "partial"
    assert state["control_requests"]["identity"]["request_id"] == "identity_retry_0001"
    assert len(state["failed_requests"]) == 1 and state["failed_attempts"] == 1
    assert state["completed_training_trials"] == 4 and state["scored_training_trials"] == 4
    assert state["incomplete_training_attempts"] == 1
    assert state["nominal_steps_allocated"] == 5 * 7_680_000
    assert state["nominal_steps_completed"] == 4 * 7_680_000
    assert artifact_hashes(failed) == before


def test_training_cost_survives_posthoc_failure(study, monkeypatch):
    def fail_analysis(**kwargs):
        raise RuntimeError("injected posthoc failure")

    monkeypatch.setattr(study_api, "run_analysis", fail_analysis)
    with pytest.raises(RuntimeError, match="posthoc"):
        evaluate(study)
    state = study_api.summarize_study(study["path"])
    assert state["completed_training_trials"] == 1 and state["scored_training_trials"] == 0
    assert state["nominal_steps_completed"] == state["nominal_steps_allocated"] == 7_680_000
    assert state["failed_attempts"] == 1 and len(state["failed_requests"]) == 1


def test_six_requests_do_not_substitute_for_all_named_controls(study):
    for index in range(6):
        evaluate(study, request_name=f"extra_{index}")
    state = study_api.summarize_study(study["path"])
    assert len(state["requests"]) == 6 and state["status"] == "partial"
    assert set(state["control_requests"]) == {"identity"}
    assert state["cache_format_check_passed"] is False
    assert len(study["training_calls"]) == 3


@pytest.mark.parametrize("field,value", [
    ("slot_consumed", False), ("new_training_trials", 0),
    ("avoided_training_steps_nominal", 23_040_000), ("request_id", "other"),
])
def test_request_slot_and_work_accounting_are_verified(study, field, value):
    evaluate(study)
    path = study["path"] / "requests/identity"
    request = read_json(path / "request.json")
    request[field] = value
    write_json(path / "request.json", request)
    resign(path)
    with pytest.raises(ValueError, match="provenance|accounting"):
        study_api.validate_request(study["path"], path, study["plan"])


@pytest.mark.parametrize("failed", [False, True])
def test_compact_export_preserves_original_hashes_and_failed_work(study, tmp_path, failed):
    if failed:
        study["fail_seed"] = 4002
        with pytest.raises(RuntimeError, match="injected"):
            evaluate(study)
    else:
        evaluate(study)
    original = artifact_hashes(study["path"])
    report = tmp_path / "report"
    summary = study_api.export_study(study=study["path"], report_dir=report)
    checks = read_json(report / "checksums.json")
    assert checks["original_artifact_sha256"] == original == artifact_hashes(study["path"])
    for relative, digest in checks["published_sha256"].items():
        assert sha256(report / relative) == digest
    assert not list(report.rglob("*.npz"))
    assert any(name.endswith("checkpoints.npz") for name in checks["original_artifact_sha256"])
    manifest = next(report.glob("raw/cache/*/attempt_*/seed_4001/training/manifest.json"))
    assert "$STUDY" in manifest.read_text()
    assert summary["status"] == "partial"
    assert summary["failed_attempts"] == int(failed)
    assert len(summary["failed_requests"]) == int(failed)
    with pytest.raises(ValueError, match="new"):
        study_api.export_study(study=study["path"], report_dir=report)
