"""A new recovery starts at 18 while retaining both earlier stopped attempts."""
from contextlib import nullcontext
import json
import sqlite3

import pytest

from test_adaptive_recovery import native as native, stopped as stopped, reserve
from shinka_crl import adaptive_recovery as prior
from shinka_crl import adaptive_search as search
from shinka_crl.pilot import read_json, sha256, write_json
from shinka_crl.reference_timing import artifact_hashes


def add_generation(fixture, work, generation, *, correct=True):
    evidence = work / f"shinka/gen_{generation}/results/evaluation"
    fixture["request"](evidence, f"continuation-{generation}")
    if not correct:
        request = read_json(evidence / "request.json")
        request.update(status="failed", error="ProgramValidationError: AST exceeds 512 nodes")
        request.pop("cache_origin")
        write_json(evidence / "request.json", request)
        write_json(evidence / "correct.json", {"correct": False, "error": request["error"]})
        write_json(evidence / "receipt.json", artifact_hashes(evidence))
    for name in ("correct.json", "metrics.json"):
        (evidence.parent / name).write_bytes((evidence / name).read_bytes())
    code = (evidence / "program.py").read_text()
    (evidence.parent.parent / "main.py").write_text(code)
    metrics = read_json(evidence / "metrics.json")
    with sqlite3.connect(work / "shinka/programs.sqlite") as db:
        db.execute("INSERT INTO programs VALUES (?,?,?,?,?,?,?,?)",
                   (f"continuation-{generation}", generation, "id-13", code, 0.5,
                    int(correct), json.dumps(metrics["public"]), json.dumps(metrics["private"])))
    with (work / "model_requests.jsonl").open("a") as stream:
        for event in ("started", "codex_exec", "finished"):
            stream.write(json.dumps({"request_id": f"continuation-{generation}",
                                     "event": event, "outcome": "success"}) + "\n")


def receipt(output, generations, *, start=18, failed=False):
    work = output / "archive"
    ids = [r["generation"] for r in search.database_snapshot(work / "shinka/programs.sqlite")]
    consumed = sorted(set(ids) | {15} | set(generations))
    return {"status": "failed" if failed else "complete", "start": start,
            "target": 25, "missing": [15], "attempted_generations": generations,
            "persisted_generations": ids, "consumed_generations": consumed,
            "initial_rng_sha256": read_json(output / "plan.json")["rng_input_sha256"],
            "final_rng_sha256": sha256(work / "shinka/rng_state.json"),
            "verified_generations": generations[:-1] if failed else generations,
            "graceful_return": True, "rng_saved": True,
            "terminal_reason": "synthetic grammar rejection" if failed else None,
            "completed_generations": len(consumed), "next_generation_to_submit": len(consumed)}


@pytest.fixture
def stopped_recovery(stopped, tmp_path, monkeypatch):
    output = tmp_path / "prior-recovery"
    prior.prepare(source=stopped["output"], output=output)
    work = output / "archive"

    def fail_at_17(*args, **kwargs):
        for generation in (16, 17):
            reserve(work, generation)
            prior.reserve_provider_request(work, generation)
            add_generation(stopped, work, generation, correct=generation == 16)
        write_json(work / "shinka/rng_state.json", {"synthetic": "fresh-after-17"})
        write_json(work / "native-recovery.json", receipt(output, [16, 17], start=16, failed=True))
        return {"returncode": 1, "stop_reason": None, "wall_seconds": 2,
                "cleanup": {"complete": True, "survivors": [], "errors": []}}

    monkeypatch.setattr(prior, "run_supervised", fail_at_17)
    with pytest.raises(ValueError, match="Recovery stopped"):
        prior.run(output)
    assert prior.summarize(output)["slots_consumed"] == 18
    return stopped, output


@pytest.fixture
def controller(monkeypatch):
    from shinka_crl import adaptive_continuation as continuation
    monkeypatch.setattr(continuation, "implementation_revision", lambda: "synthetic-revision")
    monkeypatch.setattr(continuation, "wrapper_preflight", lambda _: {"inference_calls": 0})
    monkeypatch.setattr(continuation, "read_plan", search.read_plan)
    monkeypatch.setattr(continuation, "runtime_scope", lambda _: nullcontext())
    return continuation


@pytest.fixture
def prepared(stopped_recovery, controller, tmp_path):
    fixture, source = stopped_recovery
    output = tmp_path / "continuation"
    controller.prepare(source=source, output=output)
    return fixture, source, output


def test_preparation_retains_history_and_uses_fresh_rng(prepared, controller):
    _, source, output = prepared
    plan = controller.read_continuation_plan(output)
    assert plan["allowed_generations"] == list(range(18, 25))
    assert plan["max_additional_training_trials"] == 21
    assert plan["max_additional_nominal_training_steps"] == 161280000
    assert plan["rng_input_sha256"] == sha256(source / "archive/shinka/rng_state.json")
    summary = controller.summarize(output)
    assert summary["slots_consumed"] == 18 and summary["programs_evaluated"] == 15
    assert summary["continuation_added"]["new_training_trials"] == 0
    for name in ("native-recovery.json", "recovery_slots/gen_16.json", "recovery_requests/gen_17.json"):
        assert (output / "history/archive" / name).read_bytes() == (source / "archive" / name).read_bytes()
    assert not (output / "archive/native-recovery.json").exists()
    assert not (output / "archive/recovery_slots").exists()
    assert not (output / "archive/recovery_requests").exists()


@pytest.mark.parametrize("target", ["source", "history", "program", "rng", "ledger", "plan"])
def test_changed_bindings_reject_before_execution(prepared, controller, monkeypatch, target):
    _, source, output = prepared
    paths = {"source": source / "archive/shinka/gen_16/main.py",
             "history": output / "history/archive/native-recovery.json",
             "program": output / "archive/shinka/gen_17/main.py",
             "rng": output / "archive/shinka/rng_state.json",
             "ledger": output / "archive/model_requests.jsonl", "plan": output / "plan.json"}
    with paths[target].open("a") as stream:
        stream.write("\nchanged")
    monkeypatch.setattr(controller, "run_supervised", lambda *a, **k: pytest.fail("execution started"))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        controller.run(output)


def test_inherited_failures_do_not_stop_new_attempt(prepared, controller):
    _, _, output = prepared
    assert controller.stop_reason(output / "archive") is None
    directory = output / "archive/shinka/gen_18"
    directory.mkdir()
    write_json(directory / "failure.json", {"error": "synthetic terminal failure"})
    assert controller.stop_reason(output / "archive") == "new terminal proposal failure"


def test_provider_rejects_all_consumed_slots_and_duplicate_request(prepared, controller):
    _, _, output = prepared
    work = output / "archive"
    for generation in (*range(18), 25):
        with pytest.raises(ValueError, match="outside"):
            controller.reserve_provider_request(work, generation)
    reserve(work, 18)
    path = controller.reserve_provider_request(work, 18)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        controller.reserve_provider_request(work, 18)
    assert path.read_bytes() == before


def test_partial_reservation_consumes_slot_and_failed_state_cannot_retry(prepared, controller, monkeypatch):
    _, _, output = prepared

    def interrupted(*a, **kw):
        directory = output / "archive/recovery_slots"
        directory.mkdir()
        (directory / "gen_18.json").write_text('{"generation":')
        raise KeyboardInterrupt

    monkeypatch.setattr(controller, "run_supervised", interrupted)
    with pytest.raises(KeyboardInterrupt):
        controller.run(output)
    summary = controller.summarize(output)
    assert summary["status"] == "failed" and summary["slots_consumed"] == 19
    assert summary["remaining_generations"] == list(range(19, 25))
    assert not summary["continuation_slot_receipts"][0]["receipt_complete"]
    with pytest.raises(ValueError, match="Only a prepared"):
        controller.run(output)


def test_complete_endpoint_preserves_source_and_counts_history_once(prepared, controller, monkeypatch):
    fixture, source, output = prepared
    before = prior.inventory(source)
    base = prior.summarize(source)
    work = output / "archive"

    def execute(command, *, env, **kwargs):
        assert env["SHINKA_RECOVERY_WORK"] == str(work)
        assert "continuation_headless.py" in env["SHINKA_HEADLESS_COMMAND"]
        for generation in range(18, 25):
            reserve(work, generation)
            controller.reserve_provider_request(work, generation)
            add_generation(fixture, work, generation)
        write_json(work / "native-recovery.json", receipt(output, list(range(18, 25))))
        return {"returncode": 0, "stop_reason": None, "wall_seconds": 3,
                "cleanup": {"complete": True}}

    monkeypatch.setattr(controller, "run_supervised", execute)
    assert controller.run(output)["status"] == "complete"
    summary = controller.summarize(output)
    assert summary["slots_consumed"] == 25 and summary["programs_evaluated"] == 22
    assert len(search.database_snapshot(work / "shinka/programs.sqlite")) == 24
    assert summary["remaining_generations"] == []
    assert summary["continuation_requested_generations"] == list(range(18, 25))
    assert summary["proposal_usage"]["guarded_requests"] == 24
    assert summary["proposal_usage"]["codex_cli_launches"] == 23
    assert summary["session_wall_seconds"] == base["session_wall_seconds"] + 3
    assert summary["sessions"][:-1] == base["sessions"]
    assert prior.inventory(source) == before
    assert not (work / "shinka/gen_15/main.py").exists()
    with pytest.raises(ValueError, match="Only a prepared"):
        controller.run(output)


def test_prepared_export_binds_zero_new_work(prepared, controller, tmp_path):
    _, _, output = prepared
    report = tmp_path / "freeze"
    summary = controller.export(output=output, report=report)
    assert summary["status"] == "prepared"
    assert summary["continuation_reserved_generations"] == []
    checks = read_json(report / "checksums.json")
    assert checks["prepared_only"]
    assert all(sha256(report / p) == h for p, h in checks["published_sha256"].items())


def test_check_route_has_no_reservations(controller, monkeypatch):
    monkeypatch.delenv("SHINKA_RECOVERY_WORK", raising=False)
    monkeypatch.delenv("SHINKA_RECOVERY_GENERATION", raising=False)
    calls = []
    monkeypatch.setattr(controller.runpy, "run_path", lambda _: {"main": lambda args: calls.append(args) or 0})
    assert controller.provider_main(["--check"]) == 0
    assert calls == [["--check"]]


def test_output_and_export_cannot_modify_original_ancestor(prepared, controller):
    fixture, source, output = prepared
    nested = fixture["output"] / "nested-continuation"
    before = prior.inventory(fixture["output"])
    with pytest.raises(ValueError, match="separate"):
        controller.prepare(source=source, output=nested)
    with pytest.raises(ValueError, match="independent"):
        controller.export(output=output, report=nested)
    assert not nested.exists() and prior.inventory(fixture["output"]) == before


def test_new_files_in_consumed_generation_rejected_after_seal(prepared, controller):
    _, _, output = prepared
    (output / "archive/shinka/gen_17/extra.json").write_text("{}")
    state = read_json(output / "state.json")
    state["status"] = "failed"
    controller.seal(output, state)
    with pytest.raises(ValueError, match="Inherited generation file set"):
        controller.summarize(output)


def test_resealed_budget_change_is_rejected(prepared, controller):
    _, _, output = prepared
    plan = read_json(output / "plan.json")
    plan["max_additional_training_trials"] = 24
    write_json(output / "plan.json", plan)
    write_json(output / "plan-receipt.json", {"plan_sha256": sha256(output / "plan.json")})
    with pytest.raises(ValueError, match="budget"):
        controller.read_continuation_plan(output)


def test_failed_export_retains_new_failure_and_both_histories(prepared, controller, monkeypatch, tmp_path):
    _, _, output = prepared
    work = output / "archive"

    def failure(*args, **kwargs):
        reserve(work, 18)
        root = work / "shinka/gen_18"
        root.mkdir()
        write_json(root / "failure.json", {"error": "synthetic failed proposal"})
        return {"returncode": 1, "stop_reason": "new terminal proposal failure", "wall_seconds": 1,
                "cleanup": {"complete": True}}

    monkeypatch.setattr(controller, "run_supervised", failure)
    with pytest.raises(ValueError, match="Continuation stopped"):
        controller.run(output)
    report = tmp_path / "failed-report"
    summary = controller.export(output=output, report=report)
    assert summary["status"] == "failed" and summary["slots_consumed"] == 19
    assert summary["continuation_added"]["new_training_trials"] == 0
    for generation in (14, 17):
        assert (report / f"raw/continuation/archive/shinka/gen_{generation}/main.py").exists()
    assert (report / "raw/continuation/history/archive/native-recovery.json").exists()
    assert (report / "raw/continuation/history/controller/plan.json").exists()
    checks = read_json(report / "checksums.json")
    assert all(sha256(report / p) == h for p, h in checks["published_sha256"].items())
    assert not list(report.rglob("*.sqlite"))


def test_success_exit_without_completed_slots_is_rejected(prepared, controller, monkeypatch):
    _, _, output = prepared

    def incomplete(*args, **kwargs):
        write_json(output / "archive/native-recovery.json", receipt(output, []))
        return {"returncode": 0, "stop_reason": None, "wall_seconds": 1,
                "cleanup": {"complete": True}}

    monkeypatch.setattr(controller, "run_supervised", incomplete)
    with pytest.raises(ValueError, match="Incomplete native continuation completion receipt"):
        controller.run(output)
    assert read_json(output / "state.json")["status"] == "failed"
