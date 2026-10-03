"""Independently reviewed attempts consume only the remaining allocation."""
from contextlib import nullcontext
import json

import pytest

from test_adaptive_continuation import (
    native as native,
    stopped as stopped,
    stopped_recovery as stopped_recovery,
    controller as _continuation_fixture,
    add_generation,
    receipt,
    reserve,
)
from shinka_crl import adaptive_endpoint as endpoint
from shinka_crl import adaptive_search as search
from shinka_crl.pilot import read_json, sha256, write_json

continuation_controller = _continuation_fixture


@pytest.fixture
def stopped_continuation(stopped_recovery, continuation_controller, tmp_path, monkeypatch):
    fixture, source = stopped_recovery
    output = tmp_path / "continuation"
    continuation_controller.prepare(source=source, output=output)
    work = output / "archive"

    def execute(*args, **kwargs):
        for generation in range(18, 21):
            reserve(work, generation)
            continuation_controller.reserve_provider_request(work, generation)
            add_generation(fixture, work, generation, correct=generation != 20)
        write_json(work / "shinka/rng_state.json", {"synthetic": "fresh-after-20"})
        write_json(work / "native-recovery.json", receipt(output, [18, 19, 20], failed=True))
        return {"returncode": 1, "stop_reason": None, "wall_seconds": 3,
                "cleanup": {"complete": True, "survivors": [], "errors": []}}

    monkeypatch.setattr(continuation_controller, "run_supervised", execute)
    with pytest.raises(ValueError, match="Continuation stopped"):
        continuation_controller.run(output)
    return fixture, output


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr(endpoint, "implementation_revision", lambda: "synthetic-endpoint-revision")
    monkeypatch.setattr(endpoint, "wrapper_preflight", lambda _: {"inference_calls": 0})
    monkeypatch.setattr(endpoint, "read_plan", search.read_plan)
    monkeypatch.setattr(endpoint, "runtime_scope", lambda _: nullcontext())
    return endpoint


@pytest.fixture
def prepared(stopped_continuation, controller, tmp_path):
    fixture, source = stopped_continuation
    output = tmp_path / "endpoint"
    controller.prepare(source=source, output=output)
    return fixture, source, output


def execute_attempt(fixture, output, controller, monkeypatch, *, end=25, failed=False):
    work = output / "archive"
    start = read_json(output / "plan.json")["start_generation"]

    def execute(command, *, env, **kwargs):
        assert env["SHINKA_RECOVERY_WORK"] == str(work)
        assert "endpoint_headless.py" in env["SHINKA_HEADLESS_COMMAND"]
        for generation in range(start, end):
            reserve(work, generation)
            controller.reserve_provider_request(work, generation)
            add_generation(fixture, work, generation, correct=not (failed and generation == end - 1))
        write_json(work / "shinka/rng_state.json", {"synthetic": f"fresh-after-{end - 1}"})
        write_json(work / "native-recovery.json",
                   receipt(output, list(range(start, end)), start=start, failed=failed))
        return {"returncode": int(failed), "stop_reason": None, "wall_seconds": 4,
                "cleanup": {"complete": True, "survivors": [], "errors": []}}

    monkeypatch.setattr(controller, "run_supervised", execute)
    if failed:
        with pytest.raises(ValueError, match="Endpoint stopped"):
            controller.run(output)
    else:
        assert controller.run(output)["status"] == "complete"
    return controller.summarize(output)


def test_prepare_binds_checkpoint_and_retains_recursive_history(prepared, controller):
    _, source, output = prepared
    plan = controller.read_endpoint_plan(output)
    assert plan["start_generation"] == 21
    assert plan["allowed_generations"] == [21, 22, 23, 24]
    assert plan["inherited_invalid_generations"] == [14, 17, 20]
    assert plan["inherited_model_request_groups"] == 20
    assert plan["max_additional_training_trials"] == 12
    assert plan["max_additional_nominal_training_steps"] == 92160000
    assert plan["rng_input_sha256"] == sha256(source / "archive/shinka/rng_state.json")
    assert controller.inventory(source / "history") == controller.inventory(
        output / "history/controller/history")
    for name in ("native-recovery.json", "recovery_slots/gen_18.json", "recovery_requests/gen_20.json"):
        assert (output / "history/archive" / name).read_bytes() == (source / "archive" / name).read_bytes()
    summary = controller.summarize(output)
    assert summary["slots_consumed"] == 21 and summary["programs_evaluated"] == 17
    assert summary["endpoint_added"]["new_training_trials"] == 0
    assert summary["allocation_status"] == "open"
    assert summary["endpoint_status"] == "unresolved"


@pytest.mark.parametrize("target", ["source", "history", "program", "rng", "ledger", "plan", "extra"])
def test_tampering_rejected_before_execution(prepared, controller, monkeypatch, target):
    _, source, output = prepared
    paths = {"source": source / "archive/shinka/gen_20/main.py",
             "history": output / "history/controller/history/archive/native-recovery.json",
             "program": output / "archive/shinka/gen_19/main.py",
             "rng": output / "archive/shinka/rng_state.json",
             "ledger": output / "archive/model_requests.jsonl", "plan": output / "plan.json",
             "extra": output / "archive/shinka/gen_20/extra.json"}
    with paths[target].open("a") as stream:
        stream.write("\nchanged")
    monkeypatch.setattr(controller, "run_supervised", lambda *a, **k: pytest.fail("execution started"))
    with pytest.raises((ValueError, json.JSONDecodeError)):
        controller.run(output)


def test_provider_rejects_consumed_skipped_and_repeated_requests(prepared, controller):
    _, _, output = prepared
    work = output / "archive"
    for generation in (*range(21), 25):
        with pytest.raises(ValueError, match="outside"):
            controller.reserve_provider_request(work, generation)
    reserve(work, 22)
    with pytest.raises(ValueError, match="first unused"):
        controller.reserve_provider_request(work, 22)
    reserve(work, 21)
    path = controller.reserve_provider_request(work, 21)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="first unused"):
        controller.reserve_provider_request(work, 21)
    assert path.read_bytes() == before


def test_historical_failures_do_not_stop_new_attempt(prepared, controller):
    _, _, output = prepared
    work = output / "archive"
    assert controller.stop_reason(work) is None
    (work / "shinka/gen_21").mkdir()
    write_json(work / "shinka/gen_21/failure.json", {"error": "synthetic"})
    assert controller.stop_reason(work) == "new terminal proposal failure"


def test_complete_allocation_counts_history_once(prepared, controller, monkeypatch):
    fixture, source, output = prepared
    before = controller.inventory(source)
    _, base = controller.source_summary(source)
    summary = execute_attempt(fixture, output, controller, monkeypatch)
    assert summary["slots_consumed"] == 25 and summary["programs_evaluated"] == 21
    assert len(read_json(output / "state.json")["programs"]) == 24
    assert summary["remaining_generations"] == []
    assert summary["endpoint_status"] == "allocation_exhausted_complete"
    assert summary["endpoint_resolution_required"]
    assert summary["sessions"][:-1] == base["sessions"]
    assert summary["session_wall_seconds"] == base["session_wall_seconds"] + 4
    assert summary["proposal_usage"]["guarded_requests"] == 24
    assert summary["proposal_usage"]["codex_cli_launches"] == 23
    assert controller.inventory(source) == before
    with pytest.raises(ValueError, match="Only a prepared"):
        controller.run(output)


def test_failed_attempt_needs_fresh_review_and_new_path(prepared, controller, monkeypatch, tmp_path):
    fixture, _, output = prepared
    first = execute_attempt(fixture, output, controller, monkeypatch, end=22, failed=True)
    assert first["remaining_generations"] == [22, 23, 24]
    with pytest.raises(ValueError, match="Only a prepared"):
        controller.run(output)
    second = tmp_path / "second-endpoint"
    controller.prepare(source=output, output=second)
    plan = controller.read_endpoint_plan(second)
    assert plan["start_generation"] == 22 and plan["max_additional_training_trials"] == 9
    assert plan["max_additional_nominal_training_steps"] == 69120000
    assert plan["inherited_invalid_generations"] == [14, 17, 20, 21]
    assert controller.inventory(output / "history") == controller.inventory(
        second / "history/controller/history")
    before = controller.inventory(output)
    summary = execute_attempt(fixture, second, controller, monkeypatch)
    assert summary["slots_consumed"] == 25 and summary["programs_evaluated"] == 20
    assert summary["session_wall_seconds"] == first["session_wall_seconds"] + 4
    assert controller.inventory(output) == before
    assert not (second / "archive/shinka/gen_25").exists()


def test_last_slot_grammar_failure_exhausts_allocation_without_success(prepared, controller, monkeypatch, tmp_path):
    fixture, _, output = prepared
    summary = execute_attempt(fixture, output, controller, monkeypatch, failed=True)
    assert summary["status"] == "failed"
    assert summary["allocation_status"] == "exhausted"
    assert summary["endpoint_status"] == "allocation_exhausted_with_grammar_stop"
    assert summary["endpoint_resolution_required"]
    assert summary["remaining_generations"] == []
    with pytest.raises(ValueError, match="No unused slots"):
        controller.prepare(source=output, output=tmp_path / "impossible")


@pytest.mark.parametrize("bad", ["cleanup", "rng", "error", "receipt", "extra"])
def test_unreviewable_failed_sources_cannot_prepare(prepared, controller, monkeypatch, tmp_path, bad):
    fixture, _, output = prepared
    execute_attempt(fixture, output, controller, monkeypatch, end=22, failed=True)
    state = read_json(output / "state.json")
    if bad == "cleanup":
        state["sessions"][0]["cleanup"]["survivors"] = [123]
    elif bad == "rng":
        native = read_json(output / "archive/native-recovery.json")
        native["rng_saved"] = False
        write_json(output / "archive/native-recovery.json", native)
    elif bad == "error":
        state["sessions"][0]["verification_error"] = "synthetic verification failure"
    elif bad == "receipt":
        (output / "archive/recovery_requests/gen_21.json").write_text('{"generation":')
    else:
        (output / "archive/shinka/gen_21/extra.json").write_text("{}")
    if bad != "extra":
        controller.seal(output, state)
    with pytest.raises(ValueError):
        controller.prepare(source=output, output=tmp_path / "unreviewed")


def test_partial_reservation_consumes_slot_but_cannot_be_continued(prepared, controller, monkeypatch, tmp_path):
    _, _, output = prepared

    def interrupted(*a, **kw):
        directory = output / "archive/recovery_slots"
        directory.mkdir()
        (directory / "gen_21.json").write_text('{"generation":')
        raise KeyboardInterrupt

    monkeypatch.setattr(controller, "run_supervised", interrupted)
    with pytest.raises(KeyboardInterrupt):
        controller.run(output)
    summary = controller.summarize(output)
    assert summary["slots_consumed"] == 22 and summary["allocation_status"] == "open"
    assert not summary["endpoint_slot_receipts"][0]["receipt_complete"]
    with pytest.raises(ValueError):
        controller.prepare(source=output, output=tmp_path / "not-reviewable")


def test_prepared_export_is_zero_work_and_binds_all_files(prepared, controller, tmp_path):
    _, _, output = prepared
    report = tmp_path / "freeze"
    summary = controller.export(output=output, report=report)
    assert summary["status"] == "prepared" and summary["endpoint_reserved_generations"] == []
    checks = read_json(report / "checksums.json")
    assert checks["prepared_only"]
    assert all(sha256(report / p) == h for p, h in checks["published_sha256"].items())


def test_all_ancestor_output_paths_are_protected(prepared, controller):
    fixture, source, output = prepared
    ancestor = fixture["output"]
    before = controller.inventory(ancestor)
    with pytest.raises(ValueError, match="separate"):
        controller.prepare(source=source, output=ancestor / "nested")
    with pytest.raises(ValueError, match="independent"):
        controller.export(output=output, report=ancestor / "nested")
    assert controller.inventory(ancestor) == before


def test_check_route_makes_no_reservation(controller, monkeypatch):
    monkeypatch.delenv("SHINKA_RECOVERY_WORK", raising=False)
    monkeypatch.delenv("SHINKA_RECOVERY_GENERATION", raising=False)
    calls = []
    monkeypatch.setattr(controller.runpy, "run_path", lambda _: {"main": lambda args: calls.append(args) or 0})
    assert controller.provider_main(["--check"]) == 0 and calls == [["--check"]]


@pytest.mark.parametrize("cycle", ["self", "pair"])
def test_cyclic_source_chain_rejected(tmp_path, controller, cycle):
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    write_json(first / "plan.json", {"protocol": controller.VERSION, "start_generation": 22,
                                    "source_archive": str(first if cycle == "self" else second)})
    write_json(second / "plan.json", {"protocol": controller.VERSION, "start_generation": 21,
                                     "source_archive": str(first)})
    with pytest.raises(ValueError, match="Cyclic"):
        controller.source_ancestors(first)


@pytest.mark.parametrize("field,value", [("start_generation", 20),
                                        ("max_additional_training_trials", 13),
                                        ("max_additional_nominal_training_steps", 92160001)])
def test_resealed_budget_change_rejected(prepared, controller, field, value):
    _, _, output = prepared
    plan = read_json(output / "plan.json")
    plan[field] = value
    write_json(output / "plan.json", plan)
    write_json(output / "plan-receipt.json", {"plan_sha256": sha256(output / "plan.json")})
    with pytest.raises(ValueError):
        controller.read_endpoint_plan(output)


def test_exhausted_allocation_without_graceful_finalization_stays_unresolved(
        prepared, controller, monkeypatch):
    fixture, _, output = prepared
    execute_attempt(fixture, output, controller, monkeypatch, failed=True)
    native = read_json(output / "archive/native-recovery.json")
    native["rng_saved"] = False
    write_json(output / "archive/native-recovery.json", native)
    controller.seal(output, read_json(output / "state.json"))
    summary = controller.summarize(output)
    assert summary["allocation_status"] == "exhausted"
    assert summary["endpoint_status"] == "unresolved"
