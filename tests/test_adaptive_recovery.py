"""Recovery binds an immutable stopped source and consumes attempts exactly once."""
from contextlib import nullcontext
import json
import sqlite3

import pytest

from test_adaptive_search import native as archive_fixture
from shinka_crl import adaptive_recovery as recovery
from shinka_crl import adaptive_search as search
from shinka_crl.pilot import read_json, sha256, write_json
from shinka_crl.reference_timing import artifact_hashes


native = archive_fixture


@pytest.fixture
def stopped(native, monkeypatch):
    source = native["output"]
    write_json(native["study"] / "plan-receipt.json",
               {"plan_sha256": sha256(native["study"] / "plan.json")})
    search.run_search(output=source, study=native["study"], target=13)
    native["execute"](search.native_command(source, 15), log=source / "extra.log", env={},
                      timeout=1, shinka=source / "shinka")
    directory = source / "shinka/gen_14/results/evaluation"
    request = read_json(directory / "request.json")
    request.update(status="failed", error="ProgramValidationError: AST exceeds 512 nodes")
    request.pop("cache_origin")
    write_json(directory / "request.json", request)
    write_json(directory / "correct.json", {"correct": False, "error": request["error"]})
    write_json(directory / "receipt.json", artifact_hashes(directory))
    (directory.parent / "correct.json").write_bytes((directory / "correct.json").read_bytes())
    with sqlite3.connect(source / "shinka/programs.sqlite") as db:
        db.execute("UPDATE programs SET correct=0 WHERE generation=14")
    canceled = source / "shinka/gen_15"
    canceled.mkdir()
    (canceled / "results").mkdir()  # Native creates this empty directory before requesting code.
    (canceled / ".generation_lock").touch()
    with (source / "model_requests.jsonl").open("a") as stream:
        for event in ("started", "finished"):
            stream.write(json.dumps({"request_id": "canceled-15", "event": event,
                                     "outcome": "provider_failure", "returncode": -15}) + "\n")
    state = read_json(source / "state.json")
    state.update(status="failed", programs=search.database_snapshot(source / "shinka/programs.sqlite"),
                 artifact_sha256=search.artifact_receipts(source),
                 rng_sha256=sha256(source / "shinka/rng_state.json"),
                 model_requests_sha256=sha256(source / "model_requests.jsonl"))
    state["sessions"].append({"index": 2, "target": 25, "status": "failed", "wall_seconds": 1})
    write_json(source / "state.json", state)
    monkeypatch.setattr(recovery, "read_plan", search.read_plan)
    monkeypatch.setattr(recovery, "runtime_scope", lambda _: nullcontext())
    monkeypatch.setattr(recovery, "wrapper_preflight", lambda _: {"inference_calls": 0})
    monkeypatch.setattr(recovery, "implementation_revision", lambda: "synthetic-test-revision")
    return native


@pytest.fixture
def prepared(stopped, tmp_path):
    output = tmp_path / "recovery"
    recovery.prepare(source=stopped["output"], output=output)
    return stopped, output


def test_prepare_preserves_original_and_independent_database(stopped, tmp_path):
    source = stopped["output"]
    before = recovery.inventory(source)
    calls = len(stopped["calls"])
    output = tmp_path / "copy"
    state = recovery.prepare(source=source, output=output)
    assert state["status"] == "prepared" and not state["sessions"]
    assert len(stopped["calls"]) == calls and recovery.inventory(source) == before
    plan = recovery.read_recovery_plan(output)
    assert plan["allowed_generations"] == list(range(16, 25))
    assert plan["no_row_consumed_generations"] == [15]
    assert plan["rng_policy"] == recovery.RNG_POLICY
    assert (source / "shinka/programs.sqlite").stat().st_ino != (
        output / "archive/shinka/programs.sqlite").stat().st_ino
    summary = recovery.summarize(output)
    assert summary["slots_consumed"] == 16 and summary["programs_evaluated"] == 14
    assert summary["recovery_added"]["new_training_trials"] == 0


@pytest.mark.parametrize("kind", ["plan", "source", "copy", "ledger", "rng", "reservation"])
def test_mutation_rejected_before_any_execution(prepared, monkeypatch, kind):
    native, output = prepared
    work = output / "archive"
    if kind == "plan":
        plan = read_json(output / "plan.json")
        plan["allowed_generations"] = list(range(15, 25))
        write_json(output / "plan.json", plan)
    elif kind == "source":
        (native["output"] / "shinka/gen_14/main.py").write_text("changed")
    elif kind == "copy":
        (work / "shinka/gen_14/main.py").write_text("changed")
    elif kind == "ledger":
        (work / "model_requests.jsonl").write_text("")
    elif kind == "rng":
        (work / "shinka/rng_state.json").write_text("{}")
    else:
        (work / "recovery_slots").mkdir()
        write_json(work / "recovery_slots/gen_16.json", {"generation": 16})
    monkeypatch.setattr(recovery, "run_supervised", lambda *a, **kw: pytest.fail("execution started"))
    with pytest.raises(ValueError):
        recovery.run(output)


def test_resealed_invalid_rng_policy_is_rejected(prepared):
    _, output = prepared
    plan = read_json(output / "plan.json")
    plan["rng_policy"] = "pretend uninterrupted"
    write_json(output / "plan.json", plan)
    write_json(output / "plan-receipt.json", {"plan_sha256": sha256(output / "plan.json")})
    with pytest.raises(ValueError, match="RNG policy"):
        recovery.read_recovery_plan(output)


def test_symlink_copy_and_existing_destination_rejected(stopped, tmp_path):
    source = stopped["output"]
    link = source / "unexpected-link"
    link.symlink_to(tmp_path)
    with pytest.raises(ValueError, match="symlink"):
        recovery.prepare(source=source, output=tmp_path / "new")
    link.unlink()
    with pytest.raises(ValueError, match="fresh"):
        recovery.prepare(source=source, output=source)


def test_controller_lock_excludes_duplicate_launch(prepared):
    _, output = prepared
    with recovery.execution_lock(output):
        with pytest.raises(ValueError, match="Another recovery"):
            with recovery.execution_lock(output):
                pytest.fail("duplicate lock")


def test_export_cannot_race_active_controller(prepared, tmp_path):
    _, output = prepared
    with recovery.execution_lock(output):
        with pytest.raises(ValueError, match="Another recovery"):
            recovery.export(output=output, report=tmp_path / "report")


def test_prepared_export_contains_freeze_with_no_new_execution(prepared, tmp_path):
    _, output = prepared
    report = tmp_path / "freeze"
    summary = recovery.export(output=output, report=report)
    assert summary["status"] == "prepared"
    assert summary["recovery_requested_generations"] == []
    assert summary["recovery_reserved_generations"] == []
    assert read_json(report / "checksums.json")["prepared_only"]
    assert sorted(p.name for p in report.rglob("*") if p.is_file()) == sorted([
        "summary.json", "recovery-plan.json", "plan.json", "plan-receipt.json", "state.json", "checksums.json"])


def test_partial_reservation_json_still_consumes_slot(prepared, monkeypatch):
    _, output = prepared
    work = output / "archive"

    def interrupted(*a, **kw):
        directory = work / "recovery_slots"
        directory.mkdir()
        (directory / "gen_16.json").write_text('{"generation":')
        raise KeyboardInterrupt

    monkeypatch.setattr(recovery, "run_supervised", interrupted)
    with pytest.raises(KeyboardInterrupt):
        recovery.run(output)
    summary = recovery.summarize(output)
    assert summary["slots_consumed"] == 17
    assert summary["recovery_slot_receipts"][0]["receipt_complete"] is False


def test_failed_native_receipt_retains_reserved_but_unscheduled_slot(prepared, monkeypatch, tmp_path):
    _, output = prepared
    work = output / "archive"

    def failed(*a, **kw):
        directory = work / "recovery_slots"
        directory.mkdir()
        (directory / "gen_16.json").write_text('{"generation":')
        receipt = native_receipt(output, [])
        receipt.update(status="failed", consumed_generations=list(range(17)),
                       completed_generations=16, next_generation_to_submit=16,
                       terminal_reason="Reservation failed before native scheduling")
        write_json(work / "native-recovery.json", receipt)
        return {"returncode": 1, "stop_reason": None, "wall_seconds": 1,
                "cleanup": {"complete": True}}

    monkeypatch.setattr(recovery, "run_supervised", failed)
    with pytest.raises(ValueError, match="Recovery stopped"):
        recovery.run(output)
    summary = recovery.export(output=output, report=tmp_path / "report")
    assert summary["status"] == "failed" and summary["slots_consumed"] == 17
    assert summary["recovery_slot_receipts"][0]["receipt_complete"] is False


def reserve(work, generation):
    directory = work / "recovery_slots"
    directory.mkdir(exist_ok=True)
    write_json(directory / f"gen_{generation}.json", {"generation": generation})


def native_receipt(output, generations):
    work = output / "archive"
    ids = [r["generation"] for r in search.database_snapshot(work / "shinka/programs.sqlite")]
    return {"status": "complete", "start": 16, "target": 25, "missing": [15],
            "attempted_generations": generations, "persisted_generations": ids,
            "consumed_generations": sorted(set(ids) | {15} | set(generations)),
            "initial_rng_sha256": read_json(output / "plan.json")["rng_input_sha256"],
            "final_rng_sha256": sha256(work / "shinka/rng_state.json"),
            "verified_generations": generations, "graceful_return": True, "rng_saved": True,
            "terminal_reason": None, "completed_generations": 25, "next_generation_to_submit": 25}


def test_request_gate_is_exclusive_and_rejects_consumed_slots(prepared):
    _, output = prepared
    work = output / "archive"
    reserve(work, 16)
    path = recovery.reserve_provider_request(work, 16)
    assert read_json(path)["generation"] == 16
    with pytest.raises(FileExistsError):
        recovery.reserve_provider_request(work, 16)
    for generation in (0, 14, 15, 25):
        with pytest.raises(ValueError, match="outside"):
            recovery.reserve_provider_request(work, generation)
    with pytest.raises(FileNotFoundError):
        recovery.reserve_provider_request(work, 17)


def test_provider_gate_delegates_once_to_unchanged_guard(prepared, monkeypatch):
    _, output = prepared
    work = output / "archive"
    reserve(work, 16)
    monkeypatch.setenv("SHINKA_RECOVERY_WORK", str(work))
    monkeypatch.setenv("SHINKA_RECOVERY_GENERATION", "16")
    calls = []

    def load(path):
        assert path == str(recovery.REPO_ROOT / "scripts/subscription_headless.py")
        return {"main": lambda argv: calls.append("guard") or 1}

    monkeypatch.setattr(recovery.runpy, "run_path", load)
    assert recovery.provider_main([]) == 1
    with pytest.raises(FileExistsError):
        recovery.provider_main([])
    assert calls == ["guard"]
    assert read_json(work / "recovery_requests/gen_16.json")["status"] == "reserved"


def test_native_provider_availability_probe_does_not_reserve_a_slot(monkeypatch):
    monkeypatch.delenv("SHINKA_RECOVERY_WORK", raising=False)
    monkeypatch.delenv("SHINKA_RECOVERY_GENERATION", raising=False)
    calls = []
    monkeypatch.setattr(recovery.runpy, "run_path", lambda _: {
        "main": lambda argv: calls.append(argv) or 0})
    assert recovery.provider_main(["--check"]) == 0
    assert calls == [["--check"]]


def test_reservation_before_mkdir_remains_consumed_after_interrupt(prepared, monkeypatch):
    _, output = prepared
    work = output / "archive"

    def interrupted(*a, **kw):
        reserve(work, 16)
        raise KeyboardInterrupt

    monkeypatch.setattr(recovery, "run_supervised", interrupted)
    with pytest.raises(KeyboardInterrupt):
        recovery.run(output)
    summary = recovery.summarize(output)
    assert summary["status"] == "failed" and summary["slots_consumed"] == 17
    assert summary["recovery_reserved_generations"] == [16]
    assert summary["recovery_requested_generations"] == []
    assert read_json(work / "state.json")["status"] == "failed"
    with pytest.raises(ValueError, match="Only a prepared"):
        recovery.run(output)


def test_old_failure_is_allowed_but_new_failure_stops(prepared):
    _, output = prepared
    work = output / "archive"
    assert recovery.stop_reason(work) is None
    reserve(work, 16)
    path = work / "shinka/gen_16"
    path.mkdir()
    write_json(path / "failure.json", {"error": "terminal proposal"})
    assert recovery.stop_reason(work) == "new terminal proposal failure"


def test_terminal_monitor_allows_bounded_grace_for_native_receipts(prepared, monkeypatch):
    _, output = prepared
    now = [0.]
    monkeypatch.setattr(recovery.time, "monotonic", lambda: now[0])
    monitor = recovery.terminal_monitor(output / "archive", 10.)
    assert monitor() is None  # inherited failure is exempt
    monkeypatch.setattr(recovery, "stop_reason", lambda _: "new terminal evaluation failure")
    assert monitor() is None
    now[0] = 9.9
    assert monitor() is None
    now[0] = 10.
    assert monitor() == "new terminal evaluation failure; native finalization grace expired"


def test_failed_export_preserves_partial_costs_and_original_bindings(prepared, monkeypatch, tmp_path):
    _, output = prepared
    work = output / "archive"

    def failure(*a, **kw):
        reserve(work, 16)
        directory = work / "shinka/gen_16"
        directory.mkdir()
        write_json(directory / "failure.json", {"error": "fixture terminal failure"})
        return {"returncode": 1, "stop_reason": "new terminal proposal failure", "wall_seconds": 2}

    monkeypatch.setattr(recovery, "run_supervised", failure)
    with pytest.raises(ValueError, match="Recovery stopped"):
        recovery.run(output)
    report = tmp_path / "report"
    summary = recovery.export(output=output, report=report)
    assert summary["status"] == "failed" and summary["slots_consumed"] == 17
    assert summary["recovery_added"]["new_training_trials"] == 0
    assert summary["terminal_proposal_failures"] == ["shinka/gen_16/failure.json"]
    assert (report / "raw/recovery/archive/shinka/gen_14/main.py").exists()
    checks = read_json(report / "checksums.json")
    assert all(sha256(report / p) == h for p, h in checks["published_sha256"].items())
    assert not list(report.rglob("*.sqlite"))


def test_complete_native_exit_without_slots_is_failed(prepared, monkeypatch):
    _, output = prepared
    work = output / "archive"

    def incomplete(*a, **kw):
        write_json(work / "native-recovery.json", native_receipt(output, []))
        return {"returncode": 0, "stop_reason": None, "wall_seconds": 1, "cleanup": {"complete": True}}

    monkeypatch.setattr(recovery, "run_supervised", incomplete)
    with pytest.raises(ValueError, match="Incomplete native recovery completion receipt"):
        recovery.run(output)
    assert read_json(output / "state.json")["status"] == "failed"


def test_success_counts_consumed_no_row_slot_without_changing_source(prepared, monkeypatch):
    native, output = prepared
    source, work = native["output"], output / "archive"
    original = recovery.inventory(source)

    def execute(command, *, env, **kwargs):
        assert env["SHINKA_RECOVERY_WORK"] == str(work)
        assert env["SHINKA_SUBSCRIPTION_LEDGER"] == str(work / "model_requests.jsonl")
        assert "recovery_headless.py" in env["SHINKA_HEADLESS_COMMAND"]
        for generation in range(16, 25):
            reserve(work, generation)
            recovery.reserve_provider_request(work, generation)
            evidence = work / f"shinka/gen_{generation}/results/evaluation"
            native["request"](evidence, f"recovery-{generation}")
            for name in ("correct.json", "metrics.json"):
                (evidence.parent / name).write_bytes((evidence / name).read_bytes())
            code = (evidence / "program.py").read_text()
            (evidence.parent.parent / "main.py").write_text(code)
            metrics = read_json(evidence / "metrics.json")
            with sqlite3.connect(work / "shinka/programs.sqlite") as db:
                db.execute("INSERT INTO programs VALUES (?,?,?,?,?,?,?,?)",
                           (f"recovery-{generation}", generation, "id-13", code, 0.5, 1,
                            json.dumps(metrics["public"]), json.dumps(metrics["private"])))
            with (work / "model_requests.jsonl").open("a") as stream:
                for event in ("started", "codex_exec", "finished"):
                    stream.write(json.dumps({"request_id": f"recovery-{generation}", "event": event,
                                             "outcome": "success"}) + "\n")
        write_json(work / "native-recovery.json", native_receipt(output, list(range(16, 25))))
        return {"returncode": 0, "stop_reason": None, "wall_seconds": 3, "cleanup": {"complete": True}}

    monkeypatch.setattr(recovery, "run_supervised", execute)
    assert recovery.run(output)["status"] == "complete"
    summary = recovery.summarize(output)
    assert summary["slots_consumed"] == 25 and summary["programs_evaluated"] == 23
    assert summary["proposal_slots_consumed"] == 24 and summary["remaining_generations"] == []
    assert summary["recovery_requested_generations"] == list(range(16, 25))
    assert summary["proposal_usage"]["codex_cli_launches"] == 23
    assert summary["recovery_added"]["new_training_trials"] == 0  # all fixture cache hits
    assert recovery.inventory(source) == original
    assert not (work / "shinka/gen_15/main.py").exists()
    assert len(search.database_snapshot(work / "shinka/programs.sqlite")) == 24
    with pytest.raises(ValueError, match="Only a prepared"):
        recovery.run(output)
