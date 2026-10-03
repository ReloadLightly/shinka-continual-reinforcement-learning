"""Continue a real native failed checkpoint using synthetic proposals/evaluations."""

import os
from pathlib import Path

import pytest

from shinka_crl import adaptive_recovery_native as recovery
from shinka_crl.pilot import read_json, sha256, write_json
from shinka_crl.reference_timing import artifact_hashes
from shinka_crl.search import database_snapshot

from test_adaptive_recovery_native import assert_no_live_evaluators, native as native_fixture


native = native_fixture


@pytest.fixture
def continuation(native, monkeypatch):
    """Produce the inherited stop through the same native adapter we resume."""
    from shinka.llm.llm import AsyncLLMClient
    from shinka.llm.providers import QueryResult

    work, calls = native
    archive = work / "shinka"
    original_rng = sha256(archive / "rng_state.json")
    monkeypatch.setenv("SYNTHETIC_FAIL_GENERATION", "17")
    assert recovery.native_main(work, target=25) == 1
    inherited = read_json(work / "native-recovery.json")
    assert inherited["status"] == "failed"
    assert inherited["attempted_generations"] == [16, 17]
    assert inherited["consumed_generations"] == list(range(18))
    assert inherited["verified_generations"] == [16]
    assert inherited["rng_saved"] and inherited["graceful_return"]
    assert inherited["final_rng_sha256"] != original_rng
    monkeypatch.delenv("SYNTHETIC_FAIL_GENERATION")

    # The stub replaces the provider boundary, so create synthetic guard records
    # for those two calls to exercise relocation of all three inherited namespaces.
    (work / "recovery_requests").mkdir()
    for call in calls:
        generation = call["generation"]
        write_json(work / f"recovery_requests/gen_{generation}.json",
                   {"generation": generation, "synthetic_provider_call": True})
    history = work.parent / "history" / "archive"
    history.mkdir(parents=True)
    original_history = {}
    for name in ("native-recovery.json", "recovery_slots", "recovery_requests"):
        path = work / name
        if path.is_file():
            original_history[name] = sha256(path)
        else:
            original_history.update({f"{name}/{key}": value
                                     for key, value in artifact_hashes(path).items()})
        path.rename(history / name)
    assert artifact_hashes(history) == original_history
    historical_rows = database_snapshot(archive / "programs.sqlite")
    historical_artifacts = {generation: artifact_hashes(archive / f"gen_{generation}")
                            for generation in range(18)}
    calls.clear()
    code = (work / "task/initial.py").read_text()

    async def query(self, msg, system_msg, **kwargs):
        generation = int(os.environ["SHINKA_RECOVERY_GENERATION"])
        assert generation >= 18
        assert read_json(work / f"recovery_slots/gen_{generation}.json")["generation"] == generation
        if generation > 18:
            previous = database_snapshot(archive / "programs.sqlite")[-1]
            assert previous["generation"] == generation - 1 and previous["correct"] == 1
            assert (archive / f"gen_{generation - 1}/results/evaluation/receipt.json").is_file()
        calls.append({"generation": generation, "prompt": msg, "system": system_msg})
        if os.environ.get("SYNTHETIC_PROVIDER_FAIL") == str(generation):
            return None
        content = ("<NAME>synthetic</NAME><DESCRIPTION>Continuation fixture</DESCRIPTION>\n"
                   "```python\n" + code.replace("return sigma, memory",
                                               f"return sigma * {generation}.0, memory") + "```\n")
        return QueryResult(content=content, msg=msg, system_msg=system_msg,
                           new_msg_history=[], model_name="headless/codex?effort=medium",
                           kwargs={}, input_tokens=1, output_tokens=1)

    monkeypatch.setattr(AsyncLLMClient, "query", query)
    return {"work": work, "calls": calls, "history": history,
            "historical_hashes": original_history, "rows": historical_rows,
            "artifacts": historical_artifacts, "rng": inherited["final_rng_sha256"]}


def assert_inherited_unchanged(state):
    work = state["work"]
    assert artifact_hashes(state["history"]) == state["historical_hashes"]
    assert database_snapshot(work / "shinka/programs.sqlite")[:17] == state["rows"]
    assert {generation: artifact_hashes(work / f"shinka/gen_{generation}")
            for generation in range(18)} == state["artifacts"]
    assert_no_live_evaluators(work)
    assert "SHINKA_RECOVERY_GENERATION" not in os.environ


def test_continuation_reaches_25_after_inherited_terminal_failure(continuation):
    work = continuation["work"]
    source = Path(__import__("shinka.core.async_runner", fromlist=["__file__"]).__file__)
    source_hash = sha256(source)
    assert recovery.native_main(work, target=25, start=18, missing=(15,)) == 0
    rows = database_snapshot(work / "shinka/programs.sqlite")
    receipt = read_json(work / "native-recovery.json")
    assert len(rows) == 24 and sum(row["correct"] == 1 for row in rows) == 22
    assert [row["generation"] for row in rows] == [*range(15), *range(16, 25)]
    assert [call["generation"] for call in continuation["calls"]] == list(range(18, 25))
    assert receipt["status"] == "complete"
    assert receipt["consumed_generations"] == list(range(25))
    assert receipt["completed_generations"] == receipt["next_generation_to_submit"] == 25
    assert receipt["verified_generations"] == receipt["attempted_generations"] == list(range(18, 25))
    assert receipt["graceful_return"] and receipt["rng_saved"]
    assert receipt["initial_rng_sha256"] == continuation["rng"]
    assert receipt["final_rng_sha256"] == sha256(work / "shinka/rng_state.json")
    assert receipt["final_rng_sha256"] != continuation["rng"]
    by_id = {row["id"]: row for row in rows}
    for row in rows[17:]:
        parent = by_id[row["parent_id"]]
        assert parent["correct"] == 1 and parent["generation"] < row["generation"]
    assert not (work / "shinka/gen_25").exists()
    assert sha256(source) == source_hash
    assert_inherited_unchanged(continuation)


@pytest.mark.parametrize("failure", ["evaluation", "proposal", "receipt"])
def test_continuation_failure_at_18_stops_before_19(continuation, monkeypatch, failure):
    work = continuation["work"]
    variable = {"evaluation": "SYNTHETIC_FAIL_GENERATION",
                "proposal": "SYNTHETIC_PROVIDER_FAIL", "receipt": "SYNTHETIC_TAMPER"}[failure]
    monkeypatch.setenv(variable, "18")
    assert recovery.native_main(work, target=25, start=18, missing=(15,)) == 1
    receipt = read_json(work / "native-recovery.json")
    assert receipt["status"] == "failed" and receipt["terminal_reason"]
    assert receipt["attempted_generations"] == [18]
    assert receipt["verified_generations"] == []
    assert receipt["consumed_generations"] == list(range(19))
    assert receipt["graceful_return"] and receipt["rng_saved"]
    assert receipt["initial_rng_sha256"] == continuation["rng"]
    assert receipt["final_rng_sha256"] == sha256(work / "shinka/rng_state.json")
    assert [call["generation"] for call in continuation["calls"]] == [18]
    assert not (work / "shinka/gen_19").exists()
    assert not (work / "recovery_slots/gen_19.json").exists()
    if failure == "proposal":
        assert (work / "shinka/gen_18/failure.json").is_file()
        assert not (work / "shinka/gen_18/results/synthetic-process.json").exists()
    assert_inherited_unchanged(continuation)


def test_continuation_failure_at_24_does_not_report_success(continuation, monkeypatch):
    work = continuation["work"]
    monkeypatch.setenv("SYNTHETIC_FAIL_GENERATION", "24")
    assert recovery.native_main(work, target=25, start=18, missing=(15,)) == 1
    receipt = read_json(work / "native-recovery.json")
    assert receipt["status"] == "failed" and receipt["terminal_reason"]
    assert receipt["completed_generations"] == 25
    assert receipt["consumed_generations"] == list(range(25))
    assert receipt["verified_generations"] == list(range(18, 24))
    assert receipt["graceful_return"] and receipt["rng_saved"]
    assert [call["generation"] for call in continuation["calls"]] == list(range(18, 25))
    assert not (work / "shinka/gen_25").exists()
    assert_inherited_unchanged(continuation)


def test_continuation_duplicate_18_reservation_never_invokes_provider(continuation):
    work = continuation["work"]
    recovery._reserve(work, 18)
    original = (work / "recovery_slots/gen_18.json").read_bytes()
    assert recovery.native_main(work, target=25, start=18, missing=(15,)) == 1
    receipt = read_json(work / "native-recovery.json")
    assert continuation["calls"] == []
    assert receipt["status"] == "failed"
    assert "FileExistsError" in receipt["terminal_reason"]
    assert receipt["consumed_generations"] == list(range(19))
    assert (work / "recovery_slots/gen_18.json").read_bytes() == original
    assert not (work / "shinka/gen_18").exists()
    assert_inherited_unchanged(continuation)
