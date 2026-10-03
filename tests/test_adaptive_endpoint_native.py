"""Native final-slot orchestration, with synthetic model and evaluation calls."""
import os

import pytest

from test_adaptive_continuation_native import native as native, continuation as continuation
from test_adaptive_recovery_native import assert_no_live_evaluators
from shinka_crl import adaptive_recovery_native as recovery
from shinka_crl.pilot import read_json, sha256, write_json
from shinka_crl.reference_timing import artifact_hashes
from shinka_crl.search import database_snapshot


def retain_attempt(work, history):
    history.mkdir(parents=True)
    for name in ("native-recovery.json", "recovery_slots", "recovery_requests"):
        source = work / name
        if source.exists():
            source.rename(history / name)


@pytest.fixture
def endpoint(continuation, monkeypatch):
    from shinka.llm.llm import AsyncLLMClient
    from shinka.llm.providers import QueryResult

    work, calls = continuation["work"], continuation["calls"]
    monkeypatch.setenv("SYNTHETIC_FAIL_GENERATION", "20")
    assert recovery.native_main(work, target=25, start=18, missing=(15,)) == 1
    receipt = read_json(work / "native-recovery.json")
    assert receipt["attempted_generations"] == [18, 19, 20]
    assert receipt["verified_generations"] == [18, 19]
    assert receipt["consumed_generations"] == list(range(21))
    assert receipt["graceful_return"] and receipt["rng_saved"]
    monkeypatch.delenv("SYNTHETIC_FAIL_GENERATION")
    (work / "recovery_requests").mkdir()
    for generation in (18, 19, 20):
        write_json(work / f"recovery_requests/gen_{generation}.json", {"generation": generation})
    history = work.parent / "history"
    retain_attempt(work, history / "continuation")
    rows = database_snapshot(work / "shinka/programs.sqlite")
    hashes = {g: artifact_hashes(work / f"shinka/gen_{g}") for g in range(21)}
    calls.clear()
    first = [21]
    code = (work / "task/initial.py").read_text()

    async def query(self, msg, system_msg, **kwargs):
        generation = int(os.environ["SHINKA_RECOVERY_GENERATION"])
        assert first[0] <= generation < 25
        assert read_json(work / f"recovery_slots/gen_{generation}.json")["generation"] == generation
        if generation > first[0]:
            previous = database_snapshot(work / "shinka/programs.sqlite")[-1]
            assert previous["generation"] == generation - 1 and previous["correct"] == 1
            assert (work / f"shinka/gen_{generation - 1}/results/evaluation/receipt.json").is_file()
        calls.append({"generation": generation})
        if os.environ.get("SYNTHETIC_PROVIDER_FAIL") == str(generation):
            return None
        content = ("<NAME>synthetic</NAME><DESCRIPTION>Endpoint fixture</DESCRIPTION>\n"
                   "```python\n" + code.replace("return sigma, memory",
                                               f"return sigma * {generation}.0, memory") + "```\n")
        return QueryResult(content=content, msg=msg, system_msg=system_msg,
                           new_msg_history=[], model_name="headless/codex?effort=medium",
                           kwargs={}, input_tokens=1, output_tokens=1)

    monkeypatch.setattr(AsyncLLMClient, "query", query)
    return {"work": work, "calls": calls, "first": first, "rows": rows, "artifacts": hashes,
            "history": history, "history_sha256": artifact_hashes(history),
            "rng": sha256(work / "shinka/rng_state.json")}


def assert_history(state):
    work = state["work"]
    assert database_snapshot(work / "shinka/programs.sqlite")[:20] == state["rows"]
    assert {g: artifact_hashes(work / f"shinka/gen_{g}") for g in range(21)} == state["artifacts"]
    assert artifact_hashes(state["history"]) == state["history_sha256"]
    assert_no_live_evaluators(work)


def test_last_four_slots_finish_without_reusing_any_failure(endpoint):
    work = endpoint["work"]
    assert recovery.native_main(work, target=25, start=21, missing=(15,)) == 0
    rows = database_snapshot(work / "shinka/programs.sqlite")
    receipt = read_json(work / "native-recovery.json")
    assert [c["generation"] for c in endpoint["calls"]] == [21, 22, 23, 24]
    assert receipt["status"] == "complete"
    assert receipt["consumed_generations"] == list(range(25))
    assert receipt["verified_generations"] == [21, 22, 23, 24]
    assert len(rows) == 24 and sum(bool(r["correct"]) for r in rows) == 21
    assert receipt["initial_rng_sha256"] == endpoint["rng"]
    assert receipt["final_rng_sha256"] == sha256(work / "shinka/rng_state.json")
    assert not (work / "shinka/gen_25").exists()
    assert_history(endpoint)


@pytest.mark.parametrize("failure", ["evaluation", "proposal", "receipt"])
def test_new_failure_at21_stops_before22_and_saves_rng(endpoint, monkeypatch, failure):
    work = endpoint["work"]
    key = {"evaluation": "SYNTHETIC_FAIL_GENERATION", "proposal": "SYNTHETIC_PROVIDER_FAIL",
           "receipt": "SYNTHETIC_TAMPER"}[failure]
    monkeypatch.setenv(key, "21")
    assert recovery.native_main(work, target=25, start=21, missing=(15,)) == 1
    receipt = read_json(work / "native-recovery.json")
    assert receipt["attempted_generations"] == [21]
    assert receipt["status"] == "failed" and receipt["graceful_return"] and receipt["rng_saved"]
    assert not (work / "recovery_slots/gen_22.json").exists()
    assert not (work / "shinka/gen_22").exists()
    assert receipt["final_rng_sha256"] == sha256(work / "shinka/rng_state.json")
    assert_history(endpoint)


def test_final_slot_failure_exhausts_budget_without_claiming_native_success(endpoint, monkeypatch):
    work = endpoint["work"]
    monkeypatch.setenv("SYNTHETIC_FAIL_GENERATION", "24")
    assert recovery.native_main(work, target=25, start=21, missing=(15,)) == 1
    receipt = read_json(work / "native-recovery.json")
    assert receipt["status"] == "failed"
    assert receipt["consumed_generations"] == list(range(25))
    assert receipt["verified_generations"] == [21, 22, 23]
    assert [c["generation"] for c in endpoint["calls"]] == [21, 22, 23, 24]
    assert not (work / "shinka/gen_25").exists()
    assert_history(endpoint)


def test_duplicate21_reservation_prevents_provider_invocation(endpoint):
    work = endpoint["work"]
    recovery._reserve(work, 21)
    original = (work / "recovery_slots/gen_21.json").read_bytes()
    assert recovery.native_main(work, target=25, start=21, missing=(15,)) == 1
    assert endpoint["calls"] == []
    assert (work / "recovery_slots/gen_21.json").read_bytes() == original
    assert not (work / "shinka/gen_21").exists()
    assert_history(endpoint)
