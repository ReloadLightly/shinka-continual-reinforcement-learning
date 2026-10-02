"""Exercise frozen controls, exact resume receipts and failure accounting without inference."""

import json
from pathlib import Path
import sqlite3

import pytest

from shinka_crl import search


def test_random_pool_is_frozen_distinct_and_valid():
    pool = search.random_pool()
    assert pool == search.random_pool(20261002)
    assert pool != search.random_pool(20261003)
    assert pool["count"] == len(pool["entries"]) == 24
    assert len({tuple(e["effective_key"]) for e in pool["entries"]}) == 24
    for entry in pool["entries"]:
        assert 0.001 <= entry["settings"]["sigma"] <= 2
        assert 0.05 <= entry["settings"]["elite_ratio"] <= 0.95
        assert entry["effective_key"] != [0.5, 32]


def test_effective_identity_accounts_for_float32_rounding_and_archive_floor():
    assert search.effective_key({"sigma": 0.1, "elite_ratio": 0.501}) == (
        search.effective_key({"sigma": 0.1000000001, "elite_ratio": 0.510}))
    assert search.effective_key({"sigma": 0.1, "elite_ratio": 0.520})[1] == 33


@pytest.fixture
def runtime(monkeypatch):
    calls = []
    monkeypatch.setattr(search.os, "sched_getaffinity", lambda pid: {0, 1})
    monkeypatch.setattr(search.os, "sched_setaffinity", lambda *args: None)
    monkeypatch.setattr(search, "verify_upstream", lambda path: search.UPSTREAM_COMMIT)
    monkeypatch.setattr(search, "runtime_identity", lambda env: {"fixed": "runtime"})

    def execute(command, *, log, env, timeout, shinka=None):
        calls.append(command)
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("mock execution\n")
        if shinka:
            target = int(command[-1])
            shinka.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(shinka / "programs.sqlite") as connection:
                connection.execute("CREATE TABLE IF NOT EXISTS programs (id TEXT PRIMARY KEY, "
                                   "generation INT,parent_id TEXT,code TEXT,combined_score REAL,"
                                   "correct INT,public_metrics TEXT,private_metrics TEXT)")
                count = connection.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
                for index in range(count, target):
                    connection.execute("INSERT INTO programs VALUES (?,?,?,?,?,?,?,?)",
                                       (str(index), index, str(index - 1) if index else None,
                                        "candidate", 0.5, 1, "{}", "{}"))
                    path = shinka / f"gen_{index}"
                    path.mkdir()
                    (path / "main.py").write_text("candidate")
            search.write_json(shinka / "rng_state.json", {"mock": target})
        else:
            output = Path(command[-1])
            output.mkdir(parents=True)
            search.write_json(output / "correct.json", {"correct": True})
            search.write_json(output / "metrics.json", {"combined_score": 0.5})
        return {"returncode": 0, "stop_reason": None, "wall_seconds": 1.0}

    monkeypatch.setattr(search, "monitored_run", execute)
    return calls


def test_preparation_freezes_all_controls_before_first_proposal(tmp_path, runtime):
    output = tmp_path / "search"
    state = search.run_search(output=output, prepare_only=True)
    assert state["status"] == "prepared" and not runtime
    assert search.read_json(output / "random_pool.json")["count"] == 24
    assert len(list((output / "random_programs").glob("*.py"))) == 24
    assert search.read_json(output / "plan.json")["nominal_training_steps_per_candidate"] == 23040000


def test_native_stage_resume_preserves_generations_and_random_prefix(tmp_path, runtime):
    output = tmp_path / "search"
    search.run_search(output=output, prepare_only=True)
    first = search.run_search(output=output, target=2, resume=True)
    assert len(first["programs"]) == 2
    source = (output / "shinka/gen_0/main.py").read_bytes()
    second = search.run_search(output=output, target=5, resume=True)
    assert len(second["programs"]) == 5
    assert (output / "shinka/gen_0/main.py").read_bytes() == source
    random = search.run_search(output=output, random_count=2, resume=True)
    assert random["random_completed"] == 2
    final = search.run_search(output=output, random_count=4, resume=True)
    assert final["random_completed"] == 4 and len(runtime) == 6
    assert [s["endpoint"] for s in final["sessions"]] == [2, 5, 1, 2, 3, 4]


@pytest.mark.parametrize("changed", ["source", "program", "database", "rng", "pool"])
def test_resume_rejects_modified_frozen_or_completed_artifacts(tmp_path, runtime, monkeypatch, changed):
    output = tmp_path / "search"
    search.run_search(output=output, target=2)
    if changed == "source":
        monkeypatch.setattr(search, "runtime_identity", lambda env: {"changed": "runtime"})
    elif changed == "program":
        (output / "shinka/gen_0/main.py").write_text("tamper")
    elif changed == "database":
        with sqlite3.connect(output / "shinka/programs.sqlite") as connection:
            connection.execute("UPDATE programs SET combined_score=0.99")
    elif changed == "rng":
        (output / "shinka/rng_state.json").write_text("{}")
    else:
        (output / "random_programs/candidate_001.py").write_text("tamper")
    with pytest.raises(ValueError, match="changed"):
        search.run_search(output=output, target=5, resume=True)
    assert len(runtime) == 1


def test_failed_model_slot_cannot_be_silently_retried(tmp_path, runtime, monkeypatch):
    output = tmp_path / "search"
    search.run_search(output=output, prepare_only=True)
    def failed_request(*args, **kwargs):
        (output / "model_requests.jsonl").write_text(json.dumps(
            {"request_id": "failed-slot-1", "event": "started"}) + "\n")
        return {"returncode": -15, "stop_reason": "quota", "wall_seconds": 2}

    monkeypatch.setattr(search, "monitored_run", failed_request)
    with pytest.raises(ValueError, match="stage failed"):
        search.run_search(output=output, target=2, resume=True)
    state = search.read_json(output / "state.json")
    assert state["sessions"][0]["status"] == "failed"
    assert state["sessions"][0]["stop_reason"] == "quota"
    assert state["model_requests_sha256"] == search.sha256(output / "model_requests.jsonl")
    with pytest.raises(ValueError, match="no automatic slot retry"):
        search.run_search(output=output, target=2, resume=True)


def test_monitor_stops_on_terminal_native_failure(tmp_path):
    native = tmp_path / "shinka"
    (native / "gen_1").mkdir(parents=True)
    (native / "gen_1/failure.json").write_text(json.dumps({"failure_reason": "quota"}))
    result = search.monitored_run(
        [search.sys.executable, "-c", "import time; time.sleep(30)"],
        log=tmp_path / "session.log", env=dict(search.os.environ), timeout=5, shinka=native)
    assert result["returncode"] != 0
    assert result["stop_reason"].startswith("terminal proposal failure")
    assert result["wall_seconds"] < 5


def test_subscription_environment_excludes_key_route(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://invalid.example")
    monkeypatch.setenv("SHINKA_CRL_PYTHON", "/wrong/python")
    monkeypatch.setenv("SHINKA_CRL_UPSTREAM", "/wrong/source")
    env = search.subscription_environment("gpt-6.1-sol", 600)
    assert "OPENAI_API_KEY" not in env and "OPENAI_BASE_URL" not in env
    assert env["SHINKA_LLM_MAX_RETRIES"] == "1"
    assert env["JAX_PLATFORMS"] == "cpu"
    assert env["SHINKA_CRL_PROFILE"] == "search"
    assert env["SHINKA_CRL_PYTHON"] == str(search.DEFAULT_PYTHON)
    assert env["SHINKA_CRL_UPSTREAM"] == str(search.DEFAULT_UPSTREAM)
