"""Adaptive native stages bind immutable evaluations to sources, ancestry, and budgets."""

from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import sqlite3

import pytest

from shinka_crl import adaptive_search as search
from shinka_crl.reference_timing import artifact_hashes


def rewrite(path, change):
    value = search.read_json(path)
    change(value)
    search.write_json(path, value)


@pytest.fixture
def native(tmp_path, monkeypatch):
    """Synthetic native database and evaluator receipts, with no model or training processes."""
    repo, study, output = tmp_path / "repo", tmp_path / "study", tmp_path / "archive"
    repo.mkdir()
    study.mkdir()
    task_files = {"initial.py": "initial.py", "evaluate.py": "evaluate.py",
                  "shinka-subscription.yaml": "subscription.yaml"}
    for name in (*task_files.values(), "source.py"):
        (repo / name).write_text(f"# frozen fixture {name}\n")
    monkeypatch.setattr(search, "REPO_ROOT", repo)
    monkeypatch.setattr(search, "TASK_FILES", task_files)
    monkeypatch.setattr(search, "SOURCE_FILES", (*task_files.values(), "source.py"))
    monkeypatch.setattr(search, "runtime_scope", lambda _: nullcontext())
    evaluation = {
        "runtime": {"numerical_environment": {"PYTHONHASHSEED": None, "XLA_FLAGS": None}},
        "python": "/synthetic/python", "cpu_affinity": [0, 1], "timeout_seconds": 600,
        "source_sha256": {"evaluator.py": "frozen"},
        "profile": {"seeds": [4001, 4002, 4003], "episode_length": 500,
                    "ne": {"pop_size": 64, "num_evals": 3, "num_generations": 80}},
        "objective_version": "synthetic-objective",
    }
    search.write_json(study / "plan.json", evaluation)
    monkeypatch.setattr(search, "read_plan", lambda path: search.read_json(path / "plan.json"))
    monkeypatch.setattr(search, "runtime_fingerprint", lambda _: deepcopy(evaluation["runtime"]))
    monkeypatch.setattr(search, "runtime_identity", lambda _: {"frozen": "native runtime"})
    calls = []

    def request(path, source="identity", *, cached=True):
        path.mkdir(parents=True)
        program = f"# {source}\ndef update_sigma(sigma, stats, memory):\n    return sigma, memory\n"
        (path / "program.py").write_text(program)
        result = {
            "status": "complete", "request_id": path.name, "cache_hit": cached,
            "context_sha256": search.digest(evaluation), "slot_consumed": True,
            "evaluator_model_calls": 0,
            "candidate": {"source_sha256": search.sha256(path / "program.py"),
                          "program": {"canonical_ast_sha256": "same-identity-ast"}},
            "cache_key": "identity-cache", "cache_origin": "cache/identity-cache",
            "aggregate": {"scores": {"combined_score": {"mean": 0.5}}},
            "new_training_trials": 0 if cached else 3,
            "new_training_steps_nominal": 0 if cached else 23040000,
            "avoided_training_steps_nominal": 23040000 if cached else 0,
            "wall_seconds": 0.125,
        }
        search.write_json(path / "request.json", result)
        search.write_json(path / "correct.json", {"correct": True, "error": None})
        search.write_json(path / "metrics.json", {"combined_score": 0.5,
                          "public": {"cache_hit": cached}, "private": {"identity": "fixture"}})
        search.write_json(path / "receipt.json", artifact_hashes(path))
        return result

    def validate_request(study, path, evaluation):
        # Delegate integrity failures exactly as the real evaluator does; its deeper
        # training/cache semantics are covered by test_adaptive_evaluation.py.
        if artifact_hashes(path) != search.read_json(path / "receipt.json"):
            raise ValueError("Request artifacts changed")
        return search.read_json(path / "request.json")

    for name in search.CONTROL_IDS:
        request(study / "requests" / name)
    monkeypatch.setattr(search, "validate_request", validate_request)

    def mirror(path):
        for name in ("correct.json", "metrics.json"):
            (path.parent / name).write_bytes((path / name).read_bytes())

    def execute(command, *, log, env, timeout, shinka=None):
        calls.append({"command": command, "env": env, "timeout": timeout, "shinka": shinka})
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("synthetic subprocess\n")
        if shinka is None:
            path = Path(command[-1]) / "evaluation"
            request(path)
            mirror(path)
            return {"returncode": 0, "stop_reason": None, "wall_seconds": 0.25}
        shinka.mkdir(parents=True, exist_ok=True)
        target = int(command[-1])
        with sqlite3.connect(shinka / "programs.sqlite") as db:
            db.execute("CREATE TABLE IF NOT EXISTS programs (id TEXT PRIMARY KEY, generation INT, "
                       "parent_id TEXT,code TEXT,combined_score REAL,correct INT,"
                       "public_metrics TEXT,private_metrics TEXT,"
                       "archive_inspiration_ids TEXT,top_k_inspiration_ids TEXT)")
            count = db.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
            for generation in range(count, target):
                directory = shinka / f"gen_{generation}"
                path = directory / "results/evaluation"
                request(path, f"generation {generation}")
                mirror(path)
                code = (path / "program.py").read_text()
                (directory / "main.py").write_text(code)
                metrics = search.read_json(path / "metrics.json")
                independent = "db.parent_selection_strategy=best_of_n" in command
                parent_id = "id-0" if independent else f"id-{generation - 1}"
                db.execute("INSERT INTO programs VALUES (?,?,?,?,?,?,?,?,?,?)",
                           (f"id-{generation}", generation,
                            parent_id if generation else None, code,
                            metrics["combined_score"], 1, json.dumps(metrics["public"]),
                            json.dumps(metrics["private"]), "[]", "[]"))
                if generation:
                    with (output / "model_requests.jsonl").open("a") as writer:
                        for event in ("started", "codex_exec", "finished"):
                            writer.write(json.dumps({"request_id": f"slot-{generation}",
                                                     "event": event, "outcome": "success"}) + "\n")
        search.write_json(shinka / "rng_state.json", {"synthetic": target})
        return {"returncode": 0, "stop_reason": None, "wall_seconds": 1.0}

    monkeypatch.setattr(search, "monitored_run", execute)
    return {"repo": repo, "study": study, "output": output, "evaluation": evaluation,
            "calls": calls, "execute": execute, "request": request}


def run(native, **kwargs):
    return search.run_search(output=native["output"], study=native["study"], **kwargs)


def database_edit(native, statement, parameters=()):
    with sqlite3.connect(native["output"] / "shinka/programs.sqlite") as db:
        db.execute(statement, parameters)


def test_search_environment_restores_frozen_numerical_keys_without_static_hash_seed(monkeypatch, tmp_path):
    original = {"PYTHONHASHSEED": "20261002", "XLA_FLAGS": "ambient flags", "KEEP": "yes",
                "JAX_ENABLE_X64": "true", "PYTHONPATH": "ambient path"}
    monkeypatch.setattr(search, "subscription_environment", lambda *args: dict(original))
    frozen = {"PYTHONHASHSEED": None, "XLA_FLAGS": "frozen flags", "PYTHONPATH": None,
              "JAX_ENABLE_X64": "false", "JAX_DEFAULT_MATMUL_PRECISION": None}
    env = search.search_environment(tmp_path, {"runtime": {"numerical_environment": frozen}},
                                    "model", 600)
    assert "PYTHONHASHSEED" not in env and "PYTHONPATH" not in env
    assert env["XLA_FLAGS"] == "frozen flags" and env["JAX_ENABLE_X64"] == "false"
    assert env["KEEP"] == "yes" and env["SHINKA_ADAPTIVE_STUDY"] == str(tmp_path.resolve())
    assert original["PYTHONHASHSEED"] == "20261002"


def test_native_command_uses_frozen_staged_task_and_explicit_rng_checkpoint(tmp_path):
    command = search.native_command(tmp_path, 13)
    assert command[:3] == [search.sys.executable, "-m", "shinka_crl.search"]
    assert command[command.index("--native") + 1] == str(tmp_path / "shinka/rng_state.json")
    assert command[command.index("--task-dir") + 1] == str(tmp_path / "task")
    assert command[command.index("--config-fname") + 1] == "shinka-subscription.yaml"
    assert command[command.index("--num_generations") + 1] == "13"


def test_prepare_freezes_task_controls_and_sources_without_model_or_training(native):
    state = run(native, prepare_only=True)
    assert state["status"] == "prepared" and not native["calls"]
    plan = search.read_json(native["output"] / "plan.json")
    assert set(plan["controls"]) == set(search.CONTROL_IDS)
    assert plan["proposal_slots"] == 24 and plan["max_model_requests_per_slot"] == 1
    assert plan["evaluation_context_sha256"] == search.digest(native["evaluation"])
    search.validate_task(native["output"])


def test_independent_native_control_and_outer_seed_are_frozen_and_verified(native):
    run(native, arm="independent", outer_seed=6011)
    plan = search.read_json(native["output"] / "plan.json")
    assert plan["protocol"] == "adaptive-shinka-v2"
    assert plan["arm"] == "independent" and plan["outer_random_seed"] == 6011
    assert plan["native_selection"] == {"parent_selection_strategy": "best_of_n",
                                        "num_archive_inspirations": 0,
                                        "num_top_k_inspirations": 0}
    command = native["calls"][-1]["command"]
    assert command[command.index("--outer-seed") + 1] == "6011"
    assert "db.parent_selection_strategy=best_of_n" in command
    assert "db.num_archive_inspirations=0" in command
    assert "db.num_top_k_inspirations=0" in command
    summary = search.summarize(native["output"])
    assert summary["arm"] == "independent" and summary["outer_random_seed"] == 6011
    assert all(row["parent_id"] == "id-0" for row in summary["programs"][1:])
    assert "First adaptive" not in summary["interpretation"]
    before = len(native["calls"])
    for changed in ({"outer_seed": 6012, "arm": "independent"},
                    {"outer_seed": 6011, "arm": "evolutionary"}):
        with pytest.raises(ValueError, match="Frozen search plan"):
            run(native, resume=True, target=13, **changed)
    assert len(native["calls"]) == before


@pytest.mark.parametrize("column,value,error", [
    ("parent_id", "id-1", "identity parent"),
    ("archive_inspiration_ids", '["id-1"]', "archive inspirations"),
    ("top_k_inspiration_ids", '["id-1"]', "archive inspirations"),
])
def test_independent_control_rejects_evolving_ancestry_or_context(native, column, value, error):
    run(native, arm="independent", outer_seed=6011)
    database_edit(native, f"UPDATE programs SET {column}=? WHERE generation=2", (value,))
    with pytest.raises(ValueError, match=error):
        search.verified_rows(native["output"], search.read_json(native["output"] / "plan.json"),
                             native["evaluation"])


@pytest.mark.parametrize("kwargs,error", [
    ({"outer_seed": -1}, "Outer seed"), ({"outer_seed": 2**32}, "Outer seed"),
    ({"outer_seed": True}, "Outer seed"), ({"arm": "random"}, "Unknown adaptive search arm"),
])
def test_invalid_seed_or_arm_fails_before_any_work(native, kwargs, error):
    with pytest.raises(ValueError, match=error):
        run(native, **kwargs)
    assert not native["output"].exists() and not native["calls"]


def test_pinned_native_best_of_n_has_no_evolving_context(tmp_path):
    native_db = pytest.importorskip("shinka.database")
    sampler = pytest.importorskip("shinka.core.sampler")
    config = native_db.DatabaseConfig(db_path=str(tmp_path / "native.sqlite"), num_islands=1,
                                      **search.native_selection("independent"))
    db = native_db.ProgramDatabase(config, embedding_model=None)
    identity = "def update_sigma(sigma, stats, memory):\n    return sigma, memory\n"
    try:
        db.add(native_db.Program(id="identity", code=identity, correct=True,
                                 combined_score=0.25, public_metrics={"initial_metric": 0.25},
                                 text_feedback="Constant identity feedback", island_idx=0))
        db.add(native_db.Program(id="better-proposal", code="def marker_previous_proposal(): pass",
                                 correct=True, generation=1, parent_id="identity", island_idx=0,
                                 combined_score=0.99, public_metrics={"hidden_metric": 0.99},
                                 text_feedback="Hidden proposal feedback"))
        parent, archive, top_k, fix = db.sample_with_fix_mode(target_generation=2)
        assert parent.id == "identity" and archive == top_k == [] and fix is False
        prompt = sampler.PromptSampler(task_sys_msg="Fixed grammar", patch_types=["diff"],
                                       patch_type_probs=[1.0], use_text_feedback=True)
        system, message, patch_type = prompt.sample(parent, archive, top_k)
        assert system.startswith("Fixed grammar") and patch_type == "diff"
        assert identity in message and "initial_metric" in message
        assert "Constant identity feedback" in message
        assert "marker_previous_proposal" not in message and "hidden_metric" not in message
        assert "Hidden proposal feedback" not in message and "# Prior programs" not in message
    finally:
        db.close()


def test_staged_resume_keeps_existing_sources_and_counts_cache_reuse(native):
    first = run(native, target=5)
    assert first["status"] == "complete" and len(first["programs"]) == 5
    previous = (native["output"] / "shinka/gen_0/main.py").read_bytes()
    second = run(native, target=13, resume=True)
    assert second["status"] == "complete" and len(second["programs"]) == 13
    assert (native["output"] / "shinka/gen_0/main.py").read_bytes() == previous
    summary = search.summarize(native["output"])
    assert summary["programs_evaluated"] == 13
    assert summary["distinct_canonical_programs"] == 1
    assert summary["cache_hits"] == 13 and summary["new_training_trials"] == 0
    assert summary["proposal_usage"]["codex_cli_launches"] == 12
    assert [session["target"] for session in second["sessions"]] == [5, 13]
    assert len([call for call in native["calls"] if call["shinka"]]) == 2


@pytest.mark.parametrize("kind", ["source", "task", "database", "program", "rng", "ledger"])
def test_resume_rejects_frozen_source_task_and_completed_state_tampering(native, kind):
    run(native, target=5)
    if kind == "source":
        (native["repo"] / "source.py").write_text("changed source")
    elif kind == "task":
        (native["output"] / "task/evaluate.py").write_text("changed evaluator")
    elif kind == "database":
        database_edit(native, "UPDATE programs SET combined_score=0.9 WHERE generation=2")
    elif kind == "program":
        (native["output"] / "shinka/gen_1/main.py").write_text("different program")
    elif kind == "rng":
        (native["output"] / "shinka/rng_state.json").write_text("{}")
    else:
        with (native["output"] / "model_requests.jsonl").open("a") as writer:
            writer.write("{}\n")
    before = len(native["calls"])
    with pytest.raises(ValueError, match="changed"):
        run(native, target=13, resume=True)
    assert len(native["calls"]) == before


@pytest.mark.parametrize("target", [0, 1, 2, 4, 6, 26])
def test_undeclared_stage_is_rejected_before_any_work(native, target):
    with pytest.raises(ValueError, match="declared cumulative target"):
        run(native, target=target)
    assert not native["output"].exists() and not native["calls"]


def test_resume_requires_explicit_existing_archive_and_increasing_target(native):
    with pytest.raises(ValueError, match="missing archive"):
        run(native, resume=True)
    run(native, prepare_only=True)
    with pytest.raises(ValueError, match="explicit --resume"):
        run(native)
    run(native, target=5, resume=True)
    before = len(native["calls"])
    with pytest.raises(ValueError, match="exceed completed slots"):
        run(native, target=5, resume=True)
    assert len(native["calls"]) == before


@pytest.mark.parametrize("state_status", ["failed", "running"])
def test_interrupted_or_failed_stage_cannot_resume_automatically(native, state_status):
    run(native, prepare_only=True)
    rewrite(native["output"] / "state.json", lambda state: state.update(status=state_status))
    with pytest.raises(ValueError, match="requires review"):
        run(native, resume=True)
    assert not native["calls"]


def test_failed_prior_session_blocks_resume_even_if_outer_status_says_complete(native):
    run(native, prepare_only=True)
    rewrite(native["output"] / "state.json", lambda state: state.update(
        status="complete", sessions=[{"status": "failed"}]))
    with pytest.raises(ValueError, match="cannot be retried automatically"):
        run(native, resume=True)
    assert not native["calls"]


def test_generation_zero_only_resume_is_rejected(native, monkeypatch):
    run(native, prepare_only=True)
    row = {"generation": 0}
    rewrite(native["output"] / "state.json", lambda state: state.update(programs=[row]))
    monkeypatch.setattr(search, "validate_saved_state", lambda *args: None)
    with pytest.raises(ValueError, match="Generation-zero-only"):
        run(native, resume=True)
    assert not native["calls"]


def test_search_and_cache_roots_must_be_separate(native):
    with pytest.raises(ValueError, match="archive and evaluation cache separate"):
        search.run_search(output=native["study"] / "nested", study=native["study"])
    assert not native["calls"]


@pytest.mark.parametrize("column,value,error", [
    ("code", "different code", "Native code differs"),
    ("combined_score", 0.99, "score or feedback differs"),
    ("public_metrics", '{"unexpected": true}', "score or feedback differs"),
    ("private_metrics", '{"unexpected": true}', "score or feedback differs"),
    ("parent_id", "absent", "Invalid native ancestry"),
    ("parent_id", "id-4", "Invalid native ancestry"),
    ("correct", 0, "Failed scored candidate"),
    ("generation", 3, "Duplicate native generation"),
])
def test_verified_rows_reject_native_code_score_feedback_ancestry_mismatch(native, column, value, error):
    run(native, target=5)
    # Column names are fixed by this parametrization, never external data.
    database_edit(native, f"UPDATE programs SET {column}=? WHERE generation=2", (value,))
    plan = search.read_json(native["output"] / "plan.json")
    with pytest.raises(ValueError, match=error):
        search.verified_rows(native["output"], plan, native["evaluation"])


def test_verified_rows_reject_evaluator_receipt_tampering(native):
    run(native, target=5)
    path = native["output"] / "shinka/gen_1/results/evaluation/request.json"
    rewrite(path, lambda request: request.update(new_training_trials=7))
    with pytest.raises(ValueError, match="Request artifacts changed"):
        search.verified_rows(native["output"], search.read_json(native["output"] / "plan.json"),
                             native["evaluation"])


def test_verified_rows_reject_unreceipted_outer_result_change(native):
    run(native, target=5)
    path = native["output"] / "shinka/gen_1/results/metrics.json"
    rewrite(path, lambda metrics: metrics.update(combined_score=0.99))
    with pytest.raises(ValueError, match="Native result differs"):
        search.verified_rows(native["output"], search.read_json(native["output"] / "plan.json"),
                             native["evaluation"])


def test_initial_generation_must_reuse_identity_control(native):
    run(native, target=5)
    path = native["output"] / "shinka/gen_0/results/evaluation"
    rewrite(path / "request.json", lambda request: request.update(cache_key="different-control"))
    search.write_json(path / "receipt.json", artifact_hashes(path))
    with pytest.raises(ValueError, match="Initial slot must reuse"):
        search.verified_rows(native["output"], search.read_json(native["output"] / "plan.json"),
                             native["evaluation"])


def test_model_failure_preserves_failed_state_and_blocks_retry(native, monkeypatch):
    run(native, prepare_only=True)
    execute = native["execute"]

    def fail(command, **kwargs):
        if kwargs.get("shinka") is None:
            return execute(command, **kwargs)
        return {"returncode": -15, "stop_reason": "quota", "wall_seconds": 2.0}

    monkeypatch.setattr(search, "monitored_run", fail)
    with pytest.raises(ValueError, match="search stopped"):
        run(native, resume=True)
    state = search.read_json(native["output"] / "state.json")
    assert state["status"] == state["sessions"][-1]["status"] == "failed"
    assert state["sessions"][-1]["stop_reason"] == "quota"
    assert state["sessions"][-1]["wall_seconds"] == 2.0
    with pytest.raises(ValueError, match="requires review"):
        run(native, resume=True)
