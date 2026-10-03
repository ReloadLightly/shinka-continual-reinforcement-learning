"""Reserved handoff contracts are tested without opening reserved outcomes."""
import copy

import pytest

from shinka_crl import adaptive_validation as av
from shinka_crl.pilot import read_json, sha256, write_json
from shinka_crl.reference_timing import artifact_hashes


def candidate(name="selected", variant="ga_adaptive", sigma=None):
    return {"id": name, "variant": variant, "source": "unused.py", "source_sha256": "raw",
            "settings": None if sigma is None else {"sigma": sigma, "elite_ratio": .5},
            "program": {"canonical_ast_sha256": "ast"}}


def context():
    return {"profile": {"ne": {"pop_size": 64}}, "source_sha256": {"src/shinka_crl/adaptive.py": "adapter"}}


def test_rank_exact_mean_then_generation():
    rows = [{"generation": 4, "development_score": .500000000000001},
            {"generation": 0, "development_score": .5},
            {"generation": 1, "development_score": .5}]
    assert [r["generation"] for r in av.rank_programs(rows)] == [4, 0, 1]
    assert av.rank_programs(rows[1:])[0]["generation"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True])
def test_rank_rejects_non_numeric(value):
    with pytest.raises(ValueError):
        av.rank_programs([{"generation": 0, "development_score": value}])


def test_dedup_retains_memberships_and_raw_sources():
    left, right = candidate(), candidate("identity")
    right["source_sha256"] = "formatting-different"
    unique = av.deduplicate([left, right], context())
    assert len(unique) == 1
    assert unique[0]["memberships"] == ["selected", "identity"]
    assert right["source_sha256"] == "formatting-different"


@pytest.mark.parametrize("change", ["sigma", "archive", "method", "context"])
def test_full_recipe_not_ast_alone(change):
    left, right = candidate(), candidate("identity")
    if change == "sigma":
        right["settings"] = {"sigma": .25, "elite_ratio": .5}
    elif change == "archive":
        right["settings"] = {"sigma": .5, "elite_ratio": .25}
    elif change == "method":
        right["variant"] = "ga_static"
    if change == "context":
        different = copy.deepcopy(context())
        different["profile"]["ne"]["pop_size"] = 32
        assert av.recipe(left, context()) != av.recipe(right, different)
    else:
        assert len(av.deduplicate([left, right], context())) == 2


def test_focus_centroid_keeps_fixed_population_budget():
    value = av.recipe(candidate(variant="ga_focus"), context())
    assert value["archive_size"] == 32 and value["offspring_count"] == 31
    assert value["focus_settings"] == av.FOCUS_SEARCHER_KWARGS
    assert value["centroid_inside_population_budget"] is True


def test_profile_context_is_disjoint():
    base = {"upstream": "/upstream", "python": "/python", "cpu_affinity": [0, 1], "runtime": {}}
    reserved = av._context(base)
    diagnostic = av._context(base, diagnostic=True)
    assert reserved["profile"]["seeds"] == [5001, 5002, 5003, 5004, 5005]
    assert reserved["profile"]["ne"]["num_generations"] == 320
    assert diagnostic["profile"]["seeds"] == [3001]
    assert diagnostic["profile"]["ne"]["num_generations"] == 8
    assert reserved["objective_weights"] == diagnostic["objective_weights"] == {"active": .5, "previous": .5}


def test_endpoint_rejects_partial_before_verifier(tmp_path):
    closure = tmp_path / "closure.json"
    write_json(closure, {"status": "closed", "target_slots": 25, "slots_consumed": 21,
                         "remaining_generations": [21, 22, 23, 24], "resolution": "premature"})
    with pytest.raises(ValueError, match="complete allocation"):
        av.endpoint_candidates(closure)


def test_published_report_tamper(tmp_path):
    write_json(tmp_path / "summary.json", {"score": 0})
    write_json(tmp_path / "checksums.json", {"published_sha256": {"summary.json": sha256(tmp_path / "summary.json")}})
    closure = {"endpoint_summary_sha256": sha256(tmp_path / "summary.json"),
               "endpoint_checksums_sha256": sha256(tmp_path / "checksums.json")}
    assert av._verify_published_report(tmp_path, closure) == {"score": 0}
    write_json(tmp_path / "summary.json", {"score": 1})
    with pytest.raises(ValueError, match="differs from closure"):
        av._verify_published_report(tmp_path, closure)


def test_cross_context_receipt_forbidden(tmp_path):
    write_json(tmp_path / "summary.json", {"status": "failed", "context_sha256": "old", "trial": {}})
    write_json(tmp_path / "receipt.json", artifact_hashes(tmp_path))
    with pytest.raises(ValueError, match="Cross-profile"):
        av.verify_attempt(tmp_path, {"new": "context"}, {})


def test_failed_training_completion_stays_accounted(tmp_path, monkeypatch):
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    attempt = tmp_path / "attempt"
    plan = {"unique_recipes": [{"recipe_key": "key", "candidate": {"source": None}}],
            "timeout_seconds": 1800, "upstream": "/upstream", "python": "/python", "posthoc_episodes": 10}
    row = {"recipe_key": "key", "seed": 3001, "eval_seed": 903001}

    def train(plan, candidate, seed, training, program):
        training.mkdir()
        write_json(training / "manifest.json", {"status": "complete", "wall_seconds": 1.})

    monkeypatch.setattr(av.ae, "_run_trial", train)
    monkeypatch.setattr(av, "run_analysis", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("analysis failed")))
    summary = av._execute_trial(frozen=frozen, attempt=attempt, plan=plan, row=row,
                                deadline=av.time.monotonic() + 20)
    assert summary["status"] == "failed"
    assert read_json(attempt / "training/manifest.json")["status"] == "complete"
    assert artifact_hashes(attempt) == read_json(attempt / "receipt.json")
    assert av.verify_attempt(attempt, plan, row) == summary


def test_deadline_does_not_launch_training(tmp_path, monkeypatch):
    monkeypatch.setattr(av.ae, "_run_trial", lambda *args: pytest.fail("should not launch"))
    plan = {"unique_recipes": [{"recipe_key": "key", "candidate": {"source": None}}], "timeout_seconds": 1800}
    summary = av._execute_trial(frozen=tmp_path, attempt=tmp_path / "attempt", plan=plan,
                                row={"recipe_key": "key"}, deadline=av.time.monotonic() - 1)
    assert summary["status"] == "failed" and "ceiling" in summary["error"]


def test_resume_rejects_running_session(tmp_path):
    plan = {}
    av._seal_state(tmp_path, {"context_sha256": av.ae.digest(plan), "sessions": [{"status": "running"}]})
    with pytest.raises(ValueError, match="accounting review"):
        av._state(tmp_path, plan)


def test_resume_rejects_unrecorded_attempt(tmp_path):
    path = tmp_path / "trials/000/attempt_0001"
    path.mkdir(parents=True)
    write_json(path / "summary.json", {})
    plan = {}
    av._seal_state(tmp_path, {"context_sha256": av.ae.digest(plan), "sessions": [], "attempts": []})
    with pytest.raises(ValueError, match="Unrecorded"):
        av._state(tmp_path, plan)


@pytest.fixture
def frozen_diagnostic(tmp_path, monkeypatch):
    development = {"upstream": "/upstream", "python": "/python", "cpu_affinity": [0, 1],
                   "runtime": {"backend": "fake"}, "controls": av.ae.controls()}
    monkeypatch.setattr(av.ae, "read_plan", lambda path: development)
    monkeypatch.setattr(av, "verify_upstream", lambda path: None)
    monkeypatch.setattr(av, "native_sources", lambda path: {"source/run.py": "pinned"})
    from contextlib import nullcontext
    monkeypatch.setattr(av.ae, "runtime_scope", lambda affinity: nullcontext())
    monkeypatch.setattr(av.ae, "runtime_fingerprint", lambda python: development["runtime"])
    frozen = tmp_path / "frozen"
    av.freeze_diagnostic(study=tmp_path / "study", output=frozen)
    return frozen


def test_frozen_order_budget_and_sources(frozen_diagnostic):
    plan = av.read_plan(frozen_diagnostic)
    assert [r["memberships"] for r in plan["trial_order"]] == [["arithmetic"], ["focus"], ["static_shinka11"]]
    assert [(r["seed"], r["trial"], r["eval_seed"]) for r in plan["trial_order"]] == [(3001, 3002, 903001)] * 3
    assert plan["planned_nominal_training_steps"] == 96000
    assert plan["planned_fresh_evaluation_episodes"] == 900
    assert plan["first_block_trials"] == 3
    assert plan["unique_recipes"][1]["recipe"]["focus_settings"] == av.FOCUS_SEARCHER_KWARGS


def test_frozen_source_tamper(frozen_diagnostic):
    source = frozen_diagnostic / "programs/arithmetic.py"
    source.write_text(source.read_text() + "\n#changed\n")
    with pytest.raises(ValueError, match="handoff evidence"):
        av.read_plan(frozen_diagnostic)


@pytest.mark.parametrize("field,value", [("profile_seed", 5001), ("trial", 9999), ("eval_seed", 9999),
                                         ("planned_trials", 2), ("timeout_seconds", 1799)])
def test_changed_contract_rejected_even_with_new_receipt(frozen_diagnostic, field, value):
    plan = read_json(frozen_diagnostic / "plan.json")
    if field == "profile_seed":
        plan["profile"]["seeds"] = [value]
    elif field in ("trial", "eval_seed"):
        plan["trial_order"][0][field] = value
    else:
        plan[field] = value
    write_json(frozen_diagnostic / "plan.json", plan)
    write_json(frozen_diagnostic / "receipt.json", artifact_hashes(frozen_diagnostic))
    with pytest.raises(ValueError):
        av.read_plan(frozen_diagnostic)


def test_resume_requires_bound_review(frozen_diagnostic, tmp_path):
    plan = av.read_plan(frozen_diagnostic)
    output = tmp_path / "output"
    output.mkdir()
    av._seal_state(output, {"context_sha256": av.ae.digest(plan), "attempts": [],
                           "sessions": [{"status": "complete", "wall_seconds": 1.}]})
    with pytest.raises(ValueError, match="Resume requires review"):
        av.run(frozen=frozen_diagnostic, output=output)


def test_first_block_cannot_expand(frozen_diagnostic, tmp_path):
    with pytest.raises(ValueError, match="First block"):
        av.run(frozen=frozen_diagnostic, output=tmp_path / "output", max_trials=4)


def test_cumulative_ceiling_prevents_launch(frozen_diagnostic, tmp_path):
    plan = av.read_plan(frozen_diagnostic)
    output = tmp_path / "output"
    output.mkdir()
    av._seal_state(output, {"context_sha256": av.ae.digest(plan), "attempts": [],
                           "sessions": [{"status": "complete", "wall_seconds": 14400.}]})
    with pytest.raises(ValueError, match="Cumulative"):
        av.run(frozen=frozen_diagnostic, output=output, review_sha256=sha256(output / "state.json"),
               review_reason="reviewed exact completed state")


def test_runner_failure_review_retry_and_training_cost(frozen_diagnostic, tmp_path, monkeypatch):
    output = tmp_path / "output"
    calls = []
    scores = {"combined_score": .4, "active_score": .5, "previous_score": .3,
              "reference_metrics": {key: 0. for key in ("learning_accuracy", "forgetting",
                  "learning_minus_forgetting", "zero_shot_transfer", "normalized_curve_average")}}

    def execute(*, frozen, attempt, plan, row, deadline):
        attempt.mkdir(parents=True)
        (attempt / "training").mkdir()
        write_json(attempt / "training/manifest.json", {"status": "complete", "wall_seconds": 2.})
        result = {**row, "score": scores, "fresh_evaluation_episodes": 300}
        summary = {"status": "failed" if not calls else "complete", "context_sha256": av.ae.digest(plan),
                   "trial": row, "result": result}
        calls.append(row["index"])
        write_json(attempt / "summary.json", summary)
        write_json(attempt / "receipt.json", artifact_hashes(attempt))
        return summary

    monkeypatch.setattr(av, "_execute_trial", execute)
    monkeypatch.setattr(av, "_score_attempt", lambda attempt, plan, row: read_json(attempt / "summary.json")["result"])
    first = av.run(frozen=frozen_diagnostic, output=output)
    assert first["allocated_training_trials"] == first["completed_training_trials"] == 1
    assert first["scored_trials"] == 0 and len(first["failed_attempts"]) == 1
    review = sha256(output / "state.json")
    with pytest.raises(ValueError, match="explicit reviewed retry"):
        av.run(frozen=frozen_diagnostic, output=output, review_sha256=review, review_reason="integrity review")
    final = av.run(frozen=frozen_diagnostic, output=output, review_sha256=review,
                   review_reason="accounted training then analysis failure", retry_failed=True)
    assert calls == [0, 0, 1, 2]
    assert final["status"] == "complete" and final["scored_trials"] == 3
    assert final["completed_training_trials"] == final["allocated_training_trials"] == 4
    assert final["allocated_nominal_training_steps"] == 128000
    assert (output / "trials/000/attempt_0001/receipt.json").exists()
    assert (output / "trials/000/attempt_0002/receipt.json").exists()


def test_unresolved_complete_allocation_cannot_close(tmp_path, monkeypatch):
    from shinka_crl import adaptive_endpoint
    monkeypatch.setattr(adaptive_endpoint, "summarize", lambda path: {"endpoint_status": "unresolved"})
    closure = tmp_path / "closure.json"
    write_json(closure, {"status": "closed", "target_slots": 25, "slots_consumed": 25,
                         "remaining_generations": [], "resolution": "arbitrary text",
                         "endpoint_work": str(tmp_path / "archive")})
    with pytest.raises(ValueError, match="unresolved work"):
        av.endpoint_candidates(closure)


@pytest.mark.parametrize("location", ["inside_frozen", "inside_output", "ancestor"])
def test_export_requires_independent_paths(frozen_diagnostic, tmp_path, location):
    output = tmp_path / "output"
    output.mkdir()
    if location == "inside_frozen":
        report = frozen_diagnostic / "report"
    elif location == "inside_output":
        report = output / "report"
    else:
        report = tmp_path
    with pytest.raises(ValueError, match="independent paths"):
        av.export(frozen=frozen_diagnostic, output=output, report=report)


def test_whole_trial_deadline_covers_external_metadata_probe():
    import signal
    import subprocess
    import sys
    started = av.time.monotonic()
    with pytest.raises(TimeoutError, match="session deadline"):
        with av.trial_deadline(av.time.monotonic() + .05):
            subprocess.run([sys.executable, "-c", "import time;time.sleep(10)"], check=True)
    assert av.time.monotonic() - started < 2
    assert signal.getitimer(signal.ITIMER_REAL) == (0., 0.)


def test_failed_attempt_raw_episodes_are_counted(frozen_diagnostic, tmp_path, monkeypatch):
    plan = av.read_plan(frozen_diagnostic)
    output = tmp_path / "output"
    attempt = output / "trials/000/attempt_0001"
    (attempt / "training").mkdir(parents=True)
    (attempt / "analysis").mkdir()
    write_json(attempt / "training/manifest.json", {"status": "complete", "wall_seconds": 1.})
    write_json(attempt / "analysis/manifest.json", {"status": "failed", "wall_seconds": 2.})
    write_json(attempt / "analysis/evaluation.json", {"per_task": [{"returns": [1., 2.], "prev_returns": [3., 4.]}]})
    state = {"sessions": [{"wall_seconds": 3.}], "attempts": [{"path": "trials/000/attempt_0001"}]}
    monkeypatch.setattr(av, "_state", lambda output, plan: (state, {}, [{"failed": True}]))
    summary = av.summarize(frozen=frozen_diagnostic, output=output)
    assert summary["realized_fresh_evaluation_episodes"] == 4
    assert summary["scored_fresh_evaluation_episodes"] == 0
    assert summary["analysis_wall_seconds"] == 2.
    assert summary["completed_nominal_training_steps"] == plan["trial_steps_nominal"]
    assert "no reserved outcomes" in summary["interpretation"]


def test_native_evaluation_episodes_count_before_copy(frozen_diagnostic, tmp_path, monkeypatch):
    output = tmp_path / "output"
    directory = output / "trials/000/attempt_0001/analysis"
    (directory / "evaluation-input").mkdir(parents=True)
    write_json(directory / "evaluation-input/evaluation.json", {"per_task": [{"returns": [1., 2.]}]})
    state = {"sessions": [], "attempts": [{"path": "trials/000/attempt_0001"}]}
    monkeypatch.setattr(av, "_state", lambda output, plan: (state, {}, []))
    assert av.summarize(frozen=frozen_diagnostic, output=output)["realized_fresh_evaluation_episodes"] == 2
    write_json(directory / "evaluation.json", {"per_task": [{"returns": [1., 2.]}]})
    assert av.summarize(frozen=frozen_diagnostic, output=output)["realized_fresh_evaluation_episodes"] == 2
    write_json(directory / "evaluation.json", {"per_task": [{"returns": [3.]}]})
    with pytest.raises(ValueError, match="copies disagree"):
        av.summarize(frozen=frozen_diagnostic, output=output)
