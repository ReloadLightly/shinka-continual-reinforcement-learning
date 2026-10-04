"""Closed-endpoint adaptive handoff and a separate, bounded reserved comparison.

No search/control plan is changed. Every completed trial is re-scored from its
training and fresh checkpoint evidence before reuse. Interrupted attempts are
immutable and require an explicit reviewed retry in a new directory.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import math
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import time

from shinka_crl import adaptive_evaluation as ae
from shinka_crl.adaptive import GRAMMAR_VERSION, load_program
from shinka_crl.analysis import run_analysis
from shinka_crl.baseline_contract import FOCUS_SEARCHER_KWARGS
from shinka_crl.experiment import REPO_ROOT, UPSTREAM_COMMIT, load_profile, nominal_training_steps, verify_upstream
from shinka_crl.pilot import read_json, require, sha256, write_json
from shinka_crl.reference_timing import artifact_files, artifact_hashes, utc_now
from shinka_crl.search import SEARCH_THREAD_ENV

VERSION = "adaptive-reserved-validation-v1"
REPEATED_VERSION = "adaptive-repeated-validation-v1"
SOURCE_FILES = (*ae.SOURCE_FILES, "src/shinka_crl/adaptive_validation.py",
                "scripts/run_adaptive_validation.py", "scripts/report_adaptive_validation.py")
CONDITIONS = ("selected", *ae.CONTROL_IDS)
LIMIT = 4 * 3600


def source_hashes():
    return {name: sha256(REPO_ROOT / name) for name in SOURCE_FILES}


def native_sources(upstream: Path) -> dict:
    names = subprocess.check_output(["git", "-C", str(upstream), "ls-files", "source", "scripts"], text=True).splitlines()
    return {name: sha256(upstream / name) for name in names if Path(name).suffix in {".py", ".yaml", ".yml"}}


@contextmanager
def trial_deadline(deadline: float):
    """Bound metadata probes and subprocesses together, including native helpers."""
    remaining = deadline - time.monotonic()
    require(remaining > 0, "Cumulative validation session ceiling reached")
    require(signal.getitimer(signal.ITIMER_REAL) == (0., 0.), "An existing process timer prevents deadline enforcement")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise TimeoutError("Cumulative validation session deadline reached")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.)
        signal.signal(signal.SIGALRM, previous)


def _published(path: Path) -> None:
    """Refuse reserved outcomes until the exact preregistration is pushed."""
    relative = path.resolve().relative_to(REPO_ROOT)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    for item in artifact_files(path):
        name = str(relative / item.relative_to(path))
        saved = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=REPO_ROOT)
        require(saved == item.read_bytes(), "Commit the exact frozen handoff before execution")
    upstream = subprocess.check_output(["git", "rev-parse", "@{upstream}"], cwd=REPO_ROOT, text=True).strip()
    require(subprocess.run(["git", "merge-base", "--is-ancestor", revision, upstream],
                           cwd=REPO_ROOT, check=False).returncode == 0,
            "Push the frozen handoff before reserved execution")


def _verify_published_report(report: Path, closure: dict) -> dict:
    require(sha256(report / "summary.json") == closure["endpoint_summary_sha256"]
            and sha256(report / "checksums.json") == closure["endpoint_checksums_sha256"],
            "Endpoint publication differs from closure")
    checksums = read_json(report / "checksums.json")
    for name, expected in checksums["published_sha256"].items():
        path = (report / name).resolve()
        require(path.is_relative_to(report.resolve()) and sha256(path) == expected,
                "Published endpoint evidence changed")
    return read_json(report / "summary.json")


def rank_programs(rows: list[dict]) -> list[dict]:
    require(bool(rows) and len({r["generation"] for r in rows}) == len(rows), "Invalid endpoint candidates")
    require(all(type(r["generation"]) is int and type(r["development_score"]) in (int, float)
                and math.isfinite(r["development_score"]) for r in rows), "Invalid development score")
    return sorted(rows, key=lambda r: (-r["development_score"], r["generation"]))


def endpoint_candidates(closure_path: Path) -> tuple[dict, dict, list[dict]]:
    from shinka_crl import adaptive_endpoint as endpoint
    closure = read_json(closure_path)
    require(closure["status"] == "closed" and closure["target_slots"] == closure["slots_consumed"] == 25
            and closure["remaining_generations"] == [] and bool(closure["resolution"].strip()),
            "Reserved handoff requires an explicitly closed complete allocation")
    work = Path(closure["endpoint_work"]).resolve()
    require(work.name == "archive", "Expected reviewed endpoint archive")
    summary = endpoint.summarize(work.parent)
    require(summary["endpoint_status"] in {"allocation_exhausted_complete", "allocation_exhausted_with_grammar_stop"}
            and summary["allocation_status"] == "exhausted" and summary["slots_consumed"] == 25
            and summary["target_slots"] == 25 and not summary["remaining_generations"]
            and not summary["unpersisted_requests"] and summary["incomplete_training_attempts"] == 0
            and not summary["terminal_proposal_failures"], "Endpoint contains unresolved work")
    require(all(r["receipt_verified"] and r["status"] == "failed"
                and r["error"] == "ProgramValidationError: AST exceeds 512 nodes"
                for r in summary["failed_requests"]), "Unreviewed endpoint failure")
    report = Path(closure["endpoint_report"])
    if not report.is_absolute():
        report = REPO_ROOT / report
    published = _verify_published_report(report, closure)
    # Exported summaries may redact machine-local paths in session metadata.
    for field in ("slots_consumed", "programs_evaluated", "programs", "failed_requests",
                  "incomplete_training_attempts", "remaining_generations"):
        require(published[field] == summary[field], f"Published endpoint {field} differs from verified source")
    original = read_json(work / "plan.json")
    study = Path(original["evaluation_study"])
    development = ae.read_plan(study)
    require(development["profile"]["seeds"] == [4001, 4002, 4003], "Development partition changed")
    candidates = []
    for row in summary["programs"]:
        request_dir = work / f"shinka/gen_{row['generation']}/results/evaluation"
        request = ae.validate_request(study, request_dir, development)
        result = ae.validate_cache(study / request["cache_origin"], development, request["cache_key"])
        score = statistics.mean(t["score"]["combined_score"] for t in result["trials"])
        require(request == row["request"] and score == request["aggregate"]["scores"]["combined_score"]["mean"],
                "Endpoint score does not match rederived evidence")
        candidates.append({"generation": row["generation"], "native_id": row["id"],
                           "development_score": score, "development_seeds": [4001, 4002, 4003],
                           "source": str(request_dir / "program.py"), "program": request["candidate"]["program"],
                           "request_receipt_sha256": sha256(request_dir / "receipt.json"),
                           "cache_receipt_sha256": request["cache_receipt_sha256"]})
    require(any(r["generation"] == 0 for r in candidates), "Identity is missing from endpoint selection")
    return closure, development, rank_programs(candidates)


def recipe(candidate: dict, context: dict) -> dict:
    """Execution identity includes initialization and method, never AST alone."""
    population = context["profile"]["ne"]["pop_size"]
    static = candidate["settings"] or {"sigma": .5, "elite_ratio": .5}
    focus = candidate["variant"] == "ga_focus"
    return {"native_method": "ga_focus" if focus else "ga", "variant": candidate["variant"],
            "program_identity": candidate.get("program", {}).get("canonical_ast_sha256")
                                or candidate["source_sha256"],
            "sigma_initial": static["sigma"], "archive_fraction": static["elite_ratio"],
            "archive_size": max(1, int(population * static["elite_ratio"])),
            "offspring_count": population - max(1, int(population * static["elite_ratio"])) - int(focus),
            "memory_initial": [0., 0., 0., 0.] if candidate["variant"] == "ga_adaptive" else None,
            "focus_settings": FOCUS_SEARCHER_KWARGS if focus else None,
            "centroid_inside_population_budget": focus,
            "adapter": {"grammar": GRAMMAR_VERSION, "width_bounds": [.001, 2.],
                        "source_sha256": context["source_sha256"]["src/shinka_crl/adaptive.py"]}
                       if candidate["variant"] == "ga_adaptive" else None,
            "settings": candidate["settings"],
            "native_fixed": {"hidden_dims": [16, 16], "num_params": 386, "objective": "mean",
                             "obs_norm": False, "first_task_clean": True, "task_warmup": 0,
                             "init_around_mean": False, "refresh": True, "variation": "gaussian",
                             "cross_over_rate": 0., "schedule": "switch", "noise_range": .5},
            "evaluation_context_sha256": ae.digest(context)}


def deduplicate(conditions: list[dict], context: dict) -> list[dict]:
    unique = []
    for candidate in conditions:
        identity = recipe(candidate, context)
        key = ae.digest(identity)
        match = next((r for r in unique if r["recipe_key"] == key), None)
        if match is None:
            unique.append({"recipe_key": key, "recipe": identity, "candidate": candidate,
                           "memberships": [candidate["id"]]})
        else:
            match["memberships"].append(candidate["id"])
    return unique


def _context(development: dict, *, diagnostic: bool = False, repeated: bool = False) -> dict:
    profile = copy.deepcopy(development["reserved_validation_profile"]) if repeated else load_profile("adaptive-validation")
    if diagnostic:
        profile = load_profile("adaptive-gate-switching")
        profile.update(num_phases=4, ne={"num_generations": 8, "task_interval": 2,
                                       "pop_size": 8, "num_evals": 1})
    return {"protocol_version": REPEATED_VERSION if repeated else VERSION,
            "kind": "repeated_reserved" if repeated else "diagnostic" if diagnostic else "reserved",
            "profile": profile, "objective_version": "adaptive-active-previous-v1",
            "objective_weights": {"active": .5, "previous": .5}, "trial_offset": 1,
            "eval_seed_offset": 900000, "posthoc_episodes": 10,
            "upstream": development["upstream"], "upstream_commit": UPSTREAM_COMMIT,
            "python": development["python"], "timeout_seconds": 1800,
            "cpu_affinity": development["cpu_affinity"], "thread_environment": SEARCH_THREAD_ENV,
            "runtime": development["runtime"], "source_sha256": source_hashes()}


def _write_freeze(output, context, conditions, *, provenance, protocol_path):
    require(not output.exists(), "Handoff requires a fresh directory")
    verify_upstream(Path(context["upstream"]))
    with ae.runtime_scope(context["cpu_affinity"]):
        require(ae.runtime_fingerprint(context["python"]) == context["runtime"], "Frozen runtime changed")
    output.mkdir(parents=True)
    (output / "programs").mkdir()
    frozen_conditions = []
    for candidate in conditions:
        candidate = copy.deepcopy(candidate)
        if candidate["source"]:
            source = Path(candidate["source"])
            if not source.is_absolute():
                source = REPO_ROOT / source
            target = output / "programs" / f"{candidate['id']}.py"
            require(sha256(source) == candidate["source_sha256"], "Candidate source changed")
            shutil.copyfile(source, target)
            candidate["source"] = str(target.relative_to(output))
            if candidate["variant"] == "ga_adaptive":
                candidate["program"] = load_program(target).metadata()
        frozen_conditions.append(candidate)
    unique = deduplicate(frozen_conditions, context)
    trials = [{"index": len(unique) * i + j, "recipe_key": item["recipe_key"],
               "memberships": item["memberships"], "seed": seed, "trial": seed + 1,
               "eval_seed": seed + 900000}
              for i, seed in enumerate(context["profile"]["seeds"]) for j, item in enumerate(unique)]
    steps = nominal_training_steps(context["profile"], "ga")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    if context["kind"] in {"reserved", "repeated_reserved"}:
        for name in SOURCE_FILES:
            require(subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=REPO_ROOT)
                    == (REPO_ROOT / name).read_bytes(), "Commit tested validation implementation before freeze")
    shutil.copyfile(protocol_path, output / "protocol.md")
    plan = {**context, "implementation_revision": revision,
            "native_source_sha256": native_sources(Path(context["upstream"])), "conditions": frozen_conditions,
            "unique_recipes": unique, "trial_order": trials, "provenance": provenance,
            "protocol_sha256": sha256(output / "protocol.md"), "session_limit_seconds": LIMIT,
            "first_block_trials": len(unique), "trial_steps_nominal": steps,
            "planned_trials": len(trials), "planned_nominal_training_steps": steps * len(trials),
            "planned_fresh_evaluation_episodes": 300 * len(trials),
            "reporting": {"primary": "paired five-seed combined-score selected minus focus mean and sample SD",
                          "secondary": ["identity", "arithmetic", "static_shinka11", "static_random24"],
                          "inference": "descriptive only; no significance threshold or promotion rule",
                          "early_stop": "integrity or fixed resource limit only; never score-dependent",
                          "deadline": "whole-trial wall deadline includes metadata probes; terminal receipt finalization overhead is retained in session cost"}}
    if context["kind"] == "repeated_reserved":
        allocation = provenance["seed_allocation"]
        require(sha256(Path(allocation["source"])) == allocation["sha256"],
                "Fresh seed allocation evidence changed before finalist freeze")
        shutil.copyfile(allocation["source"], output / "seed-allocation.json")
        require(sha256(output / "seed-allocation.json") == allocation["sha256"],
                "Fresh seed allocation evidence changed while freezing")
        plan["reporting"] = {
            "primary": "Within each outer-search repetition, evolutionary minus independent finalist mean combined score across the five common fresh evaluation seeds",
            "secondary": ["identity", "arithmetic", "focus"],
            "inference": "Report individual outer-search differences and their mean/sample SD; evaluation seeds are not independent search repetitions; no significance threshold or promotion rule",
            "early_stop": "integrity or preregistered cumulative experiment deadline only; never score-dependent",
        }
    write_json(output / "plan.json", plan)
    write_json(output / "receipt.json", artifact_hashes(output))
    return plan


def freeze_repeated(*, study: Path, archives: list[Path], output: Path, protocol_path: Path) -> dict:
    """Freeze all search finalists together before opening the new evaluation split."""
    from shinka_crl import adaptive_search as search
    development = ae.read_plan(study.resolve())
    require(development["protocol_version"] == ae.PARTITION_VERSION,
            "Repeated search needs a separately allocated fresh evaluation partition")
    selections, conditions, identities = [], [], set()
    for archive in archives:
        archive = archive.resolve()
        summary = search.summarize(archive)
        plan = read_json(archive / "plan.json")
        state = read_json(archive / "state.json")
        require(state["status"] == "complete" and summary["slots_consumed"] == 5
                and summary["programs_evaluated"] == 5
                and not summary["failed_requests"] and not summary["terminal_proposal_failures"],
                "Every five-slot search must be complete before the common finalist freeze")
        require(plan["evaluation_context_sha256"] == ae.digest(development), "Search evaluation context differs")
        arm, seed = plan.get("arm", "evolutionary"), plan["outer_random_seed"]
        require(arm in {"evolutionary", "independent"} and (seed, arm) not in identities,
                "Duplicated or unsupported search arm")
        identities.add((seed, arm))
        ranked = rank_programs([
            {"generation": row["generation"],
             "development_score": row["request"]["aggregate"]["scores"]["combined_score"]["mean"],
             "request": row["request"]} for row in summary["programs"]])
        require({row["generation"] for row in ranked} == set(range(5)),
                "Five-slot selection must include the identity and every proposal")
        winner = ranked[0]
        request = winner["request"]
        source = archive / "shinka" / f"gen_{winner['generation']}" / "main.py"
        require(sha256(source) == request["candidate"]["source_sha256"]
                and load_program(source).metadata() == request["candidate"]["program"],
                "Selected source differs from the verified development request")
        identifier = f"{arm}_{seed}"
        conditions.append({"id": identifier, "variant": "ga_adaptive", "source": str(source),
                           "source_sha256": sha256(source), "settings": None,
                           "program": request["candidate"]["program"]})
        selections.append({"id": identifier, "arm": arm, "outer_seed": seed,
                           "archive": str(archive), "plan_sha256": sha256(archive / "plan.json"),
                           "state_sha256": sha256(archive / "state.json"),
                           "generation": winner["generation"], "development_score": winner["development_score"],
                           "source_sha256": sha256(source),
                           "ranked_candidates": [{k: r[k] for k in ("generation", "development_score")} for r in ranked]})
    seeds = sorted({seed for seed, _ in identities})
    require(len(seeds) >= 2 and identities == {(seed, arm) for seed in seeds for arm in ("evolutionary", "independent")},
            "Require at least two complete paired outer-search repetitions")
    conditions.extend(copy.deepcopy(c) for c in development["controls"] if c["id"] in {"identity", "arithmetic", "focus"})
    provenance = {"development_plan": development, "development_plan_sha256": ae.digest(development),
                  "seed_allocation": {"source": str(study.resolve() / "seed-allocation.json"),
                                      "sha256": development["seed_allocation"]["sha256"]},
                  "selected_searches": selections,
                  "selection": "Highest exact combined development mean among all five slots; earlier generation breaks exact ties; all finalists frozen before any fresh evaluation"}
    return _write_freeze(output.resolve(), _context(development, repeated=True), conditions,
                         provenance=provenance, protocol_path=protocol_path.resolve())


def freeze(*, closure_path: Path, output: Path) -> dict:
    closure, development, ranked = endpoint_candidates(closure_path.resolve())
    selected = ranked[0]
    conditions = [{"id": "selected", "variant": "ga_adaptive", "source": selected["source"],
                   "source_sha256": selected["program"]["source_sha256"], "settings": None,
                   "program": selected["program"]}, *copy.deepcopy(development["controls"])]
    require(conditions[1:] == ae.controls(), "Control memberships or settings changed")
    provenance = {"closure": closure, "closure_sha256": sha256(closure_path),
                  "development_plan_sha256": ae.digest(development), "ranked_candidates": ranked,
                  "selected_generation": selected["generation"], "selected_native_id": selected["native_id"],
                  "selected_development_score": selected["development_score"],
                  "selection": "exact unrounded combined mean, earlier generation breaks exact ties"}
    return _write_freeze(output.resolve(), _context(development), conditions,
                         provenance=provenance, protocol_path=REPO_ROOT / "docs/adaptive-validation.md")


def freeze_diagnostic(*, study: Path, output: Path) -> dict:
    development = ae.read_plan(study.resolve())
    conditions = [copy.deepcopy(c) for c in development["controls"] if c["id"] in
                  ("arithmetic", "focus", "static_shinka11")]
    return _write_freeze(output.resolve(), _context(development, diagnostic=True), conditions,
                         provenance={"purpose": "reduced implementation diagnostic; no reserved outcomes"},
                         protocol_path=REPO_ROOT / "docs/adaptive-validation.md")


def read_plan(frozen: Path) -> dict:
    require(artifact_hashes(frozen) == read_json(frozen / "receipt.json"), "Frozen handoff evidence changed")
    plan = read_json(frozen / "plan.json")
    require(plan["protocol_version"] in {VERSION, REPEATED_VERSION} and plan["source_sha256"] == source_hashes(),
            "Validation implementation changed")
    require(plan["session_limit_seconds"] == LIMIT and plan["timeout_seconds"] == 1800
            and plan["posthoc_episodes"] == 10 and plan["objective_weights"] == {"active": .5, "previous": .5}
            and plan["objective_version"] == "adaptive-active-previous-v1"
            and plan["thread_environment"] == SEARCH_THREAD_ENV and plan["upstream_commit"] == UPSTREAM_COMMIT,
            "Validation numerical contract changed")
    require(plan["native_source_sha256"] == native_sources(Path(plan["upstream"])), "Pinned native implementation changed")
    if plan["kind"] == "repeated_reserved":
        development = plan["provenance"]["development_plan"]
        require(plan["protocol_version"] == REPEATED_VERSION
                and development["protocol_version"] == ae.PARTITION_VERSION
                and ae.digest(development) == plan["provenance"]["development_plan_sha256"]
                and plan["profile"] == development["reserved_validation_profile"], "Repeated validation partition changed")
        allocation = plan["provenance"]["seed_allocation"]
        require(sha256(frozen / "seed-allocation.json") == allocation["sha256"]
                == development["seed_allocation"]["sha256"], "Fresh seed allocation evidence changed")
        ae._validate_allocation(read_json(frozen / "seed-allocation.json"),
                                development["profile"]["seeds"], plan["profile"]["seeds"])
        expected_development = load_profile("adaptive-search")
        expected_development["seeds"] = development["profile"]["seeds"]
        require(development["profile"] == expected_development
                and development["source_sha256"] == ae.source_hashes()
                and development["controls"] == ae.controls(), "Repeated development contract changed")
        expected_profile = load_profile("adaptive-validation")
        expected_profile["seeds"] = development["reserved_validation_profile"]["seeds"]
        require(plan["profile"] == expected_profile and len(plan["profile"]["seeds"]) == 5
                and not set(plan["profile"]["seeds"]) & set(development["profile"]["seeds"]),
                "Repeated validation scientific profile changed")
        require(all(plan[key] == value for key, value in _context(development, repeated=True).items()),
                "Repeated validation context changed")
        selections = plan["provenance"]["selected_searches"]
        identities = {(s["outer_seed"], s["arm"]) for s in selections}
        seeds = {seed for seed, _ in identities}
        require(len(selections) == len(identities) and len(seeds) >= 2
                and identities == {(seed, arm) for seed in seeds for arm in ("evolutionary", "independent")},
                "Repeated validation requires complete search pairs")
        require([c["id"] for c in plan["conditions"]] == [s["id"] for s in selections] + ["identity", "arithmetic", "focus"],
                "Repeated finalist or control memberships changed")
        for candidate, selection in zip(plan["conditions"], selections):
            ranked = rank_programs(selection["ranked_candidates"])
            require(candidate["id"] == f"{selection['arm']}_{selection['outer_seed']}"
                    and candidate["variant"] == "ga_adaptive" and candidate["settings"] is None
                    and candidate["source"] == f"programs/{candidate['id']}.py"
                    and candidate["source_sha256"] == selection["source_sha256"]
                    and {row["generation"] for row in ranked} == set(range(5))
                    and selection["generation"] == ranked[0]["generation"]
                    and selection["development_score"] == ranked[0]["development_score"],
                    "Repeated finalist selection changed")
        for candidate in plan["conditions"][len(selections):]:
            expected = copy.deepcopy(next(c for c in development["controls"] if c["id"] == candidate["id"]))
            if expected["source"]:
                expected["source"] = f"programs/{expected['id']}.py"
                if expected["variant"] == "ga_adaptive":
                    expected["program"] = load_program(frozen / expected["source"]).metadata()
            require(candidate == expected, "Repeated validation fixed control changed")
    elif plan["kind"] == "reserved":
        require(plan["protocol_version"] == VERSION, "Legacy validation version changed")
        require(plan["profile"] == load_profile("adaptive-validation")
                and tuple(c["id"] for c in plan["conditions"]) == CONDITIONS,
                "Reserved partition or controls changed")
    else:
        require(plan["kind"] == "diagnostic" and plan["profile"]["seeds"] == [3001]
                and plan["profile"]["name"] == "adaptive-gate-switching", "Invalid diagnostic partition")
    context = {key: plan[key] for key in _context(
        plan["provenance"]["development_plan"] if plan["kind"] == "repeated_reserved" else plan,
        diagnostic=plan["kind"] == "diagnostic", repeated=plan["kind"] == "repeated_reserved")}
    require(plan["unique_recipes"] == deduplicate(plan["conditions"], context), "Full recipe identities changed")
    unique = plan["unique_recipes"]
    expected_trials = [{"index": len(unique) * i + j, "recipe_key": item["recipe_key"],
                        "memberships": item["memberships"], "seed": seed, "trial": seed + 1,
                        "eval_seed": seed + 900000}
                       for i, seed in enumerate(plan["profile"]["seeds"]) for j, item in enumerate(unique)]
    steps = nominal_training_steps(plan["profile"], "ga")
    require(plan["trial_order"] == expected_trials and plan["first_block_trials"] == len(unique)
            and plan["trial_steps_nominal"] == steps and plan["planned_trials"] == len(expected_trials)
            and plan["planned_nominal_training_steps"] == len(expected_trials) * steps
            and plan["planned_fresh_evaluation_episodes"] == len(expected_trials) * 300,
            "Validation order or budget changed")
    for candidate in plan["conditions"]:
        if candidate["source"]:
            path = (frozen / candidate["source"]).resolve()
            require(path.is_relative_to(frozen.resolve()) and sha256(path) == candidate["source_sha256"],
                    "Frozen candidate source changed")
            if candidate["variant"] == "ga_adaptive":
                require(load_program(path).metadata() == candidate["program"], "Frozen candidate AST changed")
    return plan


def _candidate(plan, row):
    return next(r["candidate"] for r in plan["unique_recipes"] if r["recipe_key"] == row["recipe_key"])


def _score_attempt(attempt: Path, plan: dict, row: dict) -> dict:
    candidate = _candidate(plan, row)
    if candidate["source_sha256"]:
        require(sha256(attempt / "program.py") == candidate["source_sha256"], "Trial source changed")
    score = ae._score_trial(plan, candidate, row["seed"], attempt / "training", attempt / "analysis",
                            attempt / "program.py")
    evaluation = read_json(attempt / "analysis/evaluation.json")
    episodes = sum(len(values) for entry in evaluation["per_task"] for key, values in entry.items()
                   if key.endswith("returns"))
    require(episodes == 300, "Fresh evaluation count differs from native layout")
    return {**row, "score": score, "fresh_evaluation_episodes": episodes,
            "phase_returns": read_json(attempt / "analysis/summary.json")["phase_returns"],
            "training_wall_seconds": read_json(attempt / "training/manifest.json")["wall_seconds"],
            "analysis_wall_seconds": read_json(attempt / "analysis/manifest.json")["wall_seconds"]}


def verify_attempt(attempt: Path, plan: dict, row: dict) -> dict:
    require(artifact_hashes(attempt) == read_json(attempt / "receipt.json"), "Trial receipt changed")
    summary = read_json(attempt / "summary.json")
    require(summary["context_sha256"] == ae.digest(plan) and summary["trial"] == row,
            "Cross-profile/context trial reuse forbidden")
    if summary["status"] == "complete":
        require(summary["result"] == _score_attempt(attempt, plan, row), "Trial score changed")
    else:
        require(summary["status"] in {"failed", "interrupted"}, "Unsealed trial boundary requires review")
    return summary


def _execute_trial(*, frozen, attempt, plan, row, deadline):
    attempt.mkdir(parents=True)
    candidate = _candidate(plan, row)
    if candidate["source"]:
        shutil.copyfile(frozen / candidate["source"], attempt / "program.py")
    started = time.monotonic()
    summary = {"status": "running", "context_sha256": ae.digest(plan), "trial": row,
               "started_at": utc_now(), "result": None}
    write_json(attempt / "summary.json", summary)
    try:
        remaining = math.floor(deadline - time.monotonic())
        require(remaining > 0, "Validation session ceiling reached before training")
        execution = {**plan, "timeout_seconds": min(plan["timeout_seconds"], remaining)}
        ae._run_trial(execution, candidate, row["seed"], attempt / "training", attempt / "program.py")
        remaining = math.floor(deadline - time.monotonic())
        require(remaining > 0, "Validation session ceiling reached before analysis")
        run_analysis(run_dir=attempt / "training", output_dir=attempt / "analysis",
                     eval_seed=row["eval_seed"], upstream=Path(plan["upstream"]), python=plan["python"],
                     episodes=plan["posthoc_episodes"], timeout=min(plan["timeout_seconds"], remaining))
        summary.update(status="complete", result=_score_attempt(attempt, plan, row))
    except (Exception, KeyboardInterrupt) as exc:
        summary.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                       error=f"{type(exc).__name__}: {exc}")
    finally:
        summary.update(finished_at=utc_now(), wall_seconds=time.monotonic() - started)
        write_json(attempt / "summary.json", summary)
        write_json(attempt / "receipt.json", artifact_hashes(attempt))
    return summary


def _state(output, plan):
    state = read_json(output / "state.json")
    require(state["context_sha256"] == ae.digest(plan), "Validation output belongs to another context")
    require(sha256(output / "state.json") == read_json(output / "state-receipt.json")["sha256"],
            "Validation state changed")
    require(all(s["status"] != "running" for s in state["sessions"]),
            "Interrupted active session requires explicit accounting review")
    completed, failed = {}, []
    paths = []
    for record in state["attempts"]:
        row = plan["trial_order"][record["trial_index"]]
        path = output / record["path"]
        require(path.resolve().is_relative_to(output.resolve()), "Attempt path escapes output")
        result = verify_attempt(path, plan, row)
        require(record["receipt_sha256"] == sha256(path / "receipt.json"), "Attempt identity changed")
        paths.append(str(path.relative_to(output)))
        if result["status"] == "complete":
            require(row["index"] not in completed, "Repeated completed trial")
            completed[row["index"]] = result["result"]
        else:
            failed.append(record)
    require(paths == sorted(set(paths)) and set(paths) == {
        str(p.parent.relative_to(output)) for p in (output / "trials").glob("*/attempt_*/summary.json")},
        "Unrecorded, duplicated, or reordered trial attempts")
    require(sorted(completed) == list(range(len(completed))), "Resume requires contiguous whole-trial boundaries")
    return state, completed, failed


def _seal_state(output, state):
    write_json(output / "state.json", state)
    write_json(output / "state-receipt.json", {"sha256": sha256(output / "state.json")})


def run(*, frozen: Path, output: Path, max_trials: int | None = None,
        review_sha256: str | None = None, review_reason: str | None = None,
        retry_failed: bool = False) -> dict:
    frozen, output = frozen.resolve(), output.resolve()
    plan = read_plan(frozen)
    require(not output.is_relative_to(frozen) and not frozen.is_relative_to(output), "Handoff and results must be separate")
    if plan["kind"] in {"reserved", "repeated_reserved"}:
        _published(frozen)
    require(max_trials is None or type(max_trials) is int and max_trials > 0, "Positive whole-trial block required")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (output / "state.json").exists():
            require(set(p.name for p in output.iterdir()) == {".lock"}, "Fresh results directory required")
            _seal_state(output, {"context_sha256": ae.digest(plan), "frozen": str(frozen), "attempts": [], "sessions": []})
        state, completed, failures = _state(output, plan)
        require(len(completed) < plan["planned_trials"], "Validation allocation is already complete")
        if state["sessions"]:
            require(review_sha256 == sha256(output / "state.json") and bool(review_reason and review_reason.strip()),
                    "Resume requires review bound to the exact completed state")
        unresolved = [f for f in failures if f["trial_index"] not in completed]
        require(not unresolved or retry_failed, "Failed trial requires an explicit reviewed retry")
        block = max_trials or plan["first_block_trials"]
        if not state["sessions"]:
            require(block <= plan["first_block_trials"], "First block must stop after one seed's recipes")
        spent = sum(s["wall_seconds"] for s in state["sessions"])
        require(spent < LIMIT, "Cumulative validation session ceiling reached")
        with ae.runtime_scope(plan["cpu_affinity"]):
            require(ae.runtime_fingerprint(plan["python"]) == plan["runtime"], "Validation runtime changed")
            verify_upstream(Path(plan["upstream"]))
            started = time.monotonic()
            session = {"status": "running", "started_at": utc_now(), "wall_seconds": 0.,
                       "review_sha256": review_sha256, "review_reason": review_reason,
                       "requested_trials": block, "completed_trials": 0}
            state["sessions"].append(session)
            _seal_state(output, state)
            try:
                for row in plan["trial_order"][len(completed):len(completed) + block]:
                    if started + LIMIT - spent - time.monotonic() < 1:
                        session["status"] = "resource_limit"
                        break
                    number = 1 + sum(a["trial_index"] == row["index"] for a in state["attempts"])
                    relative = f"trials/{row['index']:03d}/attempt_{number:04d}"
                    with trial_deadline(started + LIMIT - spent):
                        result = _execute_trial(frozen=frozen, attempt=output / relative, plan=plan, row=row,
                                                deadline=started + LIMIT - spent)
                    state["attempts"].append({"trial_index": row["index"], "path": relative,
                                              "receipt_sha256": sha256(output / relative / "receipt.json")})
                    _seal_state(output, state)
                    if result["status"] != "complete":
                        session["status"] = result["status"]
                        break
                    session["completed_trials"] += 1
                else:
                    session["status"] = "complete"
                require(source_hashes() == plan["source_sha256"], "Implementation changed during validation")
                verify_upstream(Path(plan["upstream"]))
            finally:
                if session["status"] == "running":
                    session["status"] = "interrupted"
                session.update(wall_seconds=time.monotonic() - started, finished_at=utc_now())
                _seal_state(output, state)
    return summarize(frozen=frozen, output=output)


def summarize(*, frozen: Path, output: Path) -> dict:
    plan = read_plan(frozen)
    state, completed, failures = _state(output, plan)
    conditions = {}
    for candidate in plan["conditions"]:
        rows = [r for r in completed.values() if candidate["id"] in r["memberships"]]
        conditions[candidate["id"]] = {"trials": rows, "aggregate": ae.aggregate(rows) if rows else None}
    comparisons = {}
    if "selected" in conditions:
        selected = {r["seed"]: r for r in conditions["selected"]["trials"]}
        for name in ae.CONTROL_IDS:
            differences = [{"seed": r["seed"], "difference": selected[r["seed"]]["score"]["combined_score"]
                            - r["score"]["combined_score"]} for r in conditions[name]["trials"] if r["seed"] in selected]
            comparisons[name] = {"paired_differences": differences,
                                 "aggregate": ae.describe([r["difference"] for r in differences]) if differences else None,
                                 "primary": name == "focus", "complete_five_seed_comparison": len(differences) == 5}
    manifests = [read_json(output / a["path"] / "training/manifest.json") for a in state["attempts"]
                 if (output / a["path"] / "training/manifest.json").exists()]
    trained = sum(m["status"] == "complete" for m in manifests)
    episodes = 0
    analysis_seconds = 0.
    for attempt in state["attempts"]:
        directory = output / attempt["path"] / "analysis"
        if (directory / "manifest.json").exists():
            analysis_seconds += read_json(directory / "manifest.json").get("wall_seconds", 0.)
        episode_paths = [path for path in (directory / "evaluation.json", directory / "evaluation-input/evaluation.json")
                         if path.exists()]
        if episode_paths:
            require(len({sha256(path) for path in episode_paths}) == 1,
                    "Analysis copies disagree about realized episode evidence")
            evaluation = read_json(episode_paths[0])
            episodes += sum(len(values) for entry in evaluation.get("per_task", [])
                            for key, values in entry.items() if key.endswith("returns") and isinstance(values, list))
    repetition_comparisons = []
    if plan["kind"] == "repeated_reserved":
        for seed in sorted({s["outer_seed"] for s in plan["provenance"]["selected_searches"]}):
            left = {r["seed"]: r for r in conditions[f"evolutionary_{seed}"]["trials"]}
            right = {r["seed"]: r for r in conditions[f"independent_{seed}"]["trials"]}
            differences = [{"seed": s, "difference": left[s]["score"]["combined_score"] - right[s]["score"]["combined_score"]}
                           for s in sorted(left.keys() & right.keys())]
            repetition_comparisons.append({"outer_seed": seed, "paired_seed_differences": differences,
                "aggregate": ae.describe([r["difference"] for r in differences]) if differences else None,
                "complete_five_seed_comparison": len(differences) == 5})
        for candidate in plan["provenance"]["selected_searches"]:
            selected = {r["seed"]: r for r in conditions[candidate["id"]]["trials"]}
            for name in ("identity", "arithmetic", "focus"):
                differences = [{"seed": r["seed"], "difference": selected[r["seed"]]["score"]["combined_score"] - r["score"]["combined_score"]}
                               for r in conditions[name]["trials"] if r["seed"] in selected]
                comparisons[f"{candidate['id']}_minus_{name}"] = {
                    "paired_differences": differences,
                    "aggregate": ae.describe([r["difference"] for r in differences]) if differences else None}
    return {"protocol_version": plan["protocol_version"], "kind": plan["kind"],
            "status": "complete" if len(completed) == plan["planned_trials"] else "partial",
            "conditions": conditions, "comparisons": comparisons, "sessions": state["sessions"],
            **({"repetition_comparisons": repetition_comparisons,
                "across_search_repetitions": ae.describe([r["aggregate"]["mean"] for r in repetition_comparisons])
                  if repetition_comparisons and all(r["complete_five_seed_comparison"] for r in repetition_comparisons) else None}
               if plan["kind"] == "repeated_reserved" else {}),
            "planned_trials": plan["planned_trials"], "allocated_training_trials": len(manifests),
            "completed_training_trials": trained, "scored_trials": len(completed),
            "failed_attempts": failures, "completed_nominal_training_steps": trained * plan["trial_steps_nominal"],
            "allocated_nominal_training_steps": len(manifests) * plan["trial_steps_nominal"],
            "planned_nominal_training_steps": plan["planned_nominal_training_steps"],
            "realized_fresh_evaluation_episodes": episodes,
            "scored_fresh_evaluation_episodes": sum(r["fresh_evaluation_episodes"] for r in completed.values()),
            "evaluation_accounting_note": "Realized episodes include every preserved raw episode vector, including failed attempts; terminated episodes without vectors are unmeasured",
            "analysis_wall_seconds": analysis_seconds,
            "active_session_wall_seconds": sum(s["wall_seconds"] for s in state["sessions"]),
            "training_wall_seconds": sum(m.get("wall_seconds", 0.) for m in manifests),
            "interpretation": ("Reduced implementation diagnostic on development seed 3001; no reserved outcomes"
                               if plan["kind"] == "diagnostic" else
                               "Fresh paired finalist comparison across independent outer searches; fixed common development/evaluation tasks; limited repetitions do not establish broad search-method superiority"
                               if plan["kind"] == "repeated_reserved" else
                               "Reserved transfer comparison of one development-selected rule; separate from final paper reporting")}


def export(*, frozen: Path, output: Path, report: Path) -> dict:
    frozen, output, report = frozen.resolve(), output.resolve(), report.resolve()
    require(not report.exists() and all(not left.is_relative_to(right) and not right.is_relative_to(left)
            for left, right in ((report, output), (report, frozen), (frozen, output))),
            "Report, handoff and results must use new independent paths")
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = summarize(frozen=frozen, output=output)
        before, frozen_before = artifact_hashes(output), artifact_hashes(frozen)
        report.mkdir(parents=True)
        published = {}
        for root, prefix in ((frozen, "frozen"), (output, "raw")):
            for path in artifact_files(root):
                if path.suffix not in {".json", ".jsonl", ".log", ".py", ".md"}:
                    continue
                target = report / prefix / path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                published[str(target.relative_to(report))] = sha256(target)
        write_json(report / "summary.json", summary)
        published["summary.json"] = sha256(report / "summary.json")
        require(before == artifact_hashes(output) and frozen_before == artifact_hashes(frozen),
                "Evidence changed during export")
        write_json(report / "checksums.json", {"original_artifact_sha256": before, "published_sha256": published})
    return summary
