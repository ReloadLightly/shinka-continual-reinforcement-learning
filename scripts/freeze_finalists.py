"""Freeze static finalists after the complete search, without using validation seeds."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import runpy
import shutil
import statistics
import tempfile

from shinka_crl.experiment import REPO_ROOT, UPSTREAM_COMMIT, load_profile, score_curve
from shinka_crl.search import effective_key, random_pool

SPEC = importlib.util.spec_from_file_location("finalist_report", REPO_ROOT / "scripts/report_search.py")
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)
require, sha256, read_json, contained = REPORT.require, REPORT.sha256, REPORT.read_json, REPORT.contained

PROFILE_RESOURCE = REPO_ROOT / "src/shinka_crl/profiles/cartpole-validation.json"
VALIDATION_PROFILE = read_json(PROFILE_RESOURCE)


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


class Evidence:
    """Check exported bytes against receipts before trusting any selection input."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.checksums = read_json(self.root / "checksums.json")
        self.inputs = {name: sha256(self.root / name)
                       for name in ("checksums.json", "summary.json", "native_archive.json")}
        exported = set()
        for relative, receipt in self.checksums.items():
            require(relative == "raw/" + receipt["source"], "Receipt source path mismatch")
            contained(self.root, relative)
            if receipt["exported"]:
                self.path(relative)
                exported.add(relative)
        actual = {str(path.relative_to(self.root)) for path in (self.root / "raw").rglob("*")
                  if path.is_file()}
        require(actual == exported, "Missing or unreceipted raw report artifacts")

    def path(self, relative: str) -> Path:
        path = contained(self.root, relative)
        receipt = self.checksums.get(relative, {})
        require(receipt.get("exported") is True and path.is_file(),
                f"Missing exported evidence: {relative}")
        require(not path.is_symlink() and sha256(path) == receipt.get("exported_sha256"),
                f"Changed exported evidence: {relative}")
        self.inputs[relative] = receipt["exported_sha256"]
        return path

    def json(self, relative: str):
        return read_json(self.path(relative))

    def original(self, relative: str, digest: str) -> None:
        require(self.checksums.get(relative, {}).get("original_sha256") == digest,
                f"Original receipt mismatch: {relative}")


def validate_endpoint(report_dir: Path) -> tuple[dict, list[dict], Evidence]:
    """Reconstruct a complete 25/24 endpoint from portable, compact raw evidence."""
    evidence = Evidence(report_dir)
    summary = read_json(evidence.root / "summary.json")
    archive = read_json(evidence.root / "native_archive.json")
    plan, state = evidence.json("raw/plan.json"), evidence.json("raw/state.json")
    require(summary["status"] == state["status"] == "complete", "Search is not complete")
    require(plan["proposal_slots"] == 24 and state["random_completed"] == 24,
            "Require the complete 25-program / 24-control endpoint")
    native = archive["programs"]
    require([p["generation"] for p in native] == list(range(25))
            and len({p["id"] for p in native}) == 25 and all(p["correct"] for p in native),
            "Require exactly 25 successful native generation slots")
    require(not summary["failed_proposals"] and not any(
        name.endswith("/failure.json") for name in evidence.checksums),
        "Failed proposal evidence cannot be silently replaced")
    sessions = state["sessions"]
    require(sessions and all(s["status"] == "complete" and s["returncode"] == 0
                            and not s.get("stop_reason") for s in sessions),
            "Incomplete or failed execution session")
    require([s["endpoint"] for s in sessions if s["arm"] == "shinka"][-1] == 25
            and [s["endpoint"] for s in sessions if s["arm"] == "random"] == list(range(1, 25)),
            "Missing search or random execution slots")
    profile = load_profile("search")
    require(summary["profile"] == plan["profile"] == profile
            and summary["upstream_commit"] == plan["upstream_commit"] == UPSTREAM_COMMIT,
            "Search protocol or upstream pin mismatch")
    pool = random_pool(plan["random_pool_seed"])
    require(plan["random_pool"] == evidence.json("raw/random_pool.json") == pool,
            "Random candidates differ from the frozen pool")
    for entry in pool["entries"]:
        require(sha256(evidence.path("raw/" + entry["program_path"])) == entry["program_sha256"],
                "Frozen random source changed")
    for name, digest in state["artifact_sha256"].items():
        evidence.original("raw/" + name, digest)
    evidence.original("raw/model_requests.jsonl", state["model_requests_sha256"])
    evidence.original("raw/shinka/rng_state.json", state["rng_sha256"])
    usage = REPORT.summarize_usage(archive, evidence.path("raw/model_requests.jsonl"))
    require(all(usage[key] == 24 for key in ("adapter_attempts", "codex_exec_launches",
                                          "successful_adapter_responses"))
            and usage["failed_adapter_attempts"] == usage["incomplete_adapter_attempts"] == 0,
            "Proposal request ledger does not account for exactly 24 successful slots")
    require(summary["model_usage"] == usage, "Reported proposal usage differs from raw ledger")
    ids = {program["id"]: program for program in native}
    saved = state["programs"]
    require(len(saved) == 25, "Saved archive has missing generation slots")
    for program, recorded in zip(native, saved):
        for key in ("id", "generation", "parent_id", "code", "combined_score", "correct"):
            require(program[key] == recorded[key], "Saved and exported native archive differ")
        for key in ("public_metrics", "private_metrics"):
            require(program[key] == json.loads(recorded[key]), "Saved native metrics differ")
        parent, generation = program["parent_id"], program["generation"]
        require((generation == 0 and parent is None) or
                (parent in ids and ids[parent]["generation"] < generation), "Invalid parentage")
    generation_dirs = {name.split("/")[2] for name in evidence.checksums
                       if name.startswith("raw/shinka/gen_")}
    random_dirs = {name.split("/")[2] for name in evidence.checksums
                   if name.startswith("raw/random/candidate_")}
    require(generation_dirs == {f"gen_{i}" for i in range(25)}
            and random_dirs == {f"candidate_{i:03d}" for i in range(1, 25)},
            "Missing or unexpected raw evaluation slots")
    expected = {("shinka", i) for i in range(25)} | {("random", i) for i in range(1, 25)}
    require(len(summary["rows"]) == 49
            and {(r["arm"], r["index"]) for r in summary["rows"]} == expected,
            "Missing, duplicated or unexpected scored slots")
    parse = runpy.run_path(str(REPO_ROOT / "tasks/cartpole_ga/evaluate.py"))["parse_ga_config"]
    rows, vectors = [], {}
    for row in summary["rows"]:
        arm, index = row["arm"], row["index"]
        relative = (f"shinka/gen_{index}/main.py" if arm == "shinka"
                    else pool["entries"][index - 1]["program_path"])
        require(row["program_path"] == relative, "Candidate assigned to a different slot")
        program = evidence.path("raw/" + relative)
        settings, program_hash = parse(program), sha256(program)
        require(row["program_sha256"] == program_hash and row["settings"] == settings
                and row["effective_key"] == effective_key(settings), "Candidate source changed")
        if arm == "shinka":
            require(native[index]["code"] == program.read_text()
                    and row["program_id"] == native[index]["id"], "Evaluated source replaced")
            if index == 0:
                require(program_hash == plan["source_sha256"]["tasks/cartpole_ga/initial.py"],
                        "Shared default source changed")
        else:
            require(settings == pool["entries"][index - 1]["settings"], "Random settings changed")
        root = (f"raw/shinka/gen_{index}/results" if arm == "shinka"
                else f"raw/random/candidate_{index:03d}")
        metrics = evidence.json(root + "/metrics.json")
        require(evidence.json(root + "/correct.json") == {"correct": True, "error": None}
                and row["correct"] and not row["incomplete_seeds"], "Failed candidate evaluation")
        private, public = metrics["private"], metrics["public"]
        provenance = private["provenance"]
        require(provenance["program_sha256"] == program_hash
                and provenance["evaluator_sha256"] == plan["source_sha256"]["tasks/cartpole_ga/evaluate.py"]
                and provenance["profile"] == profile
                and provenance["profile_sha256"] == REPORT.canonical_hash(profile)
                and provenance["cpu_affinity"] == plan["cpu_affinity"]
                and provenance["thread_environment"] == plan["thread_environment"],
                "Candidate evaluation provenance differs from frozen protocol")
        require([s["seed"] for s in private["seed_results"]] == profile["seeds"]
                and [s["seed"] for s in private["seed_attempts"]] == profile["seeds"]
                and [s["seed"] for s in row["completed_seeds"]] == profile["seeds"],
                "Incomplete or replaced development seed slots")
        scores = []
        for attempt, recorded, published in zip(private["seed_attempts"], private["seed_results"],
                                               row["completed_seeds"]):
            seed = attempt["seed"]
            seed_root = root + f"/seed_{seed}"
            require(attempt["status"] == "complete" and attempt["trial"] == seed + 1,
                    "Failed or replaced development trial")
            for name, digest in attempt["artifact_sha256"].items():
                evidence.original(seed_root + "/" + name, digest)
            computed = score_curve(evidence.json(seed_root + "/training_metrics.json"),
                                   profile=profile, method="ga")
            require(recorded == {"seed": seed, "trial": seed + 1,
                                  "normalized_score": computed["normalized_score"],
                                  "mean_return": computed["mean_return"]}
                    and all(published[key] == value for key, value in recorded.items()),
                    "Published seed score differs from raw learning curve")
            result = evidence.json(seed_root + "/results.json")
            manifest = evidence.json(seed_root + "/manifest.json")
            require(result["env_steps"] == 7_680_000 and manifest["status"] == "complete"
                    and manifest["profile"] == profile and manifest["seed"] == seed
                    and manifest["trial"] == seed + 1 and manifest["ga_settings"] == settings
                    and manifest["upstream_commit"] == UPSTREAM_COMMIT,
                    "Development training protocol differs")
            config = result["config"]
            require(config["sigma"] == settings["sigma"]
                    and config["searcher_kwargs"] == {"elite_ratio": settings["elite_ratio"],
                                                       "init_around_mean": False}
                    and config["searcher_resolved"]["num_elites"] == effective_key(settings)[1],
                    "Resolved training settings differ from source")
            require(seed not in vectors or vectors[seed] == result["noise_vectors"],
                    "Development task vectors differ")
            vectors[seed] = result["noise_vectors"]
            scores.append(computed["normalized_score"])
        score = statistics.mean(scores)
        require(score == metrics["combined_score"] == row["combined_score"],
                "Candidate score differs from raw learning curves")
        require(public["profile"] == "search" and public["method"] == "ga"
                and public["seeds_completed"] == 3
                and all(public[key] == value for key, value in settings.items()),
                "Candidate public metrics differ")
        if arm == "shinka":
            require(native[index]["combined_score"] == score
                    and native[index]["public_metrics"] == public
                    and native[index]["private_metrics"] == private, "Native metrics changed")
        rows.append({"arm": arm, "index": index, "combined_score": score,
                     "program_path": relative, "program_sha256": program_hash,
                     "settings": settings, "effective_key": effective_key(settings), "correct": True})
    require(summary["comparison"] == REPORT.compare_arms(rows), "Comparison summary changed")
    require(summary["completed_seed_trials"] == 147 and summary["incomplete_seed_trials"] == 0
            and summary["training_steps_nominal_completed"] == 1_128_960_000
            and summary["training_steps_nominal_allocated"] == 1_128_960_000,
            "Endpoint compute accounting mismatch")
    return plan, rows, evidence


def select_finalists(rows: list[dict]) -> list[dict]:
    """Top two effective-distinct outcomes per arm, with the default eligible in both."""
    default = next(row for row in rows if row["arm"] == "shinka" and row["index"] == 0)
    groups = {}

    def add(row: dict, arm: str, rank: int | None) -> None:
        identity = tuple(row["effective_key"])
        if identity not in groups:
            groups[identity] = {"id": f"candidate_{len(groups) + 1:03d}",
                                **{key: row[key] for key in ("settings", "effective_key")},
                                "program_path": f"programs/{row['program_sha256']}.py",
                                "program_sha256": row["program_sha256"], "memberships": []}
        groups[identity]["memberships"].append({
            "arm": arm, "development_rank": rank, "index": row["index"],
            "combined_score": row["combined_score"], "report_program_path": row["program_path"],
            "program_path": f"programs/{row['program_sha256']}.py",
            "program_sha256": row["program_sha256"], "settings": row["settings"],
        })

    add(default, "default", None)
    for arm in ("shinka", "random"):
        pool = [default] + [r for r in rows if r["arm"] == arm and r is not default]
        ranked = sorted(pool, key=lambda r: (-r["combined_score"], r["index"], r["program_sha256"]))
        seen, rank = set(), 0
        for row in ranked:
            identity = tuple(row["effective_key"])
            if identity in seen:
                continue
            seen.add(identity)
            rank += 1
            add(row, arm, rank)
            if rank == 2:
                break
        require(rank == 2, f"Fewer than two distinct configurations for {arm}")
    return list(groups.values())


def freeze_finalists(report_dir: Path, output: Path) -> dict:
    output = output.resolve()
    require(not output.exists(), f"Refusing to overwrite frozen finalists: {output}")
    require(not output.is_relative_to(report_dir.resolve()), "Freeze must be outside the report")
    plan, rows, evidence = validate_endpoint(report_dir)
    candidates = select_finalists(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        (staging / "programs").mkdir()
        for candidate in candidates:
            for membership in candidate["memberships"]:
                source = evidence.path("raw/" + membership["report_program_path"])
                (staging / membership["program_path"]).write_bytes(source.read_bytes())
        write_json(staging / "profile.json", VALIDATION_PROFILE)
        artifact_hashes = {str(path.relative_to(staging)): sha256(path)
                           for path in sorted(staging.rglob("*")) if path.is_file()}
        manifest = {
            "schema_version": 1, "status": "frozen", "search_status": "closed_for_selection",
            "purpose": "One-time static-finalist validation; no validation observations used in selection",
            "upstream_commit": UPSTREAM_COMMIT, "profile_path": "profile.json",
            "profile": VALIDATION_PROFILE, "candidates": candidates,
            "selection": {"per_arm": 2, "include_default_always": True,
                          "eligible_pool": "shared default plus every completed candidate from that arm",
                          "rank": "descending development combined_score, ascending generation/control index, ascending source SHA256",
                          "distinct_identity": "float32(sigma), max(1,int(64*elite_ratio))",
                          "deduplication": "within each ranked arm, then globally; retain every selected arm membership",
                          "maximum_unique_finalists": 5,
                          "development_score": "mean across seeds1001–1003 of mean active-task centroid return across80 checkpoints, divided by500",
                          "search_feedback_closed": True},
            "evaluation": {"method": "ga", "seed_to_trial": "trial = seed + 1",
                           "trials": [{"seed": seed, "trial": seed + 1} for seed in VALIDATION_PROFILE["seeds"]],
                           "posthoc_episodes": 10, "posthoc_seed": "900000 + training seed",
                           "learning_accuracy": "mean own-task return across four phase-end centroids",
                           "forgetting": "each own-task return minus its fresh return after the next phase; report all three signed differences and their mean",
                           "cumulative_return": "pinned reference-grid integral at matched nominal training steps; also retain normalized and exact-dense integrals",
                           "winner_selection": "within each arm's two frozen finalists, highest mean active-task normalized score across five validation seeds; ties use lower development rank, then source SHA256",
                           "interpretation": "Validation selects once using the same active-return objective; retention metrics are reported without feeding results back to proposal search"},
            "input_report": {"name": evidence.root.name, "sha256": evidence.inputs,
                             "search_source_sha256": plan["source_sha256"],
                             "endpoint": {"shinka_programs": 25, "random_controls": 24}},
            "source_sha256": {"scripts/freeze_finalists.py": sha256(Path(__file__)),
                              "scripts/report_search.py": sha256(REPO_ROOT / "scripts/report_search.py"),
                              "src/shinka_crl/profiles/cartpole-validation.json": sha256(PROFILE_RESOURCE)},
            "artifact_sha256": artifact_hashes,
        }
        write_json(staging / "manifest.json", manifest)
        write_json(staging / "checksums.json", {**artifact_hashes,
                                               "manifest.json": sha256(staging / "manifest.json")})
        require(not output.exists(), "Finalist output appeared during preparation")
        staging.rename(output)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return validate_finalists(output)


def validate_finalists(root: Path) -> dict:
    """Verify a frozen handoff without requiring historical search runtime files."""
    root = root.resolve()
    checksums = read_json(root / "checksums.json")
    actual = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    require(actual == set(checksums) | {"checksums.json"}, "Frozen artifact inventory changed")
    for relative, digest in checksums.items():
        path = contained(root, relative)
        require(not path.is_symlink() and sha256(path) == digest, f"Frozen artifact changed: {relative}")
    manifest = read_json(root / "manifest.json")
    require(manifest["schema_version"] == 1 and manifest["status"] == "frozen"
            and manifest["search_status"] == "closed_for_selection", "Invalid finalist freeze")
    require(manifest["source_sha256"]["scripts/freeze_finalists.py"] == sha256(Path(__file__)),
            "Frozen finalist selector implementation changed")
    require(manifest["profile_path"] == "profile.json"
            and manifest["profile"] == read_json(root / "profile.json") == VALIDATION_PROFILE,
            "Frozen validation profile changed")
    require(manifest["artifact_sha256"] == {k: v for k, v in checksums.items() if k != "manifest.json"},
            "Frozen artifact receipts differ")
    candidates = manifest["candidates"]
    require(1 <= len(candidates) <= 5 and len({tuple(c["effective_key"]) for c in candidates}) == len(candidates),
            "Invalid or duplicated finalist identities")
    parse = runpy.run_path(str(REPO_ROOT / "tasks/cartpole_ga/evaluate.py"))["parse_ga_config"]
    require(len({candidate["id"] for candidate in candidates}) == len(candidates)
            and all(candidate["id"] == f"candidate_{i:03d}"
                    for i, candidate in enumerate(candidates, 1)), "Invalid finalist IDs")
    memberships = [member for candidate in candidates for member in candidate["memberships"]]
    require(sum(member["arm"] == "default" for member in memberships) == 1,
            "Missing or duplicate shared-default membership")
    for arm in ("shinka", "random"):
        require(sorted(member["development_rank"] for member in memberships if member["arm"] == arm)
                == [1, 2], "Missing or duplicate ranked finalist memberships")
    for candidate in candidates:
        settings = parse(contained(root, candidate["program_path"]))
        require(settings == candidate["settings"] and effective_key(settings) == candidate["effective_key"]
                and checksums[candidate["program_path"]] == candidate["program_sha256"],
                "Frozen finalist source identity differs")
        for member in candidate["memberships"]:
            settings = parse(contained(root, member["program_path"]))
            require(settings == member["settings"] and effective_key(settings) == candidate["effective_key"]
                    and checksums[member["program_path"]] == member["program_sha256"],
                    "Frozen membership source identity differs")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze_finalists(args.report_dir, args.output)
    print(json.dumps({"status": manifest["status"], "unique_finalists": len(manifest["candidates"]),
                      "validation_experiments_started": 0}, indent=2))


if __name__ == "__main__":
    main()
