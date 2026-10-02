"""Independently verify the seven-trial adaptive adapter gate using NumPy only.

Numerical arrays are compared by shape, dtype, and values. The report retains
content hashes for individual arrays rather than relying on compressed-file
byte identity. This checker does not run an optimizer or evaluate a policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

IDS = (
    "plain_stationary", "identity_stationary", "plain_switching", "identity_switching",
    "halving_switching", "arithmetic_switching", "focus_switching",
)
NATIVE_FIELDS = ("archive", "fitness", "sigma", "generation")
GENERATIONS, INTERVAL, POPULATION, PARAMS = 6, 3, 16, 386


def read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")
    return json.loads(Path(path).read_text(), parse_constant=reject_constant)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    require(all(array.dtype.kind in "biuf" for array in arrays.values()),
            f"Unsupported array dtype in {path}")
    return arrays


def same_arrays(left, right, label):
    require(left.shape == right.shape, f"{label}: shape mismatch")
    require(left.dtype == right.dtype, f"{label}: dtype mismatch")
    require(np.array_equal(left, right), f"{label}: numerical values differ")


def array_receipts(arrays):
    out = {}
    for name, value in sorted(arrays.items()):
        metadata = {"shape": list(value.shape), "dtype": value.dtype.str}
        header = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        digest = hashlib.sha256(header + b"\n" + value.tobytes(order="C")).hexdigest()
        out[name] = {**metadata, "semantic_sha256": digest}
    return out


def verify(suite_path):
    from shinka_crl.adaptive import load_program
    from shinka_crl.baseline_contract import validate_baseline_config

    suite_path = Path(suite_path).resolve()
    suite = read_json(suite_path)
    require(suite.get("status") == "complete", "Gate suite is not complete")
    entries = suite.get("trials", [])
    require(len(entries) == len(IDS) and {row.get("id") for row in entries} == set(IDS),
            "Expected exactly the seven declared adapter-gate trials")
    checks, trials = [], {}

    def passed(name, **details):
        checks.append({"name": name, "passed": True, **details})

    for entry in entries:
        identity = entry["id"]
        training = Path(entry["training_dir"]).resolve()
        if not (training / "results.json").is_file():
            training = training / "results"
        analysis = Path(entry["analysis_dir"]).resolve()
        result = read_json(training / "results.json")
        config = result["config"]
        manifest = read_json(training / "manifest.json")
        rows = read_json(training / "training_metrics.json")
        evaluation = read_json(analysis / "evaluation.json")
        analysis_manifest = read_json(analysis / "manifest.json")
        trajectory = load_arrays(training / "trajectory.npz")
        checkpoints = load_arrays(training / "checkpoints.npz")
        stationary = identity.endswith("stationary")
        adaptive = identity.startswith(("identity_", "halving_", "arithmetic_"))
        focus = identity.startswith("focus_")
        method = "ga_focus" if focus else "ga"
        require(manifest.get("status") == "complete", f"{identity}: incomplete training")
        require(analysis_manifest.get("status") == "complete", f"{identity}: incomplete analysis")
        require(manifest["method"] == method, f"{identity}: relabeled training method")
        validate_baseline_config(config, method)
        expected = {
            "method": method, "num_generations": 6, "task_interval": 3,
            "pop_size": 16, "num_evals": 3, "episode_length": 500,
            "num_params": 386, "num_tasks": 1 if stationary else 2,
            "task_sequence": [0, 0] if stationary else [0, 1],
            "seed": 3001, "trial": 3002, "eval_episodes": 10,
            "population_snapshot_interval": 0 if focus else 1,
            "snapshot_members": 128 if focus else 0,
            "schedule": "task0" if stationary else "switch", "task_warmup": 0,
        }
        for name, value in expected.items():
            require(config.get(name) == value, f"{identity}: unexpected config {name}")
        require(read_json(training / "config.json") == config,
                f"{identity}: config.json disagrees with results")
        require(result.get("env_steps") == 144000 and result.get("env_steps_per_generation") == 24000,
                f"{identity}: training budget differs")
        require(manifest.get("seed") == 3001 and manifest.get("trial") == 3002,
                f"{identity}: seed/trial receipt differs")
        require(len(rows) == GENERATIONS, f"{identity}: missing generation rows")
        task_indices = [0] * 6 if stationary else [0, 0, 0, 1, 1, 1]
        require([row["generation"] for row in rows] == list(range(6)),
                f"{identity}: generation order differs")
        require([row["task"] for row in rows] == task_indices,
                f"{identity}: task schedule differs")
        if not focus:
            require(trajectory["populations"].shape == (6, 16, 386),
                    f"{identity}: evaluated population snapshots are incomplete")
            same_arrays(trajectory["population_generations"], np.arange(6, dtype=np.int64),
                        f"{identity}: snapshot generations")
            same_arrays(trajectory["population_tasks"], np.asarray(task_indices, dtype=np.int64),
                        f"{identity}: snapshot tasks")
        require(set(checkpoints) == {"finalgen", "incumbent", "centroid", "noise_vectors"},
                f"{identity}: unexpected checkpoint arrays")
        for name in ("finalgen", "incumbent", "centroid"):
            require(checkpoints[name].shape == (2, 386), f"{identity}: checkpoint shape differs")
        vectors = np.asarray(result["noise_vectors"], dtype=np.float32)
        same_arrays(trajectory["noise_vectors"], vectors, f"{identity}: trajectory task vectors")
        same_arrays(checkpoints["noise_vectors"], vectors[config["task_sequence"]],
                    f"{identity}: checkpoint task vectors")
        require(all(np.isfinite(value).all() for value in (*trajectory.values(), *checkpoints.values())),
                f"{identity}: nonfinite saved policy or task vectors")
        for name, digest in analysis_manifest.get("input_sha256", {}).items():
            require(sha256(training / name) == digest, f"{identity}: analysis input changed: {name}")
        for name, digest in analysis_manifest.get("output_sha256", {}).items():
            require(sha256(analysis / name) == digest, f"{identity}: analysis output changed: {name}")
        require(set(analysis_manifest.get("input_sha256", {})) >= {
            "manifest.json", "results.json", "config.json", "training_metrics.json", "checkpoints.npz"
        }, f"{identity}: missing analysis input receipts")
        require(set(analysis_manifest.get("output_sha256", {})) >= {
            "evaluation.json", "checkpoint-metadata.json", "summary.json"
        }, f"{identity}: missing analysis output receipts")
        require(evaluation.get("method") == method and evaluation.get("trial") == 3002
                and evaluation.get("num_tasks") == 2 and evaluation.get("episodes") == 10
                and evaluation.get("eval_seed") != 3001,
                f"{identity}: fresh evaluation identity differs")
        require(evaluation.get("eval_seed") == analysis_manifest.get("eval_seed"),
                f"{identity}: fresh evaluation seed receipt differs")
        expected_sources = {"finalgen", "incumbent", "centroid"}
        require(set(evaluation.get("agent_sources", [])) == expected_sources,
                f"{identity}: checkpoint evaluation sources differ")
        evaluated = evaluation.get("per_task", [])
        require(len(evaluated) == 6 and {(row["source"], row["task_idx"]) for row in evaluated}
                == {(source, phase) for source in expected_sources for phase in range(2)},
                f"{identity}: incomplete fresh checkpoint evaluation")
        episodes = 0
        for row in evaluated:
            fields = {"returns", "prev_returns" if row["task_idx"] else "zero_shot_next_returns"}
            require(set(row) == {"source", "task_idx", *fields},
                    f"{identity}: unexpected return-vector schema")
            for field in fields:
                values = np.asarray(row[field])
                require(values.shape == (10,) and np.isfinite(values).all()
                        and (values >= 0).all() and (values <= 500).all(),
                        f"{identity}: invalid fresh evaluation returns")
                episodes += values.size

        files = {
            **{f"training/{name}": sha256(training / name) for name in (
                "manifest.json", "results.json", "config.json", "training_metrics.json",
                "trajectory.npz", "checkpoints.npz")},
            **{f"analysis/{name}": sha256(analysis / name) for name in (
                "manifest.json", "evaluation.json", "checkpoint-metadata.json", "summary.json")},
        }
        trial = {
            "result": result, "config": config, "records": rows,
            "trajectory": trajectory, "checkpoints": checkpoints, "evaluation": evaluation,
            "input_sha256": files, "posthoc_episodes": episodes,
            "array_receipts": {"trajectory": array_receipts(trajectory),
                               "checkpoints": array_receipts(checkpoints)},
        }
        if not focus:
            trace = training / "adapter-trace"
            if not trace.is_dir():
                trace = Path(entry["training_dir"]).resolve() / "adapter-trace"
            observed = read_json(trace / "manifest.json")
            require(observed.get("status") == "complete", f"{identity}: incomplete adapter trace")
            require(observed.get("algorithm_variant") == ("ga_adaptive" if adaptive else "ga"),
                    f"{identity}: wrong adapter variant")
            require(observed.get("jax_backend") == "cpu", f"{identity}: wrong backend")
            require(observed.get("upstream_commit") == manifest.get("upstream_commit")
                    == analysis_manifest.get("upstream_commit"),
                    f"{identity}: inconsistent upstream revisions")
            states, state_receipts = [], {}
            expected_names = [f"state_{generation:04d}.npz" for generation in range(7)]
            require([row["file"] for row in observed["states"]] == expected_names,
                    f"{identity}: incomplete or reordered state traces")
            require(sorted(observed["artifact_sha256"]) == expected_names,
                    f"{identity}: incomplete trace artifact receipt")
            for generation, row in enumerate(observed["states"]):
                filename = row["file"]
                require(sha256(trace / filename) == observed["artifact_sha256"][filename],
                        f"{identity}: state trace checksum differs")
                state = load_arrays(trace / filename)
                expected_fields = set(NATIVE_FIELDS)
                if adaptive:
                    expected_fields |= {"memory", "invalid_update"}
                if generation == 0:
                    expected_fields |= {"init_key", "initial_mean"}
                require(set(state) == expected_fields, f"{identity}: state fields differ")
                require(state["archive"].shape == (8, 386) and state["fitness"].shape == (8,)
                        and state["sigma"].shape == () and state["generation"].shape == (),
                        f"{identity}: invalid native state shapes")
                require(state["archive"].dtype == np.float32 and state["fitness"].dtype == np.float32
                        and state["generation"].dtype == np.int32,
                        f"{identity}: invalid native state dtypes")
                require(row["array_shapes"] == {name: list(value.shape) for name, value in state.items()},
                        f"{identity}: state shape receipt differs")
                if generation == 0:
                    require(state["init_key"].shape == (2,) and state["init_key"].dtype == np.uint32
                            and state["initial_mean"].shape == (386,)
                            and state["initial_mean"].dtype == np.float32
                            and np.isfinite(state["initial_mean"]).all(),
                            f"{identity}: invalid initialization evidence")
                require(int(state["generation"]) == generation
                        == row["completed_generations"], f"{identity}: invalid state generation")
                require(state["sigma"].dtype == np.float32 and np.isfinite(state["sigma"])
                        and row["sigma_next"] == float(state["sigma"]),
                        f"{identity}: invalid state width")
                require(np.isfinite(state["archive"]).all(), f"{identity}: nonfinite archive")
                require(np.isposinf(state["fitness"]).all() if generation == 0
                        else np.isfinite(state["fitness"]).all(), f"{identity}: invalid stored loss")
                if adaptive:
                    require(state["memory"].shape == (4,) and state["memory"].dtype == np.float32
                            and np.isfinite(state["memory"]).all(), f"{identity}: invalid memory")
                    require(state["invalid_update"].shape == ()
                            and state["invalid_update"].dtype == np.bool_
                            and not bool(state["invalid_update"]), f"{identity}: invalid update flag")
                    require(row["memory"] == state["memory"].tolist(),
                            f"{identity}: memory receipt differs")
                state_receipts[filename] = array_receipts(state)
                states.append(state)
            if adaptive:
                name = "initial.py" if identity.startswith("identity_") else identity.split("_")[0] + ".py"
                program = load_program(ROOT / "tasks" / "cartpole_adaptive" / name)
                require(observed.get("program") == program.metadata(),
                        f"{identity}: candidate identity differs")
            else:
                require(observed.get("program") is None, f"{identity}: plain GA has a candidate")
            require([float(state["sigma"]) for state in states[1:]] == [row["sigma"] for row in rows],
                    f"{identity}: raw sigma must log the post-tell next width")
            trial.update(states=states, trace=observed)
            trial["array_receipts"]["states"] = state_receipts
            files["training/adapter-trace/manifest.json"] = sha256(trace / "manifest.json")
            passed(f"{identity}: native states, source identity, next-width logging")
        trials[identity] = trial
        passed(f"{identity}: budget, task schedule, saved arrays, and fresh evaluation",
               nominal_training_steps=144000, posthoc_evaluation_episodes=episodes)

    for condition in ("stationary", "switching"):
        plain, adapted = trials[f"plain_{condition}"], trials[f"identity_{condition}"]
        require(plain["config"] == adapted["config"], f"{condition}: identity settings differ")
        require(plain["records"] == adapted["records"], f"{condition}: identity raw metrics differ")
        require({k: v for k, v in plain["result"].items() if k != "elapsed_seconds"}
                == {k: v for k, v in adapted["result"].items() if k != "elapsed_seconds"},
                f"{condition}: identity reward/task results differ")
        require(plain["evaluation"] == adapted["evaluation"],
                f"{condition}: fresh checkpoint episode returns differ")
        for generation, (a, b) in enumerate(zip(plain["states"], adapted["states"], strict=True)):
            for field in NATIVE_FIELDS + (("init_key", "initial_mean") if generation == 0 else ()):
                same_arrays(a[field], b[field], f"{condition}: identity generation {generation} {field}")
            same_arrays(b["memory"], np.zeros(4, dtype=np.float32), f"{condition}: identity memory")
            require(float(b["sigma"]) == .5, f"{condition}: identity width changed")
        for artifact in ("trajectory", "checkpoints"):
            require(set(plain[artifact]) == set(adapted[artifact]), f"{condition}: array sets differ")
            for name in plain[artifact]:
                same_arrays(plain[artifact][name], adapted[artifact][name],
                            f"{condition}: identity {artifact}/{name}")
        require(plain["trace"]["source_sha256"] == adapted["trace"]["source_sha256"]
                and plain["trace"]["thread_environment"] == adapted["trace"]["thread_environment"]
                and plain["trace"]["cpu_affinity"] == adapted["trace"]["cpu_affinity"],
                f"{condition}: identity runtime or source mismatch")
        passed(f"identity_{condition}: exact native initialization, six updates, populations, "
               "checkpoints, complete raw metrics, and fresh episode returns",
               states_compared=7, snapshot_arrays_compared=len(plain["trajectory"]),
               checkpoint_arrays_compared=len(plain["checkpoints"]))

    reference, halving = trials["identity_switching"], trials["halving_switching"]
    for generation, state in enumerate(halving["states"]):
        require(float(state["sigma"]) == 0.5 * 0.5 ** generation,
                f"Halving width incorrect after generation {generation}")
        same_arrays(state["memory"], np.full(4, generation, dtype=np.float32),
                    f"Halving memory persistence after generation {generation}")
    for field in NATIVE_FIELDS + ("init_key", "initial_mean"):
        same_arrays(reference["states"][0][field], halving["states"][0][field],
                    f"Halving initial native state {field}")
    for field in ("archive", "fitness", "generation"):
        same_arrays(reference["states"][1][field], halving["states"][1][field],
                    f"Halving first selected state {field}")
    same_arrays(reference["trajectory"]["populations"][0], halving["trajectory"]["populations"][0],
                "Halving first evaluated population")
    same_arrays(reference["trajectory"]["populations"][1, 8:],
                halving["trajectory"]["populations"][1, 8:], "Halving next re-scored archive")
    same_arrays(halving["trajectory"]["populations"][1, 8:], halving["states"][1]["archive"],
                "Halving offspring/archive ordering")
    require(not np.array_equal(reference["trajectory"]["populations"][1, :8],
                               halving["trajectory"]["populations"][1, :8]),
            "Halving did not change second-generation offspring")
    # Same key and parent/noise draw: x_full=p+.5*noise, x_half=p+.25*noise.
    # Recover p independently from the saved populations, then require a member
    # of the identical preceding archive for each child (crossover is zero).
    recovered_parents = (2 * halving["trajectory"]["populations"][1, :8]
                         - reference["trajectory"]["populations"][1, :8])
    archive = reference["states"][1]["archive"]
    parent_errors = np.max(np.abs(recovered_parents[:, None, :] - archive[None, :, :]), axis=2)
    residual = float(np.max(np.min(parent_errors, axis=1)))
    require(residual <= 2e-6, "Halving offspring are not consistent with unchanged parents/noise")
    passed("halving: memory persists across switch and width changes the next population",
           sigma_used=[float(state["sigma"]) for state in halving["states"][:-1]],
           sigma_next=[row["sigma"] for row in halving["records"]],
           parent_reconstruction_max_absolute_error=residual,
           parent_reconstruction_tolerance=2e-6)

    arithmetic = trials["arithmetic_switching"]
    for state in arithmetic["states"]:
        require(.001 <= float(state["sigma"]) <= 2, "Arithmetic width is outside harness bounds")
        same_arrays(state["memory"], np.zeros(4, dtype=np.float32), "Arithmetic unchanged memory")
    require(any(float(state["sigma"]) != .5 for state in arithmetic["states"][1:]),
            "Arithmetic control never actuated its update")
    passed("arithmetic: finite bounded executable updates and unchanged memory",
           sigma_used=[float(state["sigma"]) for state in arithmetic["states"][:-1]],
           sigma_next=[row["sigma"] for row in arithmetic["records"]])

    focus = trials["focus_switching"]
    require(focus["config"]["searcher_resolved"]["num_offspring"] == 7
            and focus["config"]["searcher_resolved"]["num_elites"] == 8,
            "Focus centroid slot was not reserved inside the population budget")
    require(all(.00001 <= row["sigma"] <= .5 for row in focus["records"]),
            "Native Focus width is outside its declared bounds")
    passed("FocusGA: true method identity, explicit settings, and fixed budget",
           offspring=7, rescored_archive=8, training_centroid=1, population=16,
           centroid_accounting="Native source and resolved configuration; no Focus population snapshot")

    for identity in IDS:
        if identity.endswith("switching"):
            same_arrays(reference["trajectory"]["noise_vectors"],
                        trials[identity]["trajectory"]["noise_vectors"],
                        f"{identity}: common switching tasks")
    passed("all switching controls share task draws and nominal training budget")
    stable_suite = {key: value for key, value in suite.items()
                    if key not in {"verification", "finished_at", "wall_seconds"}}
    suite_digest = hashlib.sha256(json.dumps(stable_suite, sort_keys=True, allow_nan=False,
                                            separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": 1, "status": "passed", "suite_projection_sha256": suite_digest,
        "suite_projection_excluded_fields": ["verification", "finished_at", "wall_seconds"],
        "checker_sha256": sha256(__file__),
        "support_source_sha256": {
            name: sha256(ROOT / name) for name in (
                "src/shinka_crl/adaptive.py", "src/shinka_crl/baseline_contract.py")
        },
        "comparison": "exact shape/dtype/numerical equality; no compressed-file equality claim",
        "array_hash": "SHA256(canonical JSON shape/dtype + newline + C-order value bytes)",
        "checks": checks,
        "trials": [{"id": identity, "input_sha256": trials[identity]["input_sha256"],
                    "array_receipts": trials[identity]["array_receipts"]} for identity in IDS],
        "nominal_training_steps": 144000 * len(IDS),
        "posthoc_evaluation_episodes": sum(trial["posthoc_episodes"] for trial in trials.values()),
        "interpretation": "Adapter correctness and control execution only; no method-ranking evidence",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"Refusing to replace existing verification output: {args.output}")
    report = verify(args.suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "checks": len(report["checks"]),
                      "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
