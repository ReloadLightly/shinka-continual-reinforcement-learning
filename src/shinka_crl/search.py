"""Frozen, staged subscription search and its preregistered random control.

The wrapper preserves native Shinka selection, patching and database behavior.
It adds source/runtime receipts, bounded stages and a fail-closed supervisor.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import runpy
import shlex
import signal
import sqlite3
import struct
import subprocess
import sys
import time

from shinka_crl.experiment import (
    DEFAULT_PYTHON, DEFAULT_UPSTREAM, REPO_ROOT, UPSTREAM_COMMIT,
    load_profile, nominal_training_steps, verify_upstream,
)
from shinka_crl.pilot import THREAD_ENV, read_json, require, sha256, write_json

RANDOM_SEED = 20261002
PROPOSAL_SLOTS = 24
TASK = REPO_ROOT / "tasks/cartpole_ga"
# Match all overrides made by the pinned native JobScheduler in both search arms.
SEARCH_THREAD_ENV = {
    **THREAD_ENV, "OMP_THREAD_LIMIT": "1", "OMP_DYNAMIC": "FALSE",
    "OMP_WAIT_POLICY": "PASSIVE", "MKL_DYNAMIC": "FALSE", "NUMEXPR_NUM_THREADS": "1",
    "NUMEXPR_MAX_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1", "GOTO_NUM_THREADS": "1",
}
SOURCE_FILES = (
    "src/shinka_crl/search.py", "scripts/run_search.py", "src/shinka_crl/pilot.py",
    "src/shinka_crl/experiment.py", "src/shinka_crl/profiles/search.json",
    "tasks/cartpole_ga/evaluate.py", "tasks/cartpole_ga/initial.py",
    "tasks/cartpole_ga/shinka-subscription.yaml", "scripts/subscription_headless.py",
    "requirements/cpu.lock", "upstream.lock.json",
)


def effective_key(settings: dict, population: int = 64) -> list:
    """Identity of the float32 mutation width and resolved upstream archive size."""
    sigma = struct.unpack("!f", struct.pack("!f", settings["sigma"]))[0]
    return [sigma, max(1, int(population * settings["elite_ratio"]))]


def program_source(settings: dict) -> str:
    return ("# EVOLVE-BLOCK-START\ndef get_ga_config():\n    return "
            + repr(settings) + "\n# EVOLVE-BLOCK-END\n")


def random_pool(seed: int = RANDOM_SEED) -> dict:
    """Generate the full control before any adaptive proposals are observed."""
    rng = random.Random(seed)
    entries, seen = [], {tuple(effective_key({"sigma": 0.5, "elite_ratio": 0.5}))}
    while len(entries) < PROPOSAL_SLOTS:
        settings = {"sigma": math.exp(rng.uniform(math.log(0.001), math.log(2.0))),
                    "elite_ratio": rng.uniform(0.05, 0.95)}
        key = effective_key(settings)
        if tuple(key) in seen:
            continue
        seen.add(tuple(key))
        index = len(entries) + 1
        entries.append({"index": index, "settings": settings, "effective_key": key,
                        "program_path": f"random_programs/candidate_{index:03d}.py",
                        "program_sha256": hashlib.sha256(
                            program_source(settings).encode()).hexdigest()})
    return {"schema_version": 1, "seed": seed, "count": PROPOSAL_SLOTS,
            "distribution": {"sigma": "log-uniform[0.001,2.0]",
                             "elite_ratio": "uniform[0.05,0.95]"},
            "effective_identity": "float32(sigma), max(1,int(64*elite_ratio))",
            "entries": entries}


def subscription_environment(model: str, timeout: int) -> dict[str, str]:
    guard = runpy.run_path(str(REPO_ROOT / "scripts/subscription_headless.py"))
    env = guard["subscription_env"](dict(os.environ))
    env.update(SEARCH_THREAD_ENV)
    env.update({"SHINKA_CODEX_MODEL": model,
                "SHINKA_HEADLESS_COMMAND": shlex.join([
                    sys.executable, str(REPO_ROOT / "scripts/subscription_headless.py")]),
                "SHINKA_CRL_PROFILE": "search", "SHINKA_CRL_TIMEOUT": str(timeout),
                "SHINKA_CRL_UPSTREAM": str(DEFAULT_UPSTREAM),
                "SHINKA_CRL_PYTHON": str(DEFAULT_PYTHON),
                "SHINKA_HEADLESS_TIMEOUT": str(timeout), "SHINKA_LLM_MAX_RETRIES": "1",
                "SHINKA_PRICING_MODE": "offline", "PYTHONHASHSEED": str(RANDOM_SEED)})
    return env


def runtime_identity(env: dict) -> dict:
    """Collect package/source versions and safe tooling metadata; never auth files."""
    probe = """
import importlib.metadata as m, json, platform, hashlib
from pathlib import Path
identity = {'python': platform.python_version(),
            'packages': {d.metadata['Name']: d.version for d in m.distributions()}}
try:
    d = m.distribution('shinka-evolve')
    identity['shinka_direct_url'] = json.loads(d.read_text('direct_url.json'))
    root = Path(d.locate_file('shinka'))
    identity['dotenv_files_present'] = [str(p) for p in
        (root.parent / '.env', Path.cwd() / '.env') if p.exists()]
    identity['shinka_source_sha256'] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*.py'))}
except m.PackageNotFoundError:
    import jax
    identity.update(jax_backend=jax.default_backend(),
                    jax_devices=[str(d) for d in jax.devices()])
print(json.dumps(identity))
"""
    runtime = {}
    for name, executable in (("harness", sys.executable), ("training", str(DEFAULT_PYTHON))):
        runtime[name] = json.loads(subprocess.check_output(
            [executable, "-c", probe], env=env, text=True, timeout=60))
    pin = read_json(REPO_ROOT / "upstream.lock.json")["shinka_evolve"]["commit"]
    require(runtime["harness"].get("shinka_direct_url", {}).get(
        "vcs_info", {}).get("commit_id") == pin, "Installed Shinka does not match source pin")
    require(runtime["training"].get("jax_backend") == "cpu", "Search requires CPU JAX")
    require(not runtime["harness"].get("dotenv_files_present"),
            "Shinka dotenv override exists; use a launch/runtime without dotenv overrides")
    check = subprocess.check_output(
        [sys.executable, str(REPO_ROOT / "scripts/subscription_headless.py"), "--check"],
        env=env, text=True, timeout=60)
    runtime["subscription_preflight"] = json.loads(check)
    guard = runpy.run_path(str(REPO_ROOT / "scripts/subscription_headless.py"))
    headless = guard["headless_command"](env)
    codex = guard["require_tool"]("codex", env)
    runtime["proposal_tools"] = {
        "codex_version": subprocess.check_output(
            [codex, "--version"], env=env, text=True, timeout=20).strip(),
        "node_version": subprocess.check_output(
            [headless[0], "--version"], env=env, text=True, timeout=20).strip(),
        "headless_cli_sha256": sha256(Path(headless[1])),
        "headless_package_sha256": sha256(Path(headless[1]).parent.parent / "package.json"),
    }
    return runtime


def make_plan(*, model: str, affinity: list[int], timeout: int, env: dict) -> dict:
    require(bool(re.fullmatch(r"gpt-[A-Za-z0-9.-]+", model)), "Invalid Codex model")
    verify_upstream(DEFAULT_UPSTREAM)
    profile = load_profile("search")
    return {
        "schema_version": 1, "purpose": "development configuration-search integration",
        "profile": profile, "upstream_commit": UPSTREAM_COMMIT,
        "source_sha256": {name: sha256(REPO_ROOT / name) for name in SOURCE_FILES},
        "model": model, "reasoning_effort": "medium", "auth": "chatgpt",
        "cpu_affinity": affinity, "thread_environment": SEARCH_THREAD_ENV,
        "training_python": str(DEFAULT_PYTHON), "harness_python": sys.executable,
        "training_timeout_seconds": timeout, "proposal_timeout_seconds": timeout,
        "proposal_slots": PROPOSAL_SLOTS, "stages": [2, 5, 13, 25],
        "outer_random_seed": RANDOM_SEED, "random_pool_seed": RANDOM_SEED,
        "random_pool": random_pool(),
        "nominal_training_steps_per_candidate": len(profile["seeds"]) *
            nominal_training_steps(profile, "ga"),
        "max_model_requests_per_slot": 1,
        "duplicates": "consume a slot and are evaluated again; charge actual seed work",
        "failure_policy": "stop on terminal proposal/evaluation failure; no automatic retry",
        "rng_policy": "save/restore Python and NumPy host RNG; restart initialization, model "
                      "responses and async ordering prevent an uninterrupted-trajectory guarantee",
        "runtime": runtime_identity(env),
    }


def database_snapshot(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=2) as connection:
        connection.row_factory = sqlite3.Row
        if not connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='programs'").fetchone():
            return []
        # children_count, archive membership and other selection state legitimately change.
        rows = connection.execute(
            "SELECT id,generation,parent_id,code,combined_score,correct,public_metrics,"
            "private_metrics FROM programs ORDER BY generation,id").fetchall()
        return [dict(row) for row in rows]


def artifact_receipts(output: Path) -> dict:
    paths = [*sorted((output / "shinka").glob("gen_*")),
             *sorted((output / "random").glob("candidate_*"))]
    return {str(path.relative_to(output)): sha256(path)
            for directory in paths for path in sorted(directory.rglob("*"))
            if path.is_file() and "__pycache__" not in path.parts}


def validate_saved_state(output: Path, state: dict) -> None:
    for name, digest in state.get("artifact_sha256", {}).items():
        path = output / name
        require(path.is_file() and sha256(path) == digest,
                f"Completed search artifact changed: {name}")
    require(database_snapshot(output / "shinka/programs.sqlite") == state.get("programs", []),
            "Search database changed after the last recorded stage")
    saved_rng = state.get("rng_sha256")
    if saved_rng:
        require(sha256(output / "shinka/rng_state.json") == saved_rng,
                "Saved host RNG state changed")
    if state.get("model_requests_sha256"):
        require(sha256(output / "model_requests.jsonl") == state["model_requests_sha256"],
                "Model request ledger changed")


def stop_process_group(process: subprocess.Popen) -> None:
    """Terminate evaluator/trainer descendants as well as the native runner."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def monitored_run(command: list[str], *, log: Path, env: dict,
                  timeout: int, shinka: Path | None = None) -> dict:
    started, reason = time.monotonic(), None
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("x") as stream:
        process = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout:
                    reason = "session timeout"
                if shinka is not None:
                    if list(shinka.glob("gen_*/failure.json")):
                        reason = "terminal proposal failure; preserve slot and stop"
                    try:
                        if any(not row["correct"] for row in database_snapshot(
                                shinka / "programs.sqlite")):
                            reason = "terminal evaluation failure; preserve slot and stop"
                    except sqlite3.OperationalError:
                        pass  # A live transaction is retried on the next monitor tick.
                if reason:
                    stop_process_group(process)
                    break
                time.sleep(0.5)
        except BaseException:
            stop_process_group(process)
            raise
    return {"returncode": process.returncode, "stop_reason": reason,
            "wall_seconds": time.monotonic() - started}


def native_command(output: Path, target: int) -> list[str]:
    return [sys.executable, "-m", "shinka_crl.search", "--native",
            str(output / "shinka/rng_state.json"), "--task-dir", str(TASK),
            "--config-fname", "shinka-subscription.yaml", "--results_dir",
            str(output / "shinka"), "--num_generations", str(target)]


def run_search(*, output: Path, model: str = "gpt-6.1-sol", cpus: int = 2,
               timeout: int = 600, target: int | None = None, random_count: int = 0,
               prepare_only: bool = False, resume: bool = False) -> dict:
    require(cpus > 0 and timeout > 0, "CPU count and timeout must be positive")
    require(target is None or target in (2, 5, 13, 25), "Use a declared target: 2,5,13,25")
    require(0 <= random_count <= PROPOSAL_SLOTS, "random_count must be in [0,24]")
    require(prepare_only or target is not None or random_count, "Choose a stage or random prefix")
    require(not (target is not None and random_count), "Run search and random stages separately")
    output = Path(output).resolve()
    affinity = sorted(os.sched_getaffinity(0))[:cpus]
    require(len(affinity) == cpus, "Requested CPU count exceeds available affinity")
    os.sched_setaffinity(0, affinity)
    env = subscription_environment(model, timeout)
    env["SHINKA_SUBSCRIPTION_LEDGER"] = str(output / "model_requests.jsonl")
    env["SHINKA_CRL_EXPECTED_THREAD_ENV"] = json.dumps(SEARCH_THREAD_ENV, sort_keys=True)
    env["SHINKA_CRL_EXPECTED_AFFINITY"] = json.dumps(affinity)
    plan = make_plan(model=model, affinity=affinity, timeout=timeout, env=env)
    if output.exists():
        require(resume, f"Output exists; explicitly use --resume: {output}")
        require(read_json(output / "plan.json") == plan,
                "Frozen search plan/source/runtime changed; use a new results directory")
        require(read_json(output / "random_pool.json") == plan["random_pool"],
                "Frozen random pool changed")
        state = read_json(output / "state.json")
        validate_saved_state(output, state)
    else:
        require(not resume, "Cannot resume a nonexistent search")
        output.mkdir(parents=True)
        write_json(output / "plan.json", plan)
        write_json(output / "random_pool.json", plan["random_pool"])
        (output / "random_programs").mkdir()
        for entry in plan["random_pool"]["entries"]:
            (output / entry["program_path"]).write_text(program_source(entry["settings"]))
        state = {"schema_version": 1, "status": "prepared", "sessions": [],
                 "programs": [], "artifact_sha256": {}, "random_completed": 0}
        write_json(output / "state.json", state)
    for entry in plan["random_pool"]["entries"]:
        require(sha256(output / entry["program_path"]) == entry["program_sha256"],
                "Frozen random candidate changed")
    if prepare_only:
        return state
    require(state["status"] != "running", "Previous session interrupted; review before new work")
    if target is not None:
        require(not any(s["arm"] == "shinka" and s["status"] != "complete"
                        for s in state["sessions"]),
                "Failed search stage requires review; no automatic slot retry")
        count = len(state["programs"])
        require(count == 0 or count >= 2, "Native Shinka cannot safely resume only generation zero")
        require(target > count, "Target must exceed completed generations")
        command = native_command(output, target)
        work = [("shinka", target, command, output / "shinka")]
    else:
        require(random_count > state["random_completed"], "Random prefix already completed")
        work = []
        for entry in plan["random_pool"]["entries"][state["random_completed"]:random_count]:
            path = output / "random" / f"candidate_{entry['index']:03d}"
            require(not path.exists(), "Incomplete random attempt requires review; cannot overwrite")
            command = [sys.executable, str(TASK / "evaluate.py"), "--program_path",
                       str(output / entry["program_path"]), "--results_dir", str(path)]
            work.append(("random", entry["index"], command, path))
    for arm, endpoint, command, path in work:
        session = {"index": len(state["sessions"]) + 1, "arm": arm,
                   "endpoint": endpoint, "command": command, "status": "running"}
        state["sessions"].append(session)
        state["status"] = "running"
        write_json(output / "state.json", state)
        log = output / "sessions" / f"session_{session['index']:03d}.log"
        print(f"Running {arm} endpoint {endpoint}; log: {log}", flush=True)
        session_started = time.monotonic()
        try:
            result = monitored_run(command, log=log, env=env,
                                   timeout=(endpoint + 1) * 4 * timeout if arm == "shinka"
                                   else (len(plan["profile"]["seeds"]) + 1) * timeout,
                                   shinka=path if arm == "shinka" else None)
            session.update(result)
            require(result["returncode"] == 0 and not result["stop_reason"],
                    f"{arm} stage failed: {result}; inspect {log}")
            if arm == "shinka":
                rows = database_snapshot(path / "programs.sqlite")
                require([r["generation"] for r in rows] == list(range(endpoint))
                        and all(r["correct"] for r in rows),
                        "Native stage did not persist exactly the bounded generation prefix")
                require(not list(path.glob("gen_*/failure.json")), "Terminal failed proposal")
                require((path / "rng_state.json").is_file(), "Missing host RNG checkpoint")
            else:
                require(read_json(path / "correct.json").get("correct") is True,
                        "Random candidate evaluation failed")
                state["random_completed"] = endpoint
            session["status"] = "complete"
            state["status"] = "complete"
        except BaseException as exc:
            session.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            state["status"] = "failed"
            raise
        finally:
            session.setdefault("wall_seconds", time.monotonic() - session_started)
            state["programs"] = database_snapshot(output / "shinka/programs.sqlite")
            state["artifact_sha256"] = artifact_receipts(output)
            rng = output / "shinka/rng_state.json"
            state["rng_sha256"] = sha256(rng) if rng.is_file() else None
            ledger = output / "model_requests.jsonl"
            state["model_requests_sha256"] = sha256(ledger) if ledger.is_file() else None
            write_json(output / "state.json", state)
    return state


def native_main(rng_path: Path, arguments: list[str], *, outer_seed: int = RANDOM_SEED) -> int:
    """Preserve native host sampling RNG across graceful stage boundaries."""
    import importlib.metadata

    require(type(outer_seed) is int and 0 <= outer_seed < 2**32,
            "Outer seed must be an unsigned 32-bit integer")
    root = Path(importlib.metadata.distribution("shinka-evolve").locate_file("shinka"))
    require(not any(path.exists() for path in (root.parent / ".env", Path.cwd() / ".env")),
            "Shinka dotenv override appeared after preparation")
    import numpy as np
    from shinka.cli.run import main

    if rng_path.is_file():
        saved = read_json(rng_path)
        require(saved.get("outer_seed", RANDOM_SEED) == outer_seed,
                "Saved native RNG outer seed changed")
        state = saved["python"]
        random.setstate((state[0], tuple(state[1]), state[2]))
        state = saved["numpy"]
        np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32),
                             state[2], state[3], state[4]))
    else:
        random.seed(outer_seed)
        np.random.seed(outer_seed)
    result = main(arguments)
    state = np.random.get_state()
    rng_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(rng_path, {"outer_seed": outer_seed, "python": random.getstate(),
                         "numpy": [state[0], state[1].tolist(), *state[2:]]})
    return result


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] != "--native":
        raise SystemExit("Use scripts/run_search.py")
    import argparse

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--outer-seed", type=int, default=RANDOM_SEED)
    native_args, remaining = parser.parse_known_args(sys.argv[3:])
    raise SystemExit(native_main(Path(sys.argv[2]), remaining, outer_seed=native_args.outer_seed))
