"""Run a restricted mutation program through the unchanged pinned GA trainer.

This production entry point writes one small, atomically updated receipt. It
does not request population snapshots or save per-generation optimizer states.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
SOURCE_FILES = (
    "scripts/adaptive_train.py", "src/shinka_crl/adaptive.py", "src/shinka_crl/experiment.py",
    "src/shinka_crl/baseline_contract.py", "src/shinka_crl/search.py", "src/shinka_crl/pilot.py",
    "requirements/cpu.lock", "upstream.lock.json",
)
ARTIFACT_FILES = ("results.json", "config.json", "training_metrics.json",
                  "checkpoints.npz", "trajectory.npz")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_receipt(path, value, *, create=False):
    """Exclusive initial creation; atomic replacement for progress and completion."""
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    if create:
        with Path(path).open("x", encoding="utf-8") as handle:
            handle.write(payload)
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".adaptive-receipt-", dir=Path(path).parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def native_argument(arguments, flag):
    require(arguments.count(flag) == 1, f"Require one explicit native {flag}")
    index = arguments.index(flag)
    require(index + 1 < len(arguments) and not arguments[index + 1].startswith("--"),
            f"Missing native {flag} value")
    return arguments[index + 1]


class StateObserver:
    """Observe host states without any random draw, policy query, or JAX trace."""

    def __init__(self, receipt, persist):
        self.receipt, self.persist = receipt, persist
        self.last_generation = -1
        self.widths = []

    def record(self, state):
        generation = int(state.generation)
        if generation == self.last_generation:
            return
        require(generation == self.last_generation + 1, "Observer skipped a native generation")
        require(state.sigma.shape == () and state.memory.shape == (4,)
                and str(state.sigma.dtype) == "float32" and str(state.memory.dtype) == "float32",
                "Adaptive state width/memory shape or dtype differs")
        width = float(state.sigma)
        memory = [float(value) for value in state.memory.tolist()]
        finite_width, finite_memory = math.isfinite(width), all(math.isfinite(value) for value in memory)
        invalid = bool(state.invalid_update)
        if generation == 0:
            require(width == .5 and memory == [0.] * 4 and not invalid,
                    "Adaptive initialization differs from the frozen contract")
        else:
            self.widths.append(width if finite_width else None)
        self.receipt.update(
            state_initialized=True, completed_generations=generation,
            sigma_next_final=width if finite_width else None,
            sigma_next_final_finite=finite_width,
            memory_final=[value if math.isfinite(value) else None for value in memory],
            memory_final_finite=finite_memory, invalid_update=invalid,
        )
        self.last_generation = generation
        self.persist()


def validate_width_artifacts(training, observer):
    from shinka_crl.baseline_contract import validate_baseline_config

    result = json.loads((training / "results.json").read_text())
    config = json.loads((training / "config.json").read_text())
    rows = json.loads((training / "training_metrics.json").read_text())
    require(result.get("config") == config, "Native configuration artifacts disagree")
    validate_baseline_config(config, "ga")
    count = config.get("num_generations")
    require(type(count) is int and count > 0 and count == observer.last_generation
            and len(rows) == count and len(observer.widths) == count,
            "Native and observed completed-generation counts differ")
    require([row.get("generation") for row in rows] == list(range(count)),
            "Native generation records are missing or reordered")
    widths = [row.get("sigma") for row in rows]
    # Float32 representation of the lower bound can exceed the decimal by a
    # few ulps. The core clips in float32 and all observed widths stay finite.
    require(all(type(value) in (int, float) and math.isfinite(value)
                and .001 <= value <= 2 for value in widths), "Invalid raw mutation-width record")
    require(widths == observer.widths, "Raw sigma must record the observed post-tell next width")
    require(observer.receipt["invalid_update"] is False
            and observer.receipt["memory_final_finite"] is True,
            "Nonfinite adaptive output cannot be successful")
    require(config.get("population_snapshot_interval") == 0,
            "Production adapter must not request diagnostic population snapshots")
    used = [.5, *widths[:-1]]
    return {
        "completed_generations": count, "raw_rows": len(rows),
        "raw_sigma_timing": "after_tell_next_generation",
        "matched_host_observations": True, "final_update_validated": True,
        "sigma_used_initial": .5, "sigma_used_final": used[-1],
        "sigma_next_final": widths[-1], "sigma_next_min": min(widths),
        "sigma_next_max": max(widths),
        "sigma_used_sha256": hashlib.sha256(json.dumps(used, separators=(",", ":")).encode()).hexdigest(),
    }


def main(argv=None):
    from shinka_crl.adaptive import AdaptiveGASearcher, load_program
    from shinka_crl.experiment import verify_upstream
    from shinka_crl.search import SEARCH_THREAD_ENV

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--program-path", type=Path, required=True)
    parser.add_argument("--receipt-path", type=Path, required=True)
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    native_args = args.native_args[1:] if args.native_args[:1] == ["--"] else args.native_args
    require(native_argument(native_args, "--method") == "ga", "Production adapter requires native GA")
    training = Path(native_argument(native_args, "--output_dir")).resolve()
    receipt_path, upstream, program_path = (path.resolve() for path in
                                           (args.receipt_path, args.upstream, args.program_path))
    require(receipt_path == training / "adaptive-manifest.json",
            "Receipt must be TRAINING_DIR/adaptive-manifest.json")
    require(not any((training / name).exists() for name in (*ARTIFACT_FILES, "resume.pkl")),
            "Native training outputs must be fresh; no mid-trial resume")
    training.mkdir(parents=True, exist_ok=True)
    sources = {name: digest(ROOT / name) for name in SOURCE_FILES}
    receipt = {
        "schema_version": 1, "algorithm_variant": "ga_adaptive", "status": "running",
        "started_at": utc_now(), "native_args": native_args,
        "launcher_invocation": [sys.executable, str(Path(__file__).resolve()),
                                *(sys.argv[1:] if argv is None else argv)],
        "program_path": str(program_path),
        "program_raw_sha256": digest(program_path) if program_path.is_file() else None,
        "source_sha256": sources,
        "state_initialized": False, "completed_generations": 0,
        "sigma_next_final": None, "memory_final": None, "invalid_update": None,
    }
    write_receipt(receipt_path, receipt, create=True)
    started = time.monotonic()
    original_environment = {key: os.environ.get(key) for key in SEARCH_THREAD_ENV}
    original_sys_path = list(sys.path)
    runner, original_factory = None, None
    observer = StateObserver(receipt, lambda: write_receipt(receipt_path, receipt))
    try:
        spec = load_program(program_path)
        receipt.update(program=spec.metadata(), upstream_commit=verify_upstream(upstream))
        os.environ.update(SEARCH_THREAD_ENV)
        sys.path.insert(0, str(upstream))
        import jax
        import source.runners.train_nes as native_runner
        import source.run as native
        from source.algorithms.ne.ga import GASearcher

        runner, original_factory = native_runner, native_runner.build_searcher
        receipt.update(jax_backend=jax.default_backend(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                       thread_environment={key: os.environ.get(key) for key in SEARCH_THREAD_ENV},
                       runtime={"python": sys.version.split()[0],
                                **{name: version(name) for name in ("jax", "jaxlib", "numpy")}})
        require(receipt["jax_backend"] == "cpu", "Production adapter requires CPU JAX")
        observer.persist()
        calls = []

        class ObservedAdaptive:
            def __init__(self, delegate):
                self.delegate = delegate

            def __getattr__(self, name):
                return getattr(self.delegate, name)

            def init(self, key, mean):
                state = self.delegate.init(key, mean)
                observer.record(state)
                return state

            def incumbent(self, state):
                # Persist completed work even if the core rejects this update.
                observer.record(state)
                return self.delegate.incumbent(state)

        def factory(method, num_params, population_size, **kwargs):
            require(method == "ga" and kwargs.get("sigma_init") == .5
                    and kwargs.get("elite_ratio") == .5
                    and kwargs.get("init_around_mean") is False
                    and kwargs.get("cross_over_rate", 0.) == 0.,
                    "Production adapter requires unchanged default CartPole GA settings")
            require(not calls, "Expected exactly one native searcher")
            delegate = original_factory(method, num_params, population_size, **kwargs)
            require(type(delegate) is GASearcher, "Expected the exact native plain GASearcher")
            calls.append(True)
            return ObservedAdaptive(AdaptiveGASearcher(delegate, spec))

        runner.build_searcher = factory
        require(native.main(native_args) == 0 and len(calls) == 1,
                "Native trainer did not complete an adaptive GA trial")
        receipt["width_validation"] = validate_width_artifacts(training, observer)
        require(sources == {name: digest(ROOT / name) for name in SOURCE_FILES},
                "Adapter sources changed during training")
        require(digest(program_path) == spec.source_sha256, "Candidate changed during training")
        verify_upstream(upstream)
        receipt["artifact_sha256"] = {name: digest(training / name) for name in ARTIFACT_FILES}
        receipt["status"] = "complete"
        return 0
    except BaseException as exc:
        receipt.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if hasattr(exc, "completed_generations"):
            require(exc.completed_generations == observer.last_generation,
                    "Invalid-update work count differs from observed state")
            receipt["failed_after_completed_generations"] = exc.completed_generations
        raise
    finally:
        if runner is not None:
            runner.build_searcher = original_factory
        sys.path[:] = original_sys_path
        for key, value in original_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        receipt.update(finished_at=utc_now(), wall_seconds=time.monotonic() - started)
        write_receipt(receipt_path, receipt)


if __name__ == "__main__":
    raise SystemExit(main())
