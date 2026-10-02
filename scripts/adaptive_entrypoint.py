"""Run the pinned GA with an optional restricted mutation program and diagnostic trace."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def main(argv=None):
    from shinka_crl.adaptive import AdaptiveGASearcher, load_program
    from shinka_crl.experiment import verify_upstream
    from shinka_crl.pilot import sha256, write_json
    from shinka_crl.search import SEARCH_THREAD_ENV

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--variant", choices=("plain", "adaptive"), required=True)
    parser.add_argument("--program-path", type=Path)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    native_args = args.native_args
    if native_args[:1] == ["--"]:
        native_args = native_args[1:]
    if (args.variant == "adaptive") != (args.program_path is not None):
        parser.error("Only the adaptive variant requires a program path")
    if (native_args.count("--method") != 1
            or native_args[native_args.index("--method") + 1:native_args.index("--method") + 2] != ["ga"]):
        parser.error("The adapter entry point accepts the native GA only")
    spec = load_program(args.program_path) if args.program_path else None
    upstream, trace = args.upstream.resolve(), args.trace_dir.resolve()
    revision = verify_upstream(upstream)
    trace.mkdir(parents=True, exist_ok=False)
    os.environ.update(SEARCH_THREAD_ENV)
    sys.path.insert(0, str(upstream))
    import numpy as np
    import jax
    import source.runners.train_nes as runner
    import source.run as native
    from source.algorithms.ne.ga import GASearcher

    sources = {name: sha256(ROOT / name) for name in (
        "scripts/adaptive_entrypoint.py", "src/shinka_crl/adaptive.py",
        "src/shinka_crl/experiment.py", "src/shinka_crl/search.py")}
    manifest = {"schema_version": 1, "status": "running", "upstream_commit": revision,
                "algorithm_variant": "ga_adaptive" if spec else "ga",
                "program": spec.metadata() if spec else None,
                "native_args": native_args, "source_sha256": sources,
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "thread_environment": {k: os.environ.get(k) for k in SEARCH_THREAD_ENV},
                "jax_backend": jax.default_backend(), "states": []}
    write_json(trace / "manifest.json", manifest)
    original_factory, calls = runner.build_searcher, []

    class TraceSearcher:
        """Host observer: native ask/tell and RNG are delegated without extra evaluation."""
        def __init__(self, delegate):
            self.delegate = delegate
            self.last_generation = -1

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def record(self, state, **extra):
            if isinstance(state.generation, jax.core.Tracer):
                raise ValueError("State trace must remain outside JIT")
            generation = int(state.generation)
            if generation == self.last_generation:
                return
            if generation != self.last_generation + 1:
                raise ValueError("State observer skipped a generation")
            arrays = {name: np.asarray(value) for name, value in state._asdict().items()}
            arrays.update({name: np.asarray(value) for name, value in extra.items()})
            filename = f"state_{generation:04d}.npz"
            np.savez_compressed(trace / filename, **arrays)
            manifest["states"].append({"completed_generations": generation, "file": filename,
                                       "sigma_next": float(state.sigma),
                                       "memory": arrays["memory"].tolist() if "memory" in arrays else None,
                                       "array_shapes": {k: list(v.shape) for k, v in arrays.items()}})
            self.last_generation = generation

        def init(self, key, mean):
            state = self.delegate.init(key, mean)
            self.record(state, init_key=jax.random.key_data(key), initial_mean=mean)
            return state

        def incumbent(self, state):
            value = self.delegate.incumbent(state)
            self.record(state)
            return value

    def factory(method, num_params, population_size, **kwargs):
        if method != "ga" or kwargs.get("elite_ratio") != .5 or kwargs.get("sigma_init") != .5:
            raise ValueError("Adaptive gate requires the unmodified default GA configuration")
        if kwargs.get("init_around_mean") is not False or kwargs.get("cross_over_rate", 0.) != 0.:
            raise ValueError("Adaptive gate requires native CartPole initialization and no crossover")
        delegate = original_factory(method, num_params, population_size, **kwargs)
        if type(delegate) is not GASearcher or calls:
            raise ValueError("Expected exactly one plain native GA searcher")
        calls.append(True)
        return TraceSearcher(AdaptiveGASearcher(delegate, spec) if spec else delegate)

    try:
        if manifest["jax_backend"] != "cpu":
            raise ValueError("This adapter protocol requires CPU JAX")
        runner.build_searcher = factory
        code = native.main(native_args)
        if code != 0 or len(calls) != 1 or len(manifest["states"]) < 2:
            raise ValueError("Native trainer did not complete an observed GA trial")
        if any(sha256(ROOT / name) != digest for name, digest in sources.items()):
            raise ValueError("Adapter sources changed during training")
        if spec and sha256(args.program_path) != spec.source_sha256:
            raise ValueError("Program changed during training")
        verify_upstream(upstream)
        manifest["status"] = "complete"
        return 0
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        if hasattr(exc, "completed_generations"):
            manifest["failed_after_completed_generations"] = exc.completed_generations
        raise
    finally:
        runner.build_searcher = original_factory
        manifest["artifact_sha256"] = {p.name: sha256(p) for p in sorted(trace.glob("*.npz"))}
        write_json(trace / "manifest.json", manifest)


if __name__ == "__main__":
    raise SystemExit(main())
