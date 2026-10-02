"""Evaluate declarative GA settings with the pinned upstream CartPole runner."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any


BOUNDS = {"sigma": (0.001, 2.0), "elite_ratio": (0.05, 0.95)}
SEARCH_PROFILES = {"smoke", "search"}


def parse_ga_config(program_path: Path) -> dict[str, float]:
    """Read exactly one literal-returning function without importing candidate code."""
    if program_path.stat().st_size > 65_536:
        raise ValueError("Candidate is larger than 64 KiB")
    tree = ast.parse(program_path.read_text(encoding="utf-8"))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Candidate must contain only def get_ga_config()")
    function = tree.body[0]
    arguments = function.args
    if (
        function.name != "get_ga_config"
        or function.decorator_list
        or function.returns is not None
        or function.type_comment is not None
        or getattr(function, "type_params", [])
        or arguments.posonlyargs
        or arguments.args
        or arguments.kwonlyargs
        or arguments.vararg is not None
        or arguments.kwarg is not None
        or arguments.defaults
        or arguments.kw_defaults
        or len(function.body) != 1
        or not isinstance(function.body[0], ast.Return)
    ):
        raise ValueError("get_ga_config must have no arguments and only a literal return")
    returned = function.body[0].value
    if not isinstance(returned, ast.Dict) or len(returned.keys) != len(BOUNDS):
        raise ValueError("Return exactly sigma and elite_ratio in a literal dictionary")
    # Validate AST keys before literal_eval, which would otherwise hide duplicates.
    keys = [key.value if isinstance(key, ast.Constant) else None for key in returned.keys]
    if keys.count("sigma") != 1 or keys.count("elite_ratio") != 1:
        raise ValueError("Return exactly sigma and elite_ratio, without duplicate keys")
    try:
        settings = ast.literal_eval(returned)
    except (ValueError, TypeError, SyntaxError, RecursionError) as exc:
        raise ValueError("GA settings must be numeric literals") from exc
    normalized = {}
    for name, (lower, upper) in BOUNDS.items():
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a numeric literal, not a boolean")
        if not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{name} must be finite and in [{lower}, {upper}]")
        normalized[name] = float(value)
    return normalized


def load_profile(name: str) -> dict[str, Any]:
    from shinka_crl.experiment import load_profile as load

    return load(name)


def default_paths() -> tuple[Path, Path]:
    from shinka_crl.experiment import DEFAULT_PYTHON, DEFAULT_UPSTREAM

    return DEFAULT_UPSTREAM, DEFAULT_PYTHON


def run_experiment(**kwargs: Any) -> dict[str, Any]:
    from shinka_crl.experiment import run_experiment as run

    return run(**kwargs)


def evaluate_program(program_path: Path, results_dir: Path) -> bool:
    """Run fixed development seeds and always write Shinka's result contract."""
    results_dir.mkdir(parents=True, exist_ok=True)
    profile_name = os.environ.get("SHINKA_CRL_PROFILE", "smoke")
    metrics: dict[str, Any] = {
        "combined_score": 0.0,
        "public": {"profile": profile_name, "method": "ga", "seeds_completed": 0},
        "private": {},
    }
    correctness: dict[str, Any] = {"correct": False, "error": None}
    try:
        if profile_name not in SEARCH_PROFILES:
            raise ValueError(
                "Shinka development profile must be smoke or search; paper-cartpole is held out"
            )
        settings = parse_ga_config(program_path)
        profile = load_profile(profile_name)
        seeds = profile["seeds"]
        if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise ValueError("Profile must contain nonempty integer seeds")
        upstream_default, python_default = default_paths()
        upstream = Path(os.environ.get("SHINKA_CRL_UPSTREAM", str(upstream_default))).resolve()
        python = os.environ.get("SHINKA_CRL_PYTHON", str(python_default))
        timeout = int(os.environ.get("SHINKA_CRL_TIMEOUT", "1800"))
        if timeout <= 0:
            raise ValueError("SHINKA_CRL_TIMEOUT must be positive")
        summaries = []
        for seed in seeds:
            # Upstream derives task offsets from trial; reserve trials 1..10 for final runs.
            trial = seed + 1
            result = run_experiment(
                profile=profile,
                method="ga",
                seed=seed,
                trial=trial,
                output_dir=results_dir / f"seed_{seed}",
                upstream=upstream,
                python=python,
                ga_settings=settings,
                timeout=timeout,
            )
            score = float(result["normalized_score"])
            mean_return = float(result["mean_return"])
            if not math.isfinite(score) or not math.isfinite(mean_return):
                raise ValueError(f"Seed {seed} returned non-finite metrics")
            summaries.append(
                {
                    "seed": seed,
                    "trial": trial,
                    "normalized_score": score,
                    "mean_return": mean_return,
                }
            )
            metrics["public"]["seeds_completed"] = len(summaries)
        metrics["combined_score"] = statistics.mean(
            result["normalized_score"] for result in summaries
        )
        metrics["public"].update(
            {
                **settings,
                "mean_return": statistics.mean(result["mean_return"] for result in summaries),
                "smoke_validation_only": profile_name == "smoke",
            }
        )
        metrics["private"]["seed_results"] = summaries
        correctness["correct"] = True
    except Exception as exc:
        metrics["combined_score"] = 0.0
        correctness["error"] = f"{type(exc).__name__}: {exc}"
    for filename, payload in (("metrics.json", metrics), ("correct.json", correctness)):
        (results_dir / filename).write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
    print(json.dumps({"combined_score": metrics["combined_score"], **correctness}, allow_nan=False))
    return correctness["correct"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program_path", type=Path, required=True)
    parser.add_argument("--results_dir", type=Path)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate candidate syntax only; no experiments or result files",
    )
    args = parser.parse_args()
    if args.validate_only:
        try:
            settings = parse_ga_config(args.program_path)
        except Exception as exc:
            print(json.dumps({"validation_only": True, "valid": False, "error": str(exc)}))
            return 1
        print(json.dumps({"validation_only": True, "valid": True, "ga_settings": settings}))
        return 0
    if args.results_dir is None:
        parser.error("--results_dir is required unless --validate-only is set")
    return 0 if evaluate_program(args.program_path, args.results_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
