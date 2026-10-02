"""Preview or run the frozen five-seed finalist comparison."""
import argparse
import json
from pathlib import Path

from shinka_crl.experiment import DEFAULT_PYTHON, DEFAULT_UPSTREAM, nominal_training_steps
from shinka_crl.validation import finalists, run_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finalists", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        frozen = finalists(args.finalists.resolve())
        n = len(frozen["candidates"]) * len(frozen["profile"]["seeds"])
        print(json.dumps({"profile": frozen["profile"], "candidates": frozen["candidates"],
                          "planned_trials": n, "nominal_training_steps": n *
                          nominal_training_steps(frozen["profile"], "ga")}, indent=2))
        return
    result = run_validation(frozen=args.finalists, output=args.results_dir, upstream=args.upstream,
                            python=args.python, cpus=args.cpus, timeout=args.timeout,
                            resume=args.resume, max_trials=args.max_trials)
    print(json.dumps({k: result[k] for k in ("status", "planned_trials", "completed_trials",
                                            "wall_seconds", "selected_winners")}, indent=2))


if __name__ == "__main__":
    main()

