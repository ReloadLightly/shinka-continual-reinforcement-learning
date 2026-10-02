"""Prepare the frozen search, execute a native Shinka stage, or evaluate random controls."""

import argparse
import json
from pathlib import Path

from shinka_crl.search import run_search


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    stage = parser.add_mutually_exclusive_group(required=True)
    stage.add_argument("--prepare-only", action="store_true", help="Freeze without inference")
    stage.add_argument("--target-generations", type=int, choices=[2, 5, 13, 25])
    stage.add_argument("--random-count", type=int, default=0, help="Cumulative frozen control prefix")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_search(output=args.results_dir, model=args.model, cpus=args.cpus,
                        timeout=args.timeout, target=args.target_generations,
                        random_count=args.random_count, prepare_only=args.prepare_only,
                        resume=args.resume)
    print(json.dumps({"status": result["status"], "programs": len(result["programs"]),
                      "random_completed": result["random_completed"],
                      "sessions": len(result["sessions"])}, indent=2))


if __name__ == "__main__":
    main()
