"""Run a bounded adaptive Shinka block through the guarded subscription route."""
import argparse
import json
from pathlib import Path

from shinka_crl.adaptive_search import ARMS, RANDOM_SEED, run_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--target-generations", type=int, choices=[5, 13, 25], default=5)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--outer-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--arm", choices=ARMS, default="evolutionary")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    state = run_search(output=args.results_dir, study=args.study_dir,
                       target=args.target_generations, model=args.model, timeout=args.timeout,
                       prepare_only=args.prepare_only, resume=args.resume,
                       outer_seed=args.outer_seed, arm=args.arm)
    print(json.dumps({"status": state["status"], "programs": len(state["programs"])}, indent=2))


if __name__ == "__main__":
    main()
