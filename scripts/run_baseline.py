"""Print a run plan by default; --execute launches the pinned upstream trainer."""

import argparse
import json
import os
from pathlib import Path
import shlex

from shinka_crl.experiment import (
    DEFAULT_PYTHON, DEFAULT_UPSTREAM, build_command, load_profile, run_experiment,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=["smoke", "search", "paper-cartpole"], default="smoke")
    parser.add_argument("--method", choices=["ga", "es", "ppo"], default="ga")
    parser.add_argument("--results-dir", type=Path, default=Path("results/baselines"))
    parser.add_argument("--upstream", type=Path,
                        default=Path(os.environ.get("SHINKA_CRL_UPSTREAM", DEFAULT_UPSTREAM)))
    parser.add_argument("--python", default=os.environ.get("SHINKA_CRL_PYTHON", str(DEFAULT_PYTHON)))
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per trial")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    profile = load_profile(args.profile)
    for index, seed in enumerate(profile["seeds"], 1):
        trial = index if args.profile == "paper-cartpole" else seed + 1
        kwargs = dict(profile=profile, method=args.method, seed=seed, trial=trial,
                      output_dir=args.results_dir / args.profile / args.method / f"trial_{trial}",
                      upstream=args.upstream, python=args.python)
        if args.execute:
            print(json.dumps(run_experiment(**kwargs, timeout=args.timeout), indent=2))
        else:
            print(shlex.join(build_command(**kwargs)))


if __name__ == "__main__":
    main()
