"""Finalize and evaluate a stopped PPO development checkpoint without further training."""

import argparse
import json
from pathlib import Path

from shinka_crl.reference_prefix import finalize_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-suite", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", type=int, default=6000)
    parser.add_argument("--allocation-seconds", type=int, default=28800)
    parser.add_argument("--prior-auxiliary-seconds", type=float, default=0,
                        help="Measured prior PPO verification/finalization cost within this allocation")
    args = parser.parse_args()
    try:
        result = finalize_prefix(suite=args.source_suite, output=args.output, endpoint=args.endpoint,
                                 allocation_seconds=args.allocation_seconds,
                                 prior_auxiliary_seconds=args.prior_auxiliary_seconds)
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"PPO prefix finalization stopped: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
