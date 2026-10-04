"""Freeze or resume the adaptive development controls without model calls."""
import argparse
import json
from pathlib import Path

from shinka_crl.adaptive_evaluation import freeze_study, read_plan, run_controls
from shinka_crl.experiment import DEFAULT_PYTHON, DEFAULT_UPSTREAM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--upstream", default=DEFAULT_UPSTREAM, type=Path)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--timeout", default=1800, type=int)
    parser.add_argument("--max-controls", type=int)
    parser.add_argument("--development-seeds", nargs=3, type=int,
                        help="Fresh development partition; requires validation seeds and allocation audit")
    parser.add_argument("--validation-seeds", nargs=5, type=int,
                        help="Reserved validation partition; changes no training/evaluation settings")
    parser.add_argument("--seed-allocation", type=Path,
                        help="JSON audit of historical and newly allocated seed/task identities")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.resume:
        if any(value is not None for value in
               (args.development_seeds, args.validation_seeds, args.seed_allocation)):
            parser.error("--resume uses the frozen partitions; omit seed allocation arguments")
        plan = read_plan(args.study_dir.resolve())
    else:
        plan = freeze_study(output=args.study_dir, upstream=args.upstream, python=args.python,
                            timeout=args.timeout, development_seeds=args.development_seeds,
                            validation_seeds=args.validation_seeds, seed_allocation=args.seed_allocation)
    result = run_controls(study=args.study_dir, max_controls=args.max_controls) if args.execute else plan
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
