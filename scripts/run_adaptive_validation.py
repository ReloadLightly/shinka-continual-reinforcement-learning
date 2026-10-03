"""Freeze, preview or execute the separate adaptive reserved comparison."""
import argparse
import json
from pathlib import Path
from shinka_crl import adaptive_validation as validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--closure", type=Path)
    parser.add_argument("--diagnostic-study", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--review-sha256")
    parser.add_argument("--review-reason")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.closure:
        result = validation.freeze(closure_path=args.closure, output=args.frozen)
    elif args.diagnostic_study:
        result = validation.freeze_diagnostic(study=args.diagnostic_study, output=args.frozen)
    else:
        result = validation.read_plan(args.frozen.resolve())
    if args.execute:
        parser.error("--output is required with --execute") if args.output is None else None
        result = validation.run(frozen=args.frozen, output=args.output, max_trials=args.max_trials,
                                review_sha256=args.review_sha256, review_reason=args.review_reason,
                                retry_failed=args.retry_failed)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
