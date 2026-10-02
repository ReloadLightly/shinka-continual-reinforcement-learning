"""Preview or execute one paper-budget GA development trial, never reporting seeds."""

import argparse
import json
from pathlib import Path

from shinka_crl.experiment import DEFAULT_PYTHON, DEFAULT_UPSTREAM
from shinka_crl.reference_timing import PROFILE, make_plan, run_reference_timing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Trainer process wall limit in seconds; default 2 hours")
    parser.add_argument("--analysis-timeout", type=int, default=600)
    parser.add_argument("--profile", choices=[PROFILE, "smoke"], default=PROFILE,
                        help="Smoke verifies instrumentation; it is not a reference result")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    kwargs = dict(output=args.results_dir, upstream=args.upstream, python=args.python,
                  cpus=args.cpus, timeout=args.timeout, analysis_timeout=args.analysis_timeout,
                  profile_name=args.profile)
    try:
        result = run_reference_timing(**kwargs) if args.execute else make_plan(**kwargs)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if not args.execute or result["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
