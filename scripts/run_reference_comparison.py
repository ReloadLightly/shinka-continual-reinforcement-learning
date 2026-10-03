"""Preview or execute the declared CartPole reference comparison."""

import argparse
import json
from pathlib import Path

from shinka_crl.experiment import DEFAULT_PYTHON, DEFAULT_UPSTREAM
from shinka_crl.reference_comparison import make_plan, run_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["development", "diagnostic", "reporting"], required=True)
    parser.add_argument("--methods", nargs="+", choices=["ga", "es", "ppo"],
                        help="Reporting: ga es ppo (default), or ga es under the PPO resource amendment")
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=21600)
    parser.add_argument("--analysis-timeout", type=int, default=1800)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    kwargs = {name: getattr(args, name) for name in ("mode", "methods", "cpus", "timeout",
                                                   "analysis_timeout", "python", "upstream")}
    try:
        result = (run_comparison(output=args.results_dir, resume=args.resume,
                                 max_trials=args.max_trials, **kwargs)
                  if args.execute else make_plan(**kwargs))
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(1, f"Reference comparison stopped: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
