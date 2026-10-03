"""Export compact scientific evidence for the resource-limited PPO development prefix."""

import argparse
import json
from pathlib import Path

from shinka_crl.reference_prefix_report import DEFAULT_BUDGET_REPORT, export_reference_prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget-report", type=Path, default=DEFAULT_BUDGET_REPORT)
    args = parser.parse_args()
    try:
        result = export_reference_prefix(args.runs_root, args.output, budget_report=args.budget_report)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"PPO prefix report rejected: {exc}\n")
    print(json.dumps({"output": str(args.output), "mode": result["mode"],
                      "experiment_kind": result["experiment_kind"],
                      "completed_reporting_trials": result["completed_reporting_trials"]}, indent=2))


if __name__ == "__main__":
    main()
