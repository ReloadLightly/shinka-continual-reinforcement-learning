"""Validate and export a completed or explicitly partial finalist comparison."""
import argparse
import json
from pathlib import Path

from shinka_crl.validation import export_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = export_validation(args.runs_root, args.output, allow_partial=args.allow_partial)
    print(json.dumps({k: report[k] for k in ("status", "completed_trials", "planned_trials",
                                            "selected_winners")}, indent=2))


if __name__ == "__main__":
    main()
