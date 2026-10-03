"""Verify and export prepared, completed or stopped adaptive continuation evidence."""
import argparse
import json
from pathlib import Path

from shinka_crl.adaptive_continuation import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = export(output=args.results_dir, report=args.report_dir)
    print(json.dumps({key: summary[key] for key in ("status", "slots_consumed", "programs_evaluated")}))


if __name__ == "__main__":
    main()
