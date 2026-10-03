"""Verify and export a completed adaptive Shinka block and its reused evidence."""
import argparse
import json
from pathlib import Path

from shinka_crl.adaptive_search import export_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = export_search(output=args.results_dir, report=args.report_dir)
    print(json.dumps({k: summary[k] for k in ("status", "programs_evaluated", "cache_hits", "new_training_trials")}, indent=2))


if __name__ == "__main__":
    main()
