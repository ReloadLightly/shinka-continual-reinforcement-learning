"""Revalidate adaptive control/cache evidence and publish compact text artifacts."""
import argparse
import json
from pathlib import Path

from shinka_crl.adaptive_evaluation import export_study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True, type=Path)
    parser.add_argument("--report-dir", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(export_study(study=args.study_dir, report_dir=args.report_dir), indent=2))
