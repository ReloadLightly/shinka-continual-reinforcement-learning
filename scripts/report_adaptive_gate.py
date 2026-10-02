"""Verify and publish compact adaptive gate evidence, retaining binary hashes."""
import argparse
import json
from pathlib import Path

from shinka_crl.adaptive_gate import export_gate
from shinka_crl.experiment import DEFAULT_PYTHON


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    args = parser.parse_args()
    print(json.dumps(export_gate(run_dir=args.run_dir, report_dir=args.report_dir, python=args.python), indent=2))


if __name__ == "__main__":
    main()
