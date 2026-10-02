"""Export immutable full or partial reference timing evidence without binary checkpoints."""

import argparse
import json
from pathlib import Path

from shinka_crl.reference_timing import export_reference_timing


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_reference_timing(run_dir=args.run_dir, report_dir=args.report_dir),
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
