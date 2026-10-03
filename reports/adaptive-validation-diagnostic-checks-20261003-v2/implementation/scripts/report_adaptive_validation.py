"""Verify and publish compact adaptive reserved validation evidence."""
import argparse
import json
from pathlib import Path
from shinka_crl.adaptive_validation import export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(frozen=args.frozen, output=args.output, report=args.report), indent=2))


if __name__ == "__main__":
    main()
