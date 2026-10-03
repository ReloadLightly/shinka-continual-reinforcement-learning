"""Prepare or execute the separately reviewed adaptive continuation controller."""
import argparse
import json
from pathlib import Path

from shinka_crl.adaptive_continuation import prepare, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        if args.source_dir is None:
            parser.error("--prepare-only requires --source-dir")
        state = prepare(source=args.source_dir, output=args.results_dir)
    else:
        if args.source_dir is not None:
            parser.error("--execute reads the source binding from the frozen continuation plan")
        state = run(args.results_dir)
    print(json.dumps({"status": state["status"], "database_rows": len(state["programs"])}))


if __name__ == "__main__":
    main()
