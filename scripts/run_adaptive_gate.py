"""Preview or execute seven reduced-budget adapter checks without model calls."""
import argparse
import json
from pathlib import Path

from shinka_crl.adaptive_gate import make_plan, run_gate
from shinka_crl.experiment import DEFAULT_PYTHON, DEFAULT_UPSTREAM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    kwargs = dict(output=args.results_dir, upstream=args.upstream, python=args.python, timeout=args.timeout)
    result = run_gate(**kwargs) if args.execute else make_plan(**kwargs)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
