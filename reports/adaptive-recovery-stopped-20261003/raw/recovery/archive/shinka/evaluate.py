"""Keep native Shinka scheduler logs outside the frozen evaluator's receipts."""

from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import shutil

from shinka_crl.experiment import REPO_ROOT


def evaluate_program(program_path: Path, results_dir: Path) -> bool:
    """Evaluate in a fresh child directory, then expose Shinka's two result files.

    Shinka owns ``job_log.out`` and ``job_log.err`` in the outer directory.
    The unchanged evaluator owns every file in ``evaluation`` and signs those
    files before returning. Mirroring its contract leaves that receipt intact.
    """
    results_dir = Path(results_dir)
    evaluation_dir = results_dir / "evaluation"
    contract_files = ("correct.json", "metrics.json")
    if evaluation_dir.exists() or any((results_dir / name).exists() for name in contract_files):
        raise ValueError("Native evaluation evidence already exists; preserve it for review")
    results_dir.mkdir(parents=True, exist_ok=True)
    evaluator = runpy.run_path(str(REPO_ROOT / "tasks/cartpole_adaptive/evaluate.py"))
    correct = evaluator["evaluate_program"](Path(program_path), evaluation_dir)
    for name in contract_files:
        source = evaluation_dir / name
        # Exclusive creation protects existing evidence even if a concurrent
        # launch passed the freshness check before this evaluator finished.
        with source.open("rb") as reader, (results_dir / name).open("xb") as writer:
            shutil.copyfileobj(reader, writer)
    return correct


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program_path", required=True, type=Path)
    parser.add_argument("--results_dir", required=True, type=Path)
    args = parser.parse_args()
    return 0 if evaluate_program(args.program_path, args.results_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
