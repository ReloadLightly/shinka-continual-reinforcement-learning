"""Shinka-compatible adaptive evaluation contract bound to an explicitly frozen study."""
import argparse
import os
from pathlib import Path

from shinka_crl.adaptive_evaluation import evaluate_candidate
from shinka_crl.pilot import write_json


def evaluate_program(program_path: Path, results_dir: Path) -> bool:
    try:
        study = os.environ.get("SHINKA_ADAPTIVE_STUDY")
        if not study:
            raise ValueError("Set SHINKA_ADAPTIVE_STUDY to a frozen adaptive-search study")
        evaluate_candidate(study=Path(study), request_dir=results_dir, program_path=program_path)
        return True
    except Exception as exc:
        results_dir.mkdir(parents=True, exist_ok=True)
        # Never replace a previously completed or failed request's evidence.
        if not (results_dir / "correct.json").exists():
            write_json(results_dir / "correct.json", {"correct": False, "error": f"{type(exc).__name__}: {exc}"})
        if not (results_dir / "metrics.json").exists():
            write_json(results_dir / "metrics.json", {"combined_score": 0., "public": {}, "private": {}})
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--program_path", required=True, type=Path)
    parser.add_argument("--results_dir", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(0 if evaluate_program(args.program_path, args.results_dir) else 1)
