"""Preview or execute the matched 18-trial CartPole pilot, with resumable stages."""

import argparse
import json
from pathlib import Path

from shinka_crl.experiment import DEFAULT_PYTHON, DEFAULT_UPSTREAM, load_profile
from shinka_crl.pilot import PROFILES, make_jobs, run_pilot, training_steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON))
    parser.add_argument("--cpus", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--max-trials", type=int, help="New completed trials in this session")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        profiles = {name: load_profile(name) for name in PROFILES}
        jobs = make_jobs(profiles)
        print(json.dumps({"jobs": jobs, "profiles": profiles, "nominal_training_steps": sum(
            training_steps(profiles[job["profile"]], job["method"]) for job in jobs)}, indent=2))
        return
    result = run_pilot(output=args.results_dir, upstream=args.upstream, python=args.python,
                       cpus=args.cpus, timeout=args.timeout, resume=args.resume,
                       max_trials=args.max_trials)
    print(json.dumps({key: result[key] for key in (
        "status", "planned_trials", "completed_trials", "wall_seconds")}, indent=2))


if __name__ == "__main__":
    main()
