"""Fetch the pinned reference checkout; optionally install its separate environment."""

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from shinka_crl.experiment import DEFAULT_UPSTREAM, REPO_ROOT, verify_upstream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--install", action="store_true",
                        help="Run uv sync --frozen in upstream (large JAX/CUDA/Torch environment)")
    args = parser.parse_args()
    target = args.directory.resolve()
    pin = json.loads((REPO_ROOT / "upstream.lock.json").read_text())["continual_neuroevolution"]
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="checkout-", dir=target.parent) as temp:
            checkout = Path(temp) / "repo"
            subprocess.run(["git", "init", str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "remote", "add", "origin", pin["url"]],
                           check=True)
            subprocess.run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin",
                            pin["commit"]], check=True)
            subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", "FETCH_HEAD"],
                           check=True)
            verify_upstream(checkout)
            checkout.rename(target)
    print(f"Pinned reference ready: {target} @ {verify_upstream(target)}")
    if args.install:
        subprocess.run(["uv", "sync", "--frozen", "--python", "3.11"], cwd=target, check=True)


if __name__ == "__main__":
    main()
