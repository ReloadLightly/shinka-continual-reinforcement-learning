"""Install and verify the hash-locked CPU subset for upstream CartPole trainers.

First fetch the pinned reference with scripts/bootstrap_upstream.py. This
separate CPython 3.11 environment contains the GA/ES/PPO gymnax runtime only;
full-suite GPU experiments still use the reference project's own uv sync.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

from shinka_crl.experiment import DEFAULT_UPSTREAM, REPO_ROOT, verify_upstream

CPU_LOCK = REPO_ROOT / "requirements" / "cpu.lock"
CPU_INPUT = REPO_ROOT / "requirements" / "cpu.in"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_lock(upstream: Path, lock: Path = CPU_LOCK, source: Path = CPU_INPUT) -> None:
    """Refuse stale lock provenance instead of silently changing dependencies."""
    text = lock.read_text()
    for label, digest in (
        ("Upstream uv.lock SHA-256", sha256(upstream / "uv.lock")),
        ("Input SHA-256", sha256(source)),
    ):
        if f"# {label}: {digest}" not in text.splitlines():
            raise ValueError(f"CPU lock provenance mismatch: {label}; regenerate the CPU lock")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("The checked CPU lock targets Linux x86_64 with CPython 3.11")
    upstream = args.upstream.resolve()
    revision = verify_upstream(upstream)
    verify_lock(upstream)
    env_dir = upstream / ".venv"
    python = env_dir / "bin" / "python"
    install_env = {**os.environ, "UV_CONCURRENT_DOWNLOADS": "2",
                   "UV_CONCURRENT_INSTALLS": "2", "UV_CONCURRENT_BUILDS": "1"}
    if not python.exists():
        subprocess.run(["uv", "venv", "--python", "3.11", str(env_dir)],
                       check=True, env=install_env)
    version = json.loads(subprocess.check_output(
        [str(python), "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"],
        text=True))
    if version != [3, 11]:
        raise SystemExit(f"CPU environment must use Python 3.11; found {version} at {python}")
    subprocess.run(["uv", "pip", "sync", "--python", str(python), "--require-hashes",
                    str(CPU_LOCK)], check=True, env=install_env)
    subprocess.run(["uv", "pip", "check", "--python", str(python)], check=True)
    preflight_env = {**os.environ, "JAX_PLATFORMS": "cpu",
                     "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OMP_NUM_THREADS": "1",
                     "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                     "XLA_FLAGS": "--xla_cpu_multi_thread_eigen=false "
                                  "intra_op_parallelism_threads=1"}
    preflight = subprocess.check_output([
        str(python), "-c",
        "import json,platform; import jax,gymnax; "
        "from source.runners import train_nes,train_ppo; "
        "gymnax.make('CartPole-v1'); "
        "devices=jax.devices(); "
        "assert devices and all(d.platform == 'cpu' for d in devices); "
        "print(json.dumps({'python':platform.python_version(), 'jax':jax.__version__, "
        "'devices':[str(d) for d in devices], 'platform':platform.platform()}))",
    ], cwd=upstream, env=preflight_env, text=True)
    info = json.loads(preflight)
    packages = json.loads(subprocess.check_output(
        ["uv", "pip", "list", "--python", str(python), "--format", "json"], text=True))
    info.update(upstream_commit=revision, upstream_lock_sha256=sha256(upstream / "uv.lock"),
                cpu_lock_sha256=sha256(CPU_LOCK), packages=packages, package_count=len(packages),
                installer=subprocess.check_output(["uv", "--version"], text=True).strip())
    (env_dir / "cpu-environment.json").write_text(json.dumps(info, indent=2) + "\n")
    verify_upstream(upstream)
    print(json.dumps({key: value for key, value in info.items() if key != "packages"}, indent=2))


if __name__ == "__main__":
    main()
