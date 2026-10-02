"""A stale CPU lock must fail before installation can alter an environment."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bootstrap_cpu", REPO / "scripts/bootstrap_cpu.py")
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("changed", ["upstream", "input"])
def test_rejects_changed_dependency_provenance(tmp_path, changed):
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    upstream_lock = upstream / "uv.lock"
    upstream_lock.write_text("version = 1\n")
    source = tmp_path / "cpu.in"
    source.write_text("jax==0.5.3\n")
    lock = tmp_path / "cpu.lock"
    lock.write_text(
        f"# Upstream uv.lock SHA-256: {digest(upstream_lock)}\n"
        f"# Input SHA-256: {digest(source)}\n"
    )
    BOOTSTRAP.verify_lock(upstream, lock, source)
    target = upstream_lock if changed == "upstream" else source
    target.write_text(target.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="CPU lock provenance mismatch"):
        BOOTSTRAP.verify_lock(upstream, lock, source)


def test_checked_lock_matches_its_direct_requirements():
    lock = (REPO / "requirements/cpu.lock").read_text()
    assert f"# Input SHA-256: {digest(REPO / 'requirements/cpu.in')}" in lock.splitlines()
