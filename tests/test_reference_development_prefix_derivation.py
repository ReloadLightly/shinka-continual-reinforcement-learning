"""Archived development evidence must survive later harness revisions."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "reports/reference-development-prefixes-20261003/derivation.py"
SPEC = importlib.util.spec_from_file_location("development_prefix_derivation", PATH)
derivation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(derivation)


def test_source_archive_identity_does_not_depend_on_current_harness_bytes(monkeypatch):
    expected = derivation.derive()
    original_digest = derivation.sha256
    changed = ROOT / "src/shinka_crl/reference_comparison.py"

    def changed_worktree_digest(path):
        return "0" * 64 if Path(path) == changed else original_digest(path)

    monkeypatch.setattr(derivation, "sha256", changed_worktree_digest)
    assert derivation.sha256(changed) == "0" * 64
    assert derivation.derive() == expected
    archives = {row["method"]: row["evidence"]["source_archive"] for row in expected["rows"]}
    assert archives["ga"]["is_recorded_runtime_commit"] is True
    assert archives["es"]["is_recorded_runtime_commit"] is False
    assert all(len(archive["source_sha256"]) == 12 for archive in archives.values())


def test_mismatched_archive_commit_is_rejected():
    protocol = derivation.read(derivation.HERE / "protocol.json")
    specification = dict(protocol["sources"]["es"])
    specification["source_archive_commit"] = protocol["sources"]["ga"]["source_archive_commit"]
    with pytest.raises(ValueError, match="Archived source"):
        derivation.verify_sources((derivation.HERE / specification["report"]).resolve(),
                                  specification, protocol)
