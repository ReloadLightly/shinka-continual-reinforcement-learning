"""Native scheduler integration without changing the frozen evaluator protocol."""

import json
from pathlib import Path
import runpy

import pytest

from shinka_crl.experiment import REPO_ROOT
from shinka_crl.pilot import read_json, write_json
from shinka_crl.reference_timing import artifact_hashes


@pytest.fixture
def adapter():
    return runpy.run_path(str(REPO_ROOT / "tasks/cartpole_adaptive/shinka_evaluate.py"))


@pytest.mark.parametrize("correct", [True, False])
def test_scheduler_logs_stay_outside_signed_evaluation(tmp_path, monkeypatch, adapter, correct):
    output = tmp_path / "results"
    output.mkdir()
    for name in ("job_log.out", "job_log.err"):
        (output / name).write_text("scheduler-owned\n")
    program = tmp_path / "main.py"
    program.write_text("def update_sigma(sigma, stats, memory):\n    return sigma, memory\n")
    called = []

    def evaluate(program_path, results_dir):
        called.append((program_path, results_dir))
        assert not results_dir.exists()
        results_dir.mkdir()
        write_json(results_dir / "request.json", {"status": "complete" if correct else "failed"})
        write_json(results_dir / "correct.json", {"correct": correct, "error": None if correct else "invalid"})
        write_json(results_dir / "metrics.json", {"combined_score": .3 if correct else 0.})
        write_json(results_dir / "receipt.json", artifact_hashes(results_dir))
        return correct

    def load(path):
        assert Path(path) == REPO_ROOT / "tasks/cartpole_adaptive/evaluate.py"
        return {"evaluate_program": evaluate}

    monkeypatch.setattr(runpy, "run_path", load)
    assert adapter["evaluate_program"](program, output) is correct
    assert called == [(program, output / "evaluation")]
    child = output / "evaluation"
    for name in ("correct.json", "metrics.json"):
        assert (output / name).read_bytes() == (child / name).read_bytes()
    receipt = read_json(child / "receipt.json")
    assert artifact_hashes(child) == receipt
    assert not any("job_log" in name for name in receipt)
    (output / "job_log.out").write_text("later scheduler output\n")
    assert artifact_hashes(child) == receipt


@pytest.mark.parametrize("existing", ["evaluation", "correct.json", "metrics.json"])
def test_existing_evidence_is_not_replaced_or_reused(tmp_path, monkeypatch, adapter, existing):
    output = tmp_path / "results"
    output.mkdir()
    target = output / existing
    target.write_text("original evidence")
    before = artifact_hashes(output)

    def forbidden(*args, **kwargs):
        pytest.fail("Existing request must not invoke the evaluator")

    monkeypatch.setattr(runpy, "run_path", forbidden)
    with pytest.raises(ValueError, match="evidence already exists"):
        adapter["evaluate_program"](tmp_path / "main.py", output)
    assert artifact_hashes(output) == before


def test_frozen_evaluator_rejects_invalid_program_with_durable_failure(tmp_path, monkeypatch, adapter):
    """The real frozen entrypoint supplies the failure contract; no training runs."""
    monkeypatch.delenv("SHINKA_ADAPTIVE_STUDY", raising=False)
    output = tmp_path / "results"
    output.mkdir()
    (output / "job_log.err").touch()
    program = tmp_path / "main.py"
    program.write_text("unavailable study should fail before candidate evaluation\n")
    assert adapter["evaluate_program"](program, output) is False
    correct = json.loads((output / "correct.json").read_text())
    assert correct["correct"] is False
    assert "SHINKA_ADAPTIVE_STUDY" in correct["error"]
    assert read_json(output / "metrics.json")["combined_score"] == 0.
    assert (output / "evaluation/correct.json").read_bytes() == (output / "correct.json").read_bytes()


def test_cli_passes_paths_and_returns_failure(tmp_path, monkeypatch, adapter):
    program = tmp_path / "main.py"
    output = tmp_path / "results"
    called = []

    def evaluate(program_path, results_dir):
        called.append((program_path, results_dir))
        return False

    monkeypatch.setitem(adapter["main"].__globals__, "evaluate_program", evaluate)
    monkeypatch.setattr("sys.argv", ["evaluate.py", "--program_path", str(program), "--results_dir", str(output)])
    assert adapter["main"]() == 1
    assert called == [(program, output)]
