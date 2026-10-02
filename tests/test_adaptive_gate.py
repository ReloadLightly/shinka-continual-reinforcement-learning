"""Gate provenance, terminal failure evidence, and caller restoration without training."""

import os
from pathlib import Path

import pytest

from shinka_crl import adaptive_gate as gate
from shinka_crl.experiment import build_command
from shinka_crl.pilot import read_json, write_json
from shinka_crl.reference_timing import artifact_hashes


@pytest.fixture
def mock_gate(monkeypatch):
    affinity = {"current": {0, 1, 2, 3}}
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(affinity["current"]))
    monkeypatch.setattr(os, "sched_setaffinity", lambda _, values: affinity.update(current=set(values)))
    monkeypatch.setattr(gate, "source_hashes", lambda: {"frozen.py": "a" * 64})
    monkeypatch.setattr(gate, "verify_upstream", lambda _: gate.UPSTREAM_COMMIT)
    monkeypatch.setattr(gate, "runtime_probe", lambda *args: {"jax_backend": "cpu"})

    def training(output, command, method, variant):
        output.mkdir(parents=True)
        write_json(output / "manifest.json", {
            "status": "complete", "command": command, "method": method,
            "algorithm_variant": variant, "wall_seconds": 2.0,
        })
        (output / "checkpoints.npz").write_bytes(b"binary checkpoint fixture")

    def wrapped(job, plan, output):
        training(Path(job["training_dir"]), job["command"], job["method"], job["algorithm_variant"])

    def focus(**kwargs):
        command = build_command(**{key: kwargs[key] for key in
                                  ("profile", "method", "seed", "trial", "output_dir", "upstream", "python")})
        training(kwargs["output_dir"], command, "ga_focus", "ga_focus")

    def analysis(**kwargs):
        output = kwargs["output_dir"]
        output.mkdir(parents=True)
        manifest = read_json(kwargs["run_dir"] / "manifest.json")
        summary = {"method": manifest["method"], "eval_seed": kwargs["eval_seed"]}
        write_json(output / "summary.json", summary)
        write_json(output / "manifest.json", {"status": "complete", "wall_seconds": 0.5})
        return summary

    def trace(suite, output, python):
        verification = {"identity": True, "actuation": True}
        write_json(output, verification)
        return verification

    monkeypatch.setattr(gate, "_run_wrapped", wrapped)
    monkeypatch.setattr(gate, "run_experiment", focus)
    monkeypatch.setattr(gate, "run_analysis", analysis)
    monkeypatch.setattr(gate, "validate_analysis", lambda **kwargs: read_json(
        kwargs["output_dir"] / "summary.json"))
    monkeypatch.setattr(gate, "check_traces", trace)
    return affinity


def test_gate_plan_is_disjoint_and_accounts_for_all_controls(tmp_path, mock_gate):
    output = tmp_path / "run"
    plan = gate.make_plan(output=output)
    assert not output.exists()
    assert (plan["seed"], plan["trial"], plan["eval_seed"]) == (3001, 3002, 903001)
    assert len(plan["trials"]) == 7
    assert plan["planned_training_steps_nominal"] == 7 * 6 * 16 * 3 * 500
    assert {job["algorithm_variant"] for job in plan["trials"]} == {"ga", "ga_adaptive", "ga_focus"}
    assert len([job for job in plan["trials"] if job["profile"]["num_tasks"] == 1]) == 2
    assert all(job["profile"]["seeds"] == [3001] for job in plan["trials"])
    assert all(job["profile"]["ne"]["task_interval"] == 3 for job in plan["trials"])
    focus = plan["trials"][-1]
    assert focus["method"] == "ga_focus" and focus["program"] is None
    assert "method=ga_focus" in focus["command"]
    for job in plan["trials"][:-1]:
        assert "population_snapshot_interval=1" in job["command"]
        assert "snapshot_members=0" in job["command"]
        assert ("--program-path" in job["command"]) == (job["program"] is not None)


@pytest.mark.parametrize("kwargs", [{"cpus": 1}, {"cpus": True}, {"timeout": 0}, {"timeout": True}])
def test_gate_rejects_invalid_runtime_before_creating_output(tmp_path, mock_gate, kwargs):
    output = tmp_path / "run"
    with pytest.raises(ValueError):
        gate.run_gate(output=output, **kwargs)
    assert not output.exists()


def test_complete_gate_restores_caller_and_exports_binary_hashes(tmp_path, mock_gate, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "caller-value")
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    run = tmp_path / "run"
    suite = gate.run_gate(output=run)
    assert suite["status"] == "complete" and len(suite["trials"]) == 7
    assert mock_gate["current"] == {0, 1, 2, 3}
    assert os.environ["OMP_NUM_THREADS"] == "caller-value"
    assert "JAX_PLATFORMS" not in os.environ
    assert artifact_hashes(run) == read_json(run / "receipt.json")
    report = tmp_path / "report"
    summary = gate.export_gate(run_dir=run, report_dir=report)
    assert summary["status"] == "complete" and len(summary["trials"]) == 7
    checks = read_json(report / "checksums.json")
    assert "plain_stationary/training/checkpoints.npz" in checks["original_artifact_sha256"]
    assert not list(report.rglob("*.npz"))
    assert "$RUN" in (report / "raw/plan.json").read_text()
    with pytest.raises(ValueError, match="new"):
        gate.export_gate(run_dir=run, report_dir=report)


def test_failed_attempt_is_preserved_and_exportable(tmp_path, mock_gate, monkeypatch):
    def fail(job, plan, output):
        training = Path(job["training_dir"])
        training.mkdir(parents=True)
        write_json(training / "manifest.json", {"status": "failed", "wall_seconds": 0.1})
        raise RuntimeError("injected trainer failure")

    monkeypatch.setattr(gate, "_run_wrapped", fail)
    run = tmp_path / "run"
    with pytest.raises(RuntimeError, match="injected"):
        gate.run_gate(output=run)
    assert mock_gate["current"] == {0, 1, 2, 3}
    suite = read_json(run / "suite.json")
    assert suite["status"] == suite["trials"][0]["status"] == "failed"
    assert len(suite["trials"]) == 1 and "injected" in suite["error"]
    assert artifact_hashes(run) == read_json(run / "receipt.json")
    summary = gate.export_gate(run_dir=run, report_dir=tmp_path / "report")
    assert summary["status"] == "failed" and summary["verification"] is None


def test_receipt_failure_still_restores_affinity_and_environment(tmp_path, mock_gate, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "caller-value")
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)

    def fail(_):
        raise OSError("injected receipt failure")

    monkeypatch.setattr(gate, "artifact_hashes", fail)
    with pytest.raises(OSError, match="receipt"):
        gate.run_gate(output=tmp_path / "run")
    assert mock_gate["current"] == {0, 1, 2, 3}
    assert os.environ["OMP_NUM_THREADS"] == "caller-value"
    assert "JAX_PLATFORMS" not in os.environ


def test_export_rejects_original_receipt_tampering(tmp_path, mock_gate):
    run = tmp_path / "run"
    gate.run_gate(output=run)
    (run / "plain_stationary/training/checkpoints.npz").write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="receipt"):
        gate.export_gate(run_dir=run, report_dir=tmp_path / "report")
    assert not (tmp_path / "report").exists()


@pytest.mark.parametrize("change,match", [
    ("source", "Source changed"), ("seed", "protocol metadata"),
    ("method", "identity"), ("status", "Incomplete trial"),
    ("command", "Training command"), ("analysis", "Analysis differs"),
])
def test_export_rederives_protocol_beyond_file_receipts(tmp_path, mock_gate, monkeypatch, change, match):
    run = tmp_path / "run"
    gate.run_gate(output=run)
    suite = read_json(run / "suite.json")
    if change == "source":
        monkeypatch.setattr(gate, "source_hashes", lambda: {"frozen.py": "b" * 64})
    elif change == "seed":
        suite["seed"] = 42
    elif change == "method":
        suite["trials"][-1]["method"] = "ga"
    elif change == "status":
        suite["trials"][0]["status"] = "failed"
    elif change == "command":
        path = run / "plain_stationary/training/manifest.json"
        manifest = read_json(path)
        manifest["command"] = ["wrong command"]
        write_json(path, manifest)
    else:
        suite["trials"][0]["analysis"]["eval_seed"] = 1
    write_json(run / "suite.json", suite)
    write_json(run / "receipt.json", artifact_hashes(run))
    with pytest.raises(ValueError, match=match):
        gate.export_gate(run_dir=run, report_dir=tmp_path / "report")
    assert not (tmp_path / "report").exists()
