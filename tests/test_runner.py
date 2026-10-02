"""Exercise artifact and failure handling without installing the RL stack."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from shinka_crl import experiment


@pytest.fixture
def upstream_checkout(tmp_path, monkeypatch):
    """A real Git checkout with its expected revision patched for the test."""
    checkout = tmp_path / "upstream"
    subprocess.run(["git", "init", "--quiet", str(checkout)], check=True)
    (checkout / "README.md").write_text("Reference implementation fixture.\n")
    subprocess.run(["git", "-C", str(checkout), "add", "README.md"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "-c", "user.name=Runner Test",
         "-c", "user.email=runner@example.invalid", "commit", "--quiet", "-m", "Fixture"],
        check=True,
    )
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
    ).strip()
    monkeypatch.setattr(experiment, "UPSTREAM_COMMIT", revision)
    return checkout, revision


def replace_trainer(monkeypatch, run):
    # Keep Git verification and Python preflight real. Patching subprocess.run
    # globally would also replace the implementation behind check_output.
    monkeypatch.setattr(
        experiment, "subprocess",
        SimpleNamespace(run=run, check_output=subprocess.check_output, STDOUT=subprocess.STDOUT),
    )


def write_smoke_metrics(command, *, truncate=False):
    output = Path(command[command.index("--output_dir") + 1])
    # Both upstream trainers use `generation`, including PPO update records.
    records = [
        {"generation": i, "task": i // 2, "centroid_task0": 8, "centroid_task1": 24}
        for i in range(4)
    ]
    if truncate:
        records.pop()
    (output / "training_metrics.json").write_text(json.dumps(records))


@pytest.mark.parametrize("method", ["ga", "es", "ppo"])
def test_runner_writes_complete_provenance_and_summary(
    tmp_path, monkeypatch, upstream_checkout, method,
):
    checkout, revision = upstream_checkout
    output = tmp_path / "run"
    launches = []

    def train(command, **kwargs):
        launches.append((command, kwargs))
        kwargs["stdout"].write("Mock trainer finished.\n")
        write_smoke_metrics(command)
        return subprocess.CompletedProcess(command, 0)

    replace_trainer(monkeypatch, train)
    result = experiment.run_experiment(
        profile=experiment.load_profile("smoke"), method=method, seed=1001, trial=1002,
        output_dir=output, upstream=checkout, python=sys.executable, timeout=17,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert result == json.loads((output / "summary.json").read_text())
    assert result["normalized_score"] == 0.5
    assert result["metric_rows"] == 4
    assert manifest["status"] == "complete"
    assert manifest["upstream_commit"] == revision
    assert manifest["seed"] == 1001
    assert manifest["trial"] == 1002
    assert manifest["profile"] == experiment.load_profile("smoke")
    assert manifest["metrics_sha256"] == hashlib.sha256(
        (output / "training_metrics.json").read_bytes(),
    ).hexdigest()
    assert manifest["wall_seconds"] >= 0
    assert len(launches) == 1
    command, options = launches[0]
    assert manifest["command"] == command
    assert options["cwd"] == checkout.resolve()
    assert options["timeout"] == 17
    assert options["check"] is True
    assert "Mock trainer finished" in (output / "train.log").read_text()


@pytest.mark.parametrize("failure", ["nonzero", "timeout", "incomplete_metrics"])
def test_runner_records_failed_runs_without_success_summary(
    tmp_path, monkeypatch, upstream_checkout, failure,
):
    checkout, _ = upstream_checkout
    output = tmp_path / "failed"

    def train(command, **kwargs):
        kwargs["stdout"].write("Trainer started.\n")
        if failure == "nonzero":
            raise subprocess.CalledProcessError(7, command)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        write_smoke_metrics(command, truncate=True)
        return subprocess.CompletedProcess(command, 0)

    replace_trainer(monkeypatch, train)
    expected_exception = {
        "nonzero": subprocess.CalledProcessError,
        "timeout": subprocess.TimeoutExpired,
        "incomplete_metrics": ValueError,
    }[failure]
    with pytest.raises(expected_exception):
        experiment.run_experiment(
            profile=experiment.load_profile("smoke"), method="ga", seed=1001,
            output_dir=output, upstream=checkout, python=sys.executable,
        )

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["error"]
    assert manifest["wall_seconds"] >= 0
    assert "metrics_sha256" not in manifest
    assert not (output / "summary.json").exists()
    assert "Trainer started" in (output / "train.log").read_text()


def test_runner_refuses_to_overwrite_existing_artifacts(
    tmp_path, monkeypatch, upstream_checkout,
):
    checkout, _ = upstream_checkout
    output = tmp_path / "existing"
    output.mkdir()
    artifact = output / "summary.json"
    artifact.write_text('{"keep": true}\n')

    def unexpected_train(*args, **kwargs):
        pytest.fail("An existing output directory must not launch another trainer")

    replace_trainer(monkeypatch, unexpected_train)
    with pytest.raises(FileExistsError):
        experiment.run_experiment(
            profile=experiment.load_profile("smoke"), method="ga", seed=1001,
            output_dir=output, upstream=checkout, python=sys.executable,
        )
    assert artifact.read_text() == '{"keep": true}\n'
    assert not (output / "manifest.json").exists()


def test_runner_rejects_modified_reference_before_creating_output(
    tmp_path, monkeypatch, upstream_checkout,
):
    checkout, _ = upstream_checkout
    (checkout / "README.md").write_text("Unexpected reference change.\n")
    output = tmp_path / "dirty-reference"

    def unexpected_train(*args, **kwargs):
        pytest.fail("A changed reference checkout must not launch training")

    replace_trainer(monkeypatch, unexpected_train)
    with pytest.raises(ValueError, match="modified tracked files"):
        experiment.run_experiment(
            profile=experiment.load_profile("smoke"), method="ga", seed=1001,
            output_dir=output, upstream=checkout, python=sys.executable,
        )
    assert not output.exists()


def test_missing_interpreter_does_not_reserve_output_directory(tmp_path, upstream_checkout):
    checkout, _ = upstream_checkout
    output = tmp_path / "missing-interpreter"
    with pytest.raises(FileNotFoundError):
        experiment.run_experiment(
            profile=experiment.load_profile("smoke"), method="ga", seed=1001,
            output_dir=output, upstream=checkout, python=str(tmp_path / "absent-python"),
        )
    assert not output.exists()
