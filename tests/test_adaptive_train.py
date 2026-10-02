"""Production adapter receipts and failure boundaries, without launching training."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest

import shinka_crl.adaptive as adaptive
import shinka_crl.experiment as experiment
from shinka_crl.search import SEARCH_THREAD_ENV


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("adaptive_train_test", ROOT / "scripts/adaptive_train.py")
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def state(generation, sigma=.5, memory=None, invalid=False):
    return types.SimpleNamespace(
        generation=np.asarray(generation, dtype=np.int32),
        sigma=np.asarray(sigma, dtype=np.float32),
        memory=np.asarray([0.] * 4 if memory is None else memory, dtype=np.float32),
        invalid_update=np.asarray(invalid),
    )


def baseline_config():
    return {
        "env": "CartPole-v1", "method": "ga", "hidden_dims": [16, 16], "num_params": 386,
        "first_task_clean": True, "task_warmup": 0, "objective": "mean", "obs_norm": False,
        "pop_size": 8, "sigma": .5, "searcher_kwargs": {"elite_ratio": .5, "init_around_mean": False},
        "searcher_resolved": {"refresh": True, "num_elites": 4, "num_offspring": 4,
                              "variation": "gaussian", "sigma": .5, "cross_over_rate": 0.},
        "num_generations": 2, "population_snapshot_interval": 0,
    }


def write_training(path, *, rows=None, config=None):
    config = baseline_config() if config is None else config
    rows = [{"generation": 0, "sigma": .25}, {"generation": 1, "sigma": .125}] if rows is None else rows
    for name, value in (("results.json", {"config": config}), ("config.json", config),
                        ("training_metrics.json", rows)):
        (path / name).write_text(json.dumps(value))


def observed_two_generations():
    observer = launcher.StateObserver({}, lambda: None)
    observer.record(state(0))
    observer.record(state(1, .25))
    observer.record(state(2, .125))
    return observer


def test_progress_is_compact_and_counts_updates_without_duplicate_observations():
    saved, receipt = [], {}
    observer = launcher.StateObserver(receipt, lambda: saved.append(dict(receipt)))
    observer.record(state(0))
    observer.record(state(1, .25, [1.] * 4))
    observer.record(state(1, .25, [1.] * 4))
    observer.record(state(2, .125, [2.] * 4))
    assert [row["completed_generations"] for row in saved] == [0, 1, 2]
    assert observer.widths == [.25, .125]
    assert receipt["memory_final"] == [2.] * 4
    assert receipt["sigma_next_final"] == .125
    assert "states" not in receipt


def test_invalid_state_preserves_completed_work_and_json_safe_values():
    receipt = {}
    observer = launcher.StateObserver(receipt, lambda: json.dumps(receipt, allow_nan=False))
    observer.record(state(0))
    observer.record(state(1, float("nan"), [float("inf"), 0, float("nan"), -1], invalid=True))
    assert receipt["completed_generations"] == 1
    assert receipt["invalid_update"] is True
    assert receipt["sigma_next_final"] is None and receipt["sigma_next_final_finite"] is False
    assert receipt["memory_final"] == [None, 0., None, -1.]
    assert receipt["memory_final_finite"] is False


def test_observer_rejects_skipped_generations_and_invalid_initialization():
    observer = launcher.StateObserver({}, lambda: None)
    with pytest.raises(ValueError, match="skipped"):
        observer.record(state(1))
    with pytest.raises(ValueError, match="initialization"):
        observer.record(state(0, .25))
    with pytest.raises(ValueError, match="shape or dtype"):
        observer.record(state(0, memory=[0] * 5))


def test_receipt_initial_creation_is_exclusive_and_failed_serialization_preserves_previous(tmp_path):
    path = tmp_path / "receipt.json"
    launcher.write_receipt(path, {"status": "running"}, create=True)
    with pytest.raises(FileExistsError):
        launcher.write_receipt(path, {"status": "changed"}, create=True)
    with pytest.raises(ValueError):
        launcher.write_receipt(path, {"width": float("nan")})
    assert json.loads(path.read_text()) == {"status": "running"}
    launcher.write_receipt(path, {"status": "complete"})
    assert json.loads(path.read_text()) == {"status": "complete"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["receipt.json"]


@pytest.mark.parametrize("arguments", [[], ["--method"], ["--method", "--output_dir", "a"],
                                      ["--method", "ga", "--method", "ga"]])
def test_native_arguments_require_one_unambiguous_explicit_value(arguments):
    with pytest.raises(ValueError):
        launcher.native_argument(arguments, "--method")


def test_width_receipt_reconstructs_used_width_and_validates_final_unused_update(tmp_path):
    write_training(tmp_path)
    checked = launcher.validate_width_artifacts(tmp_path, observed_two_generations())
    assert checked["completed_generations"] == checked["raw_rows"] == 2
    assert checked["sigma_used_initial"] == .5 and checked["sigma_used_final"] == .25
    assert checked["sigma_next_final"] == .125 and checked["final_update_validated"] is True


@pytest.mark.parametrize("rows, message", [
    ([{"generation": 0, "sigma": .5}, {"generation": 1, "sigma": .25}], "post-tell"),
    ([{"generation": 0, "sigma": .25}], "counts"),
    ([{"generation": 1, "sigma": .25}, {"generation": 0, "sigma": .125}], "reordered"),
    ([{"generation": 0, "sigma": .25}, {"generation": 1, "sigma": float("inf")}], "Invalid raw"),
])
def test_width_validation_rejects_wrong_timing_missing_rows_and_nonfinite_values(tmp_path, rows, message):
    write_training(tmp_path, rows=rows)
    with pytest.raises(ValueError, match=message):
        launcher.validate_width_artifacts(tmp_path, observed_two_generations())


def test_width_validation_rejects_snapshots_and_invalid_flag(tmp_path):
    config = baseline_config()
    config["population_snapshot_interval"] = 1
    write_training(tmp_path, config=config)
    observer = observed_two_generations()
    with pytest.raises(ValueError, match="diagnostic population"):
        launcher.validate_width_artifacts(tmp_path, observer)
    write_training(tmp_path)
    observer.receipt["invalid_update"] = True
    with pytest.raises(ValueError, match="Nonfinite"):
        launcher.validate_width_artifacts(tmp_path, observer)


def test_invalid_grammar_is_recorded_without_execution_or_completed_training(tmp_path):
    program = tmp_path / "unsafe.py"
    program.write_text("raise RuntimeError('must never execute')\n")
    training = tmp_path / "training"
    receipt_path = training / "adaptive-manifest.json"
    with pytest.raises(adaptive.ProgramValidationError):
        launcher.main(["--upstream", str(tmp_path / "unused"), "--program-path", str(program),
                       "--receipt-path", str(receipt_path), "--", "--method", "ga",
                       "--output_dir", str(training)])
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "failed" and receipt["completed_generations"] == 0
    assert receipt["state_initialized"] is False
    assert receipt["program_raw_sha256"] == launcher.digest(program)
    assert receipt["launcher_invocation"][0] == sys.executable
    assert "ProgramValidationError" in receipt["error"]


def test_existing_native_artifact_is_not_overwritten(tmp_path):
    training = tmp_path / "training"
    training.mkdir()
    result = training / "results.json"
    result.write_text("preserved evidence")
    with pytest.raises(ValueError, match="fresh"):
        launcher.main(["--upstream", str(tmp_path / "unused"), "--program-path", str(tmp_path / "unused.py"),
                       "--receipt-path", str(training / "adaptive-manifest.json"), "--",
                       "--method", "ga", "--output_dir", str(training)])
    assert result.read_text() == "preserved evidence"
    assert not (training / "adaptive-manifest.json").exists()


@pytest.mark.parametrize("fail_update", [False, True])
def test_launcher_success_and_invalid_failure_restore_factory_and_preserve_receipt(
        tmp_path, monkeypatch, fail_update):
    training = tmp_path / "training"
    program = tmp_path / "program.py"
    program.write_text("def update_sigma(sigma, stats, memory):\n    return sigma, memory\n")
    receipt_path = training / "adaptive-manifest.json"
    modules = {name: types.ModuleType(name) for name in (
        "source", "source.runners", "source.runners.train_nes", "source.run",
        "source.algorithms", "source.algorithms.ne", "source.algorithms.ne.ga", "jax")}
    for name, module in modules.items():
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    for name, module in modules.items():
        if "." in name:
            parent, attribute = name.rsplit(".", 1)
            setattr(modules[parent], attribute, module)
    modules["jax"].default_backend = lambda: "cpu"

    class NativeGA:
        pass

    class Wrapped:
        def init(self, key, mean):
            return state(0)

        def incumbent(self, current):
            if bool(current.invalid_update):
                raise adaptive.InvalidUpdateError(int(current.generation))
            return [0]

    modules["source.algorithms.ne.ga"].GASearcher = NativeGA
    def original_factory(*args, **kwargs):
        return NativeGA()
    runner = modules["source.runners.train_nes"]
    runner.build_searcher = original_factory
    monkeypatch.setattr(adaptive, "AdaptiveGASearcher", lambda delegate, spec: Wrapped())
    monkeypatch.setattr(experiment, "verify_upstream", lambda path: experiment.UPSTREAM_COMMIT)
    monkeypatch.setattr(launcher, "version", lambda name: "test")
    before = {key: os.environ.get(key) for key in SEARCH_THREAD_ENV}

    def native_main(arguments):
        assert all(os.environ[key] == value for key, value in SEARCH_THREAD_ENV.items())
        wrapped = runner.build_searcher("ga", 386, 8, sigma_init=.5, elite_ratio=.5,
                                        init_around_mean=False)
        wrapped.init(None, None)
        wrapped.incumbent(state(1, .25, [float("nan")] * 4 if fail_update else None,
                                invalid=fail_update))
        wrapped.incumbent(state(2, .125))
        write_training(training)
        for name in ("checkpoints.npz", "trajectory.npz"):
            (training / name).write_bytes(b"test fixture; array semantics checked by the core gate")
        return 0

    modules["source.run"].main = native_main
    args = ["--upstream", str(tmp_path / "upstream"), "--program-path", str(program),
            "--receipt-path", str(receipt_path), "--", "--method", "ga",
            "--output_dir", str(training)]
    if fail_update:
        with pytest.raises(adaptive.InvalidUpdateError):
            launcher.main(args)
    else:
        assert launcher.main(args) == 0
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == ("failed" if fail_update else "complete")
    assert receipt["completed_generations"] == (1 if fail_update else 2)
    assert receipt["program"]["source_sha256"] == launcher.digest(program)
    assert receipt["algorithm_variant"] == "ga_adaptive"
    assert runner.build_searcher is original_factory
    assert {key: os.environ.get(key) for key in SEARCH_THREAD_ENV} == before
    assert not any(training.glob("state_*.npz"))
    if fail_update:
        assert receipt["failed_after_completed_generations"] == 1
        assert receipt["memory_final"] == [None] * 4
        assert not (training / "results.json").exists()
    else:
        assert receipt["width_validation"]["matched_host_observations"] is True
        assert set(receipt["artifact_sha256"]) == set(launcher.ARTIFACT_FILES)
