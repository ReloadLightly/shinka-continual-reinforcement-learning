"""Evidence must reject partial, stale or internally inconsistent training artifacts."""

import importlib.util
import json
from pathlib import Path
import shutil

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("report_smoke", REPO_ROOT / "scripts/report_smoke.py")
reporter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reporter)


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def update_json(path, update):
    value = json.loads(path.read_text())
    update(value)
    write_json(path, value)


@pytest.fixture
def suite(tmp_path):
    """Explicitly synthetic upstream-shaped artifacts; never used as reported results."""
    repo = tmp_path / "repo"
    profile_path = repo / "src/shinka_crl/profiles/smoke.json"
    profile_path.parent.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / "src/shinka_crl/profiles/smoke.json", profile_path)
    shutil.copyfile(REPO_ROOT / "upstream.lock.json", repo / "upstream.lock.json")
    profile = json.loads(profile_path.read_text())
    revision = json.loads((repo / "upstream.lock.json").read_text())[
        "continual_neuroevolution"]["commit"]
    root = repo / "results/suite"
    root.mkdir(parents=True)
    write_json(root / "suite.json", {"status": "complete",
                                    "completed": ["ga", "es", "ppo", "shinka_initial"],
                                    "wall_seconds": 50.0})
    write_json(root / "environment.json", {
        "upstream_commit": revision, "jax_backend": "cpu", "python": "3.11.0",
        "jax_devices": ["TFRT_CPU_0"], "packages": {"jax": "test-fixture"},
        "source_sha256": {"src/shinka_crl/profiles/smoke.json":
                          reporter.sha256(profile_path.read_bytes())},
    })
    paths = {}
    for label in ("ga", "es", "ppo", "shinka"):
        method = "ga" if label == "shinka" else label
        path = root / (f"baselines/smoke/{method}/trial_1002"
                       if label != "shinka" else "shinka/seed_1001")
        path.mkdir(parents=True)
        paths[label] = path
        records = [{"generation": i, "task": i // 2,
                    "centroid_task0": 8.0, "centroid_task1": 24.0} for i in range(4)]
        write_json(path / "training_metrics.json", records)
        summary = {"profile": "smoke", "method": method, "seed": 1001, "trial": 1002,
                   "upstream_commit": revision, "mean_return": 16.0,
                   "normalized_score": 0.5, "metric_rows": 4}
        write_json(path / "summary.json", summary)
        manifest = {"profile": profile, "method": method, "seed": 1001, "trial": 1002,
                    "upstream_commit": revision, "status": "complete", "wall_seconds": 10.0,
                    "ga_settings": {"sigma": 0.5, "elite_ratio": 0.5}
                    if label == "shinka" else None,
                    "metrics_sha256": reporter.sha256((path / "training_metrics.json").read_bytes()),
                    "command": [str(repo / "python"), str(repo / "source/run.py"),
                                "--method", method, "--seed", "1001", "--trial", "1002",
                                "--output_dir", str(path)],
                    "python_version": "Python 3.11.0"}
        write_json(path / "manifest.json", manifest)
        config = {"env": "CartPole-v1", "method": method, "seed": 1001, "trial": 1002,
                  "schedule": "switch", "num_tasks": 2, "num_generations": 4,
                  "task_interval": 2, "episode_length": 32, "eval_episodes": 1,
                  "task_sequence": [0, 1], "noise_range": 0.5, "first_task_clean": True}
        config.update({"num_envs": 8, "num_steps": 16, "num_minibatches": 2,
                       "sigma": float("nan")} if method == "ppo" else
                      {"pop_size": 8, "num_evals": 1, "sigma": 0.5})
        write_json(path / "results.json", {"config": config,
                                          "env_steps": reporter.training_budget(profile, method),
                                          "noise_vectors": [[0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4]]})
        (path / "train.log").write_text(f"Synthetic trainer fixture wrote {path}\n")
        (path / "process.log").write_text(f"Synthetic process fixture wrote {path}\n")
        (path / "trajectory.npz").write_bytes(b"test checkpoint bytes, not a real checkpoint")
    write_json(root / "shinka/correct.json", {"correct": True, "error": None})
    write_json(root / "shinka/metrics.json", {
        "combined_score": 0.5,
        "public": {"profile": "smoke", "method": "ga", "seeds_completed": 1,
                   "smoke_validation_only": True, "sigma": 0.5, "elite_ratio": 0.5,
                   "mean_return": 16.0},
        "private": {"seed_results": [{"seed": 1001, "trial": 1002, "mean_return": 16.0,
                                      "normalized_score": 0.5}]},
    })
    return repo, root, repo / "reports/suite", paths


def refresh_metric_hash(path):
    digest = reporter.sha256((path / "training_metrics.json").read_bytes())
    update_json(path / "manifest.json", lambda value: value.update(metrics_sha256=digest))


def test_export_preserves_evidence_and_marks_redaction(suite):
    repo, root, output, paths = suite
    report = reporter.export_smoke(root, output, repo)
    checksums = json.loads((output / "checksums.json").read_text())
    assert [row["training_env_steps_nominal"] for row in report["rows"]] == [1024, 1024, 512, 1024]
    assert all(row["task_schedule"] == [0, 0, 1, 1] for row in report["rows"])
    assert "not matched" in report["caption"]
    assert all(report["validation"].values())
    for label, path in paths.items():
        exported_manifest = output / label / "manifest.json"
        assert "<repo>" in exported_manifest.read_text()
        assert str(repo) not in exported_manifest.read_text()
        assert checksums[f"{label}/manifest.json"]["redacted"] is True
        assert checksums[f"{label}/manifest.json"]["original_sha256"] == reporter.sha256(
            (path / "manifest.json").read_bytes())
        assert checksums[f"{label}/manifest.json"]["exported_sha256"] == reporter.sha256(
            exported_manifest.read_bytes())
        assert not (output / label / "trajectory.npz").exists()
        assert checksums[f"{label}/trajectory.npz"]["exported"] is False
        assert (output / label / "training_metrics.json").read_bytes() == (
            path / "training_metrics.json").read_bytes()
        assert (output / label / "results.json").read_bytes() == (path / "results.json").read_bytes()
        assert str(repo) not in (output / label / "train.log").read_text()
    assert "NaN" in (output / "ppo/results.json").read_text()
    json.loads((output / "summary.json").read_text(), parse_constant=lambda value: pytest.fail(
        f"Non-finite JSON number in derived summary: {value}"))
    assert (output / "suite.json").read_bytes() == (root / "suite.json").read_bytes()
    assert (output / "environment.json").is_file()
    assert (output / "shinka/metrics.json").is_file()
    assert json.loads((output / "summary.json").read_text()) == report


@pytest.mark.parametrize("damage", ["partial", "missing", "revision", "profile", "hash",
                                    "summary", "budget", "noise", "command", "task_sequence",
                                    "wrong_seed", "shinka_correct", "shinka_score", "source_hash"])
def test_rejects_partial_stale_or_mismatched_suite_before_export(suite, damage):
    repo, root, output, paths = suite
    ga = paths["ga"]
    if damage == "partial":
        update_json(root / "suite.json", lambda value: value.update(status="failed"))
    elif damage == "missing":
        (paths["ppo"] / "train.log").unlink()
    elif damage == "revision":
        update_json(ga / "manifest.json", lambda value: value.update(upstream_commit="stale"))
    elif damage == "profile":
        update_json(ga / "manifest.json", lambda value: value["profile"].update(episode_length=500))
    elif damage == "hash":
        (ga / "training_metrics.json").write_text("[]")
    elif damage == "summary":
        update_json(ga / "summary.json", lambda value: value.update(mean_return=31.0))
    elif damage == "budget":
        update_json(ga / "results.json", lambda value: value.update(env_steps=512))
    elif damage == "noise":
        update_json(paths["es"] / "results.json",
                    lambda value: value["noise_vectors"][1].__setitem__(0, 0.9))
    elif damage == "command":
        update_json(ga / "manifest.json", lambda value: value["command"].__setitem__(-1, "/stale"))
    elif damage == "task_sequence":
        update_json(ga / "results.json", lambda value: value["config"].update(task_sequence=[1, 0]))
    elif damage == "wrong_seed":
        update_json(ga / "results.json", lambda value: value["config"].update(seed=1000))
    elif damage == "shinka_correct":
        update_json(root / "shinka/correct.json", lambda value: value.update(correct=False))
    elif damage == "shinka_score":
        update_json(root / "shinka/metrics.json", lambda value: value.update(combined_score=0.99))
    elif damage == "source_hash":
        update_json(root / "environment.json", lambda value: value["source_sha256"].update(
            {"src/shinka_crl/profiles/smoke.json": "0" * 64}))
    with pytest.raises(ValueError):
        reporter.export_smoke(root, output, repo)
    assert not output.exists()


@pytest.mark.parametrize("damage", ["missing_row", "duplicate_row", "task_order", "nan",
                                    "infinity", "boolean", "over_cap"])
def test_rejects_corrupt_score_curves_even_when_manifest_hash_matches(suite, damage):
    repo, root, output, paths = suite
    path = paths["ga"]

    def corrupt(records):
        if damage == "missing_row":
            records.pop()
        elif damage == "duplicate_row":
            records[1] = records[0]
        elif damage == "task_order":
            records[0]["task"] = 1
        else:
            records[0]["centroid_task0"] = {"nan": float("nan"), "infinity": float("inf"),
                                             "boolean": True, "over_cap": 33}[damage]

    update_json(path / "training_metrics.json", corrupt)
    refresh_metric_hash(path)
    with pytest.raises(ValueError):
        reporter.export_smoke(root, output, repo)
    assert not output.exists()


def test_evaluator_parity_checks_inactive_centroid_scores_too(suite):
    repo, root, output, paths = suite
    path = paths["shinka"]
    update_json(path / "training_metrics.json",
                lambda records: records[0].update(centroid_task1=23.0))
    refresh_metric_hash(path)
    with pytest.raises(ValueError, match="parity"):
        reporter.export_smoke(root, output, repo)
    assert not output.exists()


@pytest.mark.parametrize("log_name", ["train.log", "process.log"])
def test_rejects_logging_collision(suite, log_name):
    repo, root, output, paths = suite
    (paths["ga"] / log_name).write_bytes(b"Trainer log\0with file-offset collision\n")
    with pytest.raises(ValueError, match="NUL bytes"):
        reporter.export_smoke(root, output, repo)
    assert not output.exists()


def test_export_refuses_existing_report(suite):
    repo, root, output, _ = suite
    output.mkdir(parents=True)
    marker = output / "keep.txt"
    marker.write_text("Do not overwrite prior evidence.\n")
    with pytest.raises(ValueError, match="overwrite"):
        reporter.export_smoke(root, output, repo)
    assert marker.read_text() == "Do not overwrite prior evidence.\n"
    assert not (output / "summary.json").exists()


def test_nominal_budget_is_derived_from_profile():
    profile = {"episode_length": 20, "ne": {"num_generations": 3, "pop_size": 5, "num_evals": 2},
               "ppo": {"num_updates": 7, "num_envs": 3, "num_steps": 11}}
    assert reporter.training_budget(profile, "ga") == 600
    assert reporter.training_budget(profile, "es") == 600
    assert reporter.training_budget(profile, "ppo") == 231
