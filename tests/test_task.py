"""Candidate parsing and Shinka artifact contracts, without training or APIs."""

import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


TASK_DIR = Path(__file__).resolve().parents[1] / "tasks" / "cartpole_ga"
SPEC = importlib.util.spec_from_file_location("cartpole_ga_evaluate", TASK_DIR / "evaluate.py")
task = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task)


def write_candidate(tmp_path, source):
    path = tmp_path / "candidate.py"
    path.write_text(source, encoding="utf-8")
    return path


def test_initial_candidate():
    assert task.parse_ga_config(TASK_DIR / "initial.py") == {"sigma": 0.5, "elite_ratio": 0.5}


def test_validate_only_requires_no_output_directory(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evaluate.py", "--program_path", str(TASK_DIR / "initial.py"), "--validate-only"],
    )
    assert task.main() == 0
    assert json.loads(capsys.readouterr().out) == {
        "validation_only": True,
        "valid": True,
        "ga_settings": {"sigma": 0.5, "elite_ratio": 0.5},
    }


@pytest.mark.parametrize(
    "source",
    [
        "import os\ndef get_ga_config():\n return {'sigma': 0.1, 'elite_ratio': 0.5}\n",
        "@print('executed')\ndef get_ga_config():\n return {'sigma': 0.1, 'elite_ratio': 0.5}\n",
        "def get_ga_config(x=print('executed')):\n return {'sigma': 0.1, 'elite_ratio': 0.5}\n",
        "def get_ga_config() -> print('executed'):\n return {'sigma': 0.1, 'elite_ratio': 0.5}\n",
        "def get_ga_config():\n print('executed')\n return {'sigma': 0.1, 'elite_ratio': 0.5}\n",
        "def get_ga_config():\n return {'sigma': float('nan'), 'elite_ratio': 0.5}\n",
        "def get_ga_config():\n return {'sigma': 0.1 + 0.1, 'elite_ratio': 0.5}\n",
        "def get_ga_config():\n return {'sigma': True, 'elite_ratio': 0.5}\n",
        "def get_ga_config():\n return {'sigma': 1e999, 'elite_ratio': 0.5}\n",
        "def get_ga_config():\n return {'sigma': 0, 'elite_ratio': 0.5}\n",
        "def get_ga_config():\n return {'sigma': 2.001, 'elite_ratio': 0.5}\n",
        "def get_ga_config():\n return {'sigma': 0.1, 'elite_ratio': 1.0}\n",
        "def get_ga_config():\n return {'sigma': 0.1, 'sigma': 0.2}\n",
        "def get_ga_config():\n return {'sigma': 0.1, 'elite_ratio': 0.5, 'budget': 1}\n",
    ],
)
def test_rejects_unsafe_or_invalid_candidates(tmp_path, source, capsys):
    with pytest.raises(ValueError):
        task.parse_ga_config(write_candidate(tmp_path, source))
    assert "executed" not in capsys.readouterr().out


def test_fixed_seed_aggregation_and_contract(tmp_path, monkeypatch):
    monkeypatch.setenv("SHINKA_CRL_PROFILE", "search")
    monkeypatch.setenv("SHINKA_CRL_UPSTREAM", str(tmp_path / "upstream"))
    monkeypatch.setenv("SHINKA_CRL_PYTHON", "/test/python")
    monkeypatch.setenv("SHINKA_CRL_TIMEOUT", "120")
    monkeypatch.setattr(task, "load_profile", lambda name: {"seeds": [1001, 1002]})
    monkeypatch.setattr(
        task, "default_paths", lambda: (tmp_path / "default", Path("/default/python"))
    )
    calls = []

    def fake_experiment(**kwargs):
        calls.append(kwargs)
        return {"normalized_score": 0.2 * len(calls), "mean_return": 100.0 * len(calls)}

    monkeypatch.setattr(task, "run_experiment", fake_experiment)
    output = tmp_path / "result"
    assert task.evaluate_program(TASK_DIR / "initial.py", output)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["combined_score"] == pytest.approx(0.3)
    assert metrics["public"]["mean_return"] == 150
    assert metrics["public"]["seeds_completed"] == 2
    assert metrics["public"]["smoke_validation_only"] is False
    assert json.loads((output / "correct.json").read_text()) == {"correct": True, "error": None}
    assert [call["seed"] for call in calls] == [1001, 1002]
    assert [call["trial"] for call in calls] == [1002, 1003]
    assert [call["output_dir"].name for call in calls] == ["seed_1001", "seed_1002"]
    assert all(
        call["method"] == "ga" and call["ga_settings"] == {"sigma": 0.5, "elite_ratio": 0.5}
        for call in calls
    )
    assert all(call["python"] == "/test/python" and call["timeout"] == 120 for call in calls)
    assert metrics["private"]["provenance"]["program_sha256"] == hashlib.sha256(
        (TASK_DIR / "initial.py").read_bytes()
    ).hexdigest()
    assert metrics["private"]["provenance"]["profile"] == {"seeds": [1001, 1002]}
    assert [row["status"] for row in metrics["private"]["seed_attempts"]] == [
        "complete", "complete"
    ]
    assert metrics["private"]["evaluation_wall_seconds"] >= 0


def test_later_failure_retains_completed_seed_scores_and_attempt_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("SHINKA_CRL_PROFILE", "search")
    monkeypatch.setattr(task, "load_profile", lambda name: {"seeds": [1001, 1002, 1003]})
    monkeypatch.setattr(task, "default_paths", lambda: (tmp_path, tmp_path / "python"))
    output = tmp_path / "result"

    def partial_experiment(**kwargs):
        kwargs["output_dir"].mkdir()
        (kwargs["output_dir"] / "process.log").write_text("recorded work\n")
        if kwargs["seed"] == 1002:
            # The first seed's evidence must exist before the next trial starts.
            progress = json.loads((output / "metrics.json").read_text())
            assert progress["public"]["seeds_completed"] == 1
            assert progress["private"]["seed_results"][0]["normalized_score"] == 0.4
            assert progress["combined_score"] == 0.0
            raise TimeoutError("second trial timed out")
        assert kwargs["seed"] == 1001
        return {"normalized_score": 0.4, "mean_return": 200.0}

    monkeypatch.setattr(task, "run_experiment", partial_experiment)
    assert not task.evaluate_program(TASK_DIR / "initial.py", output)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["combined_score"] == 0.0
    assert metrics["public"]["seeds_completed"] == 1
    assert metrics["private"]["seed_results"] == [{
        "seed": 1001, "trial": 1002, "normalized_score": 0.4, "mean_return": 200.0,
    }]
    attempts = metrics["private"]["seed_attempts"]
    assert [row["status"] for row in attempts] == ["complete", "failed"]
    assert attempts[1]["error"] == "TimeoutError: second trial timed out"
    assert all(row["wall_seconds"] >= 0 for row in attempts)
    assert all(row["artifact_sha256"] == {
        "process.log": hashlib.sha256(b"recorded work\n").hexdigest()
    } for row in attempts)


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("training failed"), {"normalized_score": float("nan"), "mean_return": 1.0}],
)
def test_failed_experiment_writes_failure_contract(tmp_path, monkeypatch, failure):
    monkeypatch.setenv("SHINKA_CRL_PROFILE", "smoke")
    monkeypatch.setattr(task, "load_profile", lambda name: {"seeds": [1001]})
    monkeypatch.setattr(task, "default_paths", lambda: (tmp_path, tmp_path / "python"))

    def fail(**kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(task, "run_experiment", fail)
    output = tmp_path / "result"
    assert not task.evaluate_program(TASK_DIR / "initial.py", output)
    assert json.loads((output / "metrics.json").read_text())["combined_score"] == 0.0
    correctness = json.loads((output / "correct.json").read_text())
    assert correctness["correct"] is False
    assert correctness["error"]


def test_final_profile_is_held_out(tmp_path, monkeypatch):
    monkeypatch.setenv("SHINKA_CRL_PROFILE", "paper-cartpole")
    output = tmp_path / "result"
    assert not task.evaluate_program(TASK_DIR / "initial.py", output)
    assert "held out" in json.loads((output / "correct.json").read_text())["error"]


def test_invalid_candidate_writes_failure_without_running(tmp_path, monkeypatch):
    monkeypatch.setenv("SHINKA_CRL_PROFILE", "smoke")
    candidate = write_candidate(tmp_path, "raise RuntimeError('must never execute')")
    output = tmp_path / "result"
    output.mkdir()
    (output / "correct.json").write_text('{"correct": true, "error": null}')
    (output / "metrics.json").write_text('{"combined_score": 1.0}')
    assert not task.evaluate_program(candidate, output)
    correctness = json.loads((output / "correct.json").read_text())
    assert correctness["correct"] is False
    assert "ValueError" in correctness["error"]
    assert json.loads((output / "metrics.json").read_text())["combined_score"] == 0.0
