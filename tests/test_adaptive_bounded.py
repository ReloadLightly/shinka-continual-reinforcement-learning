"""Actual short processes verify the repeated study's cumulative time ceiling."""
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import signal
import sys
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_adaptive_bounded.py"
spec = importlib.util.spec_from_file_location("adaptive_bounded", SCRIPT)
bounded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounded)


def invoke(path, limit=2, label="first", code="pass", *, status=False):
    command = [sys.executable, str(SCRIPT), "--budget-file", str(path),
               "--limit-seconds", str(limit)]
    command += ["--status"] if status else ["--label", label, "--", sys.executable, "-c", code]
    return subprocess.run(command, capture_output=True, text=True, timeout=12)


def test_status_does_not_start_budget_or_create_directory(tmp_path):
    path = tmp_path / "not-yet" / "budget.json"
    result = invoke(path, status=True)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"status": "not_started", "elapsed_seconds": 0.,
                                       "remaining_seconds": 2., "actions": []}
    assert not path.parent.exists()


@pytest.mark.parametrize("limit", [0, -1, "nan", "inf"])
def test_invalid_limit_never_starts_action(tmp_path, limit):
    path = tmp_path / "budget.json"
    result = invoke(path, limit)
    assert result.returncode == 2 and "positive and finite" in result.stderr
    assert not path.exists() and not (tmp_path / "actions").exists()


def test_actions_share_start_deadline_and_exhaustion_never_resets(tmp_path):
    path = tmp_path / "budget.json"
    first = invoke(path, limit=1.4, code="print('first')")
    assert first.returncode == 0, first.stderr
    original = json.loads(path.read_text())
    second = invoke(path, limit=1.4, label="second", code="print('second')")
    assert second.returncode == 0, second.stderr
    saved = json.loads(path.read_text())
    assert saved["started"] == original["started"]
    assert saved["deadline_utc_seconds"] == original["deadline_utc_seconds"]
    assert saved["deadline_boottime_seconds"] == original["deadline_boottime_seconds"]
    assert [row["status"] for row in saved["actions"]] == ["complete", "complete"]
    assert (tmp_path / "actions/first.log").read_text() == "first\n"
    assert (tmp_path / "actions/second.log").read_text() == "second\n"
    for action in saved["actions"]:
        assert action["monotonic_wall_seconds"] >= 0 and action["utc_wall_seconds"] >= 0
        assert action["returncode"] == action["child_returncode"] == 0
    changed = invoke(path, limit=10, label="cannot-reset")
    assert changed.returncode == 2 and "cannot be reset" in changed.stderr
    time.sleep(max(0., saved["deadline_utc_seconds"] - time.time()) + .05)
    before = path.read_bytes()
    exhausted = invoke(path, limit=1.4, label="third")
    assert exhausted.returncode == 2 and "exhausted" in exhausted.stderr
    assert path.read_bytes() == before and not (tmp_path / "actions/third.log").exists()


def test_duplicate_budget_controller_fails_before_any_action(tmp_path):
    path = tmp_path / "budget.json"
    with path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = invoke(path)
    assert result.returncode == 2 and "Another controller" in result.stderr
    assert not path.exists() and not (tmp_path / "actions").exists()


def test_duplicate_label_preserves_original_log_and_receipt(tmp_path):
    path = tmp_path / "budget.json"
    assert invoke(path, code="print('original')").returncode == 0
    before = path.read_bytes()
    result = invoke(path, code="print('overwrite')")
    assert result.returncode == 2 and "already exists" in result.stderr
    assert path.read_bytes() == before
    assert (tmp_path / "actions/first.log").read_text() == "original\n"


def test_deadline_cleans_grandchild_in_a_new_session(tmp_path):
    path, pid_file = tmp_path / "budget.json", tmp_path / "grandchild.pid"
    grandchild = ("import os,signal,time; "
                  "signal.signal(signal.SIGINT,signal.SIG_IGN); "
                  "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
                  f"open({str(pid_file)!r},'w').write(str(os.getpid())); time.sleep(30)")
    parent = ("import subprocess,sys,time; "
              f"subprocess.Popen([sys.executable,'-c',{grandchild!r}],start_new_session=True); "
              "time.sleep(30)")
    started = time.monotonic()
    result = invoke(path, limit=.5, code=parent)
    assert result.returncode == 124, result.stderr
    assert time.monotonic() - started < 9
    pid = int(pid_file.read_text())
    assert not (Path("/proc") / str(pid)).exists()
    saved = json.loads(path.read_text())
    action = saved["actions"][0]
    assert action["stop_reason"] == "deadline" and action["status"] == "stopped"
    assert action["returncode"] == 124
    assert any(row["pid"] == pid for row in action["descendants"])
    assert saved["observed_elapsed_seconds"] >= .5


def test_elapsed_uses_boot_clock_suspend_and_fails_closed_after_reboot_regression():
    budget = {"started": {"utc_seconds": 100, "boottime_seconds": 20, "boot_id": "one"},
              "limit_seconds": 1000, "observed_elapsed_seconds": 2}
    now = {"utc_seconds": 101, "boottime_seconds": 220, "boot_id": "one"}
    assert bounded.elapsed(budget, now) == 200
    assert bounded.elapsed(budget, {**now, "utc_seconds": 500}) == 400
    assert bounded.elapsed(budget, {**now, "boottime_seconds": 1, "boot_id": "two"}) == 2
    assert bounded.elapsed(budget, {**now, "utc_seconds": 99, "boot_id": "two"}) == 1000


def test_signal_ignores_reused_pid_identity(monkeypatch):
    calls = []
    monkeypatch.setattr(bounded.os, "pidfd_open", lambda pid: 99)
    monkeypatch.setattr(bounded.os, "close", calls.append)
    monkeypatch.setattr(bounded, "proc_record", lambda pid: {"start_ticks": 456})
    monkeypatch.setattr(bounded.signal, "pidfd_send_signal", lambda *args: pytest.fail("Reused PID signaled"))
    bounded.signal_verified(123, 455, bounded.signal.SIGTERM)
    assert calls == [99]


def test_interrupt_seals_action_and_stops_child(tmp_path):
    path, ready = tmp_path / "budget.json", tmp_path / "ready"
    command = [sys.executable, str(SCRIPT), "--budget-file", str(path),
               "--limit-seconds", "20", "--label", "interrupted", "--", sys.executable,
               "-c", f"import time; open({str(ready)!r},'w').write('ready'); time.sleep(30)"]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert ready.exists()
        os.kill(process.pid, signal.SIGTERM)
        _, error = process.communicate(timeout=8)
        assert process.returncode == 128 + signal.SIGTERM, error
        saved = json.loads(path.read_text())
        action = saved["actions"][0]
        assert action["stop_reason"] == "interrupted" and action["status"] == "stopped"
        assert action["returncode"] == 128 + signal.SIGTERM
        assert not (Path("/proc") / str(action["child_identity"]["pid"])).exists()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
