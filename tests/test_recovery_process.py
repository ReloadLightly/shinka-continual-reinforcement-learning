"""Real local process trees; no provider, model, evaluator, or trainer is launched."""

from pathlib import Path
import os
import signal
import subprocess
import sys
import threading

import pytest

from shinka_crl import recovery_process as supervisor

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux subreaper tests")


TREE = r'''
from pathlib import Path
import os, signal, sys, time
root = Path(sys.argv[1])
mode = sys.argv[2]
(root / "parent.pid").write_text(str(os.getpid()))
child = os.fork()
if child == 0:
    os.setsid()
    (root / "child.pid").write_text(str(os.getpid()))
    grandchild = os.fork()
    if grandchild == 0:
        os.setsid()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (root / "grandchild.pid").write_text(str(os.getpid()))
        while True:
            time.sleep(1)
    while not (root / "grandchild.pid").exists():
        time.sleep(.005)
    if mode == "exit":
        os._exit(0)
    while True:
        time.sleep(1)
while not (root / "grandchild.pid").exists():
    time.sleep(.005)
print("synthetic process tree ready", flush=True)
if mode == "exit":
    os._exit(0)
while True:
    time.sleep(1)
'''


@pytest.fixture
def tree(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, "TERM_GRACE_SECONDS", 0.2)
    monkeypatch.setattr(supervisor, "KILL_GRACE_SECONDS", 1.5)
    directory = tmp_path / "pids"
    directory.mkdir()
    yield directory
    # If an assertion fails, do not leave any fixture process behind.
    pids = [int(path.read_text()) for path in directory.glob("*.pid") if path.read_text()]
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    for pid in pids:
        try:
            os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            pass


def command(directory, mode="wait"):
    return [sys.executable, "-c", TREE, str(directory), mode]


def ready(directory):
    path = directory / "grandchild.pid"
    return path.exists() and bool(path.read_text())


def assert_clean(result, directory):
    assert result["cleanup"]["complete"]
    assert result["cleanup"]["subreaper_restored"]
    assert not result["cleanup"]["survivors"]
    for path in directory.glob("*.pid"):
        assert not Path(f"/proc/{int(path.read_text())}").exists(), path.name


def test_stop_check_cleans_nested_sessions_and_term_ignoring_grandchild(tree, tmp_path):
    previous = supervisor._subreaper()
    result = supervisor.run_supervised(
        command(tree), log=tmp_path / "stop.log", env=dict(os.environ), timeout=5,
        stop_check=lambda: "new failed candidate" if ready(tree) else None)
    assert result["stop_reason"] == "new failed candidate"
    assert result["returncode"] == -signal.SIGTERM
    assert int((tree / "grandchild.pid").read_text()) in result["cleanup"]["kill_sent"]
    assert len(result["cleanup"]["process_groups"]) == 3
    assert supervisor._subreaper() == previous
    assert_clean(result, tree)


def test_normal_parent_exit_still_cleans_adopted_orphans(tree, tmp_path):
    result = supervisor.run_supervised(
        command(tree, "exit"), log=tmp_path / "exit.log", env=dict(os.environ), timeout=5)
    assert result["returncode"] == 0
    assert result["stop_reason"] == "descendants remained after command exit"
    grandchild = int((tree / "grandchild.pid").read_text())
    assert grandchild in result["cleanup"]["kill_sent"]
    assert grandchild in result["cleanup"]["reaped_pids"]
    assert_clean(result, tree)


def test_timeout_cleans_all_descendants(tree, tmp_path):
    result = supervisor.run_supervised(
        command(tree), log=tmp_path / "timeout.log", env=dict(os.environ), timeout=0.5)
    assert ready(tree)
    assert result["stop_reason"] == "session timeout"
    assert .5 <= result["wall_seconds"] < 4
    assert_clean(result, tree)


def test_callback_exception_cleans_before_reraising(tree, tmp_path):
    def check():
        if ready(tree):
            raise RuntimeError("synthetic callback failure")
    with pytest.raises(RuntimeError, match="synthetic callback failure") as failure:
        supervisor.run_supervised(command(tree), log=tmp_path / "exception.log",
                                  env=dict(os.environ), timeout=5, stop_check=check)
    result = failure.value.supervision_result
    assert result["stop_reason"] == "supervisor exception: RuntimeError"
    assert_clean(result, tree)


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM])
def test_signals_clean_before_propagation_and_restore_handlers(tree, tmp_path, signum):
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    previous = supervisor._subreaper()
    sent = False

    def check():
        nonlocal sent
        if ready(tree) and not sent:
            sent = True
            os.kill(os.getpid(), signum)
    with pytest.raises(supervisor.SupervisionInterrupted) as failure:
        supervisor.run_supervised(command(tree), log=tmp_path / "signal.log",
                                  env=dict(os.environ), timeout=5, stop_check=check)
    assert failure.value.signum == signum
    assert failure.value.supervision_result["stop_reason"] == f"signal {signal.Signals(signum).name}"
    assert {sig: signal.getsignal(sig) for sig in before} == before
    assert supervisor._subreaper() == previous
    assert_clean(failure.value.supervision_result, tree)


def test_preexisting_child_is_not_signaled_or_reaped(tree, tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                 start_new_session=True)
    try:
        result = supervisor.run_supervised(
            command(tree), log=tmp_path / "unrelated.log", env=dict(os.environ), timeout=5,
            stop_check=lambda: "stop fixture" if ready(tree) else None)
        assert unrelated.poll() is None
        assert unrelated.pid not in result["cleanup"]["term_sent"]
        assert unrelated.pid not in result["cleanup"]["kill_sent"]
        assert unrelated.pid not in result["cleanup"]["reaped_pids"]
        assert_clean(result, tree)
    finally:
        unrelated.kill()
        unrelated.wait(timeout=5)


def test_normal_exit_preserves_enabled_subreaper_and_exit_code(tmp_path):
    previous = supervisor._subreaper()
    supervisor._subreaper(1)
    try:
        result = supervisor.run_supervised(
            [sys.executable, "-c", "print('local fixture'); raise SystemExit(7)"],
            log=tmp_path / "normal.log", env=dict(os.environ), timeout=5)
        assert result["returncode"] == 7 and result["stop_reason"] is None
        assert result["cleanup"]["complete"]
        assert supervisor._subreaper() == 1
        assert (tmp_path / "normal.log").read_text() == "local fixture\n"
    finally:
        supervisor._subreaper(previous)


def test_exclusive_log_refuses_overwrite_without_launch(tmp_path, monkeypatch):
    log = tmp_path / "existing.log"
    log.write_text("preserved evidence\n")
    previous = supervisor._subreaper()
    monkeypatch.setattr(supervisor.subprocess, "Popen",
                        lambda *args, **kwargs: pytest.fail("Must not launch on existing log"))
    with pytest.raises(FileExistsError) as failure:
        supervisor.run_supervised([sys.executable], log=log, env=dict(os.environ), timeout=1)
    assert log.read_text() == "preserved evidence\n"
    assert failure.value.supervision_result["cleanup"]["complete"]
    assert supervisor._subreaper() == previous


def test_spawn_failure_restores_supervisor_state(tmp_path):
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    previous = supervisor._subreaper()
    with pytest.raises(FileNotFoundError) as failure:
        supervisor.run_supervised([str(tmp_path / "absent-command")],
                                  log=tmp_path / "spawn.log", env=dict(os.environ), timeout=1)
    assert failure.value.supervision_result["returncode"] is None
    assert failure.value.supervision_result["cleanup"]["complete"]
    assert {sig: signal.getsignal(sig) for sig in before} == before
    assert supervisor._subreaper() == previous


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_unbounded_or_invalid_timeout_rejected_before_work(tmp_path, timeout):
    log = tmp_path / "unused.log"
    with pytest.raises(ValueError, match="finite and positive"):
        supervisor.run_supervised([sys.executable], log=log, env={}, timeout=timeout)
    assert not log.exists()


def test_threaded_invocation_rejected_before_work(tmp_path):
    failures = []
    def invoke():
        try:
            supervisor.run_supervised([sys.executable], log=tmp_path / "unused.log",
                                      env={}, timeout=1)
        except RuntimeError as exc:
            failures.append(str(exc))
    thread = threading.Thread(target=invoke)
    thread.start()
    thread.join(timeout=3)
    assert failures == ["Recovery supervision requires the main thread"]
    assert not (tmp_path / "unused.log").exists()


def test_cleanup_failure_is_explicit_in_exception(tmp_path, monkeypatch):
    original = supervisor._cleanup
    def report_survivor(process, tracker):
        result = original(process, tracker)
        result.update(complete=False, survivors=[{"pid": 123, "state": "D", "start_time": 1}])
        return result
    monkeypatch.setattr(supervisor, "_cleanup", report_survivor)
    with pytest.raises(supervisor.ProcessCleanupError) as failure:
        supervisor.run_supervised([sys.executable, "-c", "pass"],
                                  log=tmp_path / "incomplete.log", env={}, timeout=5)
    assert not failure.value.supervision_result["cleanup"]["complete"]
    assert failure.value.supervision_result["cleanup"]["survivors"][0]["state"] == "D"


def test_child_forked_during_shutdown_is_discovered_and_cleaned(tree, tmp_path):
    code = r'''
from pathlib import Path
import os, signal, sys, time
root = Path(sys.argv[1])
def shutdown(*_):
    if os.fork() == 0:
        os.setsid()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (root / "late.pid").write_text(str(os.getpid()))
        while True:
            time.sleep(1)
    os._exit(0)
signal.signal(signal.SIGTERM, shutdown)
(root / "parent.pid").write_text(str(os.getpid()))
while True:
    time.sleep(1)
'''
    result = supervisor.run_supervised(
        [sys.executable, "-c", code, str(tree)], log=tmp_path / "late-child.log",
        env=dict(os.environ), timeout=5,
        stop_check=lambda: "stop requested" if (tree / "parent.pid").exists() else None)
    late = int((tree / "late.pid").read_text())
    assert late in result["cleanup"]["kill_sent"]
    assert_clean(result, tree)


def test_waits_for_adopted_child_final_ledger_write(tree, tmp_path):
    code = r'''
from pathlib import Path
import os, signal, sys, time
root = Path(sys.argv[1])
(root / "parent.pid").write_text(str(os.getpid()))
if os.fork() == 0:
    os.setsid()
    def finish(*_):
        time.sleep(.08)
        (root / "ledger.txt").write_text("finished\n")
        os._exit(0)
    signal.signal(signal.SIGTERM, finish)
    (root / "child.pid").write_text(str(os.getpid()))
    while True:
        time.sleep(1)
while not (root / "child.pid").exists():
    time.sleep(.005)
os._exit(0)
'''
    result = supervisor.run_supervised(
        [sys.executable, "-c", code, str(tree)], log=tmp_path / "late-ledger.log",
        env=dict(os.environ), timeout=5)
    assert (tree / "ledger.txt").read_text() == "finished\n"
    assert result["cleanup"]["kill_sent"] == []
    assert_clean(result, tree)
