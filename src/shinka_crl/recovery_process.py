"""Linux process supervision for the separately versioned recovery controller.

The caller must be synchronous, in the main thread, and must not launch unrelated
children while this function runs. Subreaper adoption lets us discover a provider
that starts a new session and outlives its parent, even between polling ticks.
Existing children are excluded. Only /proc/*/stat is read, never process
arguments or environments. This module does not change the frozen search runner.
"""

from __future__ import annotations

from collections.abc import Callable
import ctypes
from dataclasses import dataclass
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

POLL_SECONDS = 0.05
TERM_GRACE_SECONDS = 2.0
KILL_GRACE_SECONDS = 3.0
_PR_SET_CHILD_SUBREAPER = 36
_PR_GET_CHILD_SUBREAPER = 37
_LOCK = threading.Lock()


class SupervisionInterrupted(KeyboardInterrupt):
    """SIGINT/SIGTERM received; descendants were cleaned before propagation."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"Recovery supervisor received {signal.Signals(signum).name}")


class ProcessCleanupError(RuntimeError):
    """Cleanup could not establish that all supervised descendants were gone."""


@dataclass(frozen=True)
class _Process:
    pid: int
    ppid: int
    group: int
    state: str
    born: int

    @property
    def identity(self) -> tuple[int, int]:
        return self.pid, self.born


def _read_process(pid: int) -> _Process | None:
    try:
        value = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    # comm may contain spaces or parentheses; fields after its final ')' are fixed.
    fields = value[value.rfind(")") + 2:].split()
    return _Process(pid, int(fields[1]), int(fields[2]), fields[0], int(fields[19]))


def _snapshot() -> dict[int, _Process]:
    rows = {}
    for path in Path("/proc").iterdir():
        if path.name.isdecimal():
            try:
                row = _read_process(int(path.name))
            except PermissionError:
                continue  # Unrelated users may hide their process information.
            if row is not None:
                rows[row.pid] = row
    return rows


def _subreaper(value: int | None = None) -> int:
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, *[ctypes.c_ulong] * 4]
    prctl.restype = ctypes.c_int
    saved = ctypes.c_int()
    option = _PR_GET_CHILD_SUBREAPER if value is None else _PR_SET_CHILD_SUBREAPER
    argument = ctypes.addressof(saved) if value is None else value
    if prctl(option, argument, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return saved.value if value is None else value


def _descendants(rows: dict[int, _Process], parents: set[int]) -> set[tuple[int, int]]:
    found = set()
    while parents:
        children = [row for row in rows.values() if row.ppid in parents]
        found.update(row.identity for row in children)
        parents = {row.pid for row in children}
    return found


class _Tracker:
    def __init__(self):
        self.owner = os.getpid()
        self.excluded = _descendants(_snapshot(), {self.owner})
        self.known: set[tuple[int, int]] = set()
        self.groups: set[int] = set()

    def refresh(self) -> dict[int, _Process]:
        rows = _snapshot()
        # Protect descendants of preexisting children as long as their ancestry
        # remains observable; the caller must not run unrelated spawning work.
        excluded_pids = {pid for pid, born in self.excluded
                         if pid in rows and rows[pid].born == born}
        self.excluded.update(_descendants(rows, excluded_pids))
        parents = {pid for pid, born in self.known
                   if pid in rows and rows[pid].born == born}
        for row in rows.values():
            if row.ppid == self.owner and row.identity not in self.excluded:
                self.known.add(row.identity)
                parents.add(row.pid)
        self.known.update(_descendants(rows, parents))
        live = {}
        for pid, born in self.known:
            # An unreadable known process must fail closed, not look terminated.
            row = _read_process(pid)
            if row is not None and row.born == born:
                live[pid] = row
                self.groups.add(row.group)
        return live


def _send(row: _Process, signum: int) -> bool:
    """Signal this identity across any process group, guarding against PID reuse."""
    current = _read_process(row.pid)
    if current is None or current.identity != row.identity or current.state in {"Z", "X"}:
        return False
    # pidfds bind the signal to the process rather than a subsequently reused PID.
    descriptor = os.pidfd_open(row.pid)
    try:
        current = _read_process(row.pid)
        if current is None or current.identity != row.identity:
            return False
        signal.pidfd_send_signal(descriptor, signum)
        return True
    finally:
        os.close(descriptor)


def _cleanup(process: subprocess.Popen, tracker: _Tracker) -> dict:
    started = time.monotonic()
    term_sent, kill_sent, reaped = set(), set(), set()
    errors = set()
    empty_scans = 0
    live = {}
    deadline = started + TERM_GRACE_SECONDS + KILL_GRACE_SECONDS
    while True:
        # Popen owns the direct child's wait status. Only reap adopted descendants.
        process.poll()
        live = tracker.refresh()
        for row in live.values():
            if row.pid != process.pid and row.ppid == tracker.owner and row.state == "Z":
                try:
                    pid, _ = os.waitpid(row.pid, os.WNOHANG)
                    if pid:
                        reaped.add(pid)
                except ChildProcessError:
                    pass
        live = tracker.refresh()
        if not live:
            empty_scans += 1
            if empty_scans >= 2:
                break
        else:
            empty_scans = 0
        now = time.monotonic()
        if now >= deadline:
            break
        signum = signal.SIGTERM if now - started < TERM_GRACE_SECONDS else signal.SIGKILL
        sent = term_sent if signum == signal.SIGTERM else kill_sent
        for row in live.values():
            if row.identity in sent:
                continue
            try:
                if _send(row, signum):
                    sent.add(row.identity)
            except ProcessLookupError:
                pass
            except OSError as exc:
                errors.add(f"signal {signum} to pid {row.pid}: {type(exc).__name__}")
        time.sleep(min(POLL_SECONDS, max(0., deadline - time.monotonic())))
    process.poll()
    return {"complete": not live, "tracked_processes": len(tracker.known),
            "process_groups": sorted(tracker.groups),
            "term_sent": sorted(pid for pid, _ in term_sent),
            "kill_sent": sorted(pid for pid, _ in kill_sent),
            "reaped_pids": sorted(reaped),
            "survivors": [{"pid": row.pid, "start_time": row.born, "state": row.state}
                          for row in live.values()],
            "errors": sorted(errors), "wall_seconds": time.monotonic() - started}


def run_supervised(command: list[str], *, log: Path, env: dict, timeout: float,
                   stop_check: Callable[[], str | None] | None = None) -> dict:
    """Run one command and clean its whole descendant tree on every exit path.

    Timeout and a stop_check reason return the stopped result. Exceptions are
    reraised after cleanup; SIGINT/SIGTERM raise SupervisionInterrupted. These
    exceptions expose ``supervision_result`` with the same result schema. Cleanup
    failure raises ProcessCleanupError. A normally exiting command that leaves
    descendants is stopped with an explicit reason. Cleanup grace is additional
    to the command timeout. The log is created exclusively; it is never replaced.
    """
    if not sys.platform.startswith("linux") or not Path("/proc/self/stat").is_file():
        raise RuntimeError("Recovery supervision requires Linux /proc and child subreapers")
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Recovery supervision requires the main thread")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Recovery timeout must be finite and positive")
    if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command):
        raise ValueError("Recovery command must be a nonempty argument list")
    if signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL:
        raise RuntimeError("Recovery supervision requires the default SIGCHLD handler")
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("Another recovery command is already supervised")

    started = time.monotonic()
    process = None
    tracker = None
    original_subreaper = None
    subreaper_enabled = False
    original_handlers = {}
    received = []
    failure = None
    reason = None
    cleanup = {"complete": True, "tracked_processes": 0, "process_groups": [],
               "term_sent": [], "kill_sent": [], "reaped_pids": [], "survivors": [],
               "errors": [], "wall_seconds": 0.}

    def receive(signum, _frame):
        received.append(signum)

    try:
        # Fail before spawning if this Linux kernel lacks the pidfd primitives.
        descriptor = os.pidfd_open(os.getpid())
        try:
            signal.pidfd_send_signal(descriptor, 0)
        finally:
            os.close(descriptor)
        original_subreaper = _subreaper()
        _subreaper(1)
        subreaper_enabled = True
        for signum in (signal.SIGINT, signal.SIGTERM):
            original_handlers[signum] = signal.signal(signum, receive)
        tracker = _Tracker()
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("x") as stream:
            if received:
                raise SupervisionInterrupted(received[0])
            process = subprocess.Popen(command, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            while True:
                live = tracker.refresh()
                returncode = process.poll()
                if received:
                    raise SupervisionInterrupted(received[0])
                reason = stop_check() if stop_check else None
                if reason is not None:
                    if not isinstance(reason, str) or not reason:
                        raise ValueError("stop_check must return a nonempty reason or None")
                    break
                if returncode is not None:
                    if any(pid != process.pid for pid in live):
                        reason = "descendants remained after command exit"
                    break
                if time.monotonic() - started >= timeout:
                    reason = "session timeout"
                    break
                time.sleep(min(POLL_SECONDS, max(0., timeout - (time.monotonic() - started))))
    except BaseException as exc:
        failure = exc
        reason = (f"signal {signal.Signals(exc.signum).name}"
                  if isinstance(exc, SupervisionInterrupted)
                  else f"supervisor exception: {type(exc).__name__}")
    finally:
        try:
            if process is not None and tracker is not None:
                cleanup = _cleanup(process, tracker)
        except BaseException as exc:
            cleanup["complete"] = False
            cleanup["errors"].append(f"Cleanup failed: {type(exc).__name__}")
            if failure is None:
                failure = exc
        finally:
            cleanup["subreaper_enabled"] = subreaper_enabled
            cleanup["subreaper_restored"] = False
            try:
                if original_subreaper is not None:
                    _subreaper(original_subreaper)
                    cleanup["subreaper_restored"] = _subreaper() == original_subreaper
                    if not cleanup["subreaper_restored"]:
                        cleanup["complete"] = False
                        cleanup["errors"].append("Subreaper flag differs after restoration")
            except OSError as exc:
                cleanup["complete"] = False
                cleanup["errors"].append(f"Subreaper restoration failed: {type(exc).__name__}")
            finally:
                for signum, handler in original_handlers.items():
                    signal.signal(signum, handler)
                _LOCK.release()

    if failure is None and received:
        failure = SupervisionInterrupted(received[0])
        reason = f"signal {signal.Signals(received[0]).name}"
    result = {"returncode": process.returncode if process is not None else None,
              "stop_reason": reason, "wall_seconds": time.monotonic() - started,
              "cleanup": cleanup}
    if not cleanup["complete"]:
        error = ProcessCleanupError("Recovery descendants could not be completely cleaned")
        error.supervision_result = result
        raise error from failure
    if failure is not None:
        failure.supervision_result = result
        raise failure
    return result
