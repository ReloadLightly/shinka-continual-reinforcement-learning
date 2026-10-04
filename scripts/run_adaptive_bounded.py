"""Enforce one elapsed-time budget across actions of the repeated adaptive study.

The first executed action fixes the deadline. Time between actions and suspend
time count. Status reads neither create nor reset the budget. Linux subreaper
adoption and PID-file-descriptor signals cover detached trainer descendants.
"""
from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


VERSION = "adaptive-bounded-actions-v1"
POLL_SECONDS = 0.1


def clock_snapshot() -> dict:
    utc = time.time()
    return {"utc": datetime.fromtimestamp(utc, timezone.utc).isoformat(),
            "utc_seconds": utc, "monotonic_seconds": time.monotonic(),
            "boottime_seconds": time.clock_gettime(time.CLOCK_BOOTTIME),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def elapsed(budget: dict, now: dict) -> float:
    start = budget["started"]
    clocks = [0., budget.get("observed_elapsed_seconds", 0.),
              now["utc_seconds"] - start["utc_seconds"]]
    if now["boot_id"] == start["boot_id"]:
        clocks.append(now["boottime_seconds"] - start["boottime_seconds"])
    elif now["utc_seconds"] < start["utc_seconds"]:
        # After reboot, neither the old boot clock nor a regressed UTC clock
        # establishes a remaining allowance. Preserve evidence and fail closed.
        clocks.append(budget["limit_seconds"])
    return max(clocks)


def save(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_budget(path: Path, limit: float) -> dict | None:
    if not path.exists():
        return None
    budget = json.loads(path.read_text())
    if budget["protocol"] != VERSION or budget["limit_seconds"] != limit:
        raise ValueError("Frozen budget limit changed; the existing deadline cannot be reset")
    return budget


def proc_record(pid: int) -> dict | None:
    try:
        # comm may contain spaces and parentheses; fields after its final ')'
        # begin at field 3 (state), and starttime is field 22.
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "ppid": int(fields[1]), "start_ticks": int(fields[19]),
                "state": fields[0]}
    except (FileNotFoundError, ProcessLookupError):
        return None


def discover_descendants(known: dict[int, int]) -> None:
    records = {}
    for path in Path("/proc").iterdir():
        if path.name.isdigit():
            record = proc_record(int(path.name))
            if record is not None:
                records[record["pid"]] = record
    parents = {os.getpid(), *(pid for pid, ticks in known.items()
                             if pid in records and records[pid]["start_ticks"] == ticks)}
    while True:
        additions = {pid for pid, row in records.items() if row["ppid"] in parents} - parents
        if not additions:
            break
        for pid in additions:
            known[pid] = records[pid]["start_ticks"]
        parents.update(additions)


def live_descendants(known: dict[int, int]) -> dict[int, int]:
    discover_descendants(known)
    return {pid: ticks for pid, ticks in known.items()
            if (record := proc_record(pid)) is not None
            and record["start_ticks"] == ticks and record["state"] not in ("Z", "X")}


def signal_verified(pid: int, ticks: int, signum: int) -> None:
    try:
        descriptor = os.pidfd_open(pid)
        try:
            record = proc_record(pid)
            if record is not None and record["start_ticks"] == ticks:
                signal.pidfd_send_signal(descriptor, signum)
        finally:
            os.close(descriptor)
    except ProcessLookupError:
        pass


def enable_subreaper() -> None:
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("Bounded execution requires Linux PID file descriptors")
    # PR_SET_CHILD_SUBREAPER: grandchildren are adopted here if their direct
    # parent exits, including descendants that called setsid().
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "Cannot enable child subreaper")


def cleanup(child: subprocess.Popen, known: dict[int, int]) -> None:
    discover_descendants(known)
    if child.pid in known:
        signal_verified(child.pid, known[child.pid], signal.SIGINT)
    # First give the controller time to seal its failed/partial evidence. Then
    # address remaining descendants individually, including detached sessions.
    for signum, grace in ((None, 3.), (signal.SIGTERM, 2.), (signal.SIGKILL, 1.)):
        deadline = time.monotonic() + grace
        while live := live_descendants(known):
            if signum is not None:
                for pid, ticks in live.items():
                    signal_verified(pid, ticks, signum)
            child.poll()
            if time.monotonic() >= deadline:
                break
            time.sleep(POLL_SECONDS)
    child.poll()
    if child.returncode is not None:
        # Reap adopted orphan zombies only after Popen has reaped its own child.
        while True:
            try:
                pid, _ = os.waitpid(-1, os.WNOHANG)
            except ChildProcessError:
                break
            if pid == 0:
                break
    if live_descendants(known):
        raise RuntimeError("Descendants remained alive after bounded cleanup")


def execute(path: Path, limit: float, label: str, command: list[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another controller holds this experiment budget") from exc
        budget = read_budget(path, limit)
        now = clock_snapshot()
        if budget is not None:
            if elapsed(budget, now) >= limit:
                raise ValueError("Experiment elapsed-time budget exhausted; no new action launched")
            if any(action["label"] == label for action in budget["actions"]):
                raise ValueError("Action label already exists; retain it and use a new label")
            if any(action["status"] == "running" for action in budget["actions"]):
                raise ValueError("An interrupted action needs review before further execution")
        enable_subreaper()
        actions = path.parent / "actions"
        actions.mkdir(exist_ok=True)
        log = actions / f"{label}.log"
        with log.open("x") as stream:
            started = clock_snapshot()
            if budget is None:
                budget = {"protocol": VERSION, "limit_seconds": limit, "started": started,
                          "deadline_utc_seconds": started["utc_seconds"] + limit,
                          "deadline_boottime_seconds": started["boottime_seconds"] + limit,
                          "clock_policy": "max UTC and same-boot CLOCK_BOOTTIME elapsed; gaps count",
                          "actions": []}
            action = {"label": label, "command": command, "log": str(log), "started": started,
                      "status": "running", "controller_pid": os.getpid()}
            budget["actions"].append(action)
            budget["observed_elapsed_seconds"] = elapsed(budget, started)
            save(path, budget)
            received = []
            previous = {sig: signal.signal(sig, lambda signum, frame: received.append(signum))
                        for sig in (signal.SIGINT, signal.SIGTERM)}
            known, child, reason = {}, None, None
            returncode = 125
            try:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
                record = proc_record(child.pid)
                if record is not None:
                    known[child.pid] = record["start_ticks"]
                    action["child_identity"] = {"pid": child.pid, "start_ticks": record["start_ticks"]}
                save(path, budget)
                while child.poll() is None:
                    discover_descendants(known)
                    if received:
                        reason = "interrupted"
                        break
                    if elapsed(budget, clock_snapshot()) >= limit:
                        reason = "deadline"
                        break
                    time.sleep(POLL_SECONDS)
                if reason is None and live_descendants(known):
                    reason = "unfinished_descendants"
                if reason is not None:
                    action["stop_requested"] = clock_snapshot()
                    cleanup(child, known)
                returncode = (124 if reason == "deadline" else 128 + received[0] if received
                              else 125 if reason else child.returncode)
            except BaseException as exc:
                action["error"] = f"{type(exc).__name__}: {exc}"
                reason = reason or "execution_error"
                if child is not None:
                    cleanup(child, known)
                raise
            finally:
                for sig, handler in previous.items():
                    signal.signal(sig, handler)
                finished = clock_snapshot()
                action.update(finished=finished, returncode=returncode,
                              child_returncode=child.returncode if child is not None else None,
                              stop_reason=reason, status="complete" if returncode == 0 else "stopped",
                              monotonic_wall_seconds=finished["monotonic_seconds"] - started["monotonic_seconds"],
                              utc_wall_seconds=finished["utc_seconds"] - started["utc_seconds"],
                              descendants=[{"pid": pid, "start_ticks": ticks}
                                           for pid, ticks in sorted(known.items())])
                budget["observed_elapsed_seconds"] = elapsed(budget, finished)
                save(path, budget)
            return returncode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget-file", required=True, type=Path)
    parser.add_argument("--limit-seconds", required=True, type=float)
    parser.add_argument("--label")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.limit_seconds) or args.limit_seconds <= 0:
            raise ValueError("Budget limit must be positive and finite")
        if args.status:
            budget = read_budget(args.budget_file, args.limit_seconds)
            spent = elapsed(budget, clock_snapshot()) if budget else 0.
            print(json.dumps({"status": "started" if budget else "not_started",
                              "elapsed_seconds": spent,
                              "remaining_seconds": max(0., args.limit_seconds - spent),
                              "actions": budget["actions"] if budget else []}))
            return 0
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not args.label or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", args.label):
            raise ValueError("Use a safe, unique action label")
        if not command:
            raise ValueError("An action command is required after --")
        return execute(args.budget_file.resolve(), args.limit_seconds, args.label, command)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Bounded adaptive action: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
