"""One-shot guard for the user-authorized PPO budget amendment.

This guard does not start, resume, or modify training. It preserves native
checkpoints and terminates only the identified native trainer process group.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "reference-development-ppo-20261003"
ATTEMPT = SOURCE / "trials/ppo/seed_1001/training/attempt_001"
EXPECTED_PID = 178307
EXPECTED_START = "2308949"
TARGET_UPDATES = 6000
TRAINING_SECONDS_LIMIT = 27000
TOTAL_SECONDS_LIMIT = 28800


def identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()
        return fields[19], fields[0]
    except FileNotFoundError:
        return None, None


def write(name, value):
    target = ROOT / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(target)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def events():
    journal = ATTEMPT / "phase-events.jsonl"
    parsed = []
    for line in journal.read_text().splitlines(keepends=True):
        if line.endswith("\n"):
            parsed.append(json.loads(line))
    return parsed


def main():
    with (ROOT / "guard.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = json.loads((SOURCE / "active-process.json").read_text())
        assert (active["pid"], active["start_ticks"]) == (EXPECTED_PID, EXPECTED_START)
        assert identity(EXPECTED_PID)[0] == EXPECTED_START
        assert os.getpgid(EXPECTED_PID) == EXPECTED_PID
        started = int(EXPECTED_START) / os.sysconf("SC_CLK_TCK")
        state = {
            "status": "watching", "guard_pid": os.getpid(),
            "started_at_utc": now(), "source": str(SOURCE),
            "trainer_pid": EXPECTED_PID, "trainer_start_ticks": EXPECTED_START,
            "target_updates": TARGET_UPDATES, "target_complete_phases": 4,
            "training_limit_seconds": TRAINING_SECONDS_LIMIT,
            "total_ppo_limit_seconds": TOTAL_SECONDS_LIMIT,
            "time_basis": "Monotonic process duration since original trainer launch; includes existing training",
            "reserved_seconds_for_verification_and_evaluation": 1800,
            "reason": "User requested PPO results within approximately eight hours; full 20-phase PPO trial deferred",
            "checkpoint_copies": [],
        }
        write("guard-state.json", state)
        print(json.dumps(state), flush=True)
        copied = set()
        while True:
            observed_start, process_state = identity(EXPECTED_PID)
            if observed_start != EXPECTED_START or process_state == "Z":
                state.update(status="trainer_terminal", finished_at_utc=now())
                write("guard-state.json", state)
                return
            phase_events = events()
            latest = phase_events[-1] if phase_events else None
            updates = latest["completed_updates"] if latest else 0
            if updates and updates % 3000 == 0 and updates not in copied:
                native = ATTEMPT / "resume.pkl"
                before = digest(native)
                saved = ROOT / f"resume-update-{updates}.pkl"
                shutil.copyfile(native, saved)
                assert digest(saved) == before == digest(native)
                state["checkpoint_copies"].append({
                    "completed_updates": updates, "file": saved.name,
                    "sha256": before, "phase_event": latest, "copied_at_utc": now(),
                })
                copied.add(updates)
                write("guard-state.json", state)
            elapsed = time.monotonic() - started
            reason = ("target_complete_phases_reached" if updates >= TARGET_UPDATES else
                      "training_time_allowance_reached" if elapsed >= TRAINING_SECONDS_LIMIT else None)
            if reason:
                assert identity(EXPECTED_PID)[0] == EXPECTED_START
                assert os.getpgid(EXPECTED_PID) == EXPECTED_PID
                state.update(status="termination_requested", stop_reason=reason,
                             measured_training_seconds_at_stop=elapsed,
                             latest_checkpoint_updates=updates, finished_at_utc=now())
                write("guard-state.json", state)
                os.killpg(EXPECTED_PID, signal.SIGTERM)
                print(json.dumps(state), flush=True)
                return
            time.sleep(1)


if __name__ == "__main__":
    main()
