# Reviewed adaptive-search recovery

**Status: implementation under verification; recovery plan not yet frozen or
executed.** The controller implements the
[reviewed recovery requirements](adaptive-search.md#reviewed-recovery-work)
for the [stopped target-25 archive](../reports/adaptive-shinka-stage25-stopped-20261003/summary.json).
It does not establish a completed endpoint or authorize reserved validation.
The original [search protocol](adaptive-search.md) and
[evaluator contract](adaptive-evaluation.md) remain frozen.

## Scope and retained evidence

The recovery protocol is `adaptive-shinka-reviewed-recovery-v1`. It accepts the
reviewed source archive at `results/adaptive-shinka-20261003` and creates an
independent working copy. The planned recovery directory is
`results/adaptive-shinka-recovery-20261003`; the planned public freeze is
`reports/adaptive-recovery-preflight-20261003`.

| Generations | Recorded state | Recovery treatment |
| :--- | :--- | :--- |
| 0–13 | Fourteen valid scored programs, including cached identity | Preserve sources, ancestry, scores, and receipts |
| 14 | Grammar rejection before training | Preserve the incorrect database row and failed request |
| 15 | Interrupted request before a recorded Codex launch; no program row | Keep the slot consumed without inserting a replacement row |
| 16–24 | Nine unused slots | Only these generations may receive new proposals |

The original archive, published snapshots, source files, plans, and evaluator
remain unchanged. Files are copied without writable hardlinks; SQLite is copied
through its backup API and checked for equivalent logical contents. Symlinks
inside the archive are rejected. The working copy continues to reference the
existing evaluation cache, whose verified reuse rules remain unchanged.

Native proposal selection, prompts, patching, and evaluation use the pinned
implementation. A runtime adapter changes progress accounting to include
consumed slot 15 and serializes proposals behind verified evaluation completion.
It rejects backfilling a consumed generation. The known failure at generation 14
does not stop recovery immediately; any new terminal proposal or evaluation
failure does. If all nine new slots succeed, the endpoint has 25 consumed slots,
24 database rows, and 23 valid scored programs. These counts are distinct from
the number of distinct canonical programs.

## RNG policy and bounds

The input Python and NumPy host RNG file is the sealed checkpoint from the
completed thirteen-slot stage. It predates the sampling for generations 13–15.
Recovery restores that exact file **once, before native CLI initialization**.
This is a declared deviation: it cannot reconstruct the uninterrupted trajectory,
and initialization may consume random draws before the first new proposal.
The plan binds the input hash; the native receipt records the initial and final
hashes and whether a graceful return saved updated RNG state. A failed or
interrupted execution cannot automatically resume from either RNG file.

The [controller](../src/shinka_crl/adaptive_recovery.py) fixes nine remaining
slots, one guarded request per slot, at most **27 new training trials**, and at
most **207.36 million nominal training steps**. Each uncached valid proposal
uses the unchanged three development seeds and training budget. Verified cache
hits consume slots while avoiding new training. Invalid and interrupted
attempts also consume slots; their placeholder scores are not measurements.
Reserved adaptive validation seeds 5001–5005 and final reporting seeds 42–51
remain outside this work.

The supervised native session has a **14,400-second (four-hour) timeout**.
Runtime checks and final verification are outside that interval; shutdown allows
up to two seconds for SIGTERM followed by three seconds for SIGKILL. Report
measured session and cleanup durations separately from the configured bound.
A new failure stops the session for review. No automatic retry, replacement
proposal, grammar relaxation, or enlargement of the 25-slot budget is permitted.
On terminal evidence, the native barrier prevents further proposals immediately.
The outer supervisor allows up to ten seconds for native finalization and its
RNG receipt before forcing descendant cleanup; the session timeout still applies.

## Preparation and publication

Run commands from the repository root. Preparation requires the recovery
implementation files to match a committed revision. It records that revision,
source hashes, original archive hashes, copied-state hashes, bounds, and RNG
policy in `plan.json`, with a separate plan receipt and initial state seal.

Use the recorded Codex 0.159.3 binary on the current WSL machine:

```bash
PATH="$HOME/.codex/packages/app-server-daemon/releases/0.159.3-x86_64-unknown-linux-musl/bin:$PATH" \
  .venv/bin/python scripts/run_adaptive_recovery.py \
  --source-dir results/adaptive-shinka-20261003 \
  --results-dir results/adaptive-shinka-recovery-20261003 \
  --prepare-only
```

Preparation verifies the original runtime and the recovery wrapper's `--check`
route. Native Headless also uses this route before a generation exists. It
passes directly to the unchanged subscription guard, checks tooling, login and
model configuration, and makes no model proposals or training calls. It creates
no slot or provider-request reservation. Other machines must supply the recorded
runtime rather than editing its receipt.

Export and publish the prepared plan before any execution:

```bash
.venv/bin/python scripts/report_adaptive_recovery.py \
  --results-dir results/adaptive-shinka-recovery-20261003 \
  --report-dir reports/adaptive-recovery-preflight-20261003
```

The prepared export contains the plan, state, bindings, checksums, and accounting
showing zero additional work. Historical numerical evidence remains in the
stopped-source publication. Exporting files locally does not publish them to the
repository; retain the tested implementation and public freeze together before
starting proposals.

## Execution and accounting

Execution is a separate operation, permitted once from a prepared state:

```bash
PATH="$HOME/.codex/packages/app-server-daemon/releases/0.159.3-x86_64-unknown-linux-musl/bin:$PATH" \
  .venv/bin/python scripts/run_adaptive_recovery.py \
  --results-dir results/adaptive-shinka-recovery-20261003 \
  --execute
```

The execution command reads the source binding from the frozen plan and does
not accept `--source-dir`. The controller holds an exclusive archive lock;
exports use the same lock. Execution rejects changed source, runtime, plan,
inherited evidence, or prepared state.

Before native proposal work, `archive/recovery_slots/gen_N.json` is created
exclusively and synced to disk. Before invoking the unchanged subscription
guard, a second exclusive, synced reservation is created under
`archive/recovery_requests/`. Existing reservations cannot be replaced. File
creation itself consumes the slot, including an interrupted partial receipt.
Provider reservations remain immutable; completion is recorded in the appended
model-request ledger. Counts distinguish reservations, guarded requests,
recorded Codex launches, valid programs, allocated training, completed training,
and scored trials. Backend request counts and remaining subscription allowance
are not observable.

The [Linux supervisor](../src/shinka_crl/recovery_process.py) uses child-subreaper
adoption, `/proc` process identities, and pidfd signals to clean descendants
across process groups, including providers that create new sessions or outlive
their parents. It checks for stops at 50-millisecond polling intervals. Cleanup
also runs on timeout, signals, and exceptions; prior signal handlers and
subreaper state are restored. The synchronous caller must run in its main thread
without concurrent unrelated child creation. Cleanup evidence is retained in
the session; an incomplete cleanup is an error.

The top-level recovery `state.json` and exported `summary.json` describe the
current attempt. The inherited `archive/state.json` remains failed, and its
older `archive/summary.json` remains historical. Do not reinterpret or edit
those inherited records. After success or failure, rerun the report command
with a **fresh independent `--report-dir`**; preserve the preflight publication.
The exporter verifies receipts and both inherited and added costs, retains
failure evidence, redacts local path prefixes, and hashes binary artifacts
while leaving the binaries local.

## Interfaces and checks

The Python entry points in `shinka_crl.adaptive_recovery` are
`prepare(source=..., output=...)`, `run(output)`, and
`export(output=..., report=...)`. Preparation and execution return controller
state; export returns the verified summary. Execution errors retain failed
state and prevent another execution without a new review.

The focused regression suite uses synthetic evidence, the real pinned native
runner/database/patcher/scheduler with stubbed proposal and evaluation
boundaries, and local process trees. It exercises slot-15 exclusion, retained
failures, terminal barriers, receipts, budgets, request exclusivity, and complete
descendant cleanup without model or RL-training calls:

```bash
.venv/bin/python -m pytest -q \
  tests/test_adaptive_recovery.py \
  tests/test_adaptive_recovery_native.py \
  tests/test_recovery_process.py
```

Final test counts and the public preparation receipt must be recorded before
execution. The [reserved validation handoff](adaptive-validation.md) remains a
separate proposed protocol until the search endpoint has been explicitly resolved.
