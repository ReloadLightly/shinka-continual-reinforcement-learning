# Reviewed adaptive-search recovery

**Status: executed and stopped at generation 17; 18 of 25 slots consumed.**
The [stopped export](../reports/adaptive-recovery-stopped-20261003/summary.json)
and [independent review](../reports/adaptive-recovery-stopped-review-20261003.json)
retain the valid generation 16 and grammar-rejected generation 17. No slot 18
started. The [public preparation freeze](../reports/adaptive-recovery-preflight-20261003/recovery-plan.json)
and [verification record](../reports/adaptive-recovery-verification-20261003.json)
bind the implementation, original archive, RNG decision, and zero added work
at preparation.
The controller implements the
[reviewed recovery requirements](adaptive-search.md#reviewed-recovery-work)
for the [stopped target-25 archive](../reports/adaptive-shinka-stage25-stopped-20261003/summary.json).
It does not establish a completed endpoint or authorize reserved validation.
The original [search protocol](adaptive-search.md) and
[evaluator contract](adaptive-evaluation.md) remain frozen.

## Scope and retained evidence

The recovery protocol is `adaptive-shinka-reviewed-recovery-v1`. It accepts the
reviewed source archive at `results/adaptive-shinka-20261003` and creates an
independent working copy. The recovery directory is
`results/adaptive-shinka-recovery-20261003`; the public freeze is
`reports/adaptive-recovery-preflight-20261003`.

| Generations | State at preparation | Recovery treatment |
| :--- | :--- | :--- |
| 0–13 | Fourteen valid scored programs, including cached identity | Preserve sources, ancestry, scores, and receipts |
| 14 | Grammar rejection before training | Preserve the incorrect database row and failed request |
| 15 | Interrupted request before a recorded Codex launch; no program row | Keep the slot consumed without inserting a replacement row |
| 16–24 | Nine unused slots | Only these generations may receive new proposals |

The table records the pre-execution allocation; the observed outcomes below
supersede its unused-slot counts. The original archive, published snapshots, source files, plans, and evaluator
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

Execution is a separate operation, permitted once from a prepared state.
The following command has already run for this directory; its failed state
rejects another execution:

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

## Observed execution and next checkpoint

The public freeze was committed before execution at
[`62407bc`](https://github.com/ReloadLightly/shinka-continual-reinforcement-learning/tree/62407bc).
Generation 16 was valid with 484 AST nodes and completed all three development
trials, scoring **0.44738472377061844**. Generation 17 had **515 nodes** and
failed the unchanged 512-node grammar limit before training. Its completion
barrier stopped native execution before generation 18 had a slot reservation,
provider request, or directory. The 25-slot endpoint remains incomplete.

The [review](../reports/adaptive-recovery-stopped-review-20261003.json) independently
rederives generation 16's scores and reference metrics from raw evidence and
verifies published hashes, preserved source/archive bindings, reservations,
request accounting, cleanup, and saved RNG state. Recovery added two guarded
requests and responses, three trials, **23.04 million nominal steps**, and
**900 fresh checkpoint-evaluation episodes**. Cumulative search has 18 consumed
slots, 17 database rows, 15 valid programs, 42 new trials, 322.56 million nominal
steps, and 12,600 fresh episodes. The best valid program remains generation 5.

The recorded session ran from **05:55:14.780203 to 06:00:30.701768 UTC** on
3 October 2026. Supervised native execution took **297.4673 seconds** and
returned code 1 gracefully. Cleanup took **0.1475 seconds**, with 53 tracked
processes, no survivors or errors, and no SIGTERM or SIGKILL needed. Native
saved refreshed RNG hash
`8d8fa2a390807c55874f9d17177971731f07b8ecf29329c20d722e8e0e02c20a`.
This successful finalization does not undo the declared use of stale stage-13
RNG at the beginning of recovery.

The evidence was exported with the following command. Use another fresh report
directory for a replay; both the preflight and stopped publications are immutable.

```bash
.venv/bin/python scripts/report_adaptive_recovery.py \
  --results-dir results/adaptive-shinka-recovery-20261003 \
  --report-dir reports/adaptive-recovery-stopped-20261003
```

The current controller accepts only its original reviewed source shape and
starts at generation 16. It cannot resume this failed recovery. The next
implementation must therefore use a separate controller revision, working copy,
and published plan binding this stopped checkpoint and its fresh RNG receipt.
Only **generations 18–24** may receive new requests. Retain invalid rows 14 and
17 and the no-row consumed slot 15, preserve both historical plans and all
scores, and retain the serial completion barrier and one-request reservations.
At most seven new uncached valid proposals would add **21 trials** and
**161.28 million nominal steps** under the same development protocol.

Before execution, integration tests must show that inherited failures do not
trigger a premature stop, generation 18 is first, no consumed slot is reused,
new failures save RNG when graceful and prevent the next proposal, and complete
fixtures finish at 25 consumed slots. Publish the revised source and input
bindings before any further model request. Preserve the original grammar,
prompt configuration, objective, controls, and seed partitions. Do not enlarge
the endpoint or reinterpret this partial archive as complete. Reserved validation
and final reporting seeds remain untouched.

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

The full regression suite passed 613 tests. The real preparation then exposed
an empty native `results/` directory omitted by the slot-15 fixture; preparation
stopped before creating output. The validator and fixture were corrected, and
all 23 controller tests passed again. The recorded-runtime preparation and
wrapper availability check then passed at implementation revision
[`dbdf37e`](https://github.com/ReloadLightly/shinka-continual-reinforcement-learning/commit/dbdf37eaf98c51e7e617d0cdc064312f272041a9).
The [verification record](../reports/adaptive-recovery-verification-20261003.json)
retains both preparation attempts, exact commands, revisions, and checksums.
The preserved preflight state has no recovery sessions, reservations, proposals,
or training; the separate stopped export records the subsequent execution.
After the real stopped execution, all 49 focused recovery tests and repository
lint passed again; no trainer, evaluator, or frozen controller source changed.
The [reserved validation handoff](adaptive-validation.md) remains a
separate proposed protocol until the search endpoint has been explicitly resolved.
