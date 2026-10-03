# Continuation from the eighteen-slot checkpoint

**Status: executed and stopped at generation 20; 21 of 25 slots consumed.**
The [stopped export](../reports/adaptive-continuation-stopped-20261003/summary.json)
retains valid generations 18 and 19 and rejected generation 20. Generations
21–24 remain unused. The prepared freeze below describes the allocation before
execution and remains immutable.

This protocol continues the [stopped reviewed recovery](adaptive-recovery.md)
through the seven remaining generations, 18–24. It uses a separate controller,
working copy, and pre-execution freeze. The earlier search and recovery records
remain immutable. This is development search; adaptive validation and final
reporting remain separate work.

The [prepared freeze](../reports/adaptive-continuation-preflight-20261003/continuation-plan.json)
binds implementation revision `b5a3cf1` and the original input evidence. Its
[accounting summary](../reports/adaptive-continuation-preflight-20261003/summary.json)
records zero new work. The
[verification record](../reports/adaptive-continuation-verification-20261003.json)
retains the runtime restoration and check attempts.
The [independent prepared review](../reports/adaptive-continuation-prepared-review-20261003.json)
verifies copied evidence, source bindings, unused slots, and zero added work.

## Input and fixed bounds

The source is `results/adaptive-shinka-recovery-20261003`, independently checked
in the [repository audit](../reports/repository-audit-20261003.json) and its
[original stopped review](../reports/adaptive-recovery-stopped-review-20261003.json).
It contains 18 consumed slots, 17 database rows, and 15 valid programs, including
identity. Invalid generations 14 and 17 and interrupted no-row generation 15
remain consumed. The new controller accepts only this reviewed source shape,
with verified request receipts, graceful native finalization, and complete
process cleanup.

| Bound | Value | Unit |
| :--- | ---: | :--- |
| First new generation | 18 | slot index |
| Last permitted generation | 24 | slot index |
| Maximum new guarded requests | 7 | requests |
| Maximum new training trials | 21 | seed trials |
| Maximum new nominal training steps | 161,280,000 | environment steps |
| Native session timeout | 14,400 | seconds |
| Failure finalization grace | 10 | seconds |

<sub>Table 1. Continuation allocation. The unchanged three-seed evaluator uses
7,680,000 nominal training steps per trial. Valid cache hits avoid new training;
invalid and interrupted proposals still consume their slots.</sub>

The objective, grammar, task prompts, controls, training settings, proposal
model, and seeds 4001–4003 remain fixed. Each new slot permits one guarded
request. Any new terminal failure stops execution, and the same state cannot
execute again. No retries, repairs, backfilling, or additional slots are part of
this continuation. A complete successful block would end with 25 consumed
slots, 24 database rows, and 22 valid programs; canonical duplicates are counted
separately.

## Checkpoint and history

The new archive restores the host RNG saved at the graceful generation-17
failure, with input SHA-256
`8d8fa2a390807c55874f9d17177971731f07b8ecf29329c20d722e8e0e02c20a`.
Restoration occurs once, before native initialization. Initialization can consume
draws; the earlier rollback to the stage-13 RNG remains a documented deviation.
This continuation cannot establish an uninterrupted sampling trajectory.

The independent `archive/` copy retains the native search plan, failed state,
historical summary, database rows, programs, and evaluator receipts. SQLite is
copied through its backup API and checked for logical equivalence. Earlier
recovery reservations and its native receipt are retained byte-for-byte under
`history/archive/`; earlier controller records are copied under
`history/controller/`. The new attempt writes fresh reservations and a new native
receipt. Source, copied history, generation artifacts, plan, and runtime hashes
are checked before execution and export.

The frozen [native adapter](../src/shinka_crl/adaptive_recovery_native.py) already
accepts a starting generation. The new
[controller](../src/shinka_crl/adaptive_continuation.py) invokes it with
`start=18`, `target=25`, and `missing=(15,)`. Its completion barrier ignores
inherited failures and verifies each new result before permitting another
proposal. The existing process supervisor retains descendant cleanup and
timeout behavior. Cumulative summaries include earlier work once and report
this attempt's added compute separately.

## Runtime and execution

The repository audit found that the system proposal CLI had advanced to Codex
0.160.0 while the frozen experiment requires 0.159.3. The exact
`@openai/codex@0.159.3-linux-x64` package was restored under the ignored
`results/continuation-tools/` directory and checked against the registry's
SHA-512 integrity value. The system installation and frozen runtime receipt
remain unchanged. Prepend that isolated executable directory for commands:

```bash
export PATH="$PWD/results/continuation-tools/codex-0.159.3/package/vendor/x86_64-unknown-linux-musl/bin:$PATH"
```

Preparation makes no model proposals or training calls. It requires committed
implementation files and verifies the unchanged runtime and subscription route.
Use fresh output paths for each independently reviewed attempt:

```bash
.venv/bin/python scripts/run_adaptive_continuation.py \
  --source-dir results/adaptive-shinka-recovery-20261003 \
  --results-dir results/adaptive-shinka-continuation-20261003 \
  --prepare-only

.venv/bin/python scripts/report_adaptive_continuation.py \
  --results-dir results/adaptive-shinka-continuation-20261003 \
  --report-dir reports/adaptive-continuation-preflight-20261003
```

Publish the tested implementation and exported freeze before any new proposal.
Only the prepared state can execute:

```bash
.venv/bin/python scripts/run_adaptive_continuation.py \
  --results-dir results/adaptive-shinka-continuation-20261003 --execute
```

Export completed or failed evidence to a fresh report directory, preserving the
preflight publication. Top-level continuation state and summaries describe the
new attempt; inherited records keep their original meaning.

## Verification

The controller tests construct a stopped first recovery with both grammar
failures and the interrupted slot, then exercise preparation, inherited-evidence
tampering, exclusive requests, partial reservation accounting, no retry, complete
endpoint accounting, and export checksums. Native tests use the pinned Shinka
runner, database, patcher, and scheduler with synthetic proposal and evaluation
boundaries. They establish orchestration behavior, not learning performance.

```bash
.venv/bin/python -m pytest -q \
  tests/test_adaptive_continuation.py tests/test_adaptive_continuation_native.py
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

The execution sandbox blocked local socket writes used by Python's asyncio
thread wakeups, causing native integration tests to stall. Those tests require
an environment permitting local socketpair communication. Preserve failed or
interrupted check attempts in the verification record; do not change the
experiment or native implementation to accommodate this sandbox restriction.

## Observed outcome and next checkpoint

The tested implementation was committed at
[`b5a3cf1`](https://github.com/ReloadLightly/shinka-continual-reinforcement-learning/tree/b5a3cf1)
and the reviewed input freeze was published before execution at
[`d8e6cbb`](https://github.com/ReloadLightly/shinka-continual-reinforcement-learning/tree/d8e6cbb).
The full regression suite passed 637 tests; repository lint passed. The
[verification record](../reports/adaptive-continuation-verification-20261003.json)
retains successful checks, sandbox interruptions, and the runtime restoration.

Generations 18 and 19 each contained 505 AST nodes and completed three fresh
development trials. Their combined means were **0.3943111130701171** and
**0.3980375016848246**. Generation 20 contained **536 nodes** and was rejected
under the unchanged 512-node limit before training. The completion barrier
prevented a generation-21 directory, slot reservation, or request. See
[Table 18 in the README](../README.md#stopped-generation-18-continuation), the
[raw export](../reports/adaptive-continuation-stopped-20261003/summary.json), and
the [independent outcome review](../reports/adaptive-continuation-outcome-review-20261003.json).

The attempt added six trials, 46.08 million nominal training steps, 1,800 fresh
checkpoint-evaluation episodes, and three guarded requests and responses.
Cumulative accounting records 21 consumed slots, 20 database rows, 17 valid
distinct programs including identity, 48 new trials, 368.64 million nominal
steps, and 14,400 fresh episodes. No training attempt is incomplete.

The session ran from **07:47:32.257235 to 07:54:17.264603 UTC** on 3 October
2026. Supervised native execution took **379.7382 seconds**, returning code 1
gracefully; cleanup took **0.0647 seconds** with 70 tracked processes, no
survivors or errors, and no SIGTERM or SIGKILL needed. The native adapter saved
fresh RNG SHA-256
`054c2a6c3021866e5c5787b6cef7b6861c663e9139bee626d494fe65e676b489`.
The input source, copied histories, existing results, frozen evaluator, and
upstream checkout remain intact.

The evidence was exported with the following command. Replays require a fresh
destination; preserve both existing preflight and stopped publications:

```bash
.venv/bin/python scripts/report_adaptive_continuation.py \
  --results-dir results/adaptive-shinka-continuation-20261003 \
  --report-dir reports/adaptive-continuation-stopped-20261003
```

This controller accepts only the earlier eighteen-slot source and cannot
re-execute the failed state. A further continuation requires a separate reviewed
source revision, independent working copy, and published plan binding this
checkpoint and its saved RNG. Only **generations 21–24** may receive new
requests, at most **12 new trials** and **92.16 million nominal steps**. Retain
grammar failures 14, 17, and 20, consumed no-row slot 15, both earlier plans,
this plan, and all outcomes and costs.

Before execution, test that generation 21 is first, old failures do not trigger
a premature stop, consumed slots cannot be reused, new failures prevent the
next proposal and save RNG on graceful return, and complete fixtures stop at
25 consumed slots. Four successful new programs would produce 24 database
rows and 21 valid programs; distinct canonical programs may be fewer. Preserve
the fixed grammar, objective, prompt configuration, controls, development
partition, and overall 25-slot ceiling. Reserved validation and final reporting
remain untouched while the search endpoint is unresolved.
