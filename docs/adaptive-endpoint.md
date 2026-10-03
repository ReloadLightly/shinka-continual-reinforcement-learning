# Completion of the declared adaptive search allocation

Reserved validation follows resolution of the declared 25-slot development
search. At preparation, the input checkpoint had consumed 21 slots and retained
17 valid programs, three grammar rejections, and one interrupted request. Its
[published evidence](../reports/adaptive-continuation-stopped-20261003/summary.json)
and [independent review](../reports/adaptive-continuation-outcome-review-20261003.json)
remain immutable. The four slots then remaining were generations 21–24; the
observed completion and closure are recorded below.

## Reviewed continuation

The new endpoint controller prepares an independent working copy from a fully
verified stopped checkpoint. It preserves every previous controller plan, source
receipt, program, evaluation, reservation, and request ledger. Each prepared
attempt has its own source revision and input freeze, published before any
proposal. It restores the checkpoint's gracefully saved host RNG once before
native initialization. The earlier stage-13 RNG rollback remains a deviation
from uninterrupted sampling.

The first allocation is at most four guarded requests, 12 new training trials,
and 92.16 million nominal training steps. Each slot permits one request. The
existing three-seed evaluator, objective, grammar, prompts, controls, and
training settings remain fixed. A cache hit consumes a slot while avoiding
training. A grammar rejection also consumes its slot. Failed or interrupted
slots are never retried, repaired, replaced, or backfilled.

The serial native barrier stops on each new terminal failure and prevents the
next reservation. The controller cannot execute a failed state again. A later
continuation requires a fresh preparation from the independently reviewed stop,
an independent working copy, and another published input freeze. Its starting
generation is the first unused slot, and its bounds shrink with the remaining
allocation. This is an explicit review between attempts, not an automatic retry
loop. Only gracefully finalized, sealed grammar rejections with complete cleanup
and no incomplete training are eligible; other failures need a new diagnosis.

Native sessions retain the four-hour timeout and ten-second failure-finalization
grace. They run sequentially on the original two CPUs and subscription route.
The isolated Codex 0.159.3 executable and exact runtime checks from the
[prior continuation](adaptive-continuation.md#runtime-and-execution) remain in
use. Cumulative compute and session time include every earlier attempt once.

## Endpoint resolution

Controller success and exhaustion of the search allocation are separate facts.
If the final slot fails grammar validation, the native session remains failed;
the failed proposal still exhausts its allocated slot. No slot beyond generation
24 can be requested.

After all 25 slots are consumed, independently review the complete evidence and
publish an explicit endpoint-resolution artifact. Require no unpersisted or
incomplete evaluations, complete request accounting, intact earlier source and
artifact bindings, saved RNG, and complete process cleanup. Retain every failure
and its costs. A partial archive cannot qualify merely because a session stopped.

That resolution closes all proposal feedback before selecting a finalist. Rank
all valid programs, including identity, by the exact unrounded mean combined
development score; break exact ties by earlier generation. The
[reserved validation protocol](adaptive-validation.md) then freezes the selected
source and every fixed control recipe before reserved outcomes are opened.
Neither a final grammar failure nor a negative search result permits changing
the objective, candidate set, controls, budgets, or validation partition.

## Verification

Native integration tests construct the actual earlier stopped native states
with synthetic provider and evaluator boundaries, then verify that generation
21 is first. They cover all four successful slots, failures before the last
slot, a failure at slot 24, receipt corruption, duplicate reservations, fresh RNG,
and preservation of earlier programs. These tests check orchestration and do
not provide learning results.

The new controller also tests source and plan tampering, exact remaining
budgets, recursive history preservation, independently prepared later attempts,
no automatic retries, and truthful endpoint accounting. Existing controllers
and the pinned upstream remain unchanged.

## Observed endpoint and closure

The first endpoint attempt completed generations 21–24 after publication at
[`e1bc425`](https://github.com/ReloadLightly/shinka-continual-reinforcement-learning/tree/e1bc425).
The [result export](../reports/adaptive-endpoint-complete-20261003/summary.json)
contains the complete 25-slot allocation: 21 distinct valid programs including
identity, grammar rejections at 14, 17, and 20, and the interrupted no-row slot
15. All four new requests completed without a failure or retry. The original
two-CPU numerical settings and evaluator were unchanged.

The [independent review](../reports/adaptive-endpoint-outcome-review-20261003.json)
checks all source and history bindings, the ledger, cleanup, fresh RNG, new
raw scores and costs, and the exact selection score of every valid candidate.
Generation 5 remains the development leader. The
[explicit closure](../reports/adaptive-endpoint-closure-20261003.json)
closes proposal feedback before reserved validation. The
[README](../README.md#complete-adaptive-search-endpoint) reports all new outcomes
and cumulative costs, including concurrent CPU use during this attempt.

The working archive is `results/adaptive-shinka-endpoint-20261003-round1`.
Its sealed state cannot execute again. To reproduce the text export, choose a
new destination and run:

```bash
.venv/bin/python scripts/report_adaptive_endpoint.py \
  --results-dir results/adaptive-shinka-endpoint-20261003-round1 \
  --report-dir reports/adaptive-endpoint-reexport-NEW
```

The subsequent reserved comparison completed under the
[frozen handoff](../reports/adaptive-validation-freeze-20261003/plan.json) and
[comparison protocol](adaptive-validation.md). It starts fresh populations on
the reserved partition; no development policy or score is reused as a validation
outcome. See the [README](../README.md#reserved-adaptive-finalist-validation) for
the current comparison status.
