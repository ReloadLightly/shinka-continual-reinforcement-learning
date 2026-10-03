# Completion of the declared adaptive search allocation

Reserved validation follows resolution of the declared 25-slot development
search. The input checkpoint has consumed 21 slots and retains 17 valid
programs, three grammar rejections, and one interrupted request. Its
[published evidence](../reports/adaptive-continuation-stopped-20261003/summary.json)
and [independent review](../reports/adaptive-continuation-outcome-review-20261003.json)
remain immutable. The four remaining slots are generations 21–24.

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
