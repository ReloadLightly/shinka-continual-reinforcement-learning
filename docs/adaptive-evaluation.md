# Adaptive evaluation and cache protocol

This stage freezes the evaluator for executable mutation programs. The completed
static winners remain controls. No adaptive proposal is selected from the
control comparison, and the comparison cannot establish Shinka search quality.

## Objective and partitions

The selection score is the equal-weight mean of normalized active-task centroid
return throughout training and fresh previous-task centroid return after each
switch. This implements the objective proposed in
[the adapter specification](adaptive-programs.md#6-selection-objective)
before reserved static validation. The reference evaluator supplies the phase
policies and episode returns; no evaluation feedback reaches the mutation rule.

| Partition | Training seeds | Task trials | Fresh evaluation seeds | Generations per phase |
| :--- | :--- | :--- | :--- | ---: |
| Adaptive search | 4001–4003 | 4002–4004 | 904001–904003 | 20 |
| Reserved adaptive validation | 5001–5005 | 5002–5006 | 905001–905005 | 80 |
| Final reporting | 42–51 | 1–10 | Allocated at reporting | 200 |

Both adaptive profiles use four alternating phases, population 64, three
training episodes per member, and a 500-step cap. Search uses three in-loop
evaluation episodes, validation ten; fresh post-hoc evaluation uses ten in both.
Changing the phase interval in validation tests transfer beyond the training
schedule. The [allocation audit](adaptive-seed-allocation-20261003.json) checked
4,071 local and published JSON files without inspecting outcomes to choose seeds.
Reserved validation and final reporting were unused during this control stage.

The scorer checks ordered training records, native method identity, phase/task
mapping, all episode vectors, and agreement with the independently validated
analysis summary. It reports active and previous-task score separately, along
with LA, signed forgetting at every switch, mean F, LA−F, ZT, and cumulative
metrics. Previous-task return can reflect later learning of a previously weak
task; it is not a pure measure of retained knowledge.

## Fixed controls and budget

| Control | Frozen behavior |
| :--- | :--- |
| Identity adapter | Initial width 0.5, archive fraction 0.5, unchanged memory |
| Arithmetic adapter | `sigma * exp(0.1 * (stats[4] - 0.5))` |
| Native FocusGA | Explicit paper settings with CartPole initialization; centroid inside population budget |
| Static Shinka 11 | Previously selected width 0.065 and archive fraction 0.075 |
| Static random 24 | Previously selected exact parameters from the validation receipt |

Each control consumes three development trials: 23.04 million nominal training
steps. The five-control comparison allocates 15 trials and 115.2 million steps.
Training budgets exclude in-loop diagnostics and fresh checkpoint evaluation.
All methods use the same seed/task pairs and two logical CPUs. The production
adapter records compact progress and source receipts without diagnostic population
snapshots. It validates every width update, including the final unused update.

## Verified reuse and failure accounting

Cache identity combines the canonical program AST and the complete frozen
evaluation context: profile, seed/task/evaluation mapping, objective, grammar,
upstream and evaluator source hashes, interpreter/packages, CPU affinity, numerical
environment, and timeout. Static and Focus controls use separate declared method
identities. Formatting-equivalent programs may reuse evidence; behavioral
equivalence is not inferred.

Only complete evaluations are reusable. Every hit rehashes all local artifacts,
checks the original evaluated source, rederives each seed's score from raw
evidence, and checks the aggregate. Requests preserve their own source hash and
link to the original evaluated source and receipt. Cache entries are serialized
under a study lock, preventing concurrent duplicate training.

Every request consumes a slot, including invalid and repeated proposals. The
evaluator makes no model calls; outer Shinka proposal usage must be accounted
separately. A cache hit records zero new training and the avoided nominal budget.
The cache exercise uses a differently formatted identity program and must reuse
the first identity evaluation with identical scores.

Failed requests and training attempts remain in place. A control retry creates
a fresh request and cache-attempt directory. Costs distinguish allocated training,
completed training, and scored trials, so a post-hoc failure cannot hide training
work. An interrupted attempt without a terminal receipt requires review instead
of automatic reuse. The static search retains its original retraining policy.

## Execution and replay

The completed study used source snapshot
[`fee9baa`](https://github.com/ReloadLightly/shinka-continual-reinforcement-learning/tree/fee9baa).
Replay its runners at that revision: later changes to source-hashed files are
intentionally rejected. Use a separate checkout when preserving newer work.

Freeze the plan and run only the first control:

```bash
.venv/bin/python scripts/run_adaptive_controls.py \
  --study-dir results/adaptive-controls-new --max-controls 1 --execute
```

Then resume the remaining controls and the formatting-only duplicate:

```bash
.venv/bin/python scripts/run_adaptive_controls.py \
  --study-dir results/adaptive-controls-new --resume --execute
.venv/bin/python scripts/report_adaptive_controls.py \
  --study-dir results/adaptive-controls-new --report-dir reports/adaptive-controls-new
```

Omitting `--execute` freezes the plan without training. Existing study and report
directories are never replaced. Complete controls are verified when resuming.
The Shinka-compatible task evaluator requires `SHINKA_ADAPTIVE_STUDY` to identify
this frozen study and writes `correct.json` plus `metrics.json` for each request.
Adaptive Shinka proposals use a separate stage: their own archive,
initial identity slot, and resumable targets of 5, 13, and 25 total slots.

## Observed control study

All 15 trials completed without failure: 115.2 million nominal training steps,
4,500 fresh checkpoint-evaluation episodes, and 15.21 minutes summed evaluation
time. The identity-only first stage took 2.86 minutes; resuming added the other
four controls without retraining identity. Seven requests include the five
controls, a formatting-only duplicate, and an actual command-line evaluator
check. Both diagnostic requests reused the identity cache with identical scores
and zero new training, avoiding 46.08 million nominal steps in total.

These are fixed-control development results. No adaptive Shinka proposals or
model calls occurred in this control study. The objective and arithmetic rule
were not changed after observing outcomes. The subsequent proposal experiment
used a separate Shinka archive with cumulative targets of 5, 13, and 25 slots.
The measured controls took 2.5–3.8 minutes per candidate. The
[complete adaptive endpoint](adaptive-endpoint.md) reports search outcomes and
costs under the same frozen evaluator. After explicit endpoint closure, the
[reserved handoff](../reports/adaptive-validation-freeze-20261003/plan.json)
froze the finalist and controls; that comparison is now complete. The control
study itself used no reserved outcomes.

See the [adaptive search and control results](../README.md#adaptive-search-and-fixed-controls),
the [complete evidence](../reports/adaptive-controls-20261003/summary.json),
[artifact receipts](../reports/adaptive-controls-20261003/checksums.json), and
the [first-stage export](../reports/adaptive-controls-stage1-20261003/summary.json).
