# Repeated adaptive-program search: evolutionary feedback versus independent proposals

## Question and relation to completed experiments

Does evolutionary parent selection with archive feedback improve the quality of
adaptive mutation-width programs over independent proposals from the same
initial program, at the same proposal and learner budgets?

The completed static search, adaptive search, and reserved comparisons remain
closed. Their results motivate this new question; they are not reused as
untouched test data. The [mechanism diagnostic](../reports/adaptive-mechanism-20261004/summary.json)
finds that the previously selected rule's deficit against FocusGA is principally
previous-task performance. Similar narrow mutation widths accompany both good
and poor retention. This is descriptive evidence, not a causal width ablation.
The earlier random-search control varied static settings and therefore did not
test the value of evolutionary search over adaptive programs.

## Frozen compact design

This initial study has two paired outer-search repetitions, with outer seeds
202610041 and 202610042. Each repetition contains the following two arms:

| Arm | Parent and inspiration policy | Allocation per repetition |
| :--- | :--- | ---: |
| Evolutionary | Pinned Shinka weighted parent selection and one archive plus one top-k inspiration | Identity + 4 proposal slots |
| Independent | Pinned Shinka `best_of_n`: always the initial identity parent; zero archive and top-k inspirations | Identity + 4 proposal slots |

*Table 1. Matched adaptive-program search arms. Both arms use the same initial
program, task prompt, grammar, model, patch distribution, candidate scorer,
training budgets, and three development task draws. The independent arm receives
the fixed identity's numerical/text feedback, but no later program or score.
The contrast jointly tests evolutionary parent selection and archive feedback;
it does not separate those components.*

Both arms use the existing authorized ChatGPT subscription route, the pinned
Shinka implementation, and the current GPT-6.1 Sol configuration at medium
reasoning effort. No paid model API is used. The model route is not changed
between arms. Host random seeds control the recorded native sampling state;
model responses are not claimed to be deterministically seeded.

Execute repetition 202610041 in evolutionary/independent order and repetition
202610042 in independent/evolutionary order. Each archive is new and consumes
exactly five total slots, including the shared identity. Invalid or duplicate
proposals consume slots. Complete canonical-AST matches may reuse verified
evaluations under the identical frozen context, including across arms; report
both nominal allocations and actual new training. There are at most sixteen
new proposal calls across the four archives. The broader runner's historical
13/25-slot stages are outside this study and must not be invoked.

## Learner and evaluation partitions

The unchanged adaptive-width grammar controls only the next mutation width and
four-element memory. Policy architecture, population size, archive fraction,
selection, task construction, clipping, evaluator, and score weights remain
outside candidate code. No validation return, task identity, or switch flag
enters the inner update. See the [interface](adaptive-programs.md) and
[objective](adaptive-evaluation.md).

| Role | Training seeds | Task trials | Phases | Generations per phase | Population | Nominal steps per trial |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| Development, shared by all searches | 6001–6003 | 6002–6004 | 4 | 20 | 64 | 7,680,000 |
| Fresh finalist comparison | 7001–7005 | 7002–7006 | 4 | 80 | 64 | 30,720,000 |

*Table 2. New partitions. Each individual has three training episodes capped at
500 steps. Fresh checkpoint evaluation uses ten episodes per target and the
existing evaluation seed `900000 + training_seed`. The task schedule and native
settings match the existing reduced adaptive profiles; only seed identities
change. Longer evaluation phases test transfer to a different switching
interval. These are extension trials, separate from full-budget reference
reporting.*

The [allocation audit](adaptive-repeated-seed-allocation-20261004.json) inspected
identity fields in all existing local and published JSON records before any new
training, without using scores to choose seeds. It found no previous use of the
new development or evaluation identities. Completed partitions and the original
paper reporting allocation remain excluded.

Run the five existing fixed development controls unchanged: identity,
arithmetic adaptation, native FocusGA, static Shinka 11, and static random 24.
The first identity evaluation supplies an actual reduced-budget execution check
for the new partition configuration and is retained as development evidence.
It must pass the existing artifact, source, task, and score checks before any
proposal. Check only execution correctness and cost at this gate; do not alter
settings or partitions using the observed returns.

## Selection and fresh comparison

For each archive, select the highest exact, unrounded mean combined development
score among its five slots. Include identity; break exact ties by earlier
generation. The objective remains equal-weight normalized active-task and
previous-task performance. Freeze all four selected sources and their complete
ranking records together before accessing any fresh evaluation outcome.

Evaluate the four winners plus identity, arithmetic adaptation, and FocusGA on
all five new evaluation seeds. Thus the maximum is 35 new finalist-comparison
trials; byte/AST-equivalent programs with identical complete execution recipes
may share verified trials, while retaining all experimental memberships. Fixed
static winners remain development controls and are not part of this compact
fresh comparison. Publish the exact finalist handoff before executing it.

For each outer repetition, compute evolutionary minus independent finalist
combined score on each of the five matched evaluation seeds, then average over
those seeds. Report both repetition-level differences, their mean and sample
standard deviation, all seed outcomes, learning accuracy, signed forgetting,
transfer, cumulative performance, and active/previous score components.
Evaluation seeds are not independent outer-search repetitions. Report secondary
differences from the three fixed learners; no significance threshold, automatic
promotion rule, or positive-result requirement is specified.

Two short outer searches provide an initial controlled repeated experiment,
not a precise general ranking of search methods. Both repetitions share their
development and evaluation task draws, so results are conditional on this
small task sample. Broader search spaces and larger allocations require
separate studies rather than automatic extensions of this one.

## Resource boundary and stopping

The initial allocation is **7,200 seconds of total elapsed experiment time**,
beginning at the first new development-control execution. It includes controls,
proposals, candidate evaluation, final comparison, and intervening gaps. Code
preparation and analysis of existing evidence precede this execution clock.
The bounded launcher preserves one deadline across all invocations and compares
UTC and boot-inclusive elapsed time; restarting a command does not reset the
budget. Its cleanup overhead is retained and separately identifiable.

Reserve 40 minutes of the allocation for finalist evaluation. Do not begin a
new search archive after the first 80 minutes. If measured control/search cost
makes the declared study infeasible, stop and report that concrete limitation;
do not silently change the learner budget, discard a repetition, increase the
wall limit, replace failed proposals, or choose survivors using test results.
The cap can leave a partial experiment. Integrity failures and provider failures
also stop execution for review, with existing evidence and checkpoints retained.

Existing baseline reporting remains cancelled. No GA/ES reference trial, PPO
run, new environment, or expanded rule interface is part of this allocation.
