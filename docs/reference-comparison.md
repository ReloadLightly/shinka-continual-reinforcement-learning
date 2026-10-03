# CartPole reference comparison

This experiment addresses the reproduction of the two-task CartPole cell in
*Continual Reinforcement Learning with Neuroevolution*. It includes only the
pinned reference GA, ES, and PPO. The completed static and adaptive ShinkaEvolve
studies and reserved validations remain separate, including their negative
results; none of their candidates or validation outcomes changes this
comparison's settings or reopens completed selection. The original full-budget
protocol is retained below. The October 3, 2026 resource amendment limits the
active PPO experiment to a development prefix; full-budget GA and ES reporting
remains planned.

## Protocol and partitions

The original full-budget target is the existing
[`paper-cartpole` profile](../src/shinka_crl/profiles/paper-cartpole.json),
interpreted by the [reproduction specification](reproduction-plan.md).
There are two tasks across 20 alternating phases, a 500-step episode cap,
and ten reporting trials per method. GA and ES use 4,000 generations,
population 512, and three training episodes per member. PPO uses 30,000
updates, 2,048 environments, and 50 rollout steps per update. Each trial
therefore receives 3,072,000,000 nominal training steps. The complete
30-trial comparison allocates 92,160,000,000 nominal training steps, excluding
evaluation. Baseline optimizers and architectures remain unchanged.

Reporting seeds 42–51 map in order to task trials 1–10. The original execution
order is seed-major, with GA, ES, and PPO at each seed. Methods share task draws
and nominal budgets; their internal random draws remain method-specific. Under
the amendment, GA and ES retain those reporting identities and full budgets;
the ten full-budget PPO reporting trials are deferred. No candidate program
participates in baseline execution or evaluation.

Before reporting, full-budget development measurements use seed 1001, task
trial 1002, and evaluation seed 901001. The completed
[GA development reference](../reports/reference-timing-20261002/summary.json)
is reused after verification of its original and published evidence. The
[ES development measurement](../reports/reference-development-es-20261003/summary.json)
is also complete at the full budget. PPO began with the same declared budget
and original baseline settings; its [completed four-phase prefix](../reports/reference-development-ppo-prefix-20261003/summary.json)
implements the resource amendment below. These measurements are development evidence and cannot replace
reporting trials.
The [separate reduced diagnostic](../reports/reference-comparison-diagnostic-20261003/summary.json)
checks native GA/ES/PPO execution and analysis before full runs; it is not
comparative learning evidence.

## PPO resource amendment — October 3, 2026

The user requested an approximately eight-hour limit for PPO and authorized
reducing its run. The [frozen allocation](../reports/reference-ppo-budget-20261003/protocol.json)
records the exact target and preservation policy. The active interpretation
is a total limit of eight hours
of measured active elapsed compute for this PPO work, including training
already incurred, native artifact finalization, checkpoint evaluation, and
any further PPO execution. No additional PPO reporting trials are scheduled
under this allocation. A change to that scope requires a subsequent user
instruction.

The target is the existing development trajectory through **6,000 updates:
four complete phases of 1,500 updates each**, in task order `[0, 1, 0, 1]`.
This allocates 614,400,000 nominal training environment steps. Preserve seed
1001, task trial 1002, the original task vectors, rollout size, architectures,
optimizer settings, and pinned implementation. This is a prefix of the
original schedule; it does not shorten individual phases. Changing the phase
interval to fit twenty phases would require a distinct experiment and cannot
reuse this trajectory as though it followed that schedule.

The eight-hour limit takes precedence over the 6,000-update target. Reserve
time for finalization and fresh evaluation; if the target cannot be reached
within that allocation, use the latest verified checkpoint containing complete
two-task cycles. Record the actual retained phase count and updates, and
account for all PPO compute, including work after a retained checkpoint.
Preserve the original attempt and checkpoint under their original protocol.
Write the derived prefix, its exact protocol and provenance, and its analysis
to new paths. The target was reached: the [published PPO prefix](../reports/reference-development-ppo-prefix-20261003/summary.json)
contains 6,000 updates and four phases, followed by 100 fresh evaluation
episodes. Its [exact provenance](../reports/reference-development-ppo-prefix-20261003/raw/provenance.json)
records zero additional training updates during finalization. Total accounted
PPO compute is 22,340.0237 s (6.205562 h), within the 28,800 s allocation;
all original training and previous verification costs are retained.
The [execution record](../reports/reference-ppo-budget-20261003/execution.json)
links the preserved checkpoint to the intentional stop after update 6,000.
The original suite retains its unsuccessful full-budget status because its
30,000-update target was not completed; this does not denote a training-algorithm
failure. Original records and the separately evaluated prefix remain distinct.
The [native finalization verification](../reports/reference-ppo-budget-20261003/verification.json)
confirms that the adapter can retain checkpoint policies and recorded history
without further training, then apply the existing fresh evaluator. Its measured
PPO cost, including unsuccessful verification attempts, is charged to the same
allocation.

The full twenty-phase, ten-trial PPO arm remains an uncompleted reproduction
target. Its omission prevents a full three-method reproduction claim. GA and
ES retain their planned twenty-phase, ten-trial reporting comparison. The
completed ShinkaEvolve studies, reserved outcomes, and their selection
decisions are unchanged.

## Evaluation and interpretation

The existing checkpoint analysis evaluates each phase agent with ten fresh
episodes per native evaluation target, using evaluation seed
`900000 + training_seed`. GA and ES use saved centroids; PPO uses its saved
policy. Report every trial and its learning curve, learning accuracy (LA),
signed forgetting (F), LA−F, zero-shot transfer, and cumulative return.
For the full twenty-phase trials, forgetting averages all 19 consecutive
switches, including both directions; negative differences are retained. For
the completed four-phase PPO prefix, compute the same definitions over its
three observed switches and report each phase separately. Use the actual
retained phase count if the fallback is needed. Transfer on recurring tasks
is interpreted in the context of prior task exposure.

Cumulative return follows the pinned reference's completed-update clock and
unit NE-generation integration grid. Raw reward-unit metrics and the fixed
division by 500 remain distinct from the paper's reference-based rescaling.
Without matching reference normalization values, comparisons with the paper
are restricted to the supported behavior and method ordering rather than
claims of exact normalized numerical replication.

For GA and ES reporting, report the ten individual values, mean, and sample
standard deviation. The single PPO development prefix provides no estimate
of between-trial variation. Its acquisition and early retention may be
compared descriptively with the corresponding prefixes of the completed GA
and ES development trials after verifying matching task draws and recomputing
all metrics over the same phase count and nominal budget. The
[derived GA/ES prefix analysis](../reports/reference-development-prefixes-20261003/summary.json)
reuses the existing raw episode returns and training curves without new trials.
At the completed four-phase horizon, this means 800 GA/ES generations and 6,000 PPO updates. Keep those
prefix comparisons separate from full twenty-phase method rankings.

Assess each relevant paper finding as reproduced, discrepant, or unresolved
within the observed scope. The PPO prefix can address initial acquisition and
early switches; sustained behavior across twenty phases and repeated PPO
trials remain unresolved. Do not infer reproduction of other environments or
methods from this comparison.

## Completed four-phase development comparison

The [GA/ES prefix derivation](../reports/reference-development-prefixes-20261003/protocol.json)
and [PPO prefix provenance](../reports/reference-development-ppo-prefix-20261003/raw/provenance.json)
define matched 614,400,000-step trajectories at seed 1001, task trial 1002, and
evaluation seed 901001. GA/ES statistics reuse the first four phases and their
existing fresh centroid evaluations; PPO uses its saved policies and 100 new
fresh evaluation episodes. There is one trajectory per method.

| Method | LA (return) | Signed F (return) | LA − F (return) | ZT (return) | Cum. / (steps × 500) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GA | 500.00 | 32.07 | 467.93 | 468.00 | 0.9571 |
| ES | 500.00 | 0.00 | 500.00 | 500.00 | 0.9697 |
| PPO | 500.00 | 159.67 | 340.33 | 217.43 | 0.9832 |

These values derive from the [GA/ES evidence](../reports/reference-development-prefixes-20261003/summary.json)
and [PPO evidence](../reports/reference-development-ppo-prefix-20261003/summary.json).
All four own-task means are 500 for each method. The
[PPO raw episodes](../reports/reference-development-ppo-prefix-20261003/raw/analysis/evaluation.json)
give previous-task means of 21, 500, and 500 after the three switches, so
the first switch contributes a 479-unit loss and later endpoint losses are
zero. GA's corresponding losses are 0, 96.2, and 0; ES's are all zero.
PPO's transfer means are 133.1, 19.2, and 500. Transfer on later phases includes
previously encountered tasks and should not be interpreted as wholly unseen-task
generalization.

The [learning and retention figure](../reports/figures/reference-development-prefixes-20261003.svg)
shows unsmoothed native samples; its [metric companion](../reports/figures/reference-development-prefixes-20261003-metrics.svg)
and [provenance](../reports/figures/reference-development-prefixes-20261003.json)
retain the numerical inputs. PPO has the highest normalized cumulative
active-task return in this seed, while its first-switch retention is poorer.
This descriptive ordering provides no estimate of between-trial variation and
does not establish algorithmic superiority or the full twenty-phase ranking.

The cumulative metric follows the declared reference integration grid. For
PPO, its reward-step area is 302,032,205,126.9531; integration of every dense
PPO sample instead gives 301,503,984,914.0625. The [summary](../reports/reference-development-ppo-prefix-20261003/summary.json)
retains both quantities, and the table uses the declared reference grid.

Measured elapsed training through the fourth checkpoint was 429.4531 s for GA,
277.8433 s for ES, and 22,196.1465 s for PPO. These are observed checkpoint
boundaries, including native overhead. GA/ES source-run costs remain fully
charged in their [derived evidence](../reports/reference-development-prefixes-20261003/summary.json).
For PPO, the complete original training cost is 22,197.2616 s; finalization
took 13.9266 s and fresh evaluation 6.2488 s. The [complete cost accounting](../reports/reference-development-ppo-prefix-20261003/summary.json)
also includes 107.3577 s of prior verification and the remaining analysis
overhead, for 22,340.0237 s total.

The next scientific stage is the unchanged twenty-phase GA/ES reporting
comparison, ten trials per method. The full-budget PPO arm remains deferred;
the completed development prefix does not consume any reporting trial.

## Reference findings to assess

The direct reference is [Appendix C.1, Figure 8b](https://arxiv.org/html/2610.01583v1#A3.F8),
the first-row **CartPole noise** panel. Its [original curve figure](https://arxiv.org/html/2610.01583v1/continual_curves.png)
shows GA and ES maintaining high active-task return after initial learning,
while PPO has recurring return drops at switches. Assess initial acquisition,
recovery after each switch, and whether later phases sustain performance.
Treat these as qualitative targets: no precise reference values are inferred
from the plotted curves.

Use [Figure 2b](https://arxiv.org/html/2610.01583v1#S5.F2) and
[Appendix A.3](https://arxiv.org/html/2610.01583v1#A1.S3) to assess learning
accuracy, signed forgetting, and their trade-off within this same cell.
The plotted means place ES and GA near the highest learning accuracy, with
less forgetting for ES than for GA or PPO. Check whether these directions
persist across our trials; the reference panel marks no significant
cross-family winner.
Report zero-shot transfer and cumulative performance alongside them.
Do not substitute Figure 3's two-task CartPole physics or action-reversal
panels for observation offsets, or infer that an aggregate claim across
environments must hold in this cell.

The pinned [figure assembly](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/scripts/analysis/plot_continual_combined.py)
identifies the source cell as `paper/gymnax/data/noise_2task/CartPole_v1_sigma0.5`.
Its local source is `.upstream/continual_neuroevolution/scripts/analysis/plot_continual_combined.py`;
the authors' raw trial tree is not present in this checkout. Exact numerical
agreement with their trial distribution therefore remains unresolved.
Neighborhood mechanisms, other environments, and continual PPO variants are
outside this comparison.

## Compute and preservation

Development measurements run sequentially with the existing two-CPU allocation
and numerical thread settings. Record training and analysis durations separately,
trainer peak resident memory, completed updates, and nominal training steps.
Use measured costs to schedule the remaining GA/ES reporting comparison. The
explicit PPO amendment changes its experimental horizon and claims, while
preserving its baseline settings and the original full-budget specification.
Report measured active elapsed durations separately from UTC start and finish
timestamps; elapsed calendar time is not a substitute for the compute measure.

The existing native checkpoint option preserves intermediate training state.
Completed training and analysis are verified before reuse; an analysis failure
does not require retraining a valid completed agent. Preserve unsuccessful
attempts and their compute separately. Allow only one controller for an output
directory, and use new paths for distinct experiments and exports.

## Reporting CPU allocation amendment — October 4, 2026

The first reporting seed pair began on October 4, 2026 (local date), after a
[separate GA/ES allocation diagnostic](../reports/reference-ga-es-allocation-diagnostic-20261004/summary.json)
validated training and fresh checkpoint evaluation on the eight-CPU allocation.
The diagnostic executes only GA and ES from its three-method diagnostic plan;
its partial status is explicit, and it supplies no scientific reporting trial.

In the first GA reporting attempt, the three observed phase durations were
302.3289, 260.7388, and 277.8400 s. The earlier two-CPU GA development trial had
a median noninitial phase duration of 84.9753 s. These measurements come from
different seeds and execution times, so they do not establish a causal CPU
scaling benchmark. They motivated the [recorded scheduling decision](../reports/reference-reporting-resource-amendment-20261004/decision.json)
to return to the measured two-CPU allocation while keeping the scientific
protocol fixed. The phase timings are preserved in the
[eight-CPU attempt](../reports/reference-reporting-eight-cpu-attempt-20261004/summary.json)
and [GA development record](../reports/reference-timing-20261002/summary.json).

The eight-CPU attempt was intentionally stopped with its generation-600
checkpoint and original attempt preserved. It completed no reporting trial
and no fresh post-hoc evaluation. The [execution record](../reports/reference-reporting-resource-amendment-20261004/execution.json)
documents the transition. The existing resume contract fixes CPU affinity;
therefore the checkpoint is retained as evidence, and the identical scientific
trial starts afresh in a new two-CPU suite. No completed trial is discarded.
The twenty jobs, baseline settings, seeds, task draws, training budgets,
evaluation protocol, and frozen source hashes are unchanged; `cpu_affinity`
is the only differing plan field. The scheduling decision uses measured
execution costs and does not select seeds or settings using returns.

All work in the [eight-CPU attempt](../reports/reference-reporting-eight-cpu-attempt-20261004/summary.json)
remains charged: 918.704521254 s of training, 919.669088192 s for the suite,
and at least 499,968,000 nominal attempted training steps (651 completed
generations observed). This includes work beyond the retained generation-600
checkpoint. These costs are additional to the new suite's eventual completed
trials; they are not removed by the fresh start.

## Reporting execution

The active two-CPU suite uses the existing harness with all ten reporting
seeds, twenty phases, and the full training budget per method. Its launch
command completes the first seed's GA/ES pair within the twenty-trial plan:

```bash
.venv/bin/python scripts/run_reference_comparison.py \
  --mode reporting --methods ga es \
  --results-dir results/reference-reporting-ga-es-two-cpu-20261004 \
  --cpus 2 --timeout 21600 --analysis-timeout 1800 \
  --max-trials 2 --execute
```

Continue subsequent pairs with the same arguments and `--resume`. The harness
verifies completed trials before reuse; `--max-trials 2` bounds each invocation,
not the declared number of reporting trials. Resume only the new two-CPU suite;
the preserved eight-CPU suite remains closed. The command above records the
already-started launch and must not start a duplicate controller.

This reporting allocation exposes two logical CPUs, matching the development
allocation. Record actual affinity, elapsed training and evaluation costs,
and peak trainer memory for every attempt, including the earlier eight-CPU
work. Scientific settings, task draws, and nominal training budgets remain fixed. Completion
of all twenty GA/ES trials supplies their full-budget reporting comparison;
the original three-method comparison remains incomplete while full-budget
PPO reporting is deferred.

The scientific report summarizes experimental findings and material protocol
deviations. Detailed commands, source revisions, runtime identities, raw curves,
episode returns, checkpoint hashes, and attempt records belong in the evidence
archive. Large checkpoints stay outside Git. No model calls are needed for
this reference comparison.
