# CartPole reference comparison

This experiment addresses the reproduction of the two-task CartPole cell in
*Continual Reinforcement Learning with Neuroevolution*. It includes only the
pinned reference GA, ES, and PPO. The completed static and adaptive ShinkaEvolve
studies and reserved validations remain separate, including their negative
results; none of their candidates or validation outcomes changes this
comparison's settings or reopens completed selection. The original full-budget
protocol is retained below. The October 3, 2026 resource amendment limits the
active PPO experiment to a development prefix. The October 4, 2026 scope
amendment ended GA/ES reporting after the fifth matched pair then in progress.
All five pairs are complete and published below. Baseline reporting has
stopped; no further trial is scheduled.

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
the amendments, GA and ES retain the first five reporting identities and full
per-trial budgets; the ten full-budget PPO reporting trials are deferred.
The original ten-pair GA/ES target remains uncompleted. No candidate program
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
ES retain twenty phases per trial; the later scope amendment below limits
their reporting comparison to five trials per method. The
completed ShinkaEvolve studies, reserved outcomes, and their selection
decisions are unchanged.

## Baseline scope amendment — October 4, 2026

The user directed the project to use the accumulated GA/ES evidence and focus
subsequent work on ShinkaEvolve, rather than continue a multi-hour expansion of
the baseline campaign. The [dated decision](../reports/reference-reporting-scope-amendment-20261004/decision.json)
records the instruction while ES seed 46 was active. That trial and its fresh
evaluation completed the fifth matched pair; baseline reporting then stopped.
The resulting comparison comprises GA and ES at seeds **42–46**,
task trials **1–5**, with **20 phases and 3,072,000,000 nominal training steps
per trial**. Baseline settings, task construction, evaluation, source hashes,
and all completed trajectories remain unchanged.

This is a resource and project-scope decision, not selection of trials using
their returns. Preserve every completed pair and all incurred compute,
including negative findings and unsuccessful attempts. The original frozen
plan remains a twenty-job, ten-pair plan; do not rewrite it as though five
pairs had been the original sample target. Its unexecuted seeds 47–51 remain
unexecuted, and its export remains partial against that original plan.
No further GA, ES, or PPO reporting trial is authorized by this amendment;
there is no automatic continuation after ES seed 46.

The accumulated trials already form a valid matched comparison under the full
per-trial protocol. The smaller sample changes the precision and scope of
inference, not comparability within each pair. Report five individual outcomes
per method and their dispersion; do not
claim that the original ten-pair target or three-method reproduction is
complete. Development seed 1001 and PPO's four-phase development prefix remain
separate from these reporting aggregates. Display-smoothing and reference-data
limitations remain as stated below.

The completed static and adaptive Shinka searches and reserved comparisons
remain closed. A renewed focus on Shinka does not reopen their candidate
selection or permit feedback from reserved outcomes into the completed search.
Any subsequent extension needs its own scientific question, declared budget,
controls, and appropriate untouched evaluation data; it does not require
automatically filling the unused baseline jobs.

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

For GA and ES reporting, report every completed individual value, mean, and sample
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

The twenty-phase GA/ES reporting comparison ended after five trials per method
under the October 4 scope amendment. All five pairs are complete. The
full-budget PPO arm remains deferred; the completed development prefix does
not consume any reporting trial.

## Bounded full-budget reporting results

The [cumulative reporting evidence](../reports/reference-reporting-ga-es-pair05-20261004/summary.json)
contains GA and ES at seeds 42–46, task trials 1–5, and evaluation seeds
900042–900046. Methods share task vectors within each pair, twenty alternating
phases, 4,000 generations, and 3,072,000,000 nominal training steps per trial
under the [exact frozen plan](../reports/reference-reporting-ga-es-pair05-20261004/raw/plan.json).
Ten of twenty originally planned trials are complete, five per method. This
completes the bounded comparison under the scope amendment; no later pair
will start automatically. The export remains partial against the frozen plan;
full-budget PPO reporting is deferred and the original three-method target
is incomplete. The [first-pair](../reports/reference-reporting-ga-es-pair01-20261004/summary.json),
[two-pair](../reports/reference-reporting-ga-es-pair02-20261004/summary.json),
and [four-pair](../reports/reference-reporting-ga-es-pair04-20261004/summary.json)
archives remain unchanged; their verified trials are reused in this cumulative
export.

| Method | Seed / statistic | LA (return) | Signed F (return) | LA − F (return) | ZT (return) | Cum. / (steps × 500) ↑ |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| GA | 42 | 500.00 | 489.76 | 10.24 | 10.30 | 0.8912 |
| GA | 43 | 500.00 | 132.49 | 367.51 | 354.72 | 0.9878 |
| GA | 44 | 500.00 | 20.96 | 479.04 | 500.00 | 0.9897 |
| GA | 45 | 500.00 | 42.96 | 457.04 | 450.66 | 0.9886 |
| GA | 46 | 500.00 | 467.45 | 32.55 | 28.82 | 0.9639 |
| GA | Mean ± SD | 500.00 ± 0.00 | 230.73 ± 230.24 | 269.27 ± 230.24 | 268.90 ± 233.62 | 0.9642 ± 0.0422 |
| ES | 42 | 500.00 | 489.63 | 10.37 | 10.39 | 0.9605 |
| ES | 43 | 500.00 | 5.07 | 494.93 | 477.06 | 0.9953 |
| ES | 44 | 500.00 | 0.00 | 500.00 | 488.28 | 0.9937 |
| ES | 45 | 500.00 | 0.00 | 500.00 | 500.00 | 0.9981 |
| ES | 46 | 500.00 | 488.21 | 11.79 | 11.65 | 0.9933 |
| ES | Mean ± SD | 500.00 ± 0.00 | 196.58 ± 266.87 | 303.42 ± 266.87 | 297.48 ± 261.62 | 0.9882 ± 0.0156 |

Aggregate rows show mean ± sample SD across five reporting trials, not
confidence intervals. Every fresh centroid own-task episode returns 500.
Retention spans large losses for both methods at seeds 42 and 46 and little or no
endpoint forgetting for ES at seeds 43–45. GA shows partial retention, losses
followed by recovery, and a substantial terminal loss across these task draws.
Signed forgetting and transfer average all nineteen switches per trial, with
both directions included.

GA seed 44 has perfect measured transfer but loses previous-task return at
its final checkpoint: mean 149.9, or a 350.1-unit loss. The only other
nonzero endpoint loss is 48.2 at phase 7
([raw seed-44 episodes](../reports/reference-reporting-ga-es-pair05-20261004/raw/trials/ga/seed_44/analysis/attempt_001/evaluation.json)).
Transfer uses next-task probes from phases 1–19 and omits the final policy;
previous-task and next-task probes use independent random keys even when they
refer to the same recurring task. The [pinned evaluator](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/scripts/analysis/evaluate_continual.py#L277-L296)
and [metric definitions](../src/shinka_crl/analysis.py) preserve this distinction.
GA seed 45 instead loses 288.8, 279.5, and 248.0 at phases 9, 11, and 13,
respectively, after training the clean task; the other sixteen switches have
zero measured loss. Both tasks score 500 at all final seven endpoints
([raw seed-45 episodes](../reports/reference-reporting-ga-es-pair05-20261004/raw/trials/ga/seed_45/analysis/attempt_001/evaluation.json)).
Its previous-task and next-task probe distributions can differ markedly under
independent draws: at phase 9 their means are 211.2 and 67.0.

For ES seed 45, every fresh centroid episode returns 500: 200 on the own task,
190 on the previous task, and 190 on the next task
([raw ES episodes](../reports/reference-reporting-ga-es-pair05-20261004/raw/trials/es/seed_45/analysis/attempt_001/evaluation.json)).
Its [dense record](../reports/reference-reporting-ga-es-pair05-20261004/raw/trials/es/seed_45/training/attempt_001/training_metrics.json)
contains 3,803 generations with both task means equal to 500, including every
sample from phase 2 onward. GA seed 45 has 3,623 such generations, yet loses
previous-task return within intermediate phases while current-task return
remains maximal after phase 6
([dense GA record](../reports/reference-reporting-ga-es-pair05-20261004/raw/trials/ga/seed_45/training/attempt_001/training_metrics.json)).
In-training observations use different evaluation draws and do not replace
fresh checkpoint measurements.

The final seed-46 pair again shows high active-task performance with poor
retention. GA's mean previous-task endpoint return is 32.55, versus 11.79 for
ES, while every own-task episode returns 500. Its forgetting is therefore
467.45 for GA and 488.21 for ES. This pair reverses the forgetting ordering
observed in the preceding pairs while retaining ES's higher cumulative
active-task score. The [GA](../reports/reference-reporting-ga-es-pair05-20261004/raw/trials/ga/seed_46/analysis/attempt_001/evaluation.json)
and [ES](../reports/reference-reporting-ga-es-pair05-20261004/raw/trials/es/seed_46/analysis/attempt_001/evaluation.json)
episode records retain these losses alongside their learning results.

The [curves](../reports/figures/reference-reporting-ga-es-pair05-20261004.svg),
[metric figure](../reports/figures/reference-reporting-ga-es-pair05-20261004-metrics.svg),
and [numerical provenance](../reports/figures/reference-reporting-ga-es-pair05-20261004.json)
show individual trials and arithmetic means. ES has higher cumulative
active-task performance in all five observed pairs and lower mean forgetting.
ES has lower forgetting in four pairs, but higher forgetting at seed 46.
The large dispersion and five-trial sample do not establish a general method
ranking. Task draws and training seeds both change between pairs, so their
contributions to variation are not separated. Development seed 1001 remains
excluded from reporting aggregates.

| Method | Seed | Recorded training (s) | Recorded fresh analysis (s) | Peak trainer RSS (KiB) | Fresh episodes, all sources |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GA | 42 | 1,471.850347 | 9.603423 | 866,252 | 1,740 |
| ES | 42 | 1,547.458632 | 15.842623 | 830,228 | 1,740 |
| GA | 43 | 1,718.381387 | 12.647526 | 856,220 | 1,740 |
| ES | 43 | 1,469.747567 | 11.495563 | 831,948 | 1,740 |
| GA | 44 | 1,347.026557 | 13.913382 | 857,164 | 1,740 |
| ES | 44 | 1,215.605075 | 10.742931 | 834,056 | 1,740 |
| GA | 45 | 1,456.594002 | 17.239226 | 863,268 | 1,740 |
| ES | 45 | 1,659.131222 | 19.174077 | 826,080 | 1,740 |
| GA | 46 | 1,352.595892 | 11.423535 | 856,496 | 1,740 |
| ES | 46 | 1,550.186416 | 14.292836 | 783,228 | 1,740 |

The [exact cumulative compute record](../reports/reference-reporting-ga-es-pair05-20261004/summary.json)
reports accumulated execution-duration counters of 14,959.309100890998 s for
the five pairs' suite invocations, including 14,788.577096435998 s of training
and 136.37512128600792 s of fresh analysis. These recorded counters include
all earlier invocations; they are distinct from elapsed calendar time between
UTC timestamps. The 17,400 fresh episodes
cover centroid, final-generation best member, and incumbent; primary centroid
metrics use 580 episodes per trial, or 5,800 across these ten trials. Each
source in each trial has 200 own-task, 190 previous-task, and 190 next-task
episodes. Completed nominal training totals 30,720,000,000 steps.

The preserved [earlier reporting allocation](../reports/reference-reporting-eight-cpu-attempt-20261004/summary.json)
adds 919.6690881920003 s to the recorded suite-duration accounting, bringing
its reporting total through these five pairs to 15,878.978189082998 s. This
is not a measurement of total elapsed calendar time. Component training and
analysis costs, and earlier pairs, are already included and are not added again.
The [allocation diagnostic](../reports/reference-ga-es-allocation-diagnostic-20261004/summary.json)
adds 63.13385714699689 s in a separate diagnostic category. No reporting trial
remains active. All unsuccessful-attempt compute remains
charged, without changing any trial's scientific settings.

The [timestamp accounting](../reports/reference-reporting-scope-amendment-20261004/decision.json)
records 35,558.2426 s (9.87729 h) from the first successful trainer start at
2026-10-03 23:11:52.766115 UTC to the final suite timestamp at
2026-10-04 09:04:31.008715 UTC. This calendar span includes gaps between
controllers and excludes the earlier abandoned allocation, development, and
diagnostics. It is distinct from the recorded duration counters above; the
cause of their difference is not established.

## Reference findings to assess

[Appendix C.1, Figure 8](https://arxiv.org/html/2610.01583v1#A3.F8)
provides the reference training curves; its caption identifies part (b) as
two alternating tasks. Assess initial acquisition, recovery after switches,
and sustained active-task performance in the CartPole observation-offset
setting, subject to the display and provenance qualifications below.
No precise reference values are inferred from the plotted curves.

The pinned [curve preparation](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/scripts/analysis/plot_continual_lineplots.py#L210-L216)
applies a [centered rolling median](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/scripts/plotting/make_lineplot.py#L351-L365)
to each trial, with an odd window of approximately one percent of its total
record count. It then draws the [mean across trials](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/scripts/analysis/plot_noncontinual_solve.py#L245-L259)
with a [pointwise 95% bootstrap interval](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/source/metrics/continual_metrics.py#L61-L96).
Our figures retain dense unsmoothed samples and individual trials. Median
smoothing can suppress brief return drops, and averaging changes their visual
prominence. Differences in visible switch dips therefore do not alone establish
a reproduction discrepancy. Fresh checkpoint learning and forgetting metrics
are independent of these display transformations.

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

The pinned [Figure 2 trade-off mapping](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/scripts/analysis/plot_continual_combined.py#L69-L74)
identifies our source cell as `paper/gymnax/data/noise_2task/CartPole_v1_sigma0.5`.
The exact Figure 8 curve assembly is less certain: the same script's
[`--part curves` path](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/scripts/analysis/plot_continual_combined.py#L192-L218)
loads only the [main curve panels](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/scripts/analysis/plot_continual_lineplots.py#L282-L300),
whose CartPole noise entry uses ten tasks and observation-offset scale 1.0;
the [two-task entry](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/scripts/analysis/plot_continual_lineplots.py#L76-L115)
is listed separately. That pinned path does not establish the full layout
described by the published Figure 8 caption. The authors' raw trial tree and
saved curve arrays are absent from this checkout, leaving the exact published
assembly and numerical agreement with their trial distribution unresolved.
Neighborhood mechanisms, other environments, and continual PPO variants are
outside this comparison.

## Compute and preservation

Development measurements run sequentially with the existing two-CPU allocation
and numerical thread settings. Record training and analysis durations separately,
trainer peak resident memory, completed updates, and nominal training steps.
Use measured costs to account for the bounded GA/ES reporting comparison. The
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

The two-CPU suite retains its frozen twenty-job plan, twenty phases, and the
full training budget per trial. All five allocated pairs are complete and
published above. The fifth pair at seed 46 is the final allocation under the
scope amendment. The following command is historical provenance: it launched
seed 42 within the original twenty-trial plan, and is not a directive to
start another run.

```bash
.venv/bin/python scripts/run_reference_comparison.py \
  --mode reporting --methods ga es \
  --results-dir results/reference-reporting-ga-es-two-cpu-20261004 \
  --cpus 2 --timeout 21600 --analysis-timeout 1800 \
  --max-trials 2 --execute
```

The earlier continuations used the same arguments and `--resume`. The harness
verified completed trials before reuse; `--max-trials 2` bounded each invocation,
not the declared number of reporting trials. The seed-46 continuation used
`--resume` and ended after ES completed. Do not start another controller or
resume seed 47. The preserved eight-CPU suite also remains closed.

This reporting allocation exposes two logical CPUs, matching the development
allocation. Record actual affinity, elapsed training and evaluation costs,
and peak trainer memory for every attempt, including the earlier eight-CPU
work. Scientific settings, task draws, and nominal training budgets remain
fixed. Five completed pairs supply the bounded full-budget GA/ES comparison;
the original ten-pair target and three-method reproduction remain incomplete.

The scientific report summarizes experimental findings and material protocol
deviations. Detailed commands, source revisions, runtime identities, raw curves,
episode returns, checkpoint hashes, and attempt records belong in the evidence
archive. Large checkpoints stay outside Git. No model calls are needed for
this reference comparison.
