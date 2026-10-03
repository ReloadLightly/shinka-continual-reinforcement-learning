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
and original baseline settings; its continuation now follows the resource
amendment below. These measurements are development evidence and cannot replace
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
to new paths. Prefix evaluation and compact evidence publication are pending.
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
the proposed four-phase PPO prefix, compute the same definitions over its
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
At the four-phase target, this means 800 GA/ES generations and 6,000 PPO updates. Keep those
prefix comparisons separate from full twenty-phase method rankings.

Assess each relevant paper finding as reproduced, discrepant, or unresolved
within the observed scope. The PPO prefix can address initial acquisition and
early switches; sustained behavior across twenty phases and repeated PPO
trials remain unresolved. Do not infer reproduction of other environments or
methods from this comparison.

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

The scientific report summarizes experimental findings and material protocol
deviations. Detailed commands, source revisions, runtime identities, raw curves,
episode returns, checkpoint hashes, and attempt records belong in the evidence
archive. Large checkpoints stay outside Git. No model calls are needed for
this reference comparison.
