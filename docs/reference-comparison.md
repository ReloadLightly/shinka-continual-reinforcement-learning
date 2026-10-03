# Full-budget CartPole reference comparison

This experiment addresses the reproduction of the two-task CartPole cell in
*Continual Reinforcement Learning with Neuroevolution*. It includes only the
pinned reference GA, ES, and PPO. The completed static and adaptive ShinkaEvolve
studies remain separate; none of their candidates or validation outcomes changes
this comparison's settings.

## Protocol and partitions

The authoritative budget is the existing
[`paper-cartpole` profile](../src/shinka_crl/profiles/paper-cartpole.json),
interpreted by the [reproduction specification](reproduction-plan.md).
There are two tasks across 20 alternating phases, a 500-step episode cap,
and ten reporting trials per method. GA and ES use 4,000 generations,
population 512, and three training episodes per member. PPO uses 30,000
updates, 2,048 environments, and 50 rollout steps per update. Each trial
therefore receives 3,072,000,000 nominal training steps. The complete
30-trial comparison allocates 92,160,000,000 nominal training steps, excluding
evaluation. Baseline optimizers and architectures remain unchanged.

Reporting seeds 42–51 map in order to task trials 1–10. The execution order is
seed-major, with GA, ES, and PPO at each seed. Methods share task draws and
nominal budgets; their internal random draws remain method-specific. No
candidate program participates in baseline execution or evaluation.

Before reporting, full-budget development measurements use seed 1001, task
trial 1002, and evaluation seed 901001. The completed
[GA development reference](../reports/reference-timing-20261002/summary.json)
is reused after verification of its original and published evidence. Missing
ES and PPO measurements use the same full training budgets with their original
baseline settings. The [ES development measurement](../reports/reference-development-es-20261003/summary.json)
is complete; PPO development measurement remains in progress. These measurements
are development evidence and cannot replace reporting trials.
The [separate reduced diagnostic](../reports/reference-comparison-diagnostic-20261003/summary.json)
checks native GA/ES/PPO execution and analysis before full runs; it is not
comparative learning evidence.

## Evaluation and interpretation

The existing checkpoint analysis evaluates each phase agent with ten fresh
episodes per native evaluation target, using evaluation seed
`900000 + training_seed`. GA and ES use saved centroids; PPO uses its saved
policy. Report every trial and its learning curve, learning accuracy (LA),
signed forgetting (F), LA−F, zero-shot transfer, and cumulative return.
Forgetting averages all 19 consecutive switches, including both directions;
negative differences are retained. Transfer on recurring tasks is interpreted
in the context of prior task exposure.

Cumulative return follows the pinned reference's completed-update clock and
unit NE-generation integration grid. Raw reward-unit metrics and the fixed
division by 500 remain distinct from the paper's reference-based rescaling.
Without matching reference normalization values, comparisons with the paper
are restricted to the supported behavior and method ordering rather than
claims of exact normalized numerical replication.

For each method, report the ten individual values, mean, and sample standard
deviation. Compare acquisition, loss on previously encountered tasks, and
performance after recurrent switches. Assess each relevant paper finding as
reproduced, discrepant, or unresolved within this CartPole cell. Do not infer
reproduction of other environments or methods from this comparison.

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
Use measured ES/PPO costs to schedule the reporting comparison. A resource
limit may pause the experiment but cannot silently reduce its scientific budget.

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
