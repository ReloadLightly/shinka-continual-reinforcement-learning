# Reproduction plan

The [experimental roadmap](experimental-roadmap.md) defines the immediate matched pilot, subscription-backed Shinka stages, validation split, and adaptive-rule extension. This document retains the full reproduction specification; completing the reduced search is not completion of the paper reproduction.

This project starts a reproduction of [Continual Reinforcement Learning with Neuroevolution](https://arxiv.org/abs/2610.01583), by Eleni Nisioti, Andrea Cossu, Kathrin Korte, and Sebastian Risi (arXiv:2610.01583v1, October 1, 2026). The initial scope is a CartPole comparison with upstream GA, ES, and PPO, followed by a ShinkaEvolve extension. [CPU smoke validation](../reports/smoke-20261002/summary.json), a [paper-budget GA development trial](../reports/reference-timing-20261002/summary.json), a [paper-budget ES development trial](../reports/reference-development-es-20261003/summary.json), and the [resource-limited PPO development prefix](../reports/reference-development-ppo-prefix-20261003/summary.json) are complete. The [five full-budget GA/ES reporting pairs](../reports/reference-reporting-ga-es-pair05-20261004/summary.json) are published. The October 4 scope amendment ended baseline reporting after the fifth pair; subsequent work focuses on the Shinka extension. The original full three-method reproduction remains incomplete.

The reference implementation is pinned to [`821570eb6a22db0f7aa77111b2ea541fe8fa795b`](https://github.com/eleninisioti/continual_neuroevolution/tree/821570eb6a22db0f7aa77111b2ea541fe8fa795b). Keep it as an external checkout; no top-level license was present at that revision. This repository contains the integration and experiment specification, rather than a vendored copy of that implementation.

## First experiment

The [reference comparison](reference-comparison.md) originally specified GA,
ES, and PPO with ten reporting trials per method under the full protocol below.
GA and ES development measurements are complete, and each full twenty-phase
reporting arm has five published trials. Following the user's October 3, 2026
compute limit, PPO completed a resource-limited development prefix under the
[dated amendment](reference-comparison.md#ppo-resource-amendment--october-3-2026).
The ten full-budget PPO reporting trials are deferred. The completed
ShinkaEvolve searches and reserved validations remain separate studies,
including their negative results; their selection decisions remain closed.
No selected candidate changes this reference comparison.

Under the user's [October 4 scope amendment](reference-comparison.md#baseline-scope-amendment--october-4-2026),
the ES seed-46 trial and its evaluation completed the fifth matched pair,
then baseline reporting stopped. The final allocation is GA/ES at seeds 42–46 and
task trials 1–5, retaining twenty phases and the full per-trial budget. The
frozen suite plan remains twenty jobs, including its unexecuted seeds 47–51;
the original ten-pair sample target is uncompleted. No automatic continuation
or further baseline reporting follows ES seed 46. This resource and scope
decision uses the accumulated matched evidence without selecting trials by
their returns. A five-pair comparison remains valid under the common protocol;
the smaller sample limits precision and generalization rather than invalidating
comparability. Existing negative findings and all incurred costs remain in
the record. The [dated decision](../reports/reference-reporting-scope-amendment-20261004/decision.json)
records the scope change while the final ES trial was still active.

The five reporting pairs use seeds 42–46, task trials 1–5, and evaluation
seeds 900042–900046, with matched task vectors within each pair and 3.072 billion
nominal training steps per trial. Learning accuracy is 500 in all ten trials,
but retention varies sharply. GA forgetting is 489.76, 132.49, 20.96, 42.96,
and 467.45 at seeds 42–46; ES forgetting is 489.63, 5.07, 0.00, 0.00, and
488.21. Mean ± sample SD is 230.73 ± 230.24 for GA and 196.58 ± 266.87 for
ES. Mean LA − F is 269.27 ± 230.24 versus 303.42 ± 266.87; mean transfer
is 268.90 ± 233.62 versus 297.48 ± 261.62; normalized cumulative return
is 0.9642 ± 0.0422 versus 0.9882 ± 0.0156. All quantities except normalized
cumulative return use episode-return units. The
[complete individual and aggregate outcomes](../reports/reference-reporting-ga-es-pair05-20261004/summary.json)
and [exact frozen plan](../reports/reference-reporting-ga-es-pair05-20261004/raw/plan.json)
retain reporting identities and exclude development evidence. These five-trial
estimates describe the observed comparison and variation; they do not establish
a general method ranking or complete the original ten-pair target.

The trajectories distinguish terminal forgetting from losses followed by
recovery. GA seed 44 has transfer 500 but loses 350.1 return units on the
previous task at its final checkpoint; transfer excludes the final policy,
and previous-task and next-task probes use independent draws. GA seed 45
instead loses 288.8, 279.5, and 248.0 at phases 9, 11, and 13, then retains
both tasks at all final seven endpoints. ES seed 45 returns 500 in every
fresh centroid own-task, previous-task, and next-task episode, and its dense
curve retains both tasks throughout phases 2–20. These outcomes coexist with
large losses for both methods at seeds 42 and 46. The [learning curves](../reports/figures/reference-reporting-ga-es-pair05-20261004.svg)
and [full interpretation and costs](reference-comparison.md#bounded-full-budget-reporting-results)
distinguish dense observations from fresh checkpoint metrics. The seed-46 pair
is complete and baseline reporting has stopped. ES has higher cumulative
active-task return in all five observed pairs and lower mean forgetting, but
its forgetting is higher than GA's at seed 46. The cumulative suite execution
counter is 14,959.309100890998 s. Including the earlier allocation's
919.6690881920003 s gives 15,878.978189082998 s of recorded reporting durations
through five pairs, charged once; the 63.13385714699689 s allocation diagnostic
remains a separate category. These recorded durations are distinct from elapsed
calendar time between UTC timestamps. The [timestamp record](../reports/reference-reporting-scope-amendment-20261004/decision.json)
gives 35,558.2426 s (9.87729 h) from the first successful trainer start to the
final suite timestamp, including gaps between controllers and excluding the
earlier abandoned allocation, development, and diagnostics.

Curve comparisons require the [reference display qualifications](reference-comparison.md#reference-findings-to-assess).
The pinned plotting helpers apply a centered rolling median within each trial,
then a mean and bootstrap interval across trials; our figures retain dense
unsmoothed samples. Brief switch dips can therefore look different without
establishing a reproduction discrepancy. The exact assembly of the published
Figure 8 remains unresolved because its stated panel layout is not fully
accounted for by the pinned curve path and the authors' saved arrays are absent.
These limitations concern visual comparison and do not change the independently
verified fresh endpoint learning and forgetting results.

The completed reporting trials used two logical CPUs under the
[October 4 scheduling amendment](reference-comparison.md#reporting-cpu-allocation-amendment--october-4-2026).
The initial eight-CPU GA attempt completed no reporting trial; its
generation-600 checkpoint and all incurred work remain preserved. The new
suite keeps the same twenty jobs and scientific settings, with only CPU
affinity changed. Its existing resume contract requires the fresh suite,
so no completed trial is discarded and the checkpoint is not imported across
allocations. The decision is based on measured execution cost rather than
return-based selection. Exact costs and preservation records accompany the
[scheduling decision](../reports/reference-reporting-resource-amendment-20261004/decision.json),
[execution evidence](../reports/reference-reporting-resource-amendment-20261004/execution.json),
and [original attempt](../reports/reference-reporting-eight-cpu-attempt-20261004/summary.json).
The recorded [two-CPU launch and continuation commands](reference-comparison.md#reporting-execution)
retain execution provenance for `results/reference-reporting-ga-es-two-cpu-20261004`.
They are not instructions to launch later pairs. The ES seed-46 trial and
evaluation are complete; both reporting allocations remain closed.

Use the upstream gymnax cell `CartPole-v1_sigma0.5`, alternating between the original environment and a fixed observation offset. Preserve the upstream policy, environment dynamics, task construction, and centroid evaluation. The trainer receives no explicit task-switch signal. See the paper's [experimental setup and Appendix A](https://arxiv.org/html/2610.01583v1).

The pinned [gymnax configuration](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/source/configs/gymnax.yaml) and [configuration resolver](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/source/utils/config.py) specify the original full experiment, retained as the reproduction target:

| Setting | Full experiment |
| --- | --- |
| Distinct tasks / phases | 2 tasks across 20 phases |
| GA and ES training | 200 generations per phase; 4,000 total |
| Population / training episodes | 512 candidates; 3 episodes per candidate |
| Episode cap | 500 steps |
| PPO training | 1,500 updates per phase; 30,000 total |
| PPO rollout | 2,048 environments × 50 steps |
| Nominal training budget | 3,072,000,000 environment steps per trial |
| Evaluation | 10 episodes; separate evaluation keys |
| GA baseline | `sigma=0.5`, `elite_ratio=0.5` |
| ES baseline | `sigma=0.1`, learning rate `0.05`, SGD, z-score fitness |

The YAML default is **10 phases**, not the full 20-phase experiment. Nothing implicitly doubles it. The direct upstream CLI needs `--num_phases 20 --num_tasks 2`. In the [shell launcher](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/scripts/train/run.sh), `NUM_TASKS` controls phases and `TASK_PERIOD` controls distinct tasks, despite their names. From the prepared upstream checkout, the original three-method job preview is below; its PPO arm is deferred under the amendment:

```bash
ENVS=CartPole-v1 SIGMAS=0.5 NUM_TASKS=20 TASK_PERIOD=2 \
  bash scripts/train/run.sh --dry-run --trials 10 \
  --root runs/cartpole-20phase gymnax_continual:ga,es,ppo
```

For the separate ten-distinct-task family, the corresponding flags are `--num_phases 20 --num_tasks 10`; its paper observation-offset scale differs. Do not treat it as the same experiment.

### PPO allocation amendment — October 3, 2026

The user authorized reducing PPO to obtain results within approximately eight
hours. The active allocation treats this as eight hours total of measured
active elapsed PPO compute, including training already spent, native
finalization, fresh checkpoint evaluation, and any further PPO execution.
It is not an allocation for each of ten reporting trials.

The seed-1001 trajectory completed the target of four phases and 6,000 total
updates, retaining the original 1,500-update phase interval and every baseline
setting. This prefix contains 614,400,000 nominal training steps. The
[published evidence](../reports/reference-development-ppo-prefix-20261003/summary.json)
and [exact provenance](../reports/reference-development-ppo-prefix-20261003/raw/provenance.json)
preserve all original attempts, checkpoints, and incurred compute. The saved
phase policies were finalized without further training and evaluated in 100
fresh episodes. Total accounted PPO compute is 22,340.0237 s (6.205562 h),
within the eight-hour allocation.

The [matched GA/ES/PPO prefix comparison](reference-comparison.md#completed-four-phase-development-comparison)
uses existing GA/ES evidence under its [derivation protocol](../reports/reference-development-prefixes-20261003/protocol.json).
All three methods attain learning accuracy 500. Mean signed forgetting is
32.07 for GA, 0.00 for ES, and 159.67 for PPO; PPO's first-switch loss is 479
return units, followed by full endpoint retention at the two later switches.
PPO has the highest cumulative active-task performance in this seed. These
observations describe early acquisition and retention, while a shortened PPO
trajectory cannot supply a full twenty-phase method ranking. One development trial supplies no
between-trial variation. The full three-method reproduction, PPO's later-phase
behavior, and its reporting-trial distribution remain unresolved.

## ShinkaEvolve extension

[ShinkaEvolve](https://sakanaai.github.io/ShinkaEvolve/) mutates programs using LLM proposals and an evolutionary archive. The first candidate program supplies **static GA hyperparameters**: `sigma` and `elite_ratio`. The evaluator applies them to the pinned upstream GA. This tests automated hyperparameter search; it does not yet discover an adaptive update rule or replace weight-space neuroevolution with direct program-policy evolution.

Keep population size, episode cap, generations, task sequence, evaluation budget, and allowed parameter bounds fixed outside the candidate. Use the same development trials for every candidate. Freeze the winner before evaluating on separate reporting trials. Record candidate source, selected parameters, upstream revision, seeds, resolved configuration, training metrics, selection score, failures, wall time, and LLM usage. Charge repeated trials and program-search evaluations to the search budget separately from each learner's environment-step budget.

The subsequent [adaptive-program study](adaptive-repeated-search.md) is now
complete. Two paired outer-search repetitions compare evolutionary parent
selection and archive feedback with independent proposals from identity, using
four proposals per archive. Fifteen fixed-control development trials and 48
proposal-training trials precede the common finalist freeze; 35 fresh trials
evaluate all four winners and three unchanged learners on seeds 7001–7005.
The evolutionary-minus-independent combined-score difference is
−0.0821 ± 0.0235 across the two outer repetitions. Both repetition means are
negative; five shared evaluation seeds within each repetition are not ten
independent searches. This is evidence about the compact extension protocol,
separate from the original reference reproduction.
[Frozen selections](../reports/adaptive-repeated-finalists-20261004/plan.json) ·
[Complete evaluation](../reports/adaptive-repeated-validation-20261004/summary.json) ·
[Measured costs](../reports/adaptive-repeated-accounting-20261004/summary.json).

## Milestones and acceptance criteria

1. **Validate the scaffold — complete.** Command construction, parameter bounds, metric parsing, evaluator failure behavior, and real GA/ES/PPO training have been checked. The initial Shinka candidate reproduces the GA baseline's entire two-task centroid trace. These reduced-budget runs establish pipeline execution only.
2. **Validate a matched pilot — complete.** Eighteen trials cover GA, ES, and PPO × stationary/switching conditions × three development seeds, each with 7.68 million nominal training steps and a 500-step episode cap. Saved configurations, task schedules, checkpoint evaluations, source/runtime identity, and artifact hashes validate; all methods pass the predefined stationary-learning gate. The run resumed after six trials without retraining them. [Evidence](../reports/pilot-20261002/summary.json). Reporting trials 1–10 were reserved independently of this pilot.
3. **Complete the bounded CartPole reference comparison.** The original reproduction target remains ten trials per method under the 20-phase protocol with held-out evaluation episodes. The resource-limited PPO development prefix and matched GA/ES prefix analysis are complete. The October 4 scope amendment ended GA/ES reporting after five completed matched pairs, with no further baseline launches. Preserve the original twenty-job plan and identify its uncompleted sample target. Report learning accuracy, forgetting, their difference, cumulative return, and zero-shot transfer, with dispersion across completed reporting trials. For alternating tasks, forgetting must average loss at consecutive switches in both directions; a final-checkpoint-only formula is inappropriate. Follow the paper's Appendix A.3 for definitions and normalization. These observations support a bounded full-budget GA/ES comparison, while the original three-method reproduction remains incomplete.
4. **Complete the controlled Shinka extension — compact repeated study complete.** The two paired adaptive-program searches and all 35 fresh evaluation trials are published. Existing GA/ES evidence supplies reference context. The static, adaptive, and repeated-search selections remain closed, including their negative outcomes. Further extension studies require a distinct question, declared budget, suitable controls, and an untouched evaluation partition; they cannot feed these outcomes back into completed searches or treat unequal task draws and budgets as matched. Two short outer repetitions do not establish a broad search-method ranking.
5. **Retain broader reproduction as future scope.** Further baseline trials, task variations, environments, continual PPO variants, and neighborhood analyses are outside the present allocation. They are not prerequisites for interpreting the accumulated comparisons and must not launch automatically.

Keep smoke outputs, search-development scores, and final reporting results distinguishable in artifact metadata. Record deviations from the pinned protocol before running comparisons; do not infer successful reproduction from a single favorable curve.

## Validation record · October 2, 2026

The [published smoke evidence](../reports/smoke-20261002/summary.json) contains four real CPU runs: GA, ES, PPO, and the initial GA configuration passed through the Shinka evaluator. Each run uses seed 1001, task trial 1002, four recorded steps, and the task order `[0, 0, 1, 1]`. Training budgets are intentionally unmatched (1,024 nominal steps for GA/ES versus 512 for PPO), so the observed returns do not support a method ranking.

The CPU runtime contains 45 packages, all at versions present in the pinned upstream lock. It is a Linux x86_64 / CPython 3.11 subset for this benchmark, omitting CUDA and unrelated benchmark dependencies. The reference source is unchanged. Package versions, source hashes, raw metrics, configurations, logs, and artifact checksums are included in the report.

An initial run completed numerically but exposed a logging collision: both the harness and upstream `Tee` opened `train.log`. The harness now captures stdout/stderr in `process.log`, leaving `train.log` to upstream. The initial outputs remain in the local ignored directory `results/smoke-20261002`; the published evidence comes from a fresh rerun at `results/smoke-20261002-validated`. The rerun retained the same scores and has intact logs. No LLM proposals were requested.
