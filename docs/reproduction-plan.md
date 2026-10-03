# Reproduction plan

The [experimental roadmap](experimental-roadmap.md) defines the immediate matched pilot, subscription-backed Shinka stages, validation split, and adaptive-rule extension. This document retains the full reproduction specification; completing the reduced search is not completion of the paper reproduction.

This project starts a reproduction of [Continual Reinforcement Learning with Neuroevolution](https://arxiv.org/abs/2610.01583), by Eleni Nisioti, Andrea Cossu, Kathrin Korte, and Sebastian Risi (arXiv:2610.01583v1, October 1, 2026). The initial scope is a CartPole comparison with upstream GA, ES, and PPO, followed by a ShinkaEvolve extension. [CPU smoke validation](../reports/smoke-20261002/summary.json), a [paper-budget GA development trial](../reports/reference-timing-20261002/summary.json), a [paper-budget ES development trial](../reports/reference-development-es-20261003/summary.json), and the [resource-limited PPO development prefix](../reports/reference-development-ppo-prefix-20261003/summary.json) are complete. Full-budget comparative reproduction and final reporting remain pending.

The reference implementation is pinned to [`821570eb6a22db0f7aa77111b2ea541fe8fa795b`](https://github.com/eleninisioti/continual_neuroevolution/tree/821570eb6a22db0f7aa77111b2ea541fe8fa795b). Keep it as an external checkout; no top-level license was present at that revision. This repository contains the integration and experiment specification, rather than a vendored copy of that implementation.

## First experiment

The [reference comparison](reference-comparison.md) originally specified GA,
ES, and PPO with ten reporting trials per method under the full protocol below.
GA and ES development measurements are complete, and their full twenty-phase,
ten-trial reporting arms remain planned. Following the user's October 3, 2026
compute limit, PPO completed a resource-limited development prefix under the
[dated amendment](reference-comparison.md#ppo-resource-amendment--october-3-2026).
The ten full-budget PPO reporting trials are deferred. The completed
ShinkaEvolve searches and reserved validations remain separate studies,
including their negative results; their selection decisions remain closed.
No selected candidate changes this reference comparison.

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

## Milestones and acceptance criteria

1. **Validate the scaffold — complete.** Command construction, parameter bounds, metric parsing, evaluator failure behavior, and real GA/ES/PPO training have been checked. The initial Shinka candidate reproduces the GA baseline's entire two-task centroid trace. These reduced-budget runs establish pipeline execution only.
2. **Validate a matched pilot — complete.** Eighteen trials cover GA, ES, and PPO × stationary/switching conditions × three development seeds, each with 7.68 million nominal training steps and a 500-step episode cap. Saved configurations, task schedules, checkpoint evaluations, source/runtime identity, and artifact hashes validate; all methods pass the predefined stationary-learning gate. The run resumed after six trials without retraining them. [Evidence](../reports/pilot-20261002/summary.json). Reporting trials 1–10 remain untouched.
3. **Run the CartPole reference comparison.** The full reproduction target remains ten trials per method under the 20-phase protocol with held-out evaluation episodes. The resource-limited PPO development prefix and matched GA/ES prefix analysis are complete. Under the October 3 amendment, the next stage is the full GA and ES reporting comparison; the full-budget PPO reporting arm is deferred. Report learning accuracy, forgetting, their difference, cumulative return, and zero-shot transfer, with trial uncertainty where repeated reporting trials exist. For alternating tasks, forgetting must average loss at consecutive switches in both directions; a final-checkpoint-only formula is inappropriate. Follow the paper's Appendix A.3 for definitions and normalization. Completion of the amended work does not establish completion of the original three-method reproduction.
4. **Evaluate the Shinka extension.** Compare the frozen searched configuration against the original GA and a search-budget-matched random hyperparameter search. Preserve ES and PPO as reference learners. Report individual trial outcomes as well as aggregates, and account for search cost.
5. **Expand after baseline validation.** Add other task variations and environments, continual PPO variants, and neighborhood analysis. Adaptive mutation or selection rules are a later search space requiring their own interface and controlled comparisons.

Keep smoke outputs, search-development scores, and final reporting results distinguishable in artifact metadata. Record deviations from the pinned protocol before running comparisons; do not infer successful reproduction from a single favorable curve.

## Validation record · October 2, 2026

The [published smoke evidence](../reports/smoke-20261002/summary.json) contains four real CPU runs: GA, ES, PPO, and the initial GA configuration passed through the Shinka evaluator. Each run uses seed 1001, task trial 1002, four recorded steps, and the task order `[0, 0, 1, 1]`. Training budgets are intentionally unmatched (1,024 nominal steps for GA/ES versus 512 for PPO), so the observed returns do not support a method ranking.

The CPU runtime contains 45 packages, all at versions present in the pinned upstream lock. It is a Linux x86_64 / CPython 3.11 subset for this benchmark, omitting CUDA and unrelated benchmark dependencies. The reference source is unchanged. Package versions, source hashes, raw metrics, configurations, logs, and artifact checksums are included in the report.

An initial run completed numerically but exposed a logging collision: both the harness and upstream `Tee` opened `train.log`. The harness now captures stdout/stderr in `process.log`, leaving `train.log` to upstream. The initial outputs remain in the local ignored directory `results/smoke-20261002`; the published evidence comes from a fresh rerun at `results/smoke-20261002-validated`. The rerun retained the same scores and has intact logs. No LLM proposals were requested.
