# Reproduction plan

The [experimental roadmap](experimental-roadmap.md) defines the immediate matched pilot, subscription-backed Shinka stages, validation split, and adaptive-rule extension. This document retains the full reproduction specification; completing the reduced search is not completion of the paper reproduction.

This project starts a reproduction of [Continual Reinforcement Learning with Neuroevolution](https://arxiv.org/abs/2610.01583), by Eleni Nisioti, Andrea Cossu, Kathrin Korte, and Sebastian Risi (arXiv:2610.01583v1, October 1, 2026). The initial scope is a CartPole comparison with upstream GA, ES, and PPO, followed by a ShinkaEvolve extension. [CPU smoke validation](../reports/smoke-20261002/summary.json) and one [paper-budget GA development trial](../reports/reference-timing-20261002/summary.json) are complete; full-budget comparative reproduction and final reporting remain pending.

The reference implementation is pinned to [`821570eb6a22db0f7aa77111b2ea541fe8fa795b`](https://github.com/eleninisioti/continual_neuroevolution/tree/821570eb6a22db0f7aa77111b2ea541fe8fa795b). Keep it as an external checkout; no top-level license was present at that revision. This repository contains the integration and experiment specification, rather than a vendored copy of that implementation.

## First experiment

The active [full-budget reference comparison](reference-comparison.md) includes
GA, ES, and PPO only: ten reporting trials per method under the protocol below.
It first reuses the verified GA development reference and measures missing ES/PPO
development costs. The completed ShinkaEvolve searches and reserved validations
remain separate studies; no selected candidate changes this reference comparison.

Use the upstream gymnax cell `CartPole-v1_sigma0.5`, alternating between the original environment and a fixed observation offset. Preserve the upstream policy, environment dynamics, task construction, and centroid evaluation. The trainer receives no explicit task-switch signal. See the paper's [experimental setup and Appendix A](https://arxiv.org/html/2610.01583v1).

The pinned [gymnax configuration](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/source/configs/gymnax.yaml) and [configuration resolver](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/source/utils/config.py) specify:

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

The YAML default is **10 phases**, not the full 20-phase experiment. Nothing implicitly doubles it. The direct upstream CLI needs `--num_phases 20 --num_tasks 2`. In the [shell launcher](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/scripts/train/run.sh), `NUM_TASKS` controls phases and `TASK_PERIOD` controls distinct tasks, despite their names. From the prepared upstream checkout, an equivalent job preview is:

```bash
ENVS=CartPole-v1 SIGMAS=0.5 NUM_TASKS=20 TASK_PERIOD=2 \
  bash scripts/train/run.sh --dry-run --trials 10 \
  --root runs/cartpole-20phase gymnax_continual:ga,es,ppo
```

For the separate ten-distinct-task family, the corresponding flags are `--num_phases 20 --num_tasks 10`; its paper observation-offset scale differs. Do not treat it as the same experiment.

## ShinkaEvolve extension

[ShinkaEvolve](https://sakanaai.github.io/ShinkaEvolve/) mutates programs using LLM proposals and an evolutionary archive. The first candidate program supplies **static GA hyperparameters**: `sigma` and `elite_ratio`. The evaluator applies them to the pinned upstream GA. This tests automated hyperparameter search; it does not yet discover an adaptive update rule or replace weight-space neuroevolution with direct program-policy evolution.

Keep population size, episode cap, generations, task sequence, evaluation budget, and allowed parameter bounds fixed outside the candidate. Use the same development trials for every candidate. Freeze the winner before evaluating on separate reporting trials. Record candidate source, selected parameters, upstream revision, seeds, resolved configuration, training metrics, selection score, failures, wall time, and LLM usage. Charge repeated trials and program-search evaluations to the search budget separately from each learner's environment-step budget.

## Milestones and acceptance criteria

1. **Validate the scaffold — complete.** Command construction, parameter bounds, metric parsing, evaluator failure behavior, and real GA/ES/PPO training have been checked. The initial Shinka candidate reproduces the GA baseline's entire two-task centroid trace. These reduced-budget runs establish pipeline execution only.
2. **Validate a matched pilot — complete.** Eighteen trials cover GA, ES, and PPO × stationary/switching conditions × three development seeds, each with 7.68 million nominal training steps and a 500-step episode cap. Saved configurations, task schedules, checkpoint evaluations, source/runtime identity, and artifact hashes validate; all methods pass the predefined stationary-learning gate. The run resumed after six trials without retraining them. [Evidence](../reports/pilot-20261002/summary.json). Reporting trials 1–10 remain untouched.
3. **Run the full CartPole slice.** Use ten trials, the 20-phase protocol, and held-out evaluation episodes. Report learning accuracy, forgetting, their difference, cumulative return, and zero-shot transfer with trial uncertainty. For alternating tasks, forgetting must average loss at consecutive switches in both directions; a final-checkpoint-only formula is inappropriate. Follow the paper's Appendix A.3 for definitions and normalization.
4. **Evaluate the Shinka extension.** Compare the frozen searched configuration against the original GA and a search-budget-matched random hyperparameter search. Preserve ES and PPO as reference learners. Report individual trial outcomes as well as aggregates, and account for search cost.
5. **Expand after baseline validation.** Add other task variations and environments, continual PPO variants, and neighborhood analysis. Adaptive mutation or selection rules are a later search space requiring their own interface and controlled comparisons.

Keep smoke outputs, search-development scores, and final reporting results distinguishable in artifact metadata. Record deviations from the pinned protocol before running comparisons; do not infer successful reproduction from a single favorable curve.

## Validation record · October 2, 2026

The [published smoke evidence](../reports/smoke-20261002/summary.json) contains four real CPU runs: GA, ES, PPO, and the initial GA configuration passed through the Shinka evaluator. Each run uses seed 1001, task trial 1002, four recorded steps, and the task order `[0, 0, 1, 1]`. Training budgets are intentionally unmatched (1,024 nominal steps for GA/ES versus 512 for PPO), so the observed returns do not support a method ranking.

The CPU runtime contains 45 packages, all at versions present in the pinned upstream lock. It is a Linux x86_64 / CPython 3.11 subset for this benchmark, omitting CUDA and unrelated benchmark dependencies. The reference source is unchanged. Package versions, source hashes, raw metrics, configurations, logs, and artifact checksums are included in the report.

An initial run completed numerically but exposed a logging collision: both the harness and upstream `Tee` opened `train.log`. The harness now captures stdout/stderr in `process.log`, leaving `train.log` to upstream. The initial outputs remain in the local ignored directory `results/smoke-20261002`; the published evidence comes from a fresh rerun at `results/smoke-20261002-validated`. The rerun retained the same scores and has intact logs. No LLM proposals were requested.
