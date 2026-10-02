# Shinka Continual Reinforcement Learning

Reproducing **[Continual Reinforcement Learning with Neuroevolution](https://arxiv.org/abs/2610.01583v1)** (Nisioti, Cossu, Korte & Risi, 2026), then using **[ShinkaEvolve](https://sakanaai.github.io/ShinkaEvolve/)** to search for improvements.

**Status: initial research scaffold. No training results or reproduced findings yet.** The first target is CartPole with alternating observation offsets. The repository provides pinned upstream GA, ES, and PPO launchers, budget profiles, artifact validation, and a Shinka task for searching two static GA settings. Discovering adaptive learning rules and reproducing the complete paper remain later milestones.

## Setup

Requires Git, [uv](https://docs.astral.sh/uv/), and Python 3.11 (uv can provision it).

```bash
git clone https://github.com/ReloadLightly/shinka-continual-reinforcement-learning.git
cd shinka-continual-reinforcement-learning
uv sync --frozen --group dev
uv run --frozen pytest
uv run --frozen ruff check .
```

The [GitHub Actions template](ci/github-actions.yml) runs the same checks. To enable
it, add it as `.github/workflows/checks.yml` using a GitHub session with workflow
write permission. Automated CI is not active in this initial publication.

The default environment contains the harness and development tools. The reference trainer and Shinka dependencies use separate installation steps.

```bash
# Download the exact reference revision into ignored .upstream/.
uv run --frozen python scripts/bootstrap_upstream.py

# Install its original locked dependencies in its own .venv.
# This is a large environment: upstream includes JAX/CUDA, Torch, and all benchmarks.
uv run --frozen python scripts/bootstrap_upstream.py --install
```

The upstream project uses JAX 0.5.3, while Shinka has its own dependency set. Keeping separate environments preserves the reference lock. CPU execution is supported by the upstream trainer; set `JAX_PLATFORMS=cpu` for a CPU check. Its Linux lock still installs CUDA dependencies. `SHINKA_CRL_UPSTREAM` and `SHINKA_CRL_PYTHON` can point the launchers and evaluator to an already prepared reference checkout and interpreter.

## Baselines

Commands print a run plan by default. Add `--execute` to start training. Output directories must be new; completed or failed runs are preserved.

```bash
# No training, dependencies, GPU, or API key needed for this plan.
uv run --frozen python scripts/run_baseline.py --profile smoke --method ga

# Real reduced-budget upstream check, after the upstream environment is installed.
JAX_PLATFORMS=cpu uv run --frozen python scripts/run_baseline.py \
  --profile smoke --method ga --execute

# Preview the full CartPole GA protocol and all ten reporting trials.
uv run --frozen python scripts/run_baseline.py --profile paper-cartpole --method ga
# Repeat with --method es or --method ppo.
```

| Profile | Purpose | Training seeds | Budget |
| --- | --- | --- | --- |
| `smoke` | Check that training and scoring work | 1001 | 4 generations/updates; shortened episodes; methods are not compute-matched |
| `search` | Develop GA candidates | 1001–1003 | 80 generations, 64 candidates, 4 phases |
| `paper-cartpole` | Final 20-phase CartPole slice | 42–51 | 4,000 NE generations or 30,000 PPO updates; 10 trials |

The full profile sets **20 phases and 2 distinct tasks** explicitly. Upstream's YAML default uses 10 phases. Each full trial has a nominal budget of **3,072,000,000 training environment steps**, excluding evaluation/diagnostic overhead. Choose an appropriate `--timeout` when executing it; the launcher default is 1,800 seconds per trial. The trainer gets no task-boundary signal.

Each executed trial saves `manifest.json`, `train.log`, the upstream artifacts, and `summary.json`. The manifest records the exact command, profile, seed, task trial, reference revision, interpreter version, device selection, duration, and metric-file hash. A checkout with tracked modifications is rejected. Development task trials are also disjoint from the reporting trials, because upstream draws observation offsets from the trial index.

## ShinkaEvolve task

The first search space is deliberately small: `get_ga_config()` returns a literal dictionary containing mutation `sigma` and archive `elite_ratio`. These settings feed the unchanged upstream GA. This is a hyperparameter-search extension; the candidate cannot change environment budgets, seeds, or scoring.

```bash
# Validate the candidate without training or API access.
uv run --frozen python tasks/cartpole_ga/evaluate.py \
  --program_path tasks/cartpole_ga/initial.py --validate-only

# Install the pinned Shinka source revision when ready for program search.
uv sync --frozen --extra shinka --group dev
```

See [the task instructions](tasks/cartpole_ga/README.md) for direct evaluation and the Shinka launch command. A live Shinka search requires a configured model provider and incurs model and training costs. No LLM search is started by setup or CI.

The development objective is mean active-task **centroid evaluation return**, divided by the episode cap, averaged over development seeds. This is a selection score, not the paper's complete continual-learning analysis. Reporting still needs learning accuracy, forgetting, cumulative return, zero-shot transfer, and uncertainty across trials. A selected candidate must be frozen before reporting trials; a search-budget-matched random-search control is planned.

## Sources and next steps

- [Reproduction plan](docs/reproduction-plan.md): protocol, metrics, milestones, and acceptance criteria.
- [Source pins](upstream.lock.json): paper version and exact upstream/Shinka commits.
- [Profile definitions](src/shinka_crl/profiles): fixed budgets and disjoint trial groups.
- [Reference implementation](https://github.com/eleninisioti/continual_neuroevolution): preserved as an external checkout. No top-level license was present at the pinned revision; its source is not vendored here.

Next: run the real CPU smoke checks, validate GA/ES/PPO artifacts, then reproduce the first CartPole comparison before expanding the Shinka search space.
