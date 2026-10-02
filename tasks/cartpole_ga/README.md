# Continual CartPole: GA configuration search

This is the first ShinkaEvolve extension: search over the upstream GA's static
`sigma` and `elite_ratio`. It does not yet evolve the learning algorithm, reproduce
the complete paper, or establish an improvement over its reported results.

`initial.py` contains a single literal-returning `get_ga_config()` function.
The evaluator parses its AST without importing or executing candidate code.
Only these two finite numeric values are accepted:

| Setting | Inclusive range | Initial value |
| --- | --- | --- |
| `sigma` | 0.001–2.0 | 0.5 |
| `elite_ratio` | 0.05–0.95 | 0.5 |

Imports, decorators, annotations, extra statements, duplicate keys, booleans,
calls, and other computed expressions are rejected. The experiment runner uses
the pinned upstream repository and fixed profiles from the installed project
package. Environment dimensions, training budget, seed list, scoring, and task
schedule cannot be changed through a candidate.

## Validate the candidate offline

From the project root, this needs only Python's standard library:

```bash
python3 tasks/cartpole_ga/evaluate.py \
  --program_path tasks/cartpole_ga/initial.py \
  --validate-only
```

It reports syntax validation on stdout and produces no experiment result files.
It does not train a policy or report a numerical performance score.

## Evaluate the initial configuration

Complete the project installation and pinned upstream setup in the root README,
then run:

```bash
.venv/bin/python tasks/cartpole_ga/evaluate.py \
  --program_path tasks/cartpole_ga/initial.py \
  --results_dir results/cartpole-ga-smoke
```

The default `smoke` profile runs the real upstream trainer with a small budget.
Its score verifies execution only and is not a scientific result. Set
`SHINKA_CRL_PROFILE=search` for the development search budget and fixed seeds
1001, 1002, and 1003. The `paper-cartpole` profile is deliberately rejected here:
reserve its final trials for reporting after selecting a configuration.

Development evaluations pass `trial=seed+1`, keeping the task offsets generated
by the upstream trainer separate from the final reporting trials 1–10. Each seed
gets its own output directory. `combined_score` is the mean normalized return
across development seeds; larger values are better. The normalization and
training protocol are defined by `shinka_crl.experiment`.

Supported environment overrides:

| Variable | Purpose |
| --- | --- |
| `SHINKA_CRL_PROFILE` | `smoke` (default) or `search` |
| `SHINKA_CRL_UPSTREAM` | Path to the pinned upstream checkout |
| `SHINKA_CRL_PYTHON` | Upstream environment's Python interpreter |
| `SHINKA_CRL_TIMEOUT` | Timeout in seconds per seed, default 1800 |

Successful and failed evaluations write the Shinka contract:
`metrics.json` with `combined_score`, `public`, and `private`; and `correct.json`
with `correct` and `error`. Invalid candidates and failed experiments receive
`correct: false` and `combined_score: 0.0`, and the evaluator exits nonzero.

## Evolve programs through local Codex

The planned route uses Shinka's native headless provider, which runs Codex CLI
under the local ChatGPT login. Subscription login and separately billed API-key
login are distinct [authentication modes](https://learn.chatgpt.com/docs/auth).
The [experimental roadmap](../../docs/experimental-roadmap.md) specifies the
matched pilot required before search and the random-search comparison.

The outer loop is:

1. Evaluate `initial.py` as generation zero and retain its source and score.
2. Select a parent program and archived examples using recorded scores.
3. Ask Codex for a diff or complete replacement of the permitted function.
4. Apply the proposed edit, validate it, and train fresh policy populations on
   the three fixed development seeds.
5. Store the candidate, ancestry, validity, metrics, feedback, and usage in the
   results directory and `programs.sqlite`; use them for subsequent proposals.

An outer program evaluation contains 240 inner GA generations. Shinka evolves
the source of the configuration program; GA evolves neural policy weights inside
each evaluation. No policy population carries over to the next outer program.

The pinned source's [headless provider](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/llm/providers/headless.py)
and [runner](https://github.com/SakanaAI/ShinkaEvolve/blob/9912af12d423504b8d580f4179fd15f5f88b8c50/shinka/core/async_runner.py)
establish these operational details:

- `num_generations` includes the initial program. Targets 5, 13, and 25 mean
  4, 12, and 24 cumulative proposal slots, respectively.
- Resume by increasing the target while keeping the same results directory,
  database configuration, task, and evaluator. A generation-zero-only run is
  not a reliable resume checkpoint at this revision; finish a mutation first.
- Failed proposals can leave attempt records without a scored program. Count
  attempted slots and successful evaluations separately.
- Plan breaks after completed blocks. The current evaluator cannot resume a
  partially trained candidate. Resuming the archive is not a guarantee of the
  identical stochastic trajectory an uninterrupted search would have taken.
- Native Shinka does not supervise subscription quota. Check usage after each
  block and stop on quota failures; do not switch authentication or purchase API
  usage to continue. The remaining account allowance is not visible to this repo.

The subscription configuration bounds mutation and transport retries, uses one
worker per stage, and disables embeddings, meta recommendations, novelty-model
calls, prompt evolution, and W&B. Ordinary `max_tokens` and temperature settings
do not bound the pinned headless route; model-call counts, timeout, and reasoning
effort are the applicable controls.

Headless's reported dollar cost is an API-price-equivalent estimate, not a
subscription charge or remaining-quota meter. Public package and pricing metadata
requests may still occur; they are not paid model inference. Native support and
authentication preflight are verified, but the first actual proposal and archive
resume remain experimental acceptance gates.

### Preflight and staged launch

Install the optional dependency with `uv sync --frozen --extra shinka --group dev`.
The adapter also requires Codex CLI with ChatGPT login and an installed
`@roberttlange/headless@0.6.1`. It finds that exact version in the local npm cache;
alternatively set `SHINKA_HEADLESS_CLI` to its `dist/cli.js`. If missing, install
the pinned package explicitly with `npm exec --yes --package=@roberttlange/headless@0.6.1 -- headless --help`.
That installs tooling without requesting a model completion. The guard itself
never downloads packages automatically.

The planned pilot explicitly selects GPT-6.1 Sol at medium reasoning effort.
Current [OpenAI plan documentation](https://learn.chatgpt.com/docs/pricing)
lists it for Pro. The no-inference check verifies tooling, model selection, and
login; it does not prove model entitlement or remaining quota. Without the
explicit model override below, the guard uses the local Codex model setting.
Record the resolved model and keep it fixed across all stages.

From the repository root:

```bash
export SHINKA_HEADLESS_COMMAND="$PWD/.venv/bin/python $PWD/scripts/subscription_headless.py"
export SHINKA_CODEX_MODEL=gpt-6.1-sol
export SHINKA_CRL_PROFILE=search
export SHINKA_CRL_TIMEOUT=600
export SHINKA_HEADLESS_TIMEOUT=600
export SHINKA_LLM_MAX_RETRIES=1
export SHINKA_PRICING_MODE=offline
export JAX_PLATFORMS=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# Tooling/authentication only: zero inference calls.
.venv/bin/python scripts/subscription_headless.py --check

# Launch only after the matched-pilot gates in the roadmap pass.
# CPUs 0,1 match this machine's timing calibration; adapt on other hosts.
taskset -c 0,1 .venv/bin/shinka_run \
  --task-dir tasks/cartpole_ga \
  --config-fname shinka-subscription.yaml \
  --results_dir results/subscription-pilot \
  --num_generations 5
```

After reviewing the completed first block and available quota, repeat the same
command with `--num_generations 13`, then `25`. These are cumulative targets.
Use a new results directory for a different model, evaluator, protocol, or
independent search. Keep the directory between stages of this run.

The guard removes API credentials from the subprocess environment, requires an
existing ChatGPT login, forces that login method and the built-in provider, and
preserves Codex's read-only sandbox. It does not alter stored credentials. It is
an authentication guard, not isolation from every inherited Codex integration.
It also cannot enforce account-side purchased-credit settings or inspect the
remaining included allowance. Check the usage dashboard before starting each
block, use Standard speed, and stop if the included allowance is exhausted.

The retry settings permit at most one mutation request per outer proposal slot;
Codex's own internal tool/service behavior is not a one-token-call guarantee.
Neither a subscription-backed proposal nor an archive resume has been launched
yet. The first four proposals are the explicit integration gate.

## Separate API example

The optional dependency pins ShinkaEvolve source commit
`9912af12d423504b8d580f4179fd15f5f88b8c50` (version 0.0.7). Its
[task CLI](https://sakanaai.github.io/ShinkaEvolve/cli_usage/) requires the
configuration filename explicitly:

```bash
uv sync --frozen --extra shinka --group dev
# Optional separately billed route; not the planned subscription experiment.
# Configure OPENAI_API_KEY through your environment only if choosing this route.
SHINKA_CRL_PROFILE=search uv run --frozen --extra shinka shinka_run \
  --task-dir tasks/cartpole_ga \
  --config-fname shinka.yaml \
  --results_dir results/cartpole-ga-search \
  --num_generations 10
```

`shinka.yaml` selects `gpt-4.1-mini`, disables embeddings and meta recommendations,
uses one evaluation/proposal worker, and sets a $1 API proposal budget. Model
calls already in flight may finish after that budget is reached. Change the
model and budget before a larger experiment. The project setup does not start an
LLM search automatically.

Shinka copies the evaluator into its result directory. Keep the project installed
in the interpreter that launches `shinka_run`; the evaluator imports the installed
package and does not infer repository paths from its copied file location.
