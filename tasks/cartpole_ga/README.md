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

## Start ShinkaEvolve explicitly

The optional dependency pins ShinkaEvolve source commit
`9912af12d423504b8d580f4179fd15f5f88b8c50` (version 0.0.7). Its
[task CLI](https://sakanaai.github.io/ShinkaEvolve/cli_usage/) requires the
configuration filename explicitly:

```bash
uv sync --frozen --extra shinka --group dev
# Configure OPENAI_API_KEY through your environment before launching.
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
