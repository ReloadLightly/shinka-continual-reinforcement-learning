# Adaptive mutation-width programs

This task contains the implemented program interface, fixed diagnostic rules,
and a Shinka-compatible selection evaluator. The
[adaptive search launcher](../../docs/adaptive-search.md) runs a separate native
archive with cumulative targets of 5, 13, and 25 slots.
See the [protocol](../../docs/adaptive-programs.md) for the full scientific contract.

```python
def update_sigma(sigma, stats, memory):
    return sigma, memory
```

The program runs after each evaluated GA generation. It receives a float32 width,
five current-training statistics, and four persistent memory values. It returns
the next width and memory. The harness checks finite outputs, clips width to
[0.001, 2.0], and preserves native initialization, population size, selection,
random-key progression, and environment budgets. Programs have no imports,
environment access, evaluation feedback, or task-boundary input.

| File | Purpose |
| :--- | :--- |
| `initial.py` | Identity: preserves the baseline width and memory |
| `halving.py` | Diagnostic: halves width and increments memory to test persistence |
| `arithmetic.py` | Fixed comparison: success-based multiplicative width adjustment |

These are fixed controls, not Shinka-discovered results. The bounded grammar
uses bare calls such as `exp(...)`; these names are supplied only by the validated
compiler. Do not import these programs as ordinary standalone Python modules.

Run the complete diagnostic suite with:

```bash
.venv/bin/python scripts/run_adaptive_gate.py \
  --results-dir results/adaptive-gate-new --execute
.venv/bin/python scripts/report_adaptive_gate.py \
  --run-dir results/adaptive-gate-new --report-dir reports/adaptive-gate-new
```

Omit `--execute` to inspect the frozen seven-trial plan. Outputs must be fresh;
failed attempts remain available with their logs, source hashes, and receipts.
The suite uses diagnostic seed 3001, two logical CPUs, and no model calls.

The [adaptive evaluation runbook](../../docs/adaptive-evaluation.md) defines the
new development partition, combined learning/previous-task score, five fixed
controls, and verified result cache. `evaluate.py` requires
`SHINKA_ADAPTIVE_STUDY` and writes Shinka's `correct.json` / `metrics.json`
contract. It accepts only the frozen development study; reserved validation
and final reporting trials are separate.

`shinka_evaluate.py` adapts the native scheduler's directory layout: job logs stay
outside the signed `results/evaluation` directory, and native result files mirror
the frozen evaluator's contract. `shinka-subscription.yaml` supplies the restricted
grammar and training statistics to the proposer while disabling auxiliary models.

The [local web UI](../../docs/shinka-webui.md) shows the adaptive and completed
static archives. Its generations count outer program slots, not inner GA steps.
