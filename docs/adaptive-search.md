# Adaptive Shinka proposal experiment

This archive searches executable `update_sigma(sigma, stats, memory)` rules under
the [frozen adaptive evaluator](adaptive-evaluation.md). It preserves the five
fixed controls and their development partition. It does not use reserved adaptive
validation or final reporting trials.

The first block contains five slots: cached identity at generation zero and four
model proposals. Later targets of 13 and 25 are cumulative. Shinka selects parents
and archive examples, constructs prompts, applies edits, and stores ancestry in
its native SQLite database. The repository checks source, runtime, evaluation,
cost, and resume receipts around that native loop. Each valid new rule trains
three independent GA populations from initialization; weights do not carry from
one candidate to another.

The existing subscription bridge forces ChatGPT authentication, medium reasoning,
and the declared GPT-6.1 Sol model. Embeddings, auxiliary models, prompt evolution,
and automatic mutation retries are disabled. Invalid and repeated proposals
consume slots. AST-equivalent candidates reuse fully verified cache evidence.
Terminal proposal or evaluation failures stop the stage for review and remain in
the report, including training completed before a scoring failure. No paid API
fallback is used. CLI launches are recorded separately from unobservable backend
request counts and remaining subscription allowance.

## Execution

Use the completed control study as the evaluation context:

```bash
.venv/bin/python scripts/run_adaptive_search.py \
  --study-dir results/adaptive-controls-20261003 \
  --results-dir results/adaptive-shinka-new --prepare-only
.venv/bin/python scripts/run_adaptive_search.py \
  --study-dir results/adaptive-controls-20261003 \
  --results-dir results/adaptive-shinka-new --resume --target-generations 5
.venv/bin/python scripts/report_adaptive_search.py \
  --results-dir results/adaptive-shinka-new \
  --report-dir reports/adaptive-shinka-new-stage5
```

The launcher first exercises the real native scheduler with cached identity and
no model calls. This diagnostic is separate from the five search slots. A small
adapter puts signed evaluation artifacts in `gen_N/results/evaluation`, leaving
Shinka's job logs outside the immutable receipt. Native `correct.json` and
`metrics.json` mirror the signed result exactly. All numerical environment values
match the control study, including unset values; the native Python/NumPy proposal
RNG is seeded and checkpointed separately.

The evaluation study is a shared cache that grows as new rules are evaluated.
Keep the published fixed-control report as its dated snapshot. Export subsequent
proposal stages with `report_adaptive_search.py`; the older control exporter
counts all cache work and should not be used to relabel the enlarged cache as a
new fixed-control-only study.

After a completed block, the same launch with `--resume --target-generations 13`
continues the archive. A larger target is required; existing evidence is checked
before reuse. A failed/interrupted stage cannot silently retry its consumed slot.
Restarted model responses and scheduling need not follow the exact trajectory of
an uninterrupted run.

The [local web UI](shinka-webui.md) displays the native archive and recorded
programs. Its dollar columns are token-price estimates, not subscription charges.
Selection scores remain development feedback; choosing a rule on these seeds
does not establish performance on unseen tasks or search-method superiority.
