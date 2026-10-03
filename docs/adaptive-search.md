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

## Completed first block

`results/adaptive-shinka-20261003` completed all five slots at source revision
`b2a9b01`: cached identity plus four distinct proposals, with 12 new training
trials and no failures. The original controller remained alive across the
interactive-session interruption and sealed its state and native RNG checkpoint
without a duplicate launch. Native search took 28.08 minutes;
recorded start-to-finish time was 28.78 minutes with preflight and
final checks. The [published snapshot](../reports/adaptive-shinka-stage5-20261003/summary.json)
and [README analysis](../README.md#first-adaptive-shinka-search) retain every
program, score, receipt, and cost. Generation 1 leads the proposals at 0.4361;
native FocusGA's fixed-control mean remains higher at 0.5006.

The next cumulative target is 13, with the same frozen study and existing
archive. The five-slot plot is deliberately specific to this first block;
do not overwrite its report or figure when exporting later stages.

During checkpoint recovery, the default CLI had updated from the recorded
Codex 0.159.3 to 0.160.0. The launcher correctly rejected the runtime mismatch
before a proposal or training launch. Selecting the existing 0.159.3 binary
restored exact plan equality and passed `--prepare-only` with all five saved
programs. The frozen plan was not edited, and no additional proposals ran.
On the current WSL machine, repeat that read-only continuation check with:

```bash
PATH="$HOME/.codex/packages/app-server-daemon/releases/0.159.3-x86_64-unknown-linux-musl/bin:$PATH" \
  .venv/bin/python scripts/run_adaptive_search.py \
  --study-dir results/adaptive-controls-20261003 \
  --results-dir results/adaptive-shinka-20261003 \
  --resume --target-generations 13 --prepare-only
```

For the next actual stage, use the same command without `--prepare-only`.
Other machines must provide the recorded CLI version on `PATH`; do not relax
the runtime receipt to accommodate a different version. This preflight verifies
checkpoint compatibility; actual adaptive archive continuation remains the
next experiment.
