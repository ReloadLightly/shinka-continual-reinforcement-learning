# Static finalist validation and reference timing

The static configuration search ends at 25 Shinka programs and 24 random controls.
The new experiments use separate output directories and source receipts.
Validation feedback never returns to the proposer.

## Frozen selection

Run the selector on the complete exported search endpoint:

~~~bash
.venv/bin/python scripts/freeze_finalists.py \
  --report-dir reports/search-endpoint-20261002 \
  --output results/finalists-static-local
~~~

The checked-in handoff is [reports/finalists-static-20261002](../reports/finalists-static-20261002/manifest.json); the command above reconstructs it in a fresh local directory.

The selector verifies all candidate slots, raw curves, exact source, runtime
receipts, ancestry, and the frozen random pool. It chooses two distinct
configurations per arm by development score, with the shared default eligible
in both pools and included regardless of rank. Exact score ties use lower
generation/control index, then source hash. Failed or missing slots cannot be
silently replaced. Repeated effective configurations remain charged.

Global deduplication uses float32 mutation width and the integer archive size at
population 64. The handoff preserves every selected arm membership and its exact
source. Two ratios equivalent at population 64 can yield different archive sizes
at population 512; the eventual per-arm winner therefore retains that arm's
original parameters, alongside the identity of the shared validation evaluation.

## One reserved comparison

| Component | Fixed protocol |
| :--- | :--- |
| Training seeds / task trials | 2001–2005 / 2002–2006 |
| Task phases | A–B–A–B |
| Generations per phase / total | 80 / 320 |
| Population / training episodes | 64 / 3 |
| Episode cap | 500 |
| In-loop evaluation episodes | 10 |
| Fresh checkpoint evaluation episodes | 10 |
| Post-hoc evaluation seed | 900000 + training seed |
| Nominal training steps per trial | 30,720,000 |
| Maximum unique configurations / trials | 5 / 25 |
| Maximum planned training steps | 768,000,000 |

<sub>Table 1. Static-finalist validation. Both task draws and adaptation intervals
differ from search. These are validation measurements, not final reporting.</sub>

The selection objective remains mean active-task centroid return across training
checkpoints and seeds, divided by 500. Post-hoc analysis additionally reports
learning accuracy, every consecutive switch's before/after returns and signed
forgetting, zero-shot transfer, LA − F, and the reference-grid cumulative return.
A negative mean forgetting can coexist with losses at individual switches.

Choose one winner per arm by its five-seed mean active-return score. Break exact
ties by lower development rank, then that arm membership's source hash. Retain
the default and all finalist outcomes. Higher active return does not by itself
establish improved retention.

~~~bash
# Preview without training.
.venv/bin/python scripts/run_validation.py \
  --finalists reports/finalists-static-20261002 \
  --results-dir results/validation-static-20261002

# Execute a first block of five trials, then resume the remaining trials.
.venv/bin/python scripts/run_validation.py \
  --finalists reports/finalists-static-20261002 \
  --results-dir results/validation-static-20261002 --execute --max-trials 5
.venv/bin/python scripts/run_validation.py \
  --finalists reports/finalists-static-20261002 \
  --results-dir results/validation-static-20261002 --execute --resume

.venv/bin/python scripts/report_validation.py \
  --runs-root results/validation-static-20261002 \
  --output reports/validation-static-20261002
~~~

The runner checks completed training and analysis before reuse. An interrupted
trial is retained; any required retry uses a fresh attempt directory. Resume
does not restore a partially trained population. The exporter rederives scores,
continual metrics, switch differences, and winner selection, and retains failed
attempts and their allocated work. Use a fresh report path for each snapshot;
--allow-partial labels incomplete evidence and does not select final winners.

## One paper-budget development reference

Before bulk paper-scale scheduling, measure the unchanged default GA on
development seed 1001 / task trial 1002: 20 phases, 200 generations per phase,
population 512, three training episodes, ten evaluation episodes, and a 500-step
cap. This is 3,072,000,000 nominal training steps. Reporting seeds 42–51 are not
used by this runner.

~~~bash
# Verify timing instrumentation with a small development run.
.venv/bin/python scripts/run_reference_timing.py \
  --profile smoke --results-dir results/reference-timing-smoke-20261002 --execute

# Preview, then execute the complete development reference trial.
.venv/bin/python scripts/run_reference_timing.py \
  --results-dir results/reference-timing-20261002
.venv/bin/python scripts/run_reference_timing.py \
  --results-dir results/reference-timing-20261002 --timeout 21600 --execute

.venv/bin/python scripts/report_reference_timing.py \
  --run-dir results/reference-timing-20261002 \
  --report-dir reports/reference-timing-20261002
~~~

Both experiments use two logical CPUs and the same 13 numerical-runtime settings
as static search. The timing runner uses the authors' existing checkpoint option
to observe each phase boundary. Phase 1 includes process startup and compilation;
intermediate intervals include checkpoint serialization. Final artifact writing
is recorded separately. Peak RSS is the Linux measurement for the trainer
process, excluding post-hoc analysis; it is not aggregate system memory.

The trainer is unchanged. Exact commands, runtime settings, source hashes, raw
timing events, and partial outcomes are preserved. The explicit six-hour timeout is an
execution limit, not a runtime prediction. The runner requires a fresh directory
and does not load a saved training pickle. A partial trial is timing evidence
only. One full-budget GA trial cannot establish method rankings or predict PPO
runtime.
