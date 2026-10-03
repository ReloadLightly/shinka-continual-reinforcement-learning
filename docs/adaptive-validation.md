# Proposed reserved adaptive validation

**Status: proposed, not frozen or executed.** This document proposes the
comparison after the declared 25-slot adaptive search endpoint. It does not
select a particular program, open reserved outcomes, or authorize new trials.
The search objective and its source contracts remain unchanged.

The declared endpoint has **not been reached**. The
[attempted 25-slot continuation](../reports/adaptive-shinka-stage25-stopped-20261003/summary.json)
stopped after generation 13 completed and generation 14 exceeded the frozen
512-node AST limit. Generation 15 began during shutdown; its guarded request
was cancelled before a Codex proposal launch. Both consumed slots and all
failure evidence remain part of the archive. This proposed validation handoff
is not eligible for execution until the incomplete endpoint has an explicit
documented resolution; it does not authorize using the partial archive as a
substitute endpoint.

The purpose is to assess transfer of one development-selected adaptive rule to
new task draws and a longer adaptation interval. It is a single comparison of
that rule with the five existing controls. It cannot establish the general
superiority of ShinkaEvolve, and it is separate from final paper reporting.
The [adaptive evaluation protocol](adaptive-evaluation.md),
[search runbook](adaptive-search.md), and
[reserved profile](../src/shinka_crl/profiles/adaptive-validation.json) provide
the existing scientific and numerical contract.

## Proposed candidate handoff

Close proposal feedback at the complete 25-slot endpoint before freezing the
handoff. Verify its published hashes, all consumed slots and failure records,
native ancestry, model ledger, source-to-score mapping, and saved RNG state.
Preserve the five- and thirteen-slot snapshots. A failed or incomplete endpoint
requires an explicit documented resolution; do not silently replace consumed
slots or choose from an incomplete archive under this proposal.

Select **one adaptive finalist** from all valid endpoint programs, including
generation-zero identity. Rank by the exact unrounded mean of the frozen
combined development score over seeds 4001–4003. Break an exact tie by earlier
generation. Independently rederive the scores from the published training and
checkpoint evidence before selection. Do not use a displayed rounded score,
previous-task score alone, a visually appealing trajectory, or reserved outcomes
to choose the finalist. If identity leads, retain that result.

Retain all five control memberships below. The recipes are those already
frozen by [the control declarations](../src/shinka_crl/adaptive_evaluation.py)
and [the control-study plan](../reports/adaptive-controls-20261003/raw/plan.json).

| Condition | Recipe to preserve |
| :--- | :--- |
| Selected adaptive program | Exact endpoint source; initial sigma 0.5, archive fraction 0.5, four zero memory values, unchanged adapter and width bounds |
| Identity | Frozen identity source, initial sigma 0.5, archive fraction 0.5 |
| Arithmetic | Frozen `sigma * exp(0.1 * (stats[4] - 0.5))` source; unchanged memory and the same initialization as identity |
| Native FocusGA | Pinned native method and explicit paper settings used in the control study; centroid evaluation remains inside the population budget |
| Static Shinka 11 | Frozen static source; sigma 0.065 and archive fraction 0.075 |
| Static random 24 | Frozen static source; sigma 0.224195451112929 and archive fraction 0.0921597481719689 |

<sub>Table 1. Proposed comparison conditions. Controls are retained without
retuning. FocusGA changes more than mutation width, so its comparison is between
methods rather than an isolated test of width adaptation.</sub>

Copy every selected source into the handoff and record its SHA256, canonical
AST where applicable, endpoint generation and native ID, development rank and
score, and originating report receipts. For native FocusGA, record the pinned
upstream revision, implementation hashes, and every resolved setting; there is
no standalone evolved program to hash. Preserve original control source hashes
and exact static parameters from the existing control plan.

Deduplicate only when the **full execution recipe is identical**: native method,
algorithm variant, source or canonical AST, starting sigma and memory, archive
fraction and resolved archive size, adapter settings, numerical environment,
and evaluation context. AST identity alone is insufficient across methods or
controls with different initialization or selection settings. Do not infer
behavioral equivalence. If one evaluation serves multiple identical recipes,
retain all condition memberships and source identities. Never substitute a
different candidate because of its validation performance.

## Reserved comparison and budget

| Component | Existing reserved protocol |
| :--- | :--- |
| Training seeds | 5001–5005 |
| Task trials | 5002–5006, with `trial = seed + 1` |
| Fresh evaluation seeds | 905001–905005, with `eval_seed = seed + 900000` |
| Task sequence | A–B–A–B |
| Generations per phase / total | 80 / 320 |
| Population / training episodes per member | 64 / 3 |
| Episode cap | 500 steps |
| In-loop evaluation episodes | 10 |
| Fresh episodes per native checkpoint evaluation target | 10 |
| Nominal training steps per trial | 30,720,000 |
| Maximum unique recipes / trials | 6 / 30 |
| Maximum nominal training steps | 921,600,000 |

<sub>Table 2. Proposed maximum allocation: one finalist plus five controls,
each on the same five reserved seed/task pairs. Exact recipe deduplication can
reduce the allocation. Nominal training excludes in-loop and fresh post-hoc
evaluation work.</sub>

Use the unchanged equal-weight combined objective,
`J = 0.5 * active_score + 0.5 * previous_score`, with both terms normalized by
500. Keep all four phases and all three consecutive switches. The phase length
changes from 20 in search to 80 in validation; no task identity or switch signal
is added to a program's inputs. Reuse neither development policies nor their
scores: each reserved trial starts a fresh population and memory.

Retain the existing native post-hoc evaluation targets and episode vectors.
With the same three native sources and four phase checkpoints, the current
evaluation layout uses 300 fresh episodes per trial, or at most 9,000 for this
comparison. Reconcile the realized count against the raw episode vectors.
No new model calls are needed. Final reporting seeds 42–51 and task trials
1–10 remain outside this comparison.

## Freeze and implementation before execution

The current tools do not implement this handoff.
[`freeze_finalists.py`](../scripts/freeze_finalists.py) and
[the validation runner](../src/shinka_crl/validation.py) implement the completed
static comparison. Their active-return ranking and static configuration
identity do not define adaptive selection. The current adaptive
[`read_plan`](../src/shinka_crl/adaptive_evaluation.py) deliberately requires
the search profile and cannot accept a substituted validation profile.

Implement a separate adaptive freezer and runner without weakening these
contracts or editing a sealed search/control plan. The new plan must bind the
endpoint receipts, copied candidate sources, complete control recipes, reserved
profile, objective, upstream revision, interpreter and packages, numerical
environment, CPU affinity, time limits, trial order, and reporting criterion.
Reusing verified low-level training and scoring components is appropriate;
validation must have a distinct study/cache context and fresh output paths.

Before any reserved trial, freeze the handoff, its manifest and checksums,
the implementation revision, planned budget, and this protocol with its
reporting decisions resolved. Commit and push that immutable pre-execution
record so it can be reviewed independently of outcomes. A preview must list
every recipe and seed/task/evaluation mapping and calculate the deduplicated
trial and step budgets without launching training or model calls.

Verify selection and exact ties, full-recipe deduplication, source/receipt
tampering, partition boundaries, native control settings, objective components,
and resume/failure accounting. Any required real implementation diagnostic must
use a separate development-only run, such as the existing diagnostic seed
3001, with a documented reduced budget. Do not debug implementation or choose
settings on reserved seeds. Retain the pinned upstream unchanged.

## Proposed execution and stopping limits

Run sequentially on the same two logical CPUs and frozen numerical thread
environment. Propose a fixed seed-major order, with each seed evaluating the
unique recipes in the Table 1 order. The first execution block ends after the
first seed's recipes, at most six trials, for an integrity and accounting check.
The scientific candidate set, objective, and remaining seed allocation cannot
change in response to that block's scores.

Retain the existing 1,800-second limit separately for each training process and
each fresh analysis process. As a proposed operational ceiling, allow four
hours of cumulative active validation-session time, excluding review pauses,
with a finite deadline for each launch. This is a resource limit, not a runtime
prediction or a score-dependent stopping rule. Freeze the exact limits before
execution and monitor actual elapsed time and completed work. If the ceiling
is reached, preserve and report the partial comparison; any extension requires
a recorded resource decision and must not depend on the observed ranking.

Resume only at verified whole-trial boundaries. Rehash complete training and
analysis receipts and rederive their scores before reuse. Preserve failed or
interrupted attempts in their original directories; a reviewed retry receives
a new attempt identity and remains charged. A training completion followed by
analysis failure still counts as training spent. Record allocated, completed,
and scored work separately. No automatic candidate repair, replacement,
additional seed allocation, or proposal search is part of this comparison.

## Proposed reporting criterion

Preregister the **paired five-seed mean difference in combined score between
the frozen adaptive finalist and native FocusGA** as the primary descriptive
comparison. Report the five paired differences and their mean and sample SD,
alongside both methods' separate active and previous-task components. Treat
the comparisons with identity, arithmetic, static Shinka 11, and static random
24 as secondary; report all of them regardless of direction. Matching seed/task
pairs does not imply identical internal random draws across native methods.

For every condition, publish each seed's combined, active and previous-task
score; all phase-end own-task and previous-task returns; the three signed
forgetting differences and their mean; LA, LA−F, ZT and the reference cumulative
metrics; complete unsmoothed training curves; and applied sigma traces where
defined. Link exact recipes, source hashes, raw evidence, failed attempts,
runtime and compute costs. Report dispersion across independent seed trials,
not across correlated generation checkpoints.

The development leader was selected on three seeds; validation uses five new
seeds and a different phase length. Their mean-score difference is not a
matched estimate of improvement across datasets. This proposal makes no
significance claim and defines no threshold that automatically promotes the
rule to final reporting. If inferential intervals or tests are desired, specify
their method and treatment of secondary comparisons before the first reserved
outcome; do not choose them after seeing results. A positive combined mean
alone does not demonstrate reliable retention, causal mechanism, or general
search-method superiority.

Retain a negative or mixed result without reopening the adaptive search using
validation feedback. A later final-reporting handoff and any promotion rule
require a separate declared protocol; this proposed comparison does not supply
one implicitly.
