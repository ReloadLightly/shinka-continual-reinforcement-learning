# Adaptive mutation programs

The main ShinkaEvolve extension will evolve an executable mutation-width update
from training feedback. The completed two-parameter search supplies static
baselines. This specification details the next search space in the
[experimental roadmap](experimental-roadmap.md#7-second-search-space-adaptive-mutation-programs);
the restricted adapter, fixed objective, development partition, and verified
cache are implemented. See the [evaluation protocol](adaptive-evaluation.md).
The [thirteen-slot adaptive Shinka checkpoint](../README.md#adaptive-continuation-to-thirteen-slots)
is complete; the 25-slot endpoint and reserved validation remain pending.

Keep archive size, survivor selection, policy architecture, task draws, and
interaction budgets fixed. The first program family changes only the Gaussian
width used to generate the next population.

## 1. Integration with the unchanged trainer

A repository entry point parses its own variant and program-path flags, sets
the frozen CPU environment, verifies the upstream pin, and imports the native
runner. It then passes ordinary native arguments, including **`--method ga`**,
to [`source.run.main`](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/source/run.py#L217).
`ga_adaptive` is repository metadata; upstream is never asked to parse a
nonexistent method or program flag.

Intercept `source.runners.train_nes.build_searcher` in this subprocess. The
runner imports the factory directly, so replacing only the original searchers
module's attribute after import would miss the bound reference. Call the captured
native factory with unchanged arguments, then wrap its plain `GASearcher`.
Reject an unexpected method, initial width, or archive fraction. Do not change
upstream files or install a persistent patch.

The native
[`generation_step`](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/source/runners/train_nes.py#L517)
already splits its key, generates a population, evaluates that population, and
passes its genomes and fitness to `tell`. Preserve this sequence. The wrapper
computes the declared statistics, calls native `tell`, then replaces only the
next width and program memory. It adds no random-key split or environment query.

## 2. Program and state

```python
# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    return sigma, memory
# EVOLVE-BLOCK-END
```

| Value | Contract |
| :--- | :--- |
| `sigma` | Float32 scalar, initially 0.5 |
| `stats` | Float32 vector of five current-training statistics |
| `memory` | Float32 vector of length four, initialized once to zero |
| Returned width | Finite scalar, clipped by the harness to [0.001, 2.0] |
| Returned memory | Finite float32 vector of length four |

Extend the native state with `memory` and a harness-owned `invalid_update` flag.
Preserve its `archive`, `fitness`, `sigma`, and `generation` fields. The
[`native GA`](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/source/algorithms/ne/ga.py#L81)
uses named attributes and `state._replace`, allowing delegated initialization,
mutation, selection, and policy access without copying their implementations.
Initialize with the original key and mean; append zero memory without another
key split. Keep archive fraction 0.5 and preserve memory across task switches.

The evaluated batch contains offspring first, then the re-scored old archive.
Compute these statistics before survivor selection:

| Index | Statistic |
| ---: | :--- |
| 0 | Mean fitness of the entire batch / 500 |
| 1 | Population standard deviation of batch fitness (`ddof=0`) / 500 |
| 2 | Maximum batch fitness / 500 |
| 3 | Mean current fitness of the re-scored old archive / 500 |
| 4 | Fraction of offspring strictly exceeding the median current archive fitness |

Use the current `fitness` argument, not stored `state.fitness`, which contains
negated historical values and initially contains infinities. Statistic 4 is an
archive-relative success measure; it does not compare each child with its parent.

The program receives no task identity, switch flag, phase length, generation
counter, raw observations, policy parameters, evaluation returns, or environment
handle. Its memory can still implement a counter; validation must therefore use
a different phase interval from search.

## 3. Executable grammar and invalid outputs

Accept one undecorated function with the exact three arguments, local
assignments, and a two-value return. Allow finite literals, arithmetic,
comparisons, constant-index reads, and a documented JAX operation whitelist:
`exp`, `log`, `sqrt`, `tanh`, `abs`, `minimum`, `maximum`, `clip`, `where`, and
`stack`. Reject imports, arbitrary attributes or calls, loops, comprehensions,
nested functions, recursion, and access to globals or builtins. Bound source
length, AST size, and expression depth before tracing. Grammar version
`adaptive-width-v1` permits at most 8,192 UTF-8 bytes, 512 AST nodes, 32 local
assignments, and expression depth 32. Arithmetic is limited to `+`, `-`, `*`,
and `/`; power and modulo are rejected. Calls use bare operation names (for
example `exp(...)`), never attributes such as `jnp.exp`. Only scalar or equal
length vector broadcasting is allowed. `stack` takes exactly four floating-point
scalars. Numeric literals become float32; constant indices remain integers.
Freeze these limits with the evaluator and include them in the proposer prompt.

Check scalar/vector shapes with `jax.eval_shape`. Execute the accepted function
inside the native generation JIT. Check output finiteness **before** clipping;
infinity must not become a valid width merely because clipping produces a bound.
Carry any invalid result in the harness-owned state flag.

The pinned outer loop calls `incumbent(state)` on the host immediately after the
jitted update. The wrapper can check the flag there and fail the candidate
before report-card evaluation, recording the completed-generation count. Shape
errors fail tracing. Do not silently replace an invalid program's update with a
fallback and report success.

## 4. Width logging and the two implementation checks

Set `adapts_sigma=True` so the
[`native logger`](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/source/runners/train_nes.py#L156)
reads state width instead of its initial configuration. The loop records this
value **after `tell`**: raw `sigma[g]` is the width prepared for generation `g+1`.
Preserve that raw column and derive explicitly labelled values:

```text
sigma_used[0] = 0.5
sigma_used[g] = raw_sigma[g - 1]  for g > 0
sigma_next[g] = raw_sigma[g]
```

The last update is validated but has no subsequent generation. Do not use the
hidden schedule to skip it.

1. **Identity parity.** Compare plain GA and the identity adapter on short
   stationary and switching development traces. Match native initial state,
   first asked population, complete reward/task records, archive and phase-end
   checkpoint arrays, and fresh centroid evaluations. Compare numerical arrays,
   not compressed-file bytes or timing metadata. Width must remain 0.5.
2. **Mutation actuation.** Use `return sigma * 0.5, memory + 1.0`. The memory
   counter checks persistence across a switch without influencing width. With fixed state and
   ask key, verify that archive members and parent/noise draws are preserved while
   offspring displacement changes with width. The first generation uses 0.5 and
   the second 0.25. Check the derived width log, unchanged batch/task schedule,
   and nominal budget. Also reject a nonfinite update before it can be clipped
   into a successful result.

These focused checks and a reduced real run are the adapter gate. The existing
baseline pilot does not need to be repeated.

## 5. Controls and method identity

Compare evolved programs with plain GA/identity, both frozen static winners,
the fixed rule `sigma * exp(0.1 * (stats[4] - 0.5))`, and the paper's existing
[`FocusGASearcher`](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/source/algorithms/ne/ga.py#L137)
transferred to CartPole.

Use explicit Focus settings: initial width 0.5, archive fraction 0.5, crossover
0, focus rate 0.3, width rate 0.1, target 0.9, width floor 0.00001, explorer
fraction 0.25, and CartPole initialization `init_around_mean=False`. Preserve its
native random-key progression, including its extra initialization split.

Focus changes width **and** the eligible parent pool. It evaluates the old
centroid inside the fixed population budget, replacing one offspring, and uses
that feedback to track the selected archive. Explorers retain the initial width.
Its feedback, width range, selection, and random draws differ from the proposed
interface. Treat it as a whole-method comparison, not an isolated width ablation.

The existing native GA arm selects this comparator through
`--ne_override method=ga_focus`. All numeric settings are explicit; the pinned
CartPole arm supplies boolean `init_around_mean=False`. The native override
parser does not parse booleans: passing `false` would create a truthy string.
The launcher and analysis validate the resolved boolean and all numeric settings.
Repository scoring now supports the true `ga_focus` identity throughout.

For wrapped programs, additionally record `algorithm_variant="ga_adaptive"`,
program and adapter hashes, and the exact launcher invocation. Upstream may
still record `method="ga"`; that field alone must not place an adaptive run in
the default-GA group. Match interaction and checkpoint budgets across controls,
including Focus's training centroid. Account for post-hoc evaluation separately.

## 6. Selection objective

For the new switching-task study, combine active-task learning with fresh
performance on previously trained tasks:

$$
J_{\mathrm{adaptive}}=\tfrac12 J_{\mathrm{active}}
+\tfrac12 J_{\mathrm{previous}},\qquad
J_{\mathrm{previous}}=\frac{1}{500(P-1)}
\sum_{j=0}^{P-2}R_{j,j+1}.
$$

`J_active` is the mean active-task centroid return across training, divided by
500. `R[j,j+1]` is the fresh return on phase `j`'s task after phase `j+1`, provided
by the reference evaluator's `prev_returns`. Both terms lie in [0,1]. Compute
the objective on the new adaptive development partition only. Post-hoc results
may inform the outer proposer; they never enter `update_sigma`.

This objective avoids rewarding lower early own-task performance through the
subtraction in `LA−F`. Previous-task return can nevertheless reflect later
acquisition of a poorly learned task. Report LA, all signed switch differences,
mean F, LA−F, and ZT alongside it. The objective is an extension metric;
static selection and reference-paper metrics remain fixed.

Adaptive development uses seeds **4001–4003** / task trials **4002–4004**;
reserved validation uses **5001–5005** / **5002–5006**. Fresh evaluation uses
training seed + 900000. The [allocation audit](adaptive-seed-allocation-20261003.json)
confirmed these were unused before the study was frozen.
The adapter gate alone reserves diagnostic seed 3001 / task trial 3002 and
independent evaluation seed 903001; these are not adaptive search seeds.
Do not use
reserved static validation or final-reporting outcomes to choose this objective,
grammar, search budget, or program. A single outer search evaluates its resulting
program; stronger claims about the search method require repeated outer searches.

## 7. Provenance, duplicate proposals, and implementation footprint

Use a new task directory, archive, profile, and protocol version. Record raw
source and canonical AST hashes, grammar/adapter/scorer hashes, upstream pin,
runtime fingerprint, seed/trial mapping, and outer ancestry. A score cache may
reuse only completed, verified evaluations keyed by canonical AST and all those
evaluation identities. A repeated proposal still consumes its slot and model
usage; record cache ancestry and avoided training work. Canonical AST identity
removes formatting differences, not general behavioral equivalence. Freeze this
policy before proposals; preserve the static run's original retraining policy.

| Component | Responsibility |
| :--- | :--- |
| Repository adaptive module | Source grammar, extended state, delegated GA wrapper, statistics, finite checks |
| Subprocess entry point | Parse repository flags, intercept the bound factory, call native CLI, record provenance |
| `tasks/cartpole_adaptive/` | Identity program, fixed evaluator, proposer prompt/configuration, separate archive |
| Variant reporting support | Explicit Focus identity, width timing, candidate receipts, unchanged reference metrics |
| Focused checks | Identity traces, actual width effect, invalid outputs, score terms, and duplicate-slot accounting |

Version any required shared-helper changes after current frozen runs have been
exported. The main implementation risks are patching the wrong factory binding,
using stored losses instead of current fitness, confusing next width with used
width, hiding invalid outputs through clipping, or merging distinct algorithms
under the same upstream method label.

## 8. Diagnostic protocol and next handoff

The seven-trial gate uses two phases of three generations, population 16, three
training episodes per member, and a 500-step episode cap. Each trial therefore
allocates 144,000 nominal training steps; the suite allocates 1,008,000. Saved
phase policies receive ten fresh episodes for each native evaluation target,
accounted separately from training. Execution uses two logical CPUs and the
existing frozen numerical thread environment.

| Condition | Programs or methods |
| :--- | :--- |
| Stationary | Plain GA; identity adapter |
| Switching | Plain GA; identity; halving with memory counter; arithmetic rule; native FocusGA |

The six GA-wrapper runs save every generation's full evaluated population using
the native diagnostic snapshot option. It reconstructs `ask` with the same
pre-update state and key, without additional environment evaluation. Host traces
save initialization and every completed native state. The independent NumPy
verifier compares numerical values, shapes, and dtypes, and publishes semantic
array hashes. Large binary arrays remain local. FocusGA runs directly through
the unchanged native entry point; its centroid accounting is checked against
resolved settings and pinned source rather than a population snapshot.

The source snapshot is
[`e28db07`](https://github.com/ReloadLightly/shinka-continual-reinforcement-learning/tree/e28db07).
Replay instructions are in the [task runbook](../tasks/cartpole_adaptive/README.md).
Use fresh output directories; source and artifact guards prevent silently
reinterpreting earlier evidence under changed code.

The subsequent [adaptive selection stage](adaptive-evaluation.md) is complete:
both objective terms are computed from verified training and fresh checkpoint
evidence, new search and validation partitions are allocated, and those
identities enter the canonical-AST cache key. All 15 fixed-control trials and
two cache checks passed. The next step is the separate adaptive Shinka archive,
starting with five total slots and retaining the fixed controls as baselines.
The identity gate establishes implementation correctness, not a performance
ranking or a discovered learning rule.

**Observed gate outcome:** all seven trials and 19 independent numerical checks
passed in 238.58 seconds. Both identity conditions match exactly; the halving
rule changes the next offspring, and its memory survives the task switch.
The arithmetic rule and native FocusGA completed with their declared settings.
See [raw evidence and receipts](../reports/adaptive-gate-20261003/summary.json)
and [Table 12 / Figure 5](../README.md#executable-adaptive-rule-verification).
