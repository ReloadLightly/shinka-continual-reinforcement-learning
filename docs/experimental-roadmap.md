# Experimental roadmap

This document specifies the next experiments for the CartPole reproduction and
the ShinkaEvolve extension. The immediate step is an 18-trial baseline pilot with
matched nominal training budgets and a stationary control. A staged search over
static GA settings follows only after the pilot validates learning and analysis.
Adaptive mutation programs are a separate subsequent experiment.

The numerical budgets below are proposed protocol choices, not observed results.
The existing constant-configuration evaluator is executable; the pilot profiles,
post-hoc analysis adapter, validation runner, and adaptive-program adapter still
require implementation. Source revisions remain fixed by
[`upstream.lock.json`](../upstream.lock.json). The full-paper protocol remains in
the [reproduction plan](reproduction-plan.md).

## 1. Questions and experimental separation

The reproduction asks whether the pinned GA, ES, and PPO reproduce the reference
paper's behavior under repeated CartPole task changes. The extension asks whether
LLM-guided program search selects a better GA configuration than the default and
an equally budgeted random search. Discovering a new adaptive learning rule is a
later question and needs a larger program interface.

Keep the following levels distinct: an **outer proposal** is a candidate Python
program; a **candidate evaluation** runs that program's settings on several
training seeds; an **inner generation** evaluates and updates a population of
neural-network policies. Shinka evolves the configuration program in the initial
experiment. The unchanged GA evolves policy weights inside each evaluation.

| Partition | Training seeds | Task trials | Permitted use |
| :--- | :--- | :--- | :--- |
| Development | 1001–1003 | 1002–1004 | Pilot, candidate feedback, debugging |
| Validation | 2001–2005 | 2002–2006 | One finalist comparison after search |
| Final reporting | 42–51 | 1–10 | Frozen algorithms and paper-scale protocol |

<sub>Table 1. Proposed seed partitions. The upstream task offset depends on the
task trial, so disjoint training seeds alone would not provide disjoint tasks.
Development and validation use `trial=seed+1`; reporting retains trials 1–10.</sub>

Do not feed validation outcomes back into further candidate proposals. Freeze
the selected programs, scorer, analysis definitions, and source hashes before
final reporting. If validation leads to redesign, those validation trials become
development evidence and a new untouched validation partition is required.

## 2. Matched baseline pilot

Run GA, ES, and PPO on both the alternating task and its stationary control. Both
conditions retain the 500-step episode cap and four equal checkpoint phases.

| Setting | GA | ES | PPO |
| :--- | ---: | ---: | ---: |
| Phases | 4 | 4 | 4 |
| Generations or updates per phase | 20 | 20 | 150 |
| Total generations or updates | 80 | 80 | 600 |
| Population or parallel environments | 64 | 64 | 256 |
| Training episodes per candidate | 3 | 3 | — |
| Rollout steps per update | — | — | 50 |
| Minibatches per PPO epoch | — | — | 8 |
| Episode cap | 500 | 500 | 500 |
| Evaluation episodes per checkpoint | 10 | 10 | 10 |
| Nominal training steps per trial | 7,680,000 | 7,680,000 | 7,680,000 |

<sub>Table 2. Proposed pilot budget. GA/ES: 80 × 64 × 3 × 500; PPO: 600 ×
256 × 50. Each phase receives 1,920,000 nominal training steps. Other optimizer
settings retain their pinned baseline values. PPO's reduced rollout and
minibatch counts are explicit development deviations.</sub>

The switching condition uses `CartPole-v1_sigma0.5`, two distinct tasks, and the
sequence A–B–A–B. The stationary condition uses `CartPole-v1`, one task, and
A–A–A–A. Its profile must set `num_tasks=1` as well as the stationary environment
name; its scorer must expect task zero throughout. Explicitly override its total
training length and phase interval so the upstream stationary defaults cannot
silently change the budget.

Three learners × two conditions × three development seeds give **18 trials**,
totalling **138,240,000 nominal training steps**. Evaluation, diagnostics, and
program-search training are separate costs. Nominal steps refer to the configured
episode caps for NE and collected transitions for PPO; they do not imply equal
useful transitions before termination, floating-point work, or wall time.

Acceptance requires matching resolved configurations, the expected task traces,
four phase checkpoints per trial, finite metrics, and valid post-hoc evaluations.
Show all three seed outcomes. A three-seed pilot diagnoses behavior and runtime;
it does not establish statistical superiority.

Before viewing the pilot, adopt this budget adequacy check: a method has a usable
stationary learning signal if its last-phase mean return exceeds its first-phase
mean by at least 50 reward points on at least two of three seeds, or its
last-phase mean is at least 400 on two seeds. If any baseline fails this check,
inspect its configuration and learning curve before search. If short training
is the explanation, repeat all three methods under both conditions with 80
GA/ES generations or 600 PPO updates per phase. The larger pilot costs
30,720,000 steps per trial and 552,960,000 steps across 18 trials. This gate is
an engineering criterion, not a significance test or the paper's solved-task
threshold. Do not enlarge only the failing method's budget or select budgets
using a desired ranking.

## 3. Measurements and evaluator boundaries

The reference reports each method through a single agent: the population-weight
centroid for GA, the distribution mean for ES, and the policy for PPO. Let
`theta_j` be that agent at the end of phase `j`, and `R_i,j` its mean return on
phase `i`'s task. Learning accuracy averages `R_i,i`; zero-shot transfer averages
`R_i+1,i`. For alternating tasks, forgetting averages `R_i,i − R_i,i+1` over all
consecutive switches. This includes both switching directions and allows negative
forgetting. The paper defines these quantities in
[Appendix A.3](https://arxiv.org/html/2610.01583v1#A3).

Implement these definitions explicitly in the analysis adapter:

$$
\mathrm{LA}=\frac1P\sum_{j=0}^{P-1}R_{j,j},\qquad
\mathrm{ZT}=\frac1{P-1}\sum_{j=0}^{P-2}R_{j+1,j},
$$

$$
\mathrm{F}=\frac1{P-1}\sum_{j=0}^{P-2}
\left(R_{j,j}-R_{j,j+1}\right).
$$

Here `P` counts phases, including repeated visits. Also report `LA − F`. For the
stationary control, report learning and cumulative return; task-transfer metrics
are not interpreted as transfer between distinct tasks.

Run the pinned
[`evaluate_continual.py`](https://github.com/eleninisioti/continual_neuroevolution/blob/821570eb6a22db0f7aa77111b2ea541fe8fa795b/scripts/analysis/evaluate_continual.py)
on saved checkpoints with **`--episodes 10` explicitly** and a recorded evaluation
seed independent of training. Its default is 100 episodes. Its `centroid` entries
provide the GA/ES measurements and its `final` entries provide PPO measurements.
Own-task returns supply LA, `zero_shot_next_returns` supply ZT, and the following
phase's `prev_returns` supply the second term in each forgetting difference.
The adapter must validate source labels, phase identities, and episode counts.

Do not substitute the GA's best individual, the first post-switch training
record, or a final-checkpoint-only forgetting calculation. The upstream
`zero_shot_carried_best` training field concerns the carried incumbent, which can
differ from the centroid. Preserve that field as a diagnostic with its own label.

For cumulative return, use the pinned integration convention on a common
training clock. Source inspection confirms that rows are recorded **after** the
update, so their completed-step positions are `(generation + 1) * step_size`.
In the pilot, GA/ES checkpoints are 96,000 steps apart and PPO checkpoints are
12,800 steps apart. The upstream plotter expresses that clock in NE-generation
equivalents, and its metric resamples onto the unit generation grid before
trapezoidal integration, with constant first/last tails. Our primary Cum follows
that discretization and converts its units to reward × nominal environment
steps. Also retain the integral over all logged knots as a separately labelled
diagnostic: intermediate PPO samples make it differ from the resampled metric.
A mean of unaligned GA and PPO rows is not the paper's cumulative-return metric.
Divide primary Cum by total steps and 500 only for a fixed dimensionless curve
average; the constant initial tail is not an untrained-policy evaluation.

Report raw reward units for the pilot. The fixed `/500` search normalization is
different from the paper's rescaling against an untrained reference and the best
method's mean LA. Applying the latter requires those reference measurements and
a fixed comparison set. Do not label `/500` as the paper's normalized metric.

Evaluation trajectories may measure previously seen and future tasks for
reporting, but never enter the inner learner's update. The outer search can use
development metrics; final reporting tasks and episodes remain unavailable to
the proposer.

## 4. First search space: static GA configurations

The executable candidate is a small declarative Python program:

```python
# EVOLVE-BLOCK-START
def get_ga_config():
    return {"sigma": 0.5, "elite_ratio": 0.5}
# EVOLVE-BLOCK-END
```

The [current evaluator](../tasks/cartpole_ga/evaluate.py) parses the AST without
importing or executing candidate code. It accepts exactly one undecorated,
argument-free function called `get_ga_config`, containing one literal dictionary
return with exactly these two distinct keys. Values must be finite numeric
literals, excluding booleans; imports, helpers, expressions, and extra statements
are rejected. Source size is limited to 64 KiB.

| Variable | Bounds | Default | Interpretation |
| :--- | :--- | ---: | :--- |
| `sigma` | [0.001, 2.0] | 0.5 | Gaussian mutation width |
| `elite_ratio` | [0.05, 0.95] | 0.5 | Archive share of the fixed population |

<sub>Table 3. Current search interface. The upstream GA resolves archive size as
`max(1, int(population_size * elite_ratio))`; nearby ratios may therefore describe
the same archive size at population 64.</sub>

Each candidate trains a fresh GA on all three development seeds. The current
`search` profile fixes four phases, 80 inner generations, population 64, three
training episodes, three evaluation episodes, and episode cap 500. It costs
**23,040,000 nominal training steps per candidate**, or 240 inner generations
across seeds. The pilot's ten evaluation episodes are a separate protocol.

The current selection objective is

$$
J(p)=\frac1{|S|}\sum_{s\in S}\frac1{GH}
\sum_{g=0}^{G-1}R^{\mathrm{centroid}}_{s,g,a(g)}(p),
\quad |S|=3,\ G=80,\ H=500.
$$

It rewards active-task return throughout learning and recovery. It does not
directly measure forgetting or prove a better stability–plasticity trade-off.
Keep this objective fixed throughout one search. If the larger pilot is needed,
freeze a correspondingly revised search profile and recalculate the entire
search budget before proposing candidates; do not mix scores from two budgets.

Successful evaluations write `correct.json` with `correct: true` and
`metrics.json` with `combined_score`, public metrics, and per-seed results.
Failures write `correct: false`, an error, and a zero placeholder score. Failed
programs must be excluded from scientific rankings; a placeholder is not an
observed return.

At each outer step, Shinka selects a parent and archived context, asks its
proposer for an edit, builds a candidate, evaluates it with the fixed harness,
and records source, ancestry, validity, score, and feedback. Subsequent proposals
can use this feedback. The policy populations trained during one candidate's
evaluation are not inherited by the next candidate. The source program is the
outer evolutionary individual.

## 5. Search budgets and staged execution

Use one initial program and **24 proposal slots**. Complete the run at three
checkpoints: **5 → 13 → 25 total program slots**, corresponding to the initial
program plus 4, 12, and 24 proposals. The stages contain 4, then 8, then 12 new
proposals. Retain a single archive/database and the same frozen evaluator across
these stages.

| Checkpoint | Initial programs | Cumulative proposal slots | Maximum GA seed trials |
| :--- | ---: | ---: | ---: |
| Integration gate | 1 | 4 | 15 |
| Search-behavior gate | 1 | 12 | 39 |
| Pilot endpoint | 1 | 24 | 75 |

<sub>Table 4. Proposed Shinka stages under the three-seed search profile. Counts
are cumulative and include the initial candidate. Proposal slots are an outer
budget, not a count of successful evaluations or inner GA generations.</sub>

Generate and freeze **24 distinct random configurations** from
`log(sigma) ~ Uniform(log(0.001), log(2.0))` and
`elite_ratio ~ Uniform(0.05, 0.95)` using a recorded random seed. The random-search
arm shares the initial default candidate, development seeds, training budget,
evaluation episodes, score, and promotion rule. Charge all completed candidate
training to its arm, including partially completed failures.

At maximum, 24 Shinka candidates + 24 random candidates + one shared default
produce **49 distinct configuration evaluations**, **147 GA seed trials**, and
**1,128,960,000 nominal training steps** at the current search budget. This is an
upper bound before validation and includes the shared default only once.

Distinguish proposal slots, model requests, valid distinct programs, completed
seed trials, and elapsed compute. An invalid or duplicate Shinka output consumes
its slot; identical effective configurations can reuse archived scores. A model
request can fail before producing a program, and retry/repair requests still
consume model usage. Set and record finite retry limits before launch. Never
silently continue until 24 successful improvements have appeared.

For strict evaluation-budget comparison, compare the Shinka arm to the prefix of
the frozen random pool with the same number of completed distinct candidate
evaluations, and report actual seed work when a candidate fails partway through.
If counts differ, the full 24-candidate random pool is a separately labelled
comparison. Equal proposal slots alone do not establish equal training compute.

After slot five, require valid artifact contracts, reproducible source-to-score
mapping, recorded parentage, and a verified resume. After slot 13, inspect failure
and duplicate rates and whether scores distinguish candidates. Lack of a gain
over the default is a valid result, not a reason to change the metric. Slot 25 is
the preregistered pilot endpoint.

An optional extension to **49 total Shinka slots** adds 24 further proposals and
24 further random configurations. Before the initial run, declare it conditional
on intact artifacts, at least 80% valid distinct outputs among the first 24
proposals, and the recorded compute/model-usage envelope permitting the extra
work. It must also precede validation. If extended, report both the slot-25 and
slot-49 results; do not hide the original endpoint. The extension is not currently
scheduled or launched.

Pause between completed candidates, preferably at these stage boundaries. Save
the archive/database, generation counter, candidate hashes, parentage, fixed
profile, source pins, all metrics, and cost ledger. Treat the content of a partial
candidate directory as incomplete. A future resume runner must identify completed
seed trials by their full candidate/profile/source identity and verify artifacts
before reusing them. The current evaluator creates new output directories and
does not yet provide arbitrary mid-evaluation resume.

Subscription-backed proposal execution is described in the
[task runbook](../tasks/cartpole_ga/README.md). The existing API example must not be
mistaken for a subscription-backed launch.

### Measured timing and session sizes

A real evaluation of the initial configuration on all three search seeds took
**114.69 seconds**, including process startup and compilation. Its three training
processes took 43.82, 28.46, and 42.03 seconds; the measured maximum child-process
resident set size was 739,604 KiB. This is one candidate's calibration, not a
confidence interval or an upper runtime bound. The
[timing evidence](../reports/search-timing-20261002/summary.json) records the exact
protocol and measurements.

| Stage | New candidate evaluations | Local evaluation estimate | Including assumed proposal time | Suggested session budget |
| :--- | ---: | ---: | :--- | :--- |
| Start → slot 5 | 5 | 9.56 min | 12–18 min | 15–30 min |
| Slot 5 → slot 13 | 8 | 15.29 min | 19–31 min | 20–40 min |
| Slot 13 → slot 25 | 12 | 22.94 min | 29–47 min | 30–60 min |

<sub>Table 5. Planning estimates at the observed initial-candidate speed. Proposal
time is an assumption of 0.5–2 minutes per proposal; the first stage includes one
initial evaluation and four proposals. Reviews, retries, usage-limit pauses,
hardware contention, and variable candidate runtimes can lengthen each stage.</sub>

The full 25-slot search is approximately **60–100 minutes** under those
assumptions. Twenty-four additional random configurations require approximately
**45.88 minutes** of local evaluation. Reserve **2–3 hours** for search plus the
random-search control, spread over several sessions if useful. These estimates
exclude the 18-trial baseline pilot, finalist validation, and final reporting.
Benchmark one promoted-budget trial before scheduling validation, and measure
the paper-size profile separately; training-step ratios alone do not establish
wall-time ratios.

## 6. Promotion and final evaluation

After search, select the top two valid distinct Shinka configurations and the top
two valid distinct random configurations by development score. Include the
default GA. If candidates coincide, deduplicate them and report fewer finalists;
do not replace them using validation performance.

Evaluate these at most five configurations on validation seeds 2001–2005 using
four phases of 80 generations, population 64, three training episodes, ten
evaluation episodes, and episode cap 500. This is **30,720,000 nominal training
steps per seed**, at most **25 seed trials**, and at most **768,000,000 steps**.
It probes both new task draws and a longer adaptation interval. Record complete
centroid curves and post-hoc LA, F, LA − F, and ZT.

Select one finalist per search arm using the same active-return objective on
validation; break exact ties by lower development rank, then source hash. Freeze
the selected source and configuration before the reporting trials. Report
retention metrics alongside this choice: improved active return alone cannot
support a claim of reduced forgetting. If the selected default or random arm
wins, retain that finding.

The final CartPole study compares the default GA, ES, PPO, frozen Shinka winner,
and frozen random-search winner on seeds 42–51 and task trials 1–10. Use the
20-phase `paper-cartpole` protocol, 3,072,000,000 nominal training steps per trial,
and the paper's reporting metrics. Fifty continual trials would require
153,600,000,000 nominal steps before stationary controls and diagnostics. This
stage needs separate timing and hardware planning; the development pilot is not
evidence that it fits a short local session.

Report every trial, mean performance, and uncertainty resampled over trials,
not over individual correlated checkpoints. Keep search cost separate from the
per-learner training budget. A single run of each search strategy supports a
pilot comparison; a general claim that Shinka beats random search also requires
repeated outer searches under independently seeded search trajectories.

## 7. Second search space: adaptive mutation programs

Static configuration search is a useful executable-system test and a necessary
baseline. Actual learning-rule discovery needs code whose behavior depends on
the learning process. The proposed next interface evolves only mutation-width
adaptation:

```python
# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    return sigma, memory
# EVOLVE-BLOCK-END
```

| Argument or output | Fixed contract |
| :--- | :--- |
| `sigma` | Float32 scalar; initially 0.5 |
| `stats` | Float32 vector of five training-only statistics |
| `memory` | Float32 vector of length four; initialized to zero once per trial |
| Returned width | Finite scalar, bounded by the harness to [0.001, 2.0] |
| Returned memory | Finite float32 vector with unchanged shape |

<sub>Table 6. Proposed adaptive-program interface; not yet implemented. The
identity program supplies the neutral baseline.</sub>

The five statistics, in fixed order, are population mean fitness, population
fitness standard deviation, population maximum fitness, mean fitness of the
re-scored old archive, and the fraction of offspring whose fitness exceeds the
median of that re-scored archive. Divide the first four by 500. All quantities
come from evaluations already paid for by the current generation, before
survivor selection; no new environment queries are available to the program.

Compute these summaries from the evaluated batch, complete the unchanged GA
`tell`, then call `update_sigma` once. Its width controls the next generation's
`ask`. The first generation uses width 0.5. Use a fixed archive fraction of 0.5
so the archive and batch shapes never change. Parent selection, Gaussian noise,
archive re-scoring, policy architecture, task draws, training and evaluation
budgets, and RNG progression remain owned by the harness.

The program receives no task identity, switch flag, phase length, global
generation counter, held-out evaluation return, raw observation offset, file
access, or environment handle. Memory continues through switches and is never
reset by the harness at a boundary. Provide a small documented arithmetic/JAX
operation set and validate source, shapes, finite values, runtime limits, and
side effects. Validation must include phase lengths different from search: a
recurrent program could otherwise learn a fixed schedule indirectly.

Implement a repository-owned `GASearcher` adapter at the pinned runner's
`build_searcher` boundary, without editing upstream tracked files. The adapter
must preserve the upstream initialization and random keys. Before proposing any
adaptive code, the identity program must match the default GA's complete seeded
training and evaluation traces under both stationary and switching conditions.
An adaptive test program must demonstrably change logged sigma while preserving
the training budget and task schedule.

Use a separate task directory, archive, protocol version, and seed allocation for
this experiment; the static evaluator cannot execute the function above. Before
launch, freeze a retention-sensitive objective. One proposed objective is

$$
J_{\mathrm{adaptive}}=\tfrac12 J_{\mathrm{active}}
+\tfrac12\frac{\mathrm{LA}-\mathrm{F}}{500}.
$$

This is a proposed extension objective, not the paper's score normalization.
It requires post-hoc checkpoint evaluation per candidate. Validate this scorer
on known traces and publish its cost before starting an adaptive search. Always
report the separate learning, forgetting, and active-return terms so a gain in
one cannot conceal failure in another. Compare evolved rules with the constant
rule, a frozen hand-designed rule, and the best static configuration under the
same training budget.

## 8. Next implementation deliverables

| Order | Deliverable | Acceptance evidence |
| :--- | :--- | :--- |
| 1 | Switching and stationary pilot profiles; resumable trial manifest | Exact budget arithmetic, resolved configurations, reduced real runs |
| 2 | Post-hoc evaluation and reporting adapter | Known-trace metric checks, correct centroid sources, real checkpoint evaluation |
| 3 | Eighteen-trial pilot | Raw curves, per-seed metrics, costs, adequacy-gate decision |
| 4 | Subscription-compatible proposer route and staged archive | ChatGPT authentication without separately billed API inference, candidate ancestry, validity contracts, tested stage resume |
| 5 | Frozen random pool and 5 → 13 → 25 search | Distinct proposal/evaluation counts, cost ledger, default and random comparison |
| 6 | One validation comparison and frozen finalists | Reserved trials used once, candidate hashes, all continual metrics |
| 7 | Full CartPole timing and resource plan | Measured representative paper-shape trial or segment before bulk scheduling |
| 8 | Adaptive-program adapter and neutral gate | Unchanged baseline trace, varying-sigma test, no task-boundary inputs |

<sub>Table 7. Planned deliverables. A listed gate describes required future
evidence; it does not imply the corresponding implementation or run exists.</sub>

After each substantive experiment, update the README as a scientific report:
state the frozen protocol, link compact raw evidence and hashes, distinguish
observations from interpretation, and retain failed attempts and deviations.
Preserve source pins and final reporting trials while the setup is being tested.
