# System Instructions

Evolve an executable mutation-width update for continual CartPole GA.
Maximize the fixed combined_score: equal weights on active-task centroid
return throughout training and fresh previous-task centroid return after
switches, each normalized by 500 and averaged over three development seeds.
Previous-task performance may reflect later acquisition, not only retention.
This is a reduced development study, not final reporting.

Preserve exactly one function: def update_sigma(sigma, stats, memory):
It runs after each evaluated generation; next_sigma controls the NEXT
population's mutations. Initial sigma=0.5 and memory is four float32 zeros.
Archive fraction stays 0.5. Memory persists across task switches.
stats[0]: current whole-batch mean fitness / 500.
stats[1]: whole-batch population standard deviation / 500.
stats[2]: whole-batch maximum fitness / 500.
stats[3]: mean current fitness of the re-scored old archive / 500.
stats[4]: fraction of offspring strictly above the current archive median.
This last statistic is archive-relative, not parent-matched success.
Only these training statistics are available. No task identity, switch flag,
evaluation scores, generation index, environment, or policy weights are inputs.
Seek general rules using feedback and memory, not a memorized switch schedule;
reserved validation changes the phase length.

Allowed: finite numeric constants, single local assignments, + - * /,
unary signs, single comparisons, scalar/vector broadcasting, and constant
nonnegative indexing of named vectors. Bare math calls only:
exp(x), log(x), sqrt(x), tanh(x), abs(x), minimum(x,y), maximum(x,y),
clip(x,lo,hi), where(comparison,x,y), stack([a,b,c,d]).
Return exactly (next_sigma, next_memory), a float scalar and length-four
float vector. Returning memory unchanged is valid. The harness clips finite
width to [0.001,2.0]; nonfinite outputs fail the candidate. Protect divisions,
logs and exponentials. No imports, attributes, loops, if statements, Python
conditional expressions, boolean literals, annotations, decorators, docstrings,
other functions, external state, or arbitrary calls. At most 32 assignments,
512 AST nodes, expression depth 32 and 8192 UTF-8 bytes.

Identity leaves sigma and memory unchanged. A frozen arithmetic control uses
sigma * exp(0.1 * (stats[4] - 0.5)) with unchanged memory. These and native
FocusGA and prior static winners are comparison baselines, not discoveries.
Preserve EVOLVE-BLOCK markers. Return the requested patch directly; do not
use tools or edit files yourself. Invalid and duplicate proposals consume
slots. AST-identical programs reuse verified evaluations, not extra training.

Create a novel algorithm that draws inspiration from the provided context programs but implements a fundamentally different approach.
Study the patterns and techniques from the examples, then design something new.
You MUST respond using a short summary name, description and the full code:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
Explain how you drew inspiration from the context programs and what novel approach you are implementing. Detail the key insights that led to this design.
</DESCRIPTION>

<CODE>
```{language}
# The inspired but novel algorithm implementation here.
```
</CODE>

* Keep the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the code.
* Learn from the context programs but don't copy their approaches directly.
* Combine ideas in novel ways or apply insights to different algorithmic paradigms.
* Maintain the same inputs and outputs as the original program.
* Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters to structure your response. It will be parsed afterwards.

# Previous Messages

[]

# User Request

Here are the performance metrics of a set of previously implemented programs:

# Prior programs

```python
# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    mean = clip(stats[0], 0.0, 1.0)
    best = clip(stats[2], 0.0, 1.0)
    old_mean = where(memory[3] > 0.5, memory[0], mean)
    old_best = where(memory[3] > 0.5, memory[1], best)
    progress = mean - old_mean
    stalled = exp(-40.0 * abs(progress))
    stagnation = 0.85 * memory[2] + 0.15 * stalled
    drop = clip(old_mean - stats[3] - 0.08, 0.0, 1.0)
    shock = memory[3] * drop
    difficulty = 1.0 - mean
    target = 0.015 + 0.10 * difficulty * difficulty + 0.08 * stagnation * (1.0 - best) + 0.35 * shock
    success = tanh(4.0 * (stats[4] - 0.5))
    log_width = 0.75 * log(clip(sigma, 0.001, 2.0)) + 0.25 * log(target) + 0.06 * success * difficulty
    next_sigma = clip(exp(clip(log_width, -7.0, 0.69)), 0.001, 2.0)
    next_mean = 0.8 * old_mean + 0.2 * mean
    next_best = 0.8 * old_best + 0.2 * best
    next_memory = stack([next_mean, next_best, stagnation, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: 0.44
profile: adaptive-search; objective: adaptive-active-previous-v1; cache_hit: False; scores: {'combined_score': {'mean': 0.43610000106891, 'sample_sd': 0.38004478491242727, 'n': 3}, 'active_score': {'mean': 0.5552666688044866, 'sample_sd': 0.27138263181047995, 'n': 3}, 'previous_score': {'mean': 0.31693333333333334, 'sample_sd': 0.5100315066520673, 'n': 3}}

```python
# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    mean = clip(stats[0], 0.0, 1.0)
    best = clip(stats[2], 0.0, 1.0)
    old_mean = where(memory[3] > 0.5, memory[0], mean)
    old_archive = where(memory[3] > 0.5, memory[1], clip(stats[3], 0.0, 1.0))
    progress = mean - old_mean
    stalled = exp(-40.0 * abs(progress))
    stagnation = 0.85 * memory[2] + 0.15 * stalled
    drop = clip(old_archive - stats[3] - 0.08, 0.0, 1.0)
    shock = memory[3] * drop
    difficulty = 1.0 - 0.7 * best - 0.3 * mean
    spread = clip(stats[1], 0.0, 1.0)
    archive_advantage = clip(stats[3] - mean - 0.05, 0.0, 1.0)
    exploration = 0.10 * difficulty * difficulty + 0.14 * stagnation * difficulty * exp(-6.0 * spread)
    target = 0.008 + exploration * exp(-3.0 * archive_advantage) + 0.20 * shock
    success = tanh(4.0 * (stats[4] - 0.5))
    log_width = 0.75 * log(clip(sigma, 0.001, 2.0)) + 0.25 * log(target) + 0.06 * success * difficulty
    next_sigma = clip(exp(clip(log_width, -7.0, 0.69)), 0.001, 2.0)
    next_mean = 0.8 * old_mean + 0.2 * mean
    next_archive = 0.5 * old_archive + 0.5 * clip(stats[3], 0.0, 1.0)
    next_memory = stack([next_mean, next_archive, stagnation, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: 0.49
profile: adaptive-search; objective: adaptive-active-previous-v1; cache_hit: False; scores: {'combined_score': {'mean': 0.48714444582727223, 'sample_sd': 0.4047003902585925, 'n': 3}, 'active_score': {'mean': 0.6243333360989889, 'sample_sd': 0.2516072531287761, 'n': 3}, 'previous_score': {'mean': 0.3499555555555556, 'sample_sd': 0.562959838051756, 'n': 3}}


# Current program

Here is the current program we are trying to improve (you will need to propose a new program with the same inputs and outputs as the original program, but with improved internal implementation):

```python
# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    mean = clip(stats[0], 0.0, 1.0)
    best = clip(stats[2], 0.0, 1.0)
    old_mean = where(memory[3] > 0.5, memory[0], mean)
    old_best = where(memory[3] > 0.5, memory[1], best)
    progress = mean - old_mean
    stalled = exp(-40.0 * abs(progress))
    stagnation = 0.85 * memory[2] + 0.15 * stalled
    drop = clip(old_mean - stats[3] - 0.08, 0.0, 1.0)
    shock = memory[3] * drop
    difficulty = 1.0 - mean
    spread = clip(stats[1], 0.0, 1.0)
    target = 0.012 + 0.08 * difficulty * difficulty + 0.16 * stagnation * difficulty * exp(-6.0 * spread) + 0.35 * shock
    success = tanh(4.0 * (stats[4] - 0.5))
    log_width = 0.75 * log(clip(sigma, 0.001, 2.0)) + 0.25 * log(target) + 0.06 * success * difficulty
    next_sigma = clip(exp(clip(log_width, -7.0, 0.69)), 0.001, 2.0)
    next_mean = 0.8 * old_mean + 0.2 * mean
    next_best = 0.8 * old_best + 0.2 * best
    next_memory = stack([next_mean, next_best, stagnation, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END
```

Here are the performance metrics of the program:

Combined score to maximize: 0.42
profile: adaptive-search; objective: adaptive-active-previous-v1; cache_hit: False; scores: {'combined_score': {'mean': 0.4157583345519172, 'sample_sd': 0.4150146087113926, 'n': 3}, 'active_score': {'mean': 0.5146277802149455, 'sample_sd': 0.33287258192740826, 'n': 3}, 'previous_score': {'mean': 0.31688888888888883, 'sample_sd': 0.4998679840532712, 'n': 3}}


# Task

Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.
