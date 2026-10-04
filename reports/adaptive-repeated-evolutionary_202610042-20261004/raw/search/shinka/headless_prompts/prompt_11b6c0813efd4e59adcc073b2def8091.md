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

You MUST respond using an edit name, description, and the exact SEARCH/REPLACE diff format shown below to indicate changes:

<NAME>
A shortened name summarizing the edit you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
A description and argumentation process of the edit you are proposing.
</DESCRIPTION>

<DIFF>
<<<<<<< SEARCH
# Original code to find and replace (must match exactly including indentation)
=======
# New replacement code
>>>>>>> REPLACE

</DIFF>


Example of a valid diff format:
<DIFF>
<<<<<<< SEARCH
for i in range(m):
    for j in range(p):
        for k in range(n):
            C[i, j] += A[i, k] * B[k, j]
=======
# Reorder loops for better memory access pattern
for i in range(m):
    for k in range(n):
        for j in range(p):
            C[i, j] += A[i, k] * B[k, j]
>>>>>>> REPLACE

</DIFF>

* You may only modify text that lies below a line containing "EVOLVE-BLOCK-START" and above the next "EVOLVE-BLOCK-END". Everything outside those markers is read-only.
* Do not repeat the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the SEARCH/REPLACE blocks.  
* Every block’s SEARCH section must be copied **verbatim** from the current file, including indentation.
* You can propose multiple independent edits. SEARCH/REPLACE blocks follow one after another. DO NOT ADD ANY OTHER TEXT BETWEEN THESE BLOCKS.
* Make sure the file still runs after your changes.

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
    success = clip(stats[4], 0.0, 1.0)
    ready = clip(memory[3], 0.0, 1.0)
    change = ready * (mean - memory[0])
    progress = 0.8 * memory[1] + 0.2 * change
    success_avg = where(memory[3] > 0.5, 0.75 * memory[2] + 0.25 * success, success)
    gap = maximum(0.0, best - mean)
    stalled = exp(-clip(40.0 * abs(progress), 0.0, 20.0)) * (1.0 - best)
    decline = maximum(0.0, -change - 0.08)
    archive_loss = maximum(0.0, mean - clip(stats[3], 0.0, 1.0))
    reheat = clip(2.0 * decline + 0.3 * archive_loss, 0.0, 0.5)
    success_target = 0.3 + 0.3 * mean
    rate = clip(0.12 * (success_avg - success_target) + 0.045 * stalled - 0.1 * gap - 0.4 * progress, -0.15, 0.15)
    floor = 0.008 + 0.06 * (1.0 - mean) * (1.0 - mean)
    adapted = sigma * exp(rate + reheat)
    next_sigma = clip(maximum(adapted, floor + 0.15 * reheat), 0.008, 0.8)
    next_memory = stack([mean, progress, success_avg, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: 0.68
profile: adaptive-search; objective: adaptive-active-previous-v1; cache_hit: False; scores: {'combined_score': {'mean': 0.6793666682150629, 'sample_sd': 0.05970961167313528, 'n': 3}, 'active_score': {'mean': 0.6130444475412369, 'sample_sd': 0.07707704669880819, 'n': 3}, 'previous_score': {'mean': 0.7456888888888888, 'sample_sd': 0.0665097263842192, 'n': 3}}

```python
# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    mean = clip(stats[0], 0.0, 1.0)
    best = clip(stats[2], 0.0, 1.0)
    success = clip(stats[4], 0.0, 1.0)
    ready = clip(memory[3], 0.0, 1.0)
    change = ready * (mean - memory[0])
    progress = 0.8 * memory[1] + 0.2 * change
    success_avg = where(memory[3] > 0.5, 0.75 * memory[2] + 0.25 * success, success)
    gap = maximum(0.0, best - mean)
    stalled = exp(-clip(40.0 * abs(progress), 0.0, 20.0)) * (1.0 - best)
    decline = maximum(0.0, -change - 0.08)
    archive_loss = maximum(0.0, mean - clip(stats[3], 0.0, 1.0))
    reheat = clip(2.0 * decline + 0.3 * archive_loss, 0.0, 0.5)
    rate = clip(0.12 * (success_avg - 0.3) + 0.045 * stalled - 0.1 * gap - 0.4 * progress, -0.15, 0.15)
    floor = 0.008 + 0.06 * (1.0 - mean) * (1.0 - mean)
    adapted = sigma * exp(rate + reheat)
    next_sigma = clip(maximum(adapted, floor + 0.15 * reheat), 0.008, 0.8)
    next_memory = stack([mean, progress, success_avg, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: 0.72
profile: adaptive-search; objective: adaptive-active-previous-v1; cache_hit: False; scores: {'combined_score': {'mean': 0.7177958356340727, 'sample_sd': 0.09012825412532276, 'n': 3}, 'active_score': {'mean': 0.6405250046014785, 'sample_sd': 0.06545121253534542, 'n': 3}, 'previous_score': {'mean': 0.7950666666666667, 'sample_sd': 0.19115243480880206, 'n': 3}}


# Current program

Here is the current program we are trying to improve (you will need to propose a modification to it below):

```python
# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    mean = clip(stats[0], 0.0, 1.0)
    best = clip(stats[2], 0.0, 1.0)
    success = clip(stats[4], 0.0, 1.0)
    ready = clip(memory[3], 0.0, 1.0)
    change = ready * (mean - memory[0])
    progress = 0.8 * memory[1] + 0.2 * change
    success_avg = where(memory[3] > 0.5, 0.75 * memory[2] + 0.25 * success, success)
    gap = maximum(0.0, best - mean)
    stalled = exp(-clip(40.0 * abs(progress), 0.0, 20.0)) * (1.0 - best)
    decline = maximum(0.0, -change - 0.08)
    archive_loss = maximum(0.0, mean - clip(stats[3], 0.0, 1.0))
    reheat = clip(2.0 * decline + 0.3 * archive_loss, 0.0, 0.5)
    success_target = 0.3
    rate = clip(0.12 * (success_avg - success_target) + 0.045 * stalled - 0.1 * gap - 0.4 * progress, -0.15, 0.15)
    collapsed = exp(-clip(30.0 * stats[1], 0.0, 20.0)) * (1.0 - best)
    floor = 0.008 + 0.06 * (1.0 - mean) * (1.0 - mean) + 0.04 * collapsed
    adapted = sigma * exp(rate + reheat)
    next_sigma = clip(maximum(adapted, floor + 0.15 * reheat), 0.008, 0.8)
    next_memory = stack([mean, progress, success_avg, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END
```

Here are the performance metrics of the program:

Combined score to maximize: 0.72
profile: adaptive-search; objective: adaptive-active-previous-v1; cache_hit: False; scores: {'combined_score': {'mean': 0.7177958356340727, 'sample_sd': 0.09012825412532276, 'n': 3}, 'active_score': {'mean': 0.6405250046014785, 'sample_sd': 0.06545121253534542, 'n': 3}, 'previous_score': {'mean': 0.7950666666666667, 'sample_sd': 0.19115243480880206, 'n': 3}}


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
