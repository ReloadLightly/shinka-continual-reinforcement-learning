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

Performance metrics:
Combined score to maximize: 0.42
profile: adaptive-search; objective: adaptive-active-previous-v1; cache_hit: False; scores: {'combined_score': {'mean': 0.4157583345519172, 'sample_sd': 0.4150146087113926, 'n': 3}, 'active_score': {'mean': 0.5146277802149455, 'sample_sd': 0.33287258192740826, 'n': 3}, 'previous_score': {'mean': 0.31688888888888883, 'sample_sd': 0.4998679840532712, 'n': 3}}

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


# Current program

Here is the current program we are trying to improve (you will need to propose a modification to it below):

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
    difficulty = 1.0 - mean
    target = 0.015 + 0.10 * difficulty * difficulty + 0.08 * stagnation * (1.0 - best) + 0.35 * shock
    success = tanh(4.0 * (stats[4] - 0.5))
    log_width = 0.75 * log(clip(sigma, 0.001, 2.0)) + 0.25 * log(target) + 0.06 * success * difficulty
    next_sigma = clip(exp(clip(log_width, -7.0, 0.69)), 0.001, 2.0)
    next_mean = 0.8 * old_mean + 0.2 * mean
    next_archive = 0.5 * old_archive + 0.5 * clip(stats[3], 0.0, 1.0)
    next_memory = stack([next_mean, next_archive, stagnation, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END
```

Here are the performance metrics of the program:

Combined score to maximize: 0.40
profile: adaptive-search; objective: adaptive-active-previous-v1; cache_hit: False; scores: {'combined_score': {'mean': 0.4031319455636872, 'sample_sd': 0.3891896796627328, 'n': 3}, 'active_score': {'mean': 0.5056861133495967, 'sample_sd': 0.3207696961513628, 'n': 3}, 'previous_score': {'mean': 0.3005777777777778, 'sample_sd': 0.4819368761493486, 'n': 3}}


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
