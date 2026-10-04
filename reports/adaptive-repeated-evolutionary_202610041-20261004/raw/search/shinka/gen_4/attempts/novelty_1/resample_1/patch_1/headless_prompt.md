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
    fitness = clip(stats[0], 0.0, 1.0)
    previous = where(memory[3] > 0.5, memory[0], fitness)
    change = fitness - previous
    trend = 0.75 * memory[1] + 0.25 * change
    success_previous = where(memory[3] > 0.5, memory[2], stats[4])
    success = 0.75 * success_previous + 0.25 * stats[4]
    drop = maximum(previous - fitness, 0.0)
    diversity = exp(-8.0 * clip(stats[1], 0.0, 1.0))
    stagnation = 0.5 - 0.5 * tanh(30.0 * trend)
    archive_advantage = maximum(clip(stats[3], 0.0, 1.0) - fitness, 0.0)
    elite_advantage = maximum(clip(stats[2], 0.0, 1.0) - fitness, 0.0)
    difficulty = 1.0 - fitness - 0.6 * archive_advantage - 0.2 * elite_advantage * stagnation
    target = clip(0.025 + 0.4 * difficulty * difficulty + 0.2 * difficulty * diversity * stagnation + 0.8 * drop, 0.025, 0.9)
    adjustment = 0.2 * log(target / clip(sigma, 0.001, 2.0)) + 0.08 * (success - 0.5)
    next_sigma = clip(sigma, 0.001, 2.0) * exp(clip(adjustment, -0.4, 0.4))
    next_memory = stack([0.65 * previous + 0.35 * fitness, trend, success, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: 0.75
profile: adaptive-search; objective: adaptive-active-previous-v1; cache_hit: False; scores: {'combined_score': {'mean': 0.7467916675421926, 'sample_sd': 0.12681382561161117, 'n': 3}, 'active_score': {'mean': 0.6926722239732742, 'sample_sd': 0.035850407828045175, 'n': 3}, 'previous_score': {'mean': 0.800911111111111, 'sample_sd': 0.22814036939601046, 'n': 3}}

```python
# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    fitness = clip(stats[0], 0.0, 1.0)
    previous = where(memory[3] > 0.5, memory[0], fitness)
    change = fitness - previous
    trend = 0.75 * memory[1] + 0.25 * change
    success_previous = where(memory[3] > 0.5, memory[2], stats[4])
    success = 0.75 * success_previous + 0.25 * stats[4]
    drop = maximum(previous - fitness, 0.0)
    diversity = exp(-8.0 * clip(stats[1], 0.0, 1.0))
    stagnation = 0.5 - 0.5 * tanh(30.0 * trend)
    difficulty = 1.0 - fitness
    target = clip(0.025 + 0.4 * difficulty * difficulty + 0.2 * difficulty * diversity * stagnation + 0.8 * drop, 0.025, 0.9)
    adjustment = 0.2 * log(target / clip(sigma, 0.001, 2.0)) + 0.08 * (success - 0.5)
    next_sigma = clip(sigma, 0.001, 2.0) * exp(clip(adjustment, -0.4, 0.4))
    next_memory = stack([0.65 * previous + 0.35 * fitness, trend, success, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: 0.78
profile: adaptive-search; objective: adaptive-active-previous-v1; cache_hit: False; scores: {'combined_score': {'mean': 0.7794444452563921, 'sample_sd': 0.08045044603011407, 'n': 3}, 'active_score': {'mean': 0.705555557179451, 'sample_sd': 0.04104269518368595, 'n': 3}, 'previous_score': {'mean': 0.8533333333333334, 'sample_sd': 0.12843026642241823, 'n': 3}}


# Current program

Here is the current program we are trying to improve (you will need to propose a modification to it below):

```python
# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    fitness = clip(stats[0], 0.0, 1.0)
    previous = where(memory[3] > 0.5, memory[0], fitness)
    change = fitness - previous
    trend = 0.75 * memory[1] + 0.25 * change
    success_previous = where(memory[3] > 0.5, memory[2], stats[4])
    success = 0.75 * success_previous + 0.25 * stats[4]
    drop = maximum(previous - fitness, 0.0)
    diversity = exp(-8.0 * clip(stats[1], 0.0, 1.0))
    stagnation = 0.5 - 0.5 * tanh(30.0 * trend)
    archive_advantage = maximum(clip(stats[3], 0.0, 1.0) - fitness, 0.0)
    difficulty = 1.0 - fitness - 0.6 * archive_advantage
    target = clip(0.025 + 0.4 * difficulty * difficulty + 0.2 * difficulty * diversity * stagnation + 0.8 * drop, 0.025, 0.9)
    adjustment = 0.2 * log(target / clip(sigma, 0.001, 2.0)) + 0.08 * (success - 0.5)
    next_sigma = clip(sigma, 0.001, 2.0) * exp(clip(adjustment, -0.4, 0.4))
    next_memory = stack([0.65 * previous + 0.35 * fitness, trend, success, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END
```

Here are the performance metrics of the program:

Combined score to maximize: 0.82
profile: adaptive-search; objective: adaptive-active-previous-v1; cache_hit: False; scores: {'combined_score': {'mean': 0.8191041678971714, 'sample_sd': 0.10825146840201318, 'n': 3}, 'active_score': {'mean': 0.7054527802387873, 'sample_sd': 0.10030508821487281, 'n': 3}, 'previous_score': {'mean': 0.9327555555555556, 'sample_sd': 0.11647079430452044, 'n': 3}}


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
