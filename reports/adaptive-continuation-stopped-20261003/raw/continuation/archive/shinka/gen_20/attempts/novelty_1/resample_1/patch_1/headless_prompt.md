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

Redesign the program with a different structural approach while potentially using similar core concepts.
Focus on changing the overall architecture, data flow, or program organization.
You MUST respond using a short summary name, description and the full code:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
Describe the structural changes you are making and how they improve the program's performance, maintainability, or efficiency.
</DESCRIPTION>

<CODE>
```{language}
# The structurally redesigned program here.
```
</CODE>

* Keep the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the code.
* Focus on changing the program's structure: modularization, data flow, control flow, or architectural patterns.
* The core problem-solving approach may be similar but organized differently.
* Ensure the same inputs and outputs are maintained.
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
    spread = clip(stats[1], 0.0, 1.0)
    best = clip(stats[2], 0.0, 1.0)
    archive = clip(stats[3], 0.0, 1.0)
    success = clip(stats[4], 0.0, 1.0)
    ready = clip(memory[3], 0.0, 1.0)
    old_mean = where(ready > 0.5, memory[0], mean)
    old_archive = where(ready > 0.5, memory[1], archive)
    progress = mean - old_mean
    stalled = exp(-32.0 * abs(progress))
    stagnation = clip(0.82 * memory[2] + 0.18 * stalled, 0.0, 1.0)
    difficulty = clip(1.0 - 0.65 * archive - 0.35 * mean, 0.0, 1.0)
    collapse = exp(-4.0 * spread)
    advantage = clip(archive - mean - 0.04, 0.0, 1.0)
    loss = ready * clip(old_archive - archive - 0.06, 0.0, 1.0)
    improving = clip(progress / (0.04 + spread), 0.0, 1.0)
    exploration = 0.060 * difficulty * difficulty + 0.16 * stagnation * difficulty * collapse
    target = 0.012 + exploration * exp(-2.5 * advantage - 1.2 * improving) + 0.12 * loss
    competitiveness = tanh(4.0 * (success - 0.45))
    adjusted_target = clip(target * exp(0.18 * competitiveness * difficulty), 0.003, 0.40)
    width = clip(sigma, 0.001, 2.0)
    rate = where(adjusted_target > width, 0.45, 0.20)
    log_width = (1.0 - rate) * log(width) + rate * log(adjusted_target)
    next_sigma = clip(exp(clip(log_width, -6.9, 0.69)), 0.001, 2.0)
    next_mean = 0.75 * old_mean + 0.25 * mean
    next_archive = 0.40 * old_archive + 0.60 * archive
    next_memory = stack([next_mean, next_archive, stagnation, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: 0.45
profile: adaptive-search; objective: adaptive-active-previous-v1; cache_hit: False; scores: {'combined_score': {'mean': 0.4548708347611957, 'sample_sd': 0.4247843207756544, 'n': 3}, 'active_score': {'mean': 0.5626972250779471, 'sample_sd': 0.29515376056010395, 'n': 3}, 'previous_score': {'mean': 0.34704444444444443, 'sample_sd': 0.5654763825830919, 'n': 3}}

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


# Task

Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.
