# System Instructions

Optimize two static GA settings for continual CartPole using the fixed evaluator.
The candidate must contain exactly def get_ga_config(): with no arguments,
decorators, annotations, imports, or additional statements. Its only statement
must return a literal dictionary containing sigma and elite_ratio.
sigma must be finite in [0.001, 2.0]; elite_ratio in [0.05, 0.95].
Preserve the EVOLVE-BLOCK markers. Larger combined_score is better.
This is configuration search, not evolution of a learning algorithm.
The smoke profile only verifies execution. Use the search profile for proposals.
Return the requested patch directly; do not use tools or edit files yourself.

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
def get_ga_config():
    return {"sigma": 0.077, "elite_ratio": 0.075}
# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: 0.84
profile: search; method: ga; seeds_completed: 3; sigma: 0.08; elite_ratio: 0.07; smoke_validation_only: False; mean_return: 420.49

```python
# EVOLVE-BLOCK-START
def get_ga_config():
    return {"sigma": 0.065, "elite_ratio": 0.075}


# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: 0.86
profile: search; method: ga; seeds_completed: 3; sigma: 0.07; elite_ratio: 0.07; smoke_validation_only: False; mean_return: 429.20


# Current program

Here is the current program we are trying to improve (you will need to propose a modification to it below):

```python
# EVOLVE-BLOCK-START
def get_ga_config():
    return {"sigma": 0.08, "elite_ratio": 0.075}
# EVOLVE-BLOCK-END
```

Here are the performance metrics of the program:

Combined score to maximize: 0.87
profile: search; method: ga; seeds_completed: 3; sigma: 0.08; elite_ratio: 0.07; smoke_validation_only: False; mean_return: 436.05


# Instructions

Make sure that the changes you propose are consistent with each other. For example, if you refer to a new config variable somewhere, you should also propose a change to add that variable.

Note that the changes you propose will be applied sequentially, so you should assume that the previous changes have already been applied when writing the SEARCH block.

# Task

Suggest a new idea to improve the performance of the code that is inspired by your expert knowledge of the considered subject.
Your goal is to maximize the `combined_score` of the program.
Describe each change with a SEARCH/REPLACE block.

IMPORTANT: Do not rewrite the entire program - focus on targeted improvements.
