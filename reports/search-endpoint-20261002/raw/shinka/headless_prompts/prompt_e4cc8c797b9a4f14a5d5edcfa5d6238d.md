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

Design a completely different algorithm approach to solve the same problem.
Ignore the current implementation and think of alternative algorithmic strategies that could achieve better performance.
You MUST respond using a short summary name, description and the full code:

<NAME>
A shortened name summarizing the code you are proposing. Lowercase, no spaces, underscores allowed.
</NAME>

<DESCRIPTION>
Explain the completely different algorithmic approach you are taking and why it should perform better than the current implementation.
</DESCRIPTION>

<CODE>
```{language}
# The completely new algorithm implementation here.
```
</CODE>

* Keep the markers "EVOLVE-BLOCK-START" and "EVOLVE-BLOCK-END" in the code.
* Your algorithm should solve the same problem but use a fundamentally different approach.
* Ensure the same inputs and outputs are maintained.
* Think outside the box - consider different data structures, algorithms, or paradigms.
* Use the <NAME>, <DESCRIPTION>, and <CODE> delimiters to structure your response. It will be parsed afterwards.

# Previous Messages

[]

# User Request

Here are the performance metrics of a set of previously implemented programs:

# Prior programs

```python
# EVOLVE-BLOCK-START
def get_ga_config():
    return {"sigma": 0.10, "elite_ratio": 0.15}


# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: 0.78
profile: search; method: ga; seeds_completed: 3; sigma: 0.10; elite_ratio: 0.15; smoke_validation_only: False; mean_return: 392.08

```python
# EVOLVE-BLOCK-START
def get_ga_config():
    return {"sigma": 0.035, "elite_ratio": 0.05}


# EVOLVE-BLOCK-END
```

Performance metrics:
Combined score to maximize: 0.82
profile: search; method: ga; seeds_completed: 3; sigma: 0.04; elite_ratio: 0.05; smoke_validation_only: False; mean_return: 408.50


# Current program

Here is the current program we are trying to improve (you will need to propose a new program with the same inputs and outputs as the original program, but with improved internal implementation):

```python
# EVOLVE-BLOCK-START
def get_ga_config():
    return {"sigma": 0.05, "elite_ratio": 0.075}
# EVOLVE-BLOCK-END
```

Here are the performance metrics of the program:

Combined score to maximize: 0.84
profile: search; method: ga; seeds_completed: 3; sigma: 0.05; elite_ratio: 0.07; smoke_validation_only: False; mean_return: 422.43


# Task

Rewrite the program to improve its performance on the specified metrics.
Provide the complete new program code.

IMPORTANT: Make sure your rewritten program maintains the same inputs and outputs as the original program, but with improved internal implementation.
