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