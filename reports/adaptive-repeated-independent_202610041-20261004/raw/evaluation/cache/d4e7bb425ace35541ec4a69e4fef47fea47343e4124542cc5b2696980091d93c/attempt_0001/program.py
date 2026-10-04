# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    fitness = clip(stats[0], 0.0, 1.0)
    archive = clip(stats[3], 0.0, 1.0)
    progress = memory[3] * (fitness - memory[0])
    shock = memory[3] * maximum(memory[1] - archive, 0.0)
    trend = 0.8 * memory[2] + 0.2 * progress
    plateau = exp(-40.0 * maximum(trend, 0.0))
    deficit = 1.0 - fitness
    target = 0.025 + 0.45 * deficit * deficit + 0.2 * plateau * deficit + 0.7 * shock
    correction = 0.08 * (stats[4] - 0.35)
    step = clip(0.18 * log(target / maximum(sigma, 0.001)) + correction, -0.3, 0.3)
    next_sigma = clip(sigma * exp(step), 0.001, 2.0)
    mean_memory = where(memory[3] < 0.5, fitness, 0.85 * memory[0] + 0.15 * fitness)
    archive_memory = where(memory[3] < 0.5, archive, 0.85 * memory[1] + 0.15 * archive)
    next_memory = stack([mean_memory, archive_memory, trend, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END