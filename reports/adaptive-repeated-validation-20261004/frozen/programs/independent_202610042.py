# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    mean = clip(stats[0], 0.0, 1.0)
    archive = clip(stats[3], 0.0, 1.0)
    success = clip(stats[4], 0.0, 1.0)
    ready = clip(memory[3], 0.0, 1.0)
    progress = mean - memory[0]
    decline = maximum(memory[1] - archive, 0.0) * ready
    shock = clip(4.0 * decline, 0.0, 1.0)
    stalled = where(progress < 0.003, 1.0, 0.0) * ready
    plateau = 0.9 * memory[2] + 0.1 * stalled
    quality = 0.6 * mean + 0.4 * archive
    remaining = 1.0 - quality
    target = 0.025 + 0.32 * remaining * remaining + 0.12 * plateau * remaining + 0.6 * shock
    safe_sigma = clip(sigma, 0.001, 2.0)
    adjustment = 0.16 * log(target / safe_sigma) + 0.12 * (success - 0.25)
    next_sigma = clip(safe_sigma * exp(clip(adjustment, -0.35, 0.35)), 0.001, 2.0)
    next_mean = ready * (0.85 * memory[0] + 0.15 * mean) + (1.0 - ready) * mean
    next_archive = ready * (0.85 * memory[1] + 0.15 * archive) + (1.0 - ready) * archive
    next_memory = stack([next_mean, next_archive, plateau, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END
