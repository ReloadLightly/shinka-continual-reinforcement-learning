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
    spread = clip(stats[1], 0.0, 1.0)
    difficulty = 1.0 - 0.75 * best - 0.25 * mean
    advantage = clip(stats[3] - mean - 0.04, 0.0, 1.0)
    improving = clip(progress / (0.03 + spread), 0.0, 1.0)
    exploration = 0.075 * difficulty * difficulty + 0.18 * stagnation * difficulty * exp(-5.0 * spread)
    target = 0.005 + exploration * exp(-2.5 * advantage - 1.5 * improving) + 0.22 * shock * (0.3 + 0.7 * difficulty)
    success = tanh(4.0 * (stats[4] - 0.5))
    rate = where(target > clip(sigma, 0.001, 2.0), 0.35, 0.30)
    log_width = (1.0 - rate) * log(clip(sigma, 0.001, 2.0)) + rate * log(target) + 0.04 * success * difficulty
    next_sigma = clip(exp(clip(log_width, -7.0, 0.69)), 0.001, 2.0)
    next_mean = 0.8 * old_mean + 0.2 * mean
    next_archive = 0.5 * old_archive + 0.5 * clip(stats[3], 0.0, 1.0)
    next_memory = stack([next_mean, next_archive, stagnation, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END