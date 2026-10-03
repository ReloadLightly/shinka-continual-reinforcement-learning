# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    mean = clip(stats[0], 0.0, 1.0)
    spread = clip(stats[1], 0.0, 1.0)
    best = clip(stats[2], 0.0, 1.0)
    archive = clip(stats[3], 0.0, 1.0)
    success = clip(stats[4], 0.0, 1.0)
    ready = clip(memory[3], 0.0, 1.0)
    mean_reference = where(ready > 0.5, memory[0], mean)
    archive_reference = where(ready > 0.5, memory[1], archive)
    progress = tanh((mean - mean_reference) / (0.025 + spread))
    loss = ready * clip(archive_reference - archive - 0.06, 0.0, 1.0)
    stalled = exp(-5.0 * abs(progress))
    pressure_signal = stalled * (1.0 - best) + 0.35 * clip(0.45 - success, 0.0, 1.0)
    pressure = clip(0.80 * memory[2] + 0.20 * pressure_signal, 0.0, 1.0)
    difficulty = clip(1.0 - 0.70 * best - 0.30 * mean, 0.0, 1.0)
    advantage = clip(archive - mean - 0.04, 0.0, 1.0)
    refinement = 0.008 + 0.065 * difficulty * difficulty
    exploration = 0.035 + 0.20 * difficulty * difficulty
    gate = clip(2.0 * pressure * exp(-4.0 * spread) + 2.5 * loss, 0.0, 1.0)
    blended_log = (1.0 - gate) * log(refinement) + gate * log(exploration)
    feedback = 0.12 * tanh(4.0 * (success - 0.45)) * difficulty
    target_log = blended_log + feedback - 1.5 * advantage - 0.25 * maximum(progress, 0.0)
    target = clip(exp(clip(target_log, -6.0, 0.0)) + 0.12 * loss, 0.003, 0.40)
    width = clip(sigma, 0.001, 2.0)
    rate = where(target > width, 0.45, 0.30)
    next_sigma = exp((1.0 - rate) * log(width) + rate * log(target))
    next_mean = mean_reference + 0.25 * (mean - mean_reference)
    next_archive = archive_reference + 0.65 * (archive - archive_reference)
    next_memory = stack([next_mean, next_archive, pressure, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END