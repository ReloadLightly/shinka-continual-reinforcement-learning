# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    mean = clip(stats[0], 0.0, 1.0)
    spread = clip(stats[1], 0.0, 1.0)
    archive = clip(stats[3], 0.0, 1.0)
    quality = clip(0.55 * archive + 0.45 * mean - 0.20 * spread, 0.0, 1.0)
    ready = abs(memory[3])
    previous_quality = where(ready > 0.5, memory[0], quality)
    previous_direction = where(ready > 0.5, memory[3], -1.0)
    center = where(ready > 0.5, memory[2], log(0.10))
    response = clip((quality - previous_quality) / (0.025 + 0.15 * spread), -1.0, 1.0)
    estimate = response * previous_direction
    gradient = clip(0.80 * memory[1] + 0.20 * estimate, -1.0, 1.0)
    decline = clip(previous_quality - quality - 0.08, 0.0, 1.0)
    competitiveness = tanh(4.0 * (clip(stats[4], 0.0, 1.0) - 0.5))
    difficulty = clip(1.0 - 0.70 * clip(stats[2], 0.0, 1.0) - 0.30 * mean, 0.0, 1.0)
    stalled = exp(-40.0 * abs(quality - previous_quality))
    advantage = clip(archive - mean - 0.05, 0.0, 1.0)
    exploration = 0.10 * difficulty * difficulty + 0.14 * stalled * difficulty * exp(-6.0 * spread)
    target = clip(0.008 + exploration * exp(-3.0 * advantage) + 0.20 * decline, 0.008, 0.30)
    adaptation = 0.04 * gradient + 0.025 * competitiveness * difficulty
    next_center = clip(0.75 * center + 0.25 * log(target) + adaptation, -4.83, -1.20)
    direction = -previous_direction
    probe = 0.08 + 0.12 * difficulty
    log_width = clip(next_center + probe * direction, -6.90, 0.69)
    next_sigma = exp(log_width)
    next_memory = stack([quality, gradient, next_center, direction])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END