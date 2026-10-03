# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    mean = clip(stats[0], 0.0, 1.0)
    spread = clip(stats[1], 0.0, 1.0)
    best = clip(stats[2], 0.0, 1.0)
    archive = clip(stats[3], 0.0, 1.0)
    success = clip(stats[4], 0.0, 1.0)
    ready = clip(memory[3], 0.0, 1.0)
    old_archive = where(ready > 0.5, memory[0], archive)
    quality = 0.55 * archive + 0.30 * mean + 0.15 * best
    old_quality = where(ready > 0.5, memory[1], quality)
    scale = 0.04 + spread
    innovation = tanh(3.0 * (archive - old_archive) / scale)
    progress = tanh(3.0 * (quality - old_quality) / scale)
    damage = tanh(2.0 * (archive - mean) / scale)
    competitiveness = tanh(3.0 * (success - 0.40))
    difficulty = 1.0 - best
    collapse = exp(-12.0 * spread)
    exploration = difficulty * collapse * exp(-3.0 * abs(progress))
    loss = clip((old_archive - archive) / scale, 0.0, 1.0)
    evidence = 0.30 * competitiveness + 0.35 * innovation + 0.20 * progress - 0.55 * damage + 0.45 * exploration - 0.25 * quality
    pressure = clip(0.78 * memory[2] + 0.22 * evidence, -1.0, 1.0)
    step = 0.18 * tanh(3.0 * pressure) + 0.12 * difficulty * loss
    lower = 0.003 + 0.009 * difficulty
    upper = 0.025 + 0.475 * difficulty * difficulty
    next_sigma = clip(clip(sigma, 0.001, 2.0) * exp(clip(step, -0.5, 0.5)), lower, upper)
    next_quality = 0.65 * old_quality + 0.35 * quality
    next_memory = stack([archive, next_quality, pressure, 1.0])
    return next_sigma, next_memory
# EVOLVE-BLOCK-END