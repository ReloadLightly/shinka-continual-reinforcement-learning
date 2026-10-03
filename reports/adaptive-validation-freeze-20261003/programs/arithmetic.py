# EVOLVE-BLOCK-START
def update_sigma(sigma, stats, memory):
    return sigma * exp(0.1 * (stats[4] - 0.5)), memory  # noqa: F821 - grammar operation
# EVOLVE-BLOCK-END
