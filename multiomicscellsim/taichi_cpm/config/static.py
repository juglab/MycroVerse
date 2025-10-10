
"""
    These static parameters are needed because Taichi doesn't support dynamic lists
    when called from kernels and fuctions, so we need to know beforehand the maximum
    values associated with lists of constraints and enegy terms
"""

MAX_ENERGY_TERMS = 10