import numpy as np
from functools import partial

def constant_g(t, g_max):
    return np.ones_like(t) * g_max

def triangular_g(t, g_max):
    g_min = 0.85
    return g_max - 2 * np.abs(t - .5) * (g_max-g_min)

def inverse_triangular_g(t, g_max):
    g_min = .01
    return g_min - 2 * np.abs(t - .5) * (g_min-g_max)

def decreasing_g(t, g_max):
    g_min = .1
    return g_max - np.square(t) * (g_max-g_min)

diffusivity_schedules = {
    "constant": constant_g,
    "triangular": triangular_g,
    "inverse_triangular": inverse_triangular_g,
    "decreasing": decreasing_g,
}

def get_diffusivity_schedule(schedule, g_max):
    return partial(diffusivity_schedules[schedule], g_max=g_max)
