import numpy as np

def simulate(integrator, f, y0, h, n_steps):
    y = y0.copy()

    trajectory = np.zeros((n_steps + 1, len(y)))
    trajectory[0] = y

    for i in range(n_steps):
        t = i * h
        y = integrator(f, y, h, t)
        trajectory[i + 1] = y

    return trajectory