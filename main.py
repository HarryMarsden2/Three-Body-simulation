import numpy as np
from matplotlib import pyplot as plt

from physics import derivatives
from integrators import RK4, forward_euler

'''initial conditions for the stable figure 8 orbit'''
initial_positions = np.array([
    [0.97000436, -0.24308753],
    [-0.97000436, 0.24308753],
    [0.0, 0.0]
])

initial_velocities = np.array([
    [0.466203685, 0.43236573],
    [0.466203685, 0.43236573],
    [-0.93240737, -0.86473146]
])

positions = initial_positions.copy()
velocities = initial_velocities.copy()

masses = np.array([1.0, 1.0, 1.0])
G = 1.0

y = np.concatenate([positions.flatten(),
                     velocities.flatten()]) #makes a 1D array of positions and velocities

f = lambda y, t: derivatives(y, masses, G) #creates a quick function that takes in y and t and returns the derivatives of y

