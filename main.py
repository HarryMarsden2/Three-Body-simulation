import numpy as np
from matplotlib import pyplot as plt

from physics import derivatives
from integrators import RK4, forward_euler
from simulation import simulate

'''initial conditions for the stable figure 8 orbit'''
initial_positions = np.array([
    [0.97000436, -0.24308753],
    [-0.97000436, 0.24308753],
    [0.0, 0.0]
])
# DO NOT CHANGE THESE INITIAL CONDITIONS. They are known to produce a stable figure-8 orbit.
initial_velocities = np.array([
    [0.466203685, 0.43236573],
    [0.466203685, 0.43236573],
    [-0.93240737, -0.86473146]
])

positions = initial_positions.copy()
velocities = initial_velocities.copy()

'''Time period of this orbit is T=6.32591398 (known from literature)'''
masses = np.array([1.0, 1.0, 1.0])
G = 1.0
period = 6.32591398                   # DO NOT CHANGE
n_orbits = 1                         # CAN CHANGE: number of orbits to simulate
steps_per_orbit = 1000                # CAN CHANGE: number of time steps per orbit
n_steps = n_orbits * steps_per_orbit  # total number of time steps
h = period / steps_per_orbit          # time step size
T = n_orbits * period                 # total simulation time


y = np.concatenate([positions.flatten(),
                     velocities.flatten()]) #makes a 1D array of positions and velocities

f = lambda y, t: derivatives(y, masses, G) #creates a quick function that takes in y and t and returns the derivatives of y

trajectory = simulate(RK4, f, y, h, n_steps) #simulates the trajectory using RK4 integrator


'''plottting the trajectory'''
n = len(masses)
orbit_positions = trajectory[:, :2*n].reshape(-1, n, 2) #reshapes the trajectory to get the positions of the bodies

for i in range(n):
    plt.plot(orbit_positions[:, i, 0],
             orbit_positions[:, i, 1],
             label=f"body {i+1}")

plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

error = np.linalg.norm(trajectory[-1,:2*n] - trajectory[0,:2*n]) #calculates the error in the final position compared to the initial position
print(f"Final position error: {error}")