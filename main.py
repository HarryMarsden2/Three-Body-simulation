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

rk4_trajectory = simulate(RK4, f, y, h, n_steps) #simulates the trajectory using RK4 integrator
euler_trajectory = simulate(forward_euler, f, y, h, n_steps) #simulates the trajectory using forward euler integrator

n = len(masses)
'''plottting the trajectory'''
fig, axes = plt.subplots(2, 3, figsize=(13, 5.5),
                         sharex=True, sharey=True)

methods = [("RK4", rk4_trajectory), ("Euler", euler_trajectory)]

for row, (name, trajectory) in enumerate(methods):
    positions = trajectory[:, :2*n].reshape(-1, n, 2)

    for body in range(n):
        ax = axes[row, body]

        x = positions[:, body, 0]
        y = positions[:, body, 1]

        ax.plot(x, y, color="tab:blue", label="Trajectory")

        # Starting and finishing positions
        ax.scatter(x[0], y[0], color="black",
                   s=35, zorder=5, label="Start (t=0)")

        ax.scatter(x[-1], y[-1], color="red",
                   marker="x", s=50, zorder=5,
                   label="End (t=T)")

        # Direction arrows at three points along the trajectory
        for fraction in [0.2, 0.5, 0.8]:
            i = int(fraction * (len(x) - 1))
            j = min(i + max(1, len(x)//40), len(x) - 1)

            ax.annotate("",
                        xy=(x[j], y[j]),
                        xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle="->",
                                        color="tab:blue",
                                        lw=1.5,
                                        mutation_scale=15))

        ax.set_title(f"{name} - Body {body+1}")
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        # Show tick labels on every graph
        ax.tick_params(labelbottom=True, labelleft=True)

# One legend for the entire figure
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center",
           ncol=3, frameon=False)

plt.tight_layout(rect=(0, 0, 1, 0.91))
plt.show()

rk4_error = np.linalg.norm(rk4_trajectory[-1,:2*n] - rk4_trajectory[0,:2*n]) #calculates the error in the final position compared to the initial position
euler_error = np.linalg.norm(euler_trajectory[-1,:2*n] - euler_trajectory[0,:2*n]) #calculates the error in the final position compared to the initial position

print(f"RK4 final position error: {rk4_error}")
print(f"Forward Euler final position error: {euler_error}")