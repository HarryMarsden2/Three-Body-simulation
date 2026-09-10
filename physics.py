import numpy as np

def acceleration(positions, masses, G):
    accelerations = np.zeros_like(positions)

    for i in range(len(positions)):
        for j in range(len(positions)):
            if i != j:
                r = positions[j] - positions[i]
                r_norm = np.linalg.norm(r)
                contribution = G * masses[j] * r / r_norm**3
                accelerations[i] += contribution
    return accelerations

'''dy/dt = f(y,t) where y = [x1,y1,x2,y2,x3,y3,vx1,vy1,vx2,vy2,vx3,vy3]'''
def derivatives(y,masses,G):
    n = len(masses)
    positions = y[:2*n].reshape((n, 2))    #takes positions from y and reshapes them into a 2D array
    velocities = y[2*n:].reshape((n, 2))   #takes velocities from y and reshapes them into a 2D array

    accelerations = acceleration(positions, masses, G)
    return np.concatenate([velocities, accelerations])
