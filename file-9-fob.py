# Initializing the project
# Flock of birds simulation will be carried out by using the boids simulation model.
# The three core rules of the Boids simulation are Separation, Alignment, and Cohesion. 
import math
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Computing the Position and Velocities of the Boids (step 1)

N = 10
width, height = 640, 480
pos = [width/2.0, height/2.0] + 10*np.random.rand(2*N).reshape(N, 2)
angles = 2*math.pi*np.random.rand(N)
vel = np.array(list(zip(np.sin(angles), np.cos(angles)))) # This joins two lists into a list of tuples.
# print(pos,angles,vel)

# Setting Boundary Conditions (step 2)
def applyBC(self):
# apply boundary conditions
    deltaR = 2.0
    for coord in self.pos:
        if coord[0] > width + deltaR:
            coord[0] = - deltaR
        if coord[0] < - deltaR:
            coord[0] = width + deltaR
        if coord[1] > height + deltaR:
            coord[1] = - deltaR
        if coord[1] < - deltaR:
            coord[1] = height + deltaR

# drawing boids (step 3)
# Plotting the Boid’s Body and Head
# create boids
# boids = Boids(N)
fig = plt.figure()
ax = plt.axes(xlim=(0, width), ylim=(0, height))
pts, = ax.plot([], [], markersize=10, c='k', marker='o', ls='None')
beak, = ax.plot([], [], markersize=4, c='r', marker='o', ls='None')
anim = animation.FuncAnimation(fig, tick, fargs=(pts, beak, boids),interval=50)
# Updating the Boid’s Position
vec = self.pos + 10*self.vel/self.maxVel
beak.set_data(vec.reshape(2*self.N)[::2], vec.reshape(2*self.N)[1::2])

# Applying the Rules of the Boids (step 4)
def test2(pos, radius):
# get distance matrix
    distMatrix = squareform(pdist(pos))
# apply threshold
    D = distMatrix < radius
# compute velocity
    vel = pos*D.sum(axis=1).reshape(N, 1) - D.dot(pos) # The D.sum() method sums the True values in the matrix in a column-wise fashion
    # D.dot() is just the dot product (multiplication) of the matrix and the position vector.
    return vel


