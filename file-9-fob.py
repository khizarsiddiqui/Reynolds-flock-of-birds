# Initializing the project
# Flock of birds simulation will be carried out by using the boids simulation model.
# The three core rules of the Boids simulation are Separation, Alignment, and Cohesion. 
import math
import sys, argparse
import numpy as np
from scipy.spatial.distance import squareform, pdist, cdist
import matplotlib.pyplot as plt
from numpy.linalg import norm
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

# method that applies the three rules for boids using the numpy techniques discussed earlier
def applyRules(self):
# apply rule #1: Separation
    D = distMatrix < 25.0
    vel = self.pos*D.sum(axis=1).reshape(self.N, 1) - D.dot(self.pos)
    self.limit(vel, self.maxRuleVel)
# distance threshold for alignment (different from separation)
    D = distMatrix < 50.0
# apply rule #2: Alignment
    vel2 = D.dot(self.vel)
    self.limit(vel2, self.maxRuleVel)
    vel += vel2
# apply rule #3: Cohesion
    vel3 = D.dot(self.pos) - self.pos
    self.limit(vel3, self.maxRuleVel)
    vel += vel3
    return vel

# adding a boid (step 5)
# add a "button press" event handler
cid = fig.canvas.mpl_connect('button_press_event', buttonPress)
def buttonPress(self, event):
# event handler for matplotlib button presses
# left-click to add a boid
    if event.button is 1:
        self.pos = np.concatenate((self.pos,np.array([[event.xdata, event.ydata]])),axis=0)
# generate a random velocity
        angles = 2*math.pi*np.random.rand(1)
        v = np.array(list(zip(np.sin(angles), np.cos(angles))))
        self.vel = np.concatenate((self.vel, v), axis=0)
        self.N += 1

# scattering the boids (step 6)
# right-click to scatter boids
    elif event.button is 3:
# add scattering velocity
        self.vel += 0.1*(self.pos - np.array([[event.xdata, event.ydata]]))
# command line arguments (step 7)
parser = argparse.ArgumentParser(description="Implementing Craig Reynolds's Boids...")
# add arguments
parser.add_argument('--num-boids', dest='N', required=False)
args = parser.parse_args()
# set the initial number of boids
N = 100
if args.N:
    N = int(args.N)
# create boids
boids = Boids(N)
class Boids:
#  The Boids class handles the initialization, updates the animation, and applies the rules.
    def __init__(self, N):
# initialize the Boid simulation
# initial position and velocities (before simulation begins)
        self.pos = [width/2.0, height/2.0] + 10*np.random.rand(2*N).reshape(N, 2)
# normalized random velocities (after simulation starts)
        angles = 2*math.pi*np.random.rand(N)
        self.vel = np.array(list(zip(np.sin(angles), np.cos(angles))))
        self.N = N
# minimum distance of approach
        self.minDist = 25.0
# maximum magnitude of velocities calculated by "rules"
        self.maxRuleVel = 0.03
# maximum magnitude of the final velocity
        self.maxVel = 2.0
def tick(frameNum, pts, beak, boids):
# print frameNum
# update function for animation
    boids.tick(frameNum, pts, beak)
    return pts, beak
def limitVec(self, vec, maxVal):
# limit the magnitude of the 2D vector (considering for one boid at first)
    mag = norm(vec)
    if mag > maxVal:
        vec[0], vec[1] = vec[0]*maxVal/mag, vec[1]*maxVal/mag
def limit(self, X, maxVal):
# limit the magnitude of 2D vectors in array X to maxValue (applies on all boids now)
    for vec in X:
        self.limitVec(vec, maxVal)


