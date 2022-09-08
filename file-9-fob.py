# Initializing the project
# Flock of birds simulation will be carried out by using the boids simulation model.
# The three core rules of the Boids simulation are Separation, Alignment, and Cohesion. 

# Computing the Position and Velocities of the Boids (step 1)
import math
import numpy as np
N = 10
width, height = 640, 480
pos = [width/2.0, height/2.0] + 10*np.random.rand(2*N).reshape(N, 2)
angles = 2*math.pi*np.random.rand(N)
vel = np.array(list(zip(np.sin(angles), np.cos(angles)))) # This joins two lists into a list of tuples.
# print(pos,angles,vel)