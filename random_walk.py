#! /usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


no_steps = 10000
coords = np.zeros((no_steps,3))

def get_step(coord):
    step = np.random.rand(3)-0.5
    step /= np.linalg.norm(step)
    return coord+step

for t in range(1,no_steps):
    coords[t] = get_step(coords[t-1])

r_2 = np.linalg.norm(coords,axis=1)**2
D = np.mean(r_2)/(6*no_steps)
D_array = np.cumsum(r_2)/(6*(np.arange(1,1+no_steps)**2))


print(D)
print(D_array[-1])

plt.plot(D_array)
#plt.plot(coords[:,0],coords[:,1])
plt.show()


