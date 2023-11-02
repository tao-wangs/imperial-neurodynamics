import numpy as np
import random
import matplotlib.pyplot as plt
from iznetwork import IzNetwork

# Plan
# 1. Create 100x100 excitatory module, with weights and delays 
# 2. Extend to 8 populations
# 3. Create 200x200 inhibitory population and connect to excitatory populations (1000x1000 weight matrix)
# 5. Rewiring process
# 6. Connectivitiy matrix, raster plot and mean firing rate over each probability p.


# Implementations
# 1. Create 100x100 excitatory module, with weights and delays 

N = 100
Dmax = 20 
ExNet = IzNetwork(N, Dmax)

W = np.zeros((N, N))
D = np.zeros((N, N))

all_idxs = []
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        all_idxs.append((i, j))
idx = np.array(random.sample(all_idxs, 1000))


