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

N = 100     # Number of neurons in excitatory-excitatory module
F = 17      # Scaling factor for excitatory-excitatory connections    
Dmax = 20   # Maximum conduction delay 
ExNet = IzNetwork(N, Dmax)

W = np.zeros((N, N))
D = np.zeros((N, N))

all_idxs = []
for i in range(N):
    for j in range(N):
        if i == j:
            continue
        all_idxs.append((i, j))
# idx = np.array(random.sample(all_idxs, 1000))
idxs = random.sample(all_idxs, 1000)

# Set connection weight to be 1 for randomly generated connections
# Set random delays for these connections simultaneously. 
for idx in idxs:
    W[idx] = 1*F 
    D[idx] = np.random.randint(1, Dmax+1) # Assume delays have to be integer values in range [1,20], need to double check 

# Sanity check - make sure indexes are the same in both matrices
print(np.where(W != 0)[0] == np.where(D != 0)[0])