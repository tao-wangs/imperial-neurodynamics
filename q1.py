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


N_ex = 100     # Number of neurons in excitatory module
N_in = 200     # Number of neurons in inhibitory module 
N_mod = 8      # Number of modules in the modular network 
N_net = 1000   # Number of neurons in the modular network 

F_ex_in = 50
F_in_ex = 2

# 2. Extend to 8 populations

def ExcitatoryToExcitatory(N):
    '''
    Generates excitatory-to-excitatory connections according to specification. 

    N - The number of neurons in the excitatory community. 

    W - The connectivity matrix for the excitatory-to-excitatory community. 
    D - The conduction delay matrix for the excitatory-to-excitatory community. 
    '''
    
    F = 17      # Scaling factor for excitatory-excitatory connections    
    Dmax = 20   # Maximum conduction delay 
    
    W = np.zeros((N,N))
    D = np.zeros((N,N))

    all_idxs = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            all_idxs.append((i, j))

    idxs = random.sample(all_idxs, 1000)

    # Set connection weight to be 1 for randomly generated connections
    # Set random delays for these connections simultaneously. 
    for idx in idxs:
        W[idx] = 1*F 
        D[idx] = np.random.randint(1, Dmax+1) # Assume delays have to be integer values in range [1,20], need to double check 
    
    return W, D

# 3. Create 200x200 inhibitory population and connect to excitatory populations (1000x1000 weight matrix)
  
W = np.zeros([1000, 1000])
D = np.ones([1000, 1000])

# Update excitatory-excitatory blocks
for i in range(0, 800, N_ex):
    Wblock, Dblock = ExcitatoryToExcitatory(N)
    W[i:i+N_ex, i:i+N_ex] = Wblock
    D[i:i+N_ex, i:i+N_ex] = Dblock 

# Update excitatory-to-inhibitory block
for j in range(N_mod*N_ex, N_net):
    i = np.random.randint(0, N_mod) 
    neurons = random.sample(range(N_ex), 4)
    i = neurons + N_ex*i
    W[i,j] = 50*random.uniform(0, 1)

# Update inhibitory
in_to_ex_block = F_in_ex * np.random.uniform(0, 1, (N_in, N_mod*N_ex)) 
in_to_in_block = np.random.uniform(0, 1, (N_in, N_in))
W[800:, :] = np.column_stack((in_to_ex_block, in_to_in_block))

# Inhibitory neurons cannot connect to themselves, thus set the connection weight to 0. 
W[range(800, 1000),range(800, 1000)] = 0 

net = IzNetwork(N_net, 20)
a = 0.02 * np.ones([N_net])
b = np.column_stack((0.2*np.ones([800]), 0.25*np.ones([200])))
c = -65 * np.ones([N_net])
d = np.column_stack((8*np.ones([800]), 2*np.ones([200])))
net.setParameters(a, b, c, d)
net.setWeights(W)
net.setDelays(D)
# 5. Rewiring process
# 6. Connectivitiy matrix, raster plot and mean firing rate over each probability p.