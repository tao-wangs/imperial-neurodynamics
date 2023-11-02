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

# 2. Extend to 8 populations

def ExcitatoryToExcitatory(N):
    '''
    Generates excitatory-to-excitatory connections according to specification. 

    N - The number of neurons in the excitatory community. 

    W - The connectivity matrix for the excitatory-to-excitatory community. 
    D - The conduction delay matrix for the excitatory-to-excitatory community. 
    '''

    N = 100     # Number of neurons in excitatory-excitatory module
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
    # idx = np.array(random.sample(all_idxs, 1000))
    idxs = random.sample(all_idxs, 1000)

    # Set connection weight to be 1 for randomly generated connections
    # Set random delays for these connections simultaneously. 
    for idx in idxs:
        W[idx] = 1*F 
        D[idx] = np.random.randint(1, Dmax+1) # Assume delays have to be integer values in range [1,20], need to double check 
    
    return W, D

# 3. Create 200x200 inhibitory population and connect to excitatory populations (1000x1000 weight matrix)

# Option 1: One giant block construction
exWeights, delays = [ExcitatoryToExcitatory(N) for _ in range(8)]

oneBlock  = np.ones((N, N))
zeroBlock = np.zeros((N, N))

# Block [i,j] is the connection from population i to population j
W = np.bmat([[ex1,       zeroBlock,  zeroBlock,  zeroBlock, zeroBlock, zeroBlock, zeroBlock, zeroBlock, ex-to-inhib],
             [zeroBlock, ex2,        zeroBlock,  zeroBlock, zeroBlock, zeroBlock, zeroBlock, zeroBlock, ex-to-inhib],
             [zeroBlock, zeroBlock,  ex3,        zeroBlock, zeroBlock, zeroBlock, zeroBlock, zeroBlock, ex-to-inhib],
             [zeroBlock, zeroBlock,  zeroBlock,  ex4,       zeroBlock, zeroBlock, zeroBlock, zeroBlock, ex-to-inhib],
             [zeroBlock, zeroBlock,  zeroBlock,  zeroBlock, ex5,       zeroBlock, zeroBlock, zeroBlock, ex-to-inhib],
             [zeroBlock, zeroBlock,  zeroBlock,  zeroBlock, zeroBlock, ex6,       zeroBlock, zeroBlock, ex-to-inhib],
             [zeroBlock, zeroBlock,  zeroBlock,  zeroBlock, zeroBlock, zeroBlock, ex7,       zeroBlock, ex-to-inhib],
             [zeroBlock, zeroBlock,  zeroBlock,  zeroBlock, zeroBlock, zeroBlock, zeroBlock, ex8      , ex-to-inhib],
             [inhib-to-all-excitatory,                                                                  inhib-to-all-inhib]])

# Option 2 - I think this is more intuitive:  
W = np.zeros([1000, 1000])
D = np.zeros([1000, 1000])
# Then use numpy slice notation to modify certain blocks.

# Update excitatory-excitatory blocks
for i in range(0, 800, 100):
    Wblock, Dblock = ExcitatoryToExcitatory(N)
    W[i:i+N, i:i+N] = Wblock
    D[i:i+N, i:i+N] = Dblock 

# Update excitatory-to-inhibitory block
# Reminder to self: check explanation on my notes

# Update inhibitory-to-excitatory block 
W[800:,:800] = 
D[800:,:800] = np.ones((200, 8*N))

# Update inhibitory-inhibitory block 