import numpy as np
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

# Relevant constants
N_ex = 100                    # Number of neurons in an excitatory module
N_in = 200                    # Number of neurons in the inhibitory module
N_mod = 8                     # Number of excitatory modules in the modular network
N_net = N_mod * N_ex + N_in   # Number of total neurons in the modular network

F_ex_in = 50                  # Scaling factor for excitatory-to-inhibitory connection
F_in_ex = 2                   # Scaling factor for inhibitory-to-excitatory connection

# 2. Extend to 8 populations

def ExcitatoryToExcitatory(N):
    '''
    Generates excitatory-to-excitatory connections according to specification. 

    N - The number of neurons in the excitatory community. 

    W - The connectivity matrix for the excitatory-to-excitatory community. 
    D - The conduction delay matrix for the excitatory-to-excitatory community. 
    '''
    
    F = 17         # Scaling factor for excitatory-to-excitatory connections
    Dmax = 20      # Maximum conduction delay
    N_edges = 1000 # Number of connections in this excitatory module
    
    W = np.zeros((N,N))
    D = np.ones((N,N), dtype=int)

    all_edges = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            all_edges.append((i, j))

    idxs = np.random.choice(len(all_edges), size=N_edges, replace=False)
    edges = [all_edges[idx] for idx in idxs]

    # Set connection weight to be 1 for randomly generated connections
    # Set random delays for these connections simultaneously. 
    for edge in edges:
        W[edge] = 1*F
        D[edge] = np.random.randint(1, Dmax+1) # Assume delays have to be integer values in range [1,20], need to double check
    
    return W, D

# 5. Rewiring process

def RewireConnectivity(p):
    src, tgt = np.where(W[:800, :800] > 0)
    for s, t in zip(src, tgt):
        if np.random.random() < p:
            W[s, t] = 0
            D[s, t] = 1
            # Pick index of new node to rewire to. It can't be an existing
            # connection or itself (because the total density has to be preserved)
            h = s
            while s == h or W[s, h]:
                h = np.random.randint(800)
            W[s, h] = 17
            D[s, h] = np.random.randint(1, 20+1)

# 3. Create 200x200 inhibitory population and connect to excitatory populations (1000x1000 weight matrix)
  
W = np.zeros([1000, 1000])
D = np.ones([1000, 1000], dtype=int)

# Update excitatory-to-excitatory blocks
for i in range(0, 800, N_ex):
    Wblock, Dblock = ExcitatoryToExcitatory(N_ex)
    W[i:i+N_ex, i:i+N_ex] = Wblock
    D[i:i+N_ex, i:i+N_ex] = Dblock 

# Update excitatory-to-inhibitory block
for j in range(N_mod*N_ex, N_net):
    i = np.random.randint(0, N_mod)
    neurons = np.random.randint(0, N_ex, size=4)
    i = neurons + N_ex*i
    W[i, j] = 50*np.random.uniform(0, 1)

# Update inhibitory
in_to_ex_block = F_in_ex * np.random.uniform(-1, 0, (N_in, N_mod*N_ex)) 
in_to_in_block = np.random.uniform(-1, 0, (N_in, N_in))
W[800:, :] = np.column_stack((in_to_ex_block, in_to_in_block))

# Inhibitory neurons cannot connect to themselves, thus set the connection weight to 0.
W[range(800, 1000), range(800, 1000)] = 0

RewireConnectivity(0.4)

net = IzNetwork(N_net, 20)
a = 0.02 * np.ones(N_net)
b = np.concatenate((0.2*np.ones(800), 0.25*np.ones(200)))
c = -65 * np.ones(N_net)
d = np.concatenate((8*np.ones(800), 2*np.ones(200)))
net.setParameters(a, b, c, d)
net.setWeights(W)
net.setDelays(D)

# 6. Connectivity matrix, raster plot and mean firing rate over each probability p.

# Mean firing rate 

y, x = np.where(W[:800, :800] > 0)
print(len(x))
print(len(y))
plt.scatter(x, y, s=1)
plt.xlabel('to')
plt.ylabel('from')
plt.ylim(800, 0)
plt.xlim(0, 800)
plt.show()

# T = 1000

# firing_matrix = np.zeros([T, N_mod])
# V = np.zeros((T, N_net))

# for t in range(T):
#     I = 15*np.random.poisson(0.01, N_net) 
#     net.setCurrent(I)
#     net.update()
#     V[t,:], _ = net.getState()
#     fired = V[t,:] > 29
#     for i in range(0, 8):
#         interval_sum = np.sum(fired[i*100:i*100+100])
#         firing_matrix[t, i] = interval_sum
    
# t, n = np.where(V > 29)
# plt.subplot(211)
# plt.scatter(t, n)
# plt.ylim(800, 0)
# plt.xlabel('Time (ms)')
# plt.ylabel('Neuron index')

# plt.subplot(212)
# # Downsampling time 
# windows = np.zeros([50, 8])

# for i in range(0, 1000, 20):
#     windows[int(i/20),:] = np.mean(firing_matrix[i:i+50,:], axis=0)

# for i in range(8):
#     plt.plot(np.arange(0, 1000, 20), windows[:,i], label=f"Module {i}")
#
# plt.xlabel('Time (ms)')
# plt.ylabel('Mean firing rate')
# plt.legend()
# plt.show()