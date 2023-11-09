import numpy as np
import matplotlib.pyplot as plt

from iznetwork import IzNetwork

# Relevant constants
N_ex_mod = 100                  # Number of neurons in an excitatory module
N_in_mod = 25                   # Number of neurons in the inhibitory module
N_mod = 8                       # Number of excitatory modules in the modular network
N_ex = N_mod * N_ex_mod         # Total number of excitatory neurons in the modular network
N_in = N_mod * N_in_mod         # Total number of inhibitory neurons in the modular network
N_net = N_ex + N_in             # Number of total neurons in the modular network 

F_ex_ex = 17                    # Scaling factor for excitatory-to-excitatory connection
F_ex_in = 50                    # Scaling factor for excitatory-to-inhibitory connection
F_iN_ex_mod = 2                 # Scaling factor for inhibitory-to-excitatory connection
F_iN_in_mod = 1                 # Scaling factor for inhibitory-to-inhibitory connection 

Dmax = 20                       # Maximum conduction delay


class SmallWorldModularNetwork(object):
    def __init__(self, p):
        """
        Initialise modular network.

        Inputs:
            p  -- Rewiring probability
        """

        self._p = p

        # Initialise weight and delay matrix for modular network 
        self.W = np.zeros([N_net, N_net])
        self.D = np.ones([N_net, N_net], dtype=int)

        # Update excitatory-to-excitatory blocks
        for i in range(0, N_ex, N_ex_mod):
            Wblock, Dblock = self._excitatoryToExcitatory(N_ex_mod)
            self.W[i:i + N_ex_mod, i:i + N_ex_mod] = Wblock
            self.D[i:i + N_ex_mod, i:i + N_ex_mod] = Dblock

        # Update excitatory-to-inhibitory block
        for j in range(N_ex, N_net):
            i = int(np.floor((j-N_ex)/N_in_mod))
            neurons = np.random.randint(0, N_ex_mod, size=4)
            i = neurons + N_ex_mod * i
            self.W[i, j] = F_ex_in * np.random.uniform(0, 1)

        # Update inhibitory
        in_to_ex_block = F_iN_ex_mod * np.random.uniform(-1, 0, (N_in, N_mod * N_ex_mod))
        in_to_in_block = np.random.uniform(-1, 0, (N_in, N_in))
        self.W[N_ex:, :] = np.column_stack((in_to_ex_block, in_to_in_block))

        # Inhibitory neurons cannot connect to themselves, thus set the connection weight to 0
        self.W[range(N_ex, N_net), range(N_ex, N_net)] = 0

        # Apply rewiring process
        self._rewireConnectivity()

        # Instantiate IzNetwork with parameters from 'L04 - Simple Neuron Models'
        self.net = IzNetwork(N_net, Dmax)
        a = 0.02 * np.ones(N_net)
        b = np.concatenate((0.2 * np.ones(N_ex), 0.25 * np.ones(N_in)))
        c = -65 * np.ones(N_net)
        d = np.concatenate((8 * np.ones(N_ex), 2 * np.ones(N_in)))
        self.net.setParameters(a, b, c, d)
        self.net.setWeights(self.W)
        self.net.setDelays(self.D)


    def _excitatoryToExcitatory(self, N):
        """
        Generates excitatory-to-excitatory connections according to specification.

        Inputs:
            N -- The number of neurons in the excitatory community.

        Outputs:
            W -- The connectivity matrix for the excitatory-to-excitatory community.
            D -- The conduction delay matrix for the excitatory-to-excitatory community.
        """

        # Number of connections in this excitatory module
        N_edges = 1000  

        # Initialise weight and delay matrix for excitatory commmunity
        W = np.zeros((N, N))
        D = np.ones((N, N), dtype=int)

        # Generate all possible edges to sample from 
        all_edges = []
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                all_edges.append((i, j))

        # Uniformly sample N_edges connections 
        idxs = np.random.choice(len(all_edges), size=N_edges, replace=False)
        edges = [all_edges[idx] for idx in idxs]

        # Set connection weight to be 1 for randomly generated connections
        # Set random delays for these connections simultaneously.
        for edge in edges:
            W[edge] = 1 * F_ex_ex
            D[edge] = np.random.randint(1, Dmax + 1) 

        return W, D


    def _rewireConnectivity(self):
        """
        Rewires the connections between excitatory modules with probability p.

        Inputs:
            p  -- Rewiring probability 
        """

        src, tgt = np.where(self.W[:N_ex, :N_ex] > 0)
        for s, t in zip(src, tgt):
            if np.random.random() < self._p:
                self.W[s, t] = 0
                self.D[s, t] = 1
                # Pick index of new node to rewire to. It can't be an existing
                # connection or itself (because the total density has to be preserved)
                h = s
                while s == h or self.W[s, h]:
                    h = np.random.randint(N_ex)
                self.W[s, h] = 1 * F_ex_ex
                self.D[s, h] = np.random.randint(1, Dmax + 1)


    def plotConnectivityMatrix(self):
        """
        Plots connectivity matrix for excitatory modules. 
        """

        y, x = np.where(self.W[:N_ex, :N_ex] > 0)
        plt.scatter(x, y, s=1)
        plt.title(f'p = {self._p}')
        plt.xlabel('to')
        plt.ylabel('from')
        plt.ylim(N_ex, 0)
        plt.xlim(0, N_ex)
        plt.show()

    
    def _runSimulation(self, T, transient):
        """
        Utility function to run a simulation and record the membrane potentials and firing rate. 

        Inputs:
            T             -- Time to run simulation and record activity  
            transient     -- Initial time to run simulation without recording activity
        
        Outputs: 
            firing_matrix -- Stores number of spikes per excitatory community in each time step
            V             -- Stores membrane potentials of each neuron in each time step 
        """

        # Define firing matrix to store number of spikes per community in each time step (ms)
        firing_matrix = np.zeros([T, N_mod])

        # Define matrix to store membrane potentials of each neuron in each time step (ms)
        V = np.zeros((T, N_net))

        # Run the simulation for some transient time, without recording activity
        for t in range(transient):
            I = 15 * np.random.poisson(0.01, N_net)
            self.net.setCurrent(I)
            self.net.update()

        # Run simulation for T ms and record activity 
        for t in range(T):
            I = 15 * np.random.poisson(0.01, N_net)
            self.net.setCurrent(I)
            self.net.update()
            V[t, :], _ = self.net.getState()
            fired = V[t, :] > 29
            for i in range(0, N_mod):
                interval_sum = np.sum(fired[i * N_ex_mod:i * N_ex_mod + N_ex_mod])
                firing_matrix[t, i] = interval_sum
            
        return firing_matrix, V


    def plotNeuronFiringAndMeanFiringRate(self, T, transient):
        """
        Plots raster plot for excitatory neuron firing and mean firing rate. 

        Inputs:
            T             -- Time to run simulation and record activity  
            transient     -- Initial time to run simulation without recording activity
        """

        firing_matrix, V = self._runSimulation(T, transient)

        # Plot raster plot for each excitatory neuron 
        t, n = np.where(V > 29)
        plt.subplot(211)
        plt.title(f'p = {self._p}')
        plt.scatter(t, n)
        plt.ylim(N_ex, 0)
        plt.xlim(0, T)
        plt.xlabel('Time (ms)')
        plt.ylabel('Neuron index')

        # Plot mean firing rates for each excitatory community
        plt.subplot(212)
        windows = np.zeros([50, 8])

        # Downsample the firing rates to obtain the mean by computing the 
        # average number of firings in 50ms windows shifted every 20ms.
        for i in range(0, T, 20):
            windows[int(i / 20), :] = np.mean(firing_matrix[i:i + 50, :], axis=0)

        for i in range(8):
            plt.plot(np.arange(0, T, 20), windows[:, i], label=f"Module {i}")

        plt.xlabel('Time (ms)')
        plt.ylabel('Mean firing rate')
        plt.ylim(0)
        plt.xlim(0, T)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    # Define rewiring probability
    p = 0

    # Define transient time and measurement time (ms)
    T = 1000
    transient = 50

    network = SmallWorldModularNetwork(p)
    network.plotConnectivityMatrix()
    network.plotNeuronFiringAndMeanFiringRate(T, transient)
