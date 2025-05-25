# Created by Eugene M. Izhikevich, February 25, 2003
# Excitatory neurons and Inhibitory neurons

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(2025)

# SET UP THE NETWORK
# 1 - NETWORK SIZE:
Ne = 800  # Number of excitatory neurons
Ni = 200  # Number of inhibitory neurons
SIM_TIME = 3000  # Simulation time

# 2 - GLOBAL PARAMETERS THAT SET OUR NEURON MODEL. DEFAULT IS SPIKING NEURON:
# Set initial conditions of neurons, with some variability
re = np.random.rand(Ne)  # Random values for excitatory neurons
ri = np.random.rand(Ni)  # Random values for inhibitory neurons
# Model parameters
a = np.concatenate((0.02 * np.ones(Ne), 0.02 + 0.08 * ri))
b = np.concatenate((0.2 * np.ones(Ne), 0.25 - 0.05 * ri))
c = np.concatenate((-65 + 15 * re ** 2, -65 * np.ones(Ni)))
d = np.concatenate((8 - 6 * re ** 2, 2 * np.ones(Ni)))

# 3 - SET UP THE CONNECTIVITY MATRIX: DIRECTED NETWORK
# In this construction, 1 = connection exists, 0 = no connection.
# Connectivity is set as random. Then, a fraction of connections are set to 0.

frac_delete = 0.8  # Fraction of connections to delete (set to 0)
# Create a random (Ne+Ni) x (Ne+Ni) connectivity matrix
A = np.random.rand(Ne + Ni, Ne + Ni)
# Set a fraction of connections to 0
A[A < frac_delete] = 0
A[A > 0] = 1
# Remove self-connections by zeroing the diagonal
np.fill_diagonal(A, 0)
firstA = np.copy(A)

# 4 - SET SYNAPTIC WEIGHTS (STRENGTHS) OF CONNECTIONS
# EPSC (excitatory) and IPSC (inhibitory) amplitudes
MAX_EXC_WEIGHT = 4     # Max weight for excitatory synapses
MAX_INH_WEIGHT = 0.5   # Max weight for inhibitory synapses
# W is a matrix where the first Ne columns are excitatory weights (positive),
# and the next Ni columns are inhibitory weights (negative)
W_exc = MAX_EXC_WEIGHT * np.random.rand(Ne + Ni, Ne)
W_inh = -MAX_INH_WEIGHT * np.random.rand(Ne + Ni, Ni)
W = np.hstack((W_exc, W_inh))

# 5 - Final connectivity matrix S is element-wise product of A and W
S = A * W  # Element-wise multiplication: directed, weighted connectivity

# 6 - DEFINE NOISE STRENGTH
NOISE_MAX = 3  # Strength of background noise

# MAIN SIMULATION
def main_simulation(A,SIM_TIME,name_motif):
    """Function that runs the simulation with original A"""
    # Plot the connectivity matrix
    plt.figure(figsize=(6, 6))
    plt.imshow(A, cmap='gray_r', aspect='equal')  # Binary matrix: 1s are connections
    plt.xlim([0, 1000])
    plt.ylim([0, 1000])
    plt.xlabel('Neuron')
    plt.ylabel('Neuron')
    plt.title('Connectivity matrix')
    plt.tight_layout()
    plt.savefig("Motif_figures/conn_matrix_"+name_motif+".png")

    v = -65 * np.ones(Ne + Ni)        # Initial membrane potential
    u = b * v                         # Initial recovery variable
    firings = []                      # List to store spike timings

    for t in range(1, SIM_TIME + 1):  # Simulation from t=1 to t=SIM_TIME
        I = np.concatenate((NOISE_MAX * np.random.randn(Ne),
                            2 * np.random.randn(Ni)))  # Random input (noise)

        fired = np.where(v >= 30)[0]  # Indices of neurons that fired
        if fired.size > 0:
            firings.extend([(t, neuron) for neuron in fired])
            v[fired] = c[fired]
            u[fired] += d[fired]
            I += np.sum(S[:, fired], axis=1)  # Input from fired neurons

        # Numerical integration with 0.5 ms time step (two half steps)
        v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
        v += 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)
        u += a * (b * v - u)

    # PLOT RESULTS (Raster Plot)

    firings_np = np.array(firings)  # Convert to NumPy array for plotting

    plt.figure(figsize=(10, 6))
    plt.scatter(firings_np[:, 0], firings_np[:, 1], s=7, c='black', marker='.')
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.title('Raster plot of activity')
    plt.savefig("Motif_figures/raster_plot_"+name_motif+".png")
    plt.close('all')
    # SYNC ANALYSIS

    # SIM_TIME = 1000
    # WINDOW = 100
    # num_active = np.zeros(10)
    # firings_np = np.array(firings)  # Use previously collected spikes

    # for j in range(1, 11):
    #     start_time = WINDOW * (j - 1)
    #     end_time = WINDOW * j
    #     # Select spikes in the current time window
    #     mask = (firings_np[:, 0] >= start_time) & (firings_np[:, 0] < end_time)
    #     neurons_fired = firings_np[mask][:, 1]
    #     num_active[j - 1] = len(np.unique(neurons_fired)) / (Ne + Ni)

    # sync = np.max(num_active)
    # print("Synchronization measure (max fraction of active neurons per window):", sync)

# main_simulation(firstA,SIM_TIME)

# neuron i send connection to j
#matrix 3
# 1
#2 3
motif_3a = np.array([[0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0]])
motif_3b = np.array([[0, 1, 1],
                    [1, 0, 1],
                    [0, 0, 0]])
motif_3c = np.array([[0, 1, 1],
                    [0, 0, 0],
                    [0, 0, 0]])
motif_3d = np.array([[0, 0, 1],
                    [1, 0, 1],
                    [0, 0, 0]])
#matrix 4
# 1 2
# 3 4
motif_4a = np.array([[0, 1, 1, 1],
                    [1, 0, 1, 1],
                    [1, 1, 0, 1],
                    [1, 1, 1, 0]])
motif_4b = np.array([[0, 1, 1, 0],
                    [1, 0, 1, 0],
                    [1, 1, 0, 1],
                    [0, 0, 0, 0]])
motif_4c = np.array([[0, 1, 1, 1],
                    [1, 0, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
motif_4d = np.array([[0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
motif_4e = np.array([[0, 1, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0]])
motif_4f = np.array([[0, 1, 0, 1],
                    [1, 0, 1, 1],
                    [1, 1, 0, 0],
                    [0, 0, 0, 0]])
#matrix 5
# 1 2
#3 4 5
motif_5a = np.array([[0, 1, 1, 1, 1],
                    [1, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0]])
motif_5b = np.array([[0, 1, 1, 1, 0],
                    [1, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0]])
motif_5c = np.array([[0, 1, 1, 1, 1],
                    [1, 0, 1, 1, 1],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0]])
motif_5d = np.array([[0, 0, 1, 1, 0],
                    [0, 0, 1, 1, 1],
                    [1, 1, 0, 1, 1],
                    [1, 1, 1, 0, 1],
                    [0, 1, 1, 1, 0]])
motif_5e = np.array([[0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1],
                    [1, 0, 1, 0, 1],
                    [0, 0, 1, 1, 0]])

all_motifs = [motif_3a, motif_3b, motif_3c, motif_3d,
              motif_4a, motif_4b, motif_4c, motif_4d, motif_4e, motif_4f,
              motif_5a, motif_5b, motif_5c, motif_5d, motif_5e]

all_motifs_names = [
    "motif_3a", "motif_3b", "motif_3c", "motif_3d",
    "motif_4a", "motif_4b", "motif_4c", "motif_4d", "motif_4e", "motif_4f",
    "motif_5a", "motif_5b", "motif_5c", "motif_5d", "motif_5e"
]

def simulation_all_motifs(all_motifs,all_motifs_names,A):
    """Function that runs simulations of all motifs"""
    for m in range(len(all_motifs)):
        motif = all_motifs[m]
        name_motif  = all_motifs_names[m]
        new_A = np.copy(A)
        print(name_motif)
        Nt = Ne + Ni
        neurons = np.arange(Nt)
        # Shuffle the list of neurons for random selection without removing
        np.random.shuffle(neurons)
        # Calculate max_steps based on the motif size
        motif_size = motif.shape[0]
        max_steps = Nt // motif_size
        for step in range(max_steps):
            selected = neurons[step * motif_size : (step + 1) * motif_size]
            # Assign motif block to submatrix of new_A
            new_A[np.ix_(selected, selected)] = motif
        main_simulation(new_A,SIM_TIME,name_motif)
simulation_all_motifs(all_motifs,all_motifs_names,A)
print('')
### combinations of multiple motifs
# list of motifs and the corresponding percent (total sum must be 1 or less)
combination_motifs = [motif_3a, motif_4a, motif_5a]
combination_motifs_names = ["motif_3a", "motif_4a", "motif_5a"]
percent_motifs = [0.3,0.3,0.4]
name_combination = ""
for n in range(len(combination_motifs)):
    name_combination += str(percent_motifs[n])+combination_motifs_names[n]+"_"
comb_A = np.copy(A)
print(name_combination)
Nt = Ne + Ni
list_neurons = np.arange(Nt)
# Shuffle the list of neurons for random selection without removing
np.random.shuffle(list_neurons)
steps_each_motif = [int(x * Nt) for x in percent_motifs]
idx = 0
for motif, steps in zip(combination_motifs, steps_each_motif):
    motif_size = motif.shape[0]
    max_steps = steps // motif_size
    
    for _ in range(max_steps):
        selected = list_neurons[idx : idx + motif_size]
        comb_A[np.ix_(selected, selected)] = motif
        idx += motif_size
main_simulation(comb_A,SIM_TIME,name_combination)