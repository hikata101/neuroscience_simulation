# Created by Eugene M. Izhikevich, February 25, 2003
# Excitatory neurons and Inhibitory neurons
import os
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(2025)

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

all_motifs = [motif_3a, motif_3b, motif_3c, motif_3d,
              motif_4a, motif_4b, motif_4c, motif_4d, motif_4e, motif_4f,
              motif_5a, motif_5b, motif_5c, motif_5d, motif_5e, A]

all_motifs_names = [
    "motif_3a", "motif_3b", "motif_3c", "motif_3d",
    "motif_4a", "motif_4b", "motif_4c", "motif_4d", "motif_4e", "motif_4f",
    "motif_5a", "motif_5b", "motif_5c", "motif_5d", "motif_5e", "random"
]

# MAIN SIMULATION
def main_simulation(A,SIM_TIME,name_motif,NOISE_MAX):
    """Function that runs the simulation with original A"""
    # Plot the connectivity matrix
    # plt.figure(figsize=(6, 6))
    # plt.imshow(A, cmap='gray_r', aspect='equal')  # Binary matrix: 1s are connections
    # plt.xlim([0, 1000])
    # plt.ylim([0, 1000])
    # plt.xlabel('Neuron')
    # plt.ylabel('Neuron')
    # plt.title('Connectivity matrix')
    # plt.tight_layout()
    # plt.savefig("Motif_figures/conn_matrix_"+name_motif+".png")

    S = A*W
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

    unique_elements, counts = np.unique(firings_np[:, 0], return_counts=True)
    # Display results
    # for elem, count in zip(unique_elements, counts):
    #     print(f"{elem} appears {count} times")

    # Target value
    target = 900
    # Create a boolean array where True means value equals target
    is_target = counts >= target

    # # Find where the value changes (by checking the difference)
    # diff = np.diff(is_target.astype(int))
    # # Count where a new group starts: when diff == 1
    # starts = np.where(diff == 1)[0]
    # # If the first element is the target, count it as a group start
    # num_lines = len(starts) + (is_target[0] == True)

    # Find contiguous groups of True values and count them
    # We pad with a False at both ends to detect edges
    padded = np.pad(is_target, (1, 1), constant_values=False)
    transitions = np.diff(padded.astype(int))
    # A new group starts at 0->1 transition
    num_lines = np.sum(transitions == 1)

    # print(f"Number of syncronize activity: {num_lines}")
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
    return num_lines

# main_simulation(A,SIM_TIME,"original",NOISE_MAX)

def simulation_all_motifs(all_motifs,all_motifs_names,firstA,NOISE_MAX):
    """Function that runs simulations of all motifs"""
    num_lines_each_motif = np.zeros(len(all_motifs))
    Nt = Ne + Ni
    neurons = np.arange(Nt)
    for m, motif in enumerate(all_motifs):
        name_motif  = all_motifs_names[m]
        print(name_motif)
        newA = np.ones(firstA.shape)*2
        np.fill_diagonal(newA, 0)
        nlinks = np.count_nonzero(firstA == 1)
        # Calculate max_steps based on the motif size
        motif_size = motif.shape[0]
        max_steps = Nt // motif_size
        # Shuffle the list of neurons for random selection without removing
        np.random.shuffle(neurons)
        for step in range(max_steps):
            selected = neurons[step * motif_size : (step + 1) * motif_size]
            # Assign motif block to submatrix of new_A
            newA[np.ix_(selected, selected)] = motif
        nlinks_after_motifs = np.count_nonzero(newA == 1)
        # print("before",nlinks_after_motifs)
        # Find all positions where value is 2
        positions = np.argwhere(newA == 2)
        # Randomly choose one position
        num_to_change = nlinks - nlinks_after_motifs
        # Shuffle positions in place
        np.random.shuffle(positions)
        # Select the first `num_to_change` and set them to 1
        newA[tuple(positions[:num_to_change].T)] = 1
        newA[newA==2] = 0
        # print("after",np.count_nonzero(newA == 1))
        num_lines_each_motif[m] = main_simulation(newA,SIM_TIME,name_motif,NOISE_MAX)
    return num_lines_each_motif
# simulation_all_motifs(all_motifs,all_motifs_names,firstA,NOISE_MAX=2.905)

def process_file_heatmap(data):
    # Get the output path
    output_path = os.path.join(os.path.dirname(__file__), "heatmap_data.txt")
    # Open the file for writing
    with open(output_path, 'w') as file:
        # Write each row
        for row in data:
            # Convert each element in the row to string and join with tabs
            row_values = "\t".join(map(str, row))
            # Write the row to the file with a newline
            file.write(row_values + "\n")

def heatmap_noise_variation(all_motifs,all_motifs_names,firstA,min_noise,max_noise,d_noise = 5):
    """Function that does the heatmap"""
    noise_values = np.linspace(min_noise, max_noise, d_noise)
    num_lines_all = np.zeros((len(all_motifs),d_noise))
    for n in np.arange(noise_values.shape[0]):
        print("noise_value=",noise_values[n])
        num_lines_all[:,n] = simulation_all_motifs(all_motifs,all_motifs_names,firstA,noise_values[n])
    process_file_heatmap(num_lines_all)
    plt.figure()
    plt.imshow(num_lines_all, cmap='tab10', aspect='auto')
    plt.xticks(ticks=np.arange(d_noise), labels=[f"{nv:.2f}" for nv in noise_values])
    plt.yticks(ticks=np.arange(num_lines_all.shape[0]), labels=all_motifs_names)
    # Add text annotations
    # max_val = num_lines_all.max()
    # for i in range(num_lines_all.shape[0]):
    #     for j in range(num_lines_all.shape[1]):
    #         val = num_lines_all[i, j]
    #         plt.text(j, i, f"{val:.1f}", ha='center', va='center',
    #                 color='white' if val > max_val / 2 else 'black')
    plt.colorbar(label='# lines')
    plt.title('Num.lines for values of noise')
    plt.xlabel('NOISE_MAX')
    plt.tight_layout()
    plt.savefig("Motif_figures/heatmap_lines.png")
    plt.close()

def create_random_A():
    frac_delete = 0.8  # Fraction of connections to delete (set to 0)
    # Create a random (Ne+Ni) x (Ne+Ni) connectivity matrix
    newA = np.random.rand(Ne + Ni, Ne + Ni)
    # Set a fraction of connections to 0
    newA[newA < frac_delete] = 0
    newA[newA > 0] = 2
    # Remove self-connections by zeroing the diagonal
    np.fill_diagonal(newA, 0)
    nlinks = np.count_nonzero(firstA == 1)
    nlinks_after_motifs = np.count_nonzero(newA == 2)
    # print("before",nlinks_after_motifs)
    # Randomly choose one position
    num_to_change = nlinks - nlinks_after_motifs
    if num_to_change > 0:
        # need more links
        # Find all positions where value is 0
        positions = np.argwhere(newA == 0)
        # Shuffle positions in place
        np.random.shuffle(positions)
        # Select the first `num_to_change` and set them to 1
        newA[tuple(positions[:num_to_change].T)] = 1
    else:
        #need less links
        # Find all positions where value is 0
        positions = np.argwhere(newA == 2)
        # Shuffle positions in place
        np.random.shuffle(positions)
        # Select the first `num_to_change` and set them to 0
        newA[tuple(positions[:abs(num_to_change)].T)] = 0
    newA[newA==2] = 1
    print(np.count_nonzero(newA == 1))
    return newA

# motifs3 = [A,motif_3c, motif_4a]
# motifs3_names = ["random","motif_3c", "motif_4a"]
# heatmap_noise_variation(all_motifs,all_motifs_names,firstA,2.85,3,d_noise=60)
# all_motifs_v2 = [motif_3a, motif_3b, motif_3c, motif_3d, A]
# all_motifs_names_v2 = ["motif_3a", "motif_3b", "motif_3c", "motif_3d","random"]
# all_motifs_v2 = [motif_3a, motif_3a, motif_3a, motif_3a, motif_3a, A, create_random_A(), create_random_A(), create_random_A(), create_random_A()]
# all_motifs_names_v2 = ["motif_3a1", "motif_3a2", "motif_3a3","motif_3a4", "motif_3a5","random1", "random2", "random3","random4", "random5"]
# heatmap_noise_variation(all_motifs_v2,all_motifs_names_v2,firstA,2.85,3,d_noise=10)


from scipy.stats import pearsonr
from scipy.signal import correlate
from scipy.stats import zscore

def read_heatmap_data(txt_file):
    # Get the file path
    file_path = os.path.join(os.path.dirname(__file__), txt_file)
    
    # Load the tab-separated file into a NumPy array
    data = np.loadtxt(file_path, delimiter="\t")
    return data

# Input data
# data = read_heatmap_data("heatmap_data_exemple.txt")
# data = read_heatmap_data("heatmap_data_v1.txt")
# list_heatmaps = ["heatmap_data_v2.txt","heatmap_data_v2.txt"]
# for file in list_heatmaps:
#     new_data = read_heatmap_data(file)
#     data += new_data
new_data = read_heatmap_data("heatmap_data_itr10.txt")
# data += new_data*10
# data = data/(len(list_heatmaps)+1+10)
data = new_data[:,:] - new_data[-1,:]
# print(data)
# Separate reference row
reference = zscore(data[-1])
other_rows = data[:-1]
# Compute max cross-correlation with reference
scores = []
for i, row in enumerate(other_rows):
    row_z = zscore(row)
    corr = correlate(row_z, reference, mode='full')
    max_corr = np.max(corr) / len(row)
    scores.append((i, max_corr))
print(scores)
# Sort indices by correlation score (descending)
sorted_indices = [i for i, _ in sorted(scores, key=lambda x: -x[1])]
# Reorder rows and append reference at the end
reordered_data = np.vstack([data[-1], other_rows[sorted_indices]])
reordered_data = np.vstack([data[-1], other_rows])
# Show result
# print("Reordered data (rows sorted by similarity to last row):")
# print(reordered_data)
# Optional: show mapping from new index to original
# new_to_old = sorted_indices + [len(data) - 1]
new_to_old = [len(data) - 1] + sorted_indices
# print("New to old row indices:", new_to_old)
# new_names_order = [all_motifs_names_v2[i] for i in new_to_old]
new_names_order = [all_motifs_names[i] for i in new_to_old]
# print(new_names_order)
num_lines_all = reordered_data
d_noise = 60
noise_values = np.linspace(2.85, 3.0, d_noise)
plt.figure()
plt.imshow(num_lines_all, cmap='seismic', aspect='auto')
step = 10
print(noise_values)
ticks_to_show = np.arange(0, len(noise_values), step)
labels_to_show = [f"{noise_values[i]:.2f}" for i in ticks_to_show]
plt.xticks(ticks=ticks_to_show, labels=labels_to_show)
# plt.xticks(ticks=np.arange(d_noise), labels=[f"{nv:.2f}" for nv in noise_values])
plt.yticks(ticks=np.arange(num_lines_all.shape[0]), labels=new_names_order)
# Add text annotations
# max_val = num_lines_all.max()
# for i in range(num_lines_all.shape[0]):
#     for j in range(num_lines_all.shape[1]):
#         val = num_lines_all[i, j]
#         plt.text(j, i, f"{val:.1f}", ha='center', va='center',
#                 color='white' if val > max_val / 2 else 'black')
plt.colorbar(label='# lines')
plt.title('Difference number of lines')
plt.xlabel('NOISE_MAX')
plt.tight_layout()
plt.savefig("Motif_figures/heatmap_crosscorrelation.png")
plt.close()  


### combinations of multiple motifs
# list of motifs and the corresponding percent (total sum must be 1 or less)
# combination_motifs = [motif_3a, motif_4a, motif_5a]
# combination_motifs_names = ["motif_3a", "motif_4a", "motif_5a"]
# percent_motifs = [0.3,0.3,0.4]
# name_combination = ""
# for n in range(len(combination_motifs)):
#     name_combination += str(percent_motifs[n])+combination_motifs_names[n]+"_"
# comb_A = np.copy(A)
# print(name_combination)
# Nt = Ne + Ni
# list_neurons = np.arange(Nt)
# # Shuffle the list of neurons for random selection without removing
# np.random.shuffle(list_neurons)
# steps_each_motif = [int(x * Nt) for x in percent_motifs]
# idx = 0
# for motif, steps in zip(combination_motifs, steps_each_motif):
#     motif_size = motif.shape[0]
#     max_steps = steps // motif_size
    
#     for _ in range(max_steps):
#         selected = list_neurons[idx : idx + motif_size]
#         comb_A[np.ix_(selected, selected)] = motif
#         idx += motif_size
# main_simulation(comb_A,SIM_TIME,name_combination)