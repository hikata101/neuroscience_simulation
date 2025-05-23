import numpy as np
import random
random.seed(2025)

frac_delete = 0.8  # Fraction of connections to delete
Ne = 5  # Example number of excitatory neurons (adjust as needed)
Ni = 5  # Example number of inhibitory neurons (adjust as needed)
# Create a random connection matrix
A = np.random.rand(Ne + Ni, Ne + Ni)
# Set a fraction of the connections to zero
A[A < frac_delete] = 0
# Set the remaining connections to 1
A[A > 0] = 1
# Remove self-connections (set diagonal elements to 0)
np.fill_diagonal(A, 0)
firstA = np.copy(A)

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

for motif in all_motifs:
    Nt = Ne + Ni
    list_neurons = list(range(0,Nt,1))
    # Shuffle the list of neurons for random selection without removing
    random.shuffle(list_neurons)
    # Calculate max_steps based on the motif size
    max_steps = len(list_neurons) // motif.shape[0]
    # Main loop for motif assignment
    for _ in range(max_steps):
        # Select the first `motif.shape[0]` random neurons
        selected_neurons = list_neurons[:motif.shape[0]]
        # Give the new values to the neurons following the motif structure
        for i in range(motif.shape[0]):
            for j in range(motif.shape[1]):
                A[selected_neurons[i], selected_neurons[j]] = motif[i, j]
        # Remove the used neurons (move to the next set of neurons)
        list_neurons = list_neurons[motif.shape[0]:]
    print(A==firstA)
print('')

### combinations of multiple motifs
# list of motifs and the corresponding percent (total sum must be 1 or less)
combination_motifs = [motif_3a, motif_4a, motif_5a]
percent_motifs = [0.3,0.3,0.4]
Nt = Ne + Ni
list_neurons = list(range(0,Nt,1))
# Shuffle the list of neurons for random selection without removing
random.shuffle(list_neurons)
steps_each_motif = [int(x * Nt) for x in percent_motifs]
for i in range(len(combination_motifs)):
    motif = combination_motifs[i]
    # Calculate max_steps based on the motif size
    max_steps = steps_each_motif[i] // motif.shape[0]
    # Main loop for motif assignment
    for _ in range(max_steps):
        # Select the first `motif.shape[0]` random neurons
        selected_neurons = list_neurons[:motif.shape[0]]
        # Give the new values to the neurons following the motif structure
        for i in range(motif.shape[0]):
            for j in range(motif.shape[1]):
                A[selected_neurons[i], selected_neurons[j]] = motif[i, j]
        # Remove the used neurons (move to the next set of neurons)
        list_neurons = list_neurons[motif.shape[0]:]
print(A==firstA)