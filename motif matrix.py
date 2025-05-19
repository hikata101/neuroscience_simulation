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
motif_3b = np.array([[0, 1, 1],
            [1, 0, 1],
            [0, 0, 0]])
motif = motif_3b

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