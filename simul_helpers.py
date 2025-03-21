

import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import IPython

#%%
def generate_symmetric_matrices(num_samples=100, matrix_size=20, num_classes=4,
                                        signal_strength=1.0, noise_level=1.0, random_state=None):
    """
    Generate simulated positive definite matrices with class-specific signals distributed across the entire matrix.
    
    Parameters:
    - num_samples (int): Number of samples per class.
    - matrix_size (int): Size of the square matrices (e.g., 20 for 20x20 matrices).
    - num_classes (int): Number of distinct classes.
    - signal_strength (float): Scaling factor for the class-specific signal.
    - noise_level (float): Scaling factor for the general PSD noise.
    - random_state (int or None): Seed for reproducibility.
    
    Returns:
    - matrices (numpy.ndarray): Array of shape (total_samples, matrix_size, matrix_size).
    - labels (numpy.ndarray): Array of class labels corresponding to each matrix.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    matrices = []
    labels = []
    
    total_samples = num_samples * num_classes
    
    # Generate class-specific positive definite matrices
    class_signal_matrices = []
    for _ in range(num_classes):
        # Generate a random matrix
        A = np.random.randn(matrix_size, matrix_size)
        # Create a positive definite matrix via A^T A
        S = signal_strength * np.dot(A.T, A)
        class_signal_matrices.append(S)
    
    for label in range(num_classes):
        S_class = class_signal_matrices[label]
        
        for _ in range(num_samples):
            # Generate a random matrix for noise
            B = np.random.randn(matrix_size, matrix_size)
            # Create a PSD noise matrix via B^T B
            N = noise_level * np.dot(B.T, B)
            
            # Combine the class signal and the noise
            mat = S_class + N  # Sum of two positive definite matrices is positive definite
            
            matrices.append(mat)
            labels.append(label)
    
    # Convert lists to numpy arrays
    matrices = np.array(matrices)
    labels = np.array(labels)
    
    return matrices, labels

#%%
def minimal_epsilon_graph(X):
    """
    Constructs the minimally connected epsilon-neighborhood graph from a data matrix X.
    
    Parameters:
    - X: A NumPy array of shape (n_samples, n_features), the data points.
    
    Returns:
    - A: A sparse adjacency matrix representing the minimally connected graph.
    - epsilon_min: The minimal epsilon required to ensure connectivity.
    """
    # Step 1: Compute the pairwise distance matrix
    # If X is already a distance matrix, skip this step
    distance_matrix = squareform(pdist(X, metric='euclidean'))

    # Step 2: Compute the Minimum Spanning Tree (MST)
    mst_sparse = minimum_spanning_tree(distance_matrix)
    mst_dense = mst_sparse.toarray()

    # Step 3: Find the maximal edge weight in the MST (minimal epsilon)
    epsilon_min = mst_dense[mst_dense != 0].max()

    # Step 4: Construct the epsilon-neighborhood graph using epsilon_min
    adjacency_matrix = (distance_matrix <= epsilon_min).astype(int)
    np.fill_diagonal(adjacency_matrix, 0)  # Remove self-loops

    # Convert to sparse matrix for efficiency
    adjacency_sparse = csr_matrix(adjacency_matrix)

    return adjacency_sparse, epsilon_min

#%%

def scatter3D(data, labels=None):

    IPython.get_ipython().run_line_magic('matplotlib', 'qt')
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    n_samples, n_dimensions = data.shape
    
    # If labels are not provided, create a default label vector
    if labels is None:
        labels = np.zeros(n_samples)
    
    # load data
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    
    # Create the 3D scatter plot
    scatter = ax.scatter(x, y, z, c=labels, cmap='viridis', marker='o')
    
    # Add color bar to indicate which colors correspond to which labels
    cbar = plt.colorbar(scatter, ax=ax, label='Label')
    
    # Add labels
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    
    # Show plot
    plt.tight_layout()
    plt.show()
    
   

