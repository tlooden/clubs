

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
    Generate simulated positive definite matrices with class-specific signals.
    
    Each generated matrix is a sum of:
    1. A class-specific signal matrix (signal_strength * A^T A)
    2. A random noise matrix (noise_level * B^T B)
    Both components are positive definite by construction.
    
    Parameters
    ----------
    num_samples : int, default=100
        Number of samples per class
    matrix_size : int, default=20 
        Size of the square matrices
    num_classes : int, default=4
        Number of distinct classes
    signal_strength : float, default=1.0
        Scaling factor for class-specific signal
    noise_level : float, default=1.0
        Scaling factor for random noise
    random_state : int or None, default=None
        Random seed for reproducibility
    
    Returns
    -------
    matrices : ndarray of shape (num_samples * num_classes, matrix_size, matrix_size)
        Generated symmetric positive definite matrices
    labels : ndarray of shape (num_samples * num_classes,)
        Class labels corresponding to each matrix
    """
    rng = np.random.RandomState(random_state)
    
    # Pre-allocate arrays
    total_samples = num_samples * num_classes
    matrices = np.zeros((total_samples, matrix_size, matrix_size))
    labels = np.repeat(np.arange(num_classes), num_samples)
    
    # Generate class-specific signal matrices
    class_signals = []
    for _ in range(num_classes):
        A = rng.randn(matrix_size, matrix_size)
        S = signal_strength * A.T @ A  # Using @ for matrix multiplication
        class_signals.append(S)
    
    # Generate matrices for each class
    for i in range(total_samples):
        # Add class signal
        class_idx = labels[i]
        matrices[i] = class_signals[class_idx]
        
        # Add noise
        B = rng.randn(matrix_size, matrix_size)
        N = noise_level * B.T @ B
        matrices[i] += N
    
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
    
   

