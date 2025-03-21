import sys
import os

import numpy as np
from scipy.linalg import eigh
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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

def plot_PCA(y, title='Line Plot'):
    sns.set_style('whitegrid')
    
    num_points = len(y)
    x = np.arange(1, num_points + 1)  # x runs from 1 to number of points
    
    # Determine figure size based on data size
    fig_width = max(6, min(12, num_points / 10))
    fig_height = 6  # Fixed height
    
    plt.figure(figsize=(fig_width, fig_height))
    sns.lineplot(x=x, y=y, marker='o')
    
    plt.title(title, fontsize=16)
    plt.xlabel('index', fontsize=12)
    plt.ylabel('eigenvalue', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
#%%

def spectral_embedding(similarity_matrix, num_dims=2, plot=True, random_state=None):
    """
    Performs spectral clustering on a given similarity matrix and plots the samples
    projected onto the smallest non-zero indicator eigenvectors.

    Parameters:
    - similarity_matrix (numpy.ndarray): The similarity matrix (must be square and symmetric).
    - num_clusters (int): The number of clusters to form.
    - plot (bool): Whether to plot the samples on the indicator eigenvectors.
    - random_state (int or None): Random seed for reproducibility.

    Returns:
    - labels (numpy.ndarray): Cluster labels for each sample.
    - indicator_vectors (numpy.ndarray): The indicator eigenvectors used for clustering.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Ensure the similarity matrix is square and symmetric
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "Similarity matrix must be square."
    assert np.allclose(similarity_matrix, similarity_matrix.T, atol=1e-8), "Similarity matrix must be symmetric."

    n_samples = similarity_matrix.shape[0]

    # Step 1: Compute the Degree Matrix
    degree_matrix = np.diag(similarity_matrix.sum(axis=1))

    # Step 2: Compute the Unnormalized Laplacian
    laplacian_matrix = degree_matrix - similarity_matrix

    # Step 3: Compute the Normalized Laplacian (symmetric normalization)
    with np.errstate(divide='ignore'):
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0  # Replace infinities with zeros

    L_sym = D_inv_sqrt @ laplacian_matrix @ D_inv_sqrt

    # Ensure L_sym is symmetric
    L_sym = (L_sym + L_sym.T) / 2

    # Step 4: Compute Eigenvalues and Eigenvectors
    # Use eigh, which is for symmetric matrices
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)


    # Sort eigenvalues and eigenvectors
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]


    # Identify non-zero (or significantly non-zero) eigenvalues
    epsilon = 1e-8  # Threshold for considering an eigenvalue as zero
    nonzero_indices = np.where(eigenvalues > epsilon)[0]

    # Select the smallest non-zero eigenvalues' corresponding eigenvectors
    indicator_vectors = eigenvectors[:, nonzero_indices[:num_dims]]

    return indicator_vectors

#%%

def plot_multiscatter(data, labels=None, saveloc=None):
    """
    Generates scatter plots for multidimensional data and adds histograms on the diagonal.
    
    Parameters:
    - data: A 2D NumPy array of shape (n_samples, n_dimensions).
    - labels: A vector (or list) of labels for each point, used for coloring the points based on their cluster.
    """
    n_samples, n_dimensions = data.shape
    
    # If labels are not provided, create a default label vector
    if labels is None:
        labels = np.zeros(n_samples)
    
    # Get a list of unique labels for coloring purposes
    unique_labels = np.unique(labels)
    
    # Define color map
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0, 1, len(unique_labels)))
    label_colors = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Set up the subplots grid
    fig, axes = plt.subplots(n_dimensions, n_dimensions, figsize=(15, 15))
    
    handles = []
    plot_labels = []

    # Loop over the grid and populate scatter plots and diagonal histograms
    for i in range(n_dimensions):
        for j in range(n_dimensions):
            ax = axes[i, j]
            
            # Diagonal case: Plot a histogram (or a KDE plot) for the distribution of each dimension
            if i == j:
                sns.histplot(data[:, i], ax=ax, kde=True, color="gray")
                ax.set_title(f'Dimension {i + 1} Distribution')
            
            # Off-diagonal case: Create pairwise scatter plots for all combinations of dimensions
            else:
                for label in unique_labels:
                    points = data[labels == label]
                    sc = ax.scatter(points[:, j], points[:, i], label=f'Label {label}', color=label_colors[label], alpha=0.7)
                    
                    # Collect handles and labels for the legend
                    if (i == 0) and (j == 1):  # Only collect from one axis to avoid duplicates
                        handles.append(sc)
                        plot_labels.append(f'Label {label}')
                ax.set_xlabel(f'Dim {j + 1}')
                ax.set_ylabel(f'Dim {i + 1}')
    
    # Add a single legend outside the subplots
    fig.legend(handles, plot_labels, loc='upper right', bbox_to_anchor=(1.1, 0.9), fontsize=30)
    
    
    
    plt.suptitle('Scatter Plot Matrix with Diagonal Histograms')
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    if saveloc != None:
        plt.savefig(saveloc)
    
    plt.show()
    
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
    
#%%

def select_columns(matrix, n):
    # Initialize an empty list to store selected column indices
    selected_columns = []
    
    # Total number of columns in the matrix
    num_columns = matrix.shape[1]
    
    # Select columns following the pattern
    for i in range(n):
        if i % 2 == 0:
            # Even index (0, 2, 4, ...) - take columns from the start
            selected_columns.append(i // 2)
        else:
            # Odd index (1, 3, 5, ...) - take columns from the end
            selected_columns.append(num_columns - 1 - (i // 2))
    
    # Extract the selected columns and concatenate them horizontally
    selected_matrix = matrix[:, selected_columns]
    
    # Optionally, flatten to a 1D vector if desired
    selected_matrix = selected_matrix.reshape(-1, order='F')  # Uncomment if needed
    
    return selected_matrix
    

