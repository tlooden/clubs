#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:32:38 2024

Cluster learning underpinned by SPADE (CLUBS) toolkit

@author: triloo
"""

import numpy as np
from kneed import KneeLocator
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt

#%%

def clubs(targets, reference = None, DRdim = 2, embeddingdim=4, gamma=0.1, random_state = None):
    
    # subject specific spade
    features = spadefeatures(targets, reference, DRdim)
    features_corr = np.corrcoef(features)
    
    # sparsify
    A = rbf_kernel(features_corr, gamma=gamma)
    
    # spectral embedding
    indicator_vectors =  spectral_embedding(A, embeddingdim)
    
    # find cluster dimensionality
    clusterdim = clusterdim_estimate(features)
    
    # clustering
    kmeans = KMeans(n_clusters=clusterdim, init='k-means++', n_init=10, random_state= random_state)
    labels = kmeans.fit_predict(indicator_vectors)
    
    return labels, clusterdim, indicator_vectors

#%%

# enter reference matrix and target matrices, returns dim spade features
def spadefeatures(targets, reference = None, DRdim=2):
    
    # if no reference is presented, take mean of the targets.
    if reference is None:
        reference = np.mean(targets,0)

    for i in range(len(targets)):
        csp_eigvecs, proj_eigvals1, proj_eigvals2 = basic_csp(reference, targets[i])
        
        features=features_csp(csp_eigvecs,targets)
        features2=select_columns(features,DRdim)
        
        if i==0:
            allfeatures=features2
        else:
            allfeatures=np.vstack([allfeatures,features2])
            
    return allfeatures

#%%

def clusterdim_estimate(X):
    
    pca = PCA(n_components=10)
    pca.fit(X)
    X_pca=pca.transform(X)
    plot_PCA(pca.explained_variance_ratio_, 'PCA')
    
    # find bending point in the graph
    y_knee=pca.explained_variance_ratio_
    x_knee=np.arange(len(y_knee))
    
    kneedle = KneeLocator(x_knee, y_knee, curve='convex', direction='decreasing')
    knee = kneedle.knee
    
    if knee == 1:
        knee = 2
    
    return knee


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
    

#%%

def spectral_embedding(similarity_matrix, num_dims):
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
def basic_csp(C1,C2):
#inputs: C1 and C2, 2 covariance matrices of same size
#outputs: csp_eigvecs, projected_eigvals1, projected_eigvals2, (using diag(w'*C*w))
    eigvals, csp_eigvecs = eigh(C1, C1+C2, eigvals_only=False)
    projected_eigvals1 = np.diag(np.dot(np.dot(np.transpose(csp_eigvecs),C1),csp_eigvecs))
    projected_eigvals2 = np.diag(np.dot(np.dot(np.transpose(csp_eigvecs),C2),csp_eigvecs))
    return csp_eigvecs, projected_eigvals1, projected_eigvals2

#%%
def features_csp(csp_eigvecs,covariances):
#inputs: csp_eigvecs is output of basic_csp,
#        covariances is a 3d numpy array of covariance matrices,with first dimension de number of covariances
#output: log-variance of data projections to csp_eigvecs
    features=np.ndarray(shape=(covariances.shape[0],covariances.shape[1]))
    for i in range(covariances.shape[0]):
        features[i,:]= np.log(np.diag(np.dot(np.dot(np.transpose(csp_eigvecs),np.squeeze(covariances[i,:,:])),csp_eigvecs)))
    return features
