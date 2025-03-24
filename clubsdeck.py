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
from sklearn.cluster import KMeans

from visualization import plot_multiscatter, plot_PCA

def clubs(targets, reference=None, DRdim=2, embeddingdim=4, gamma=0.1, random_state=None):
    """
    Performs CLUBS clustering on target matrices.
    
    Parameters:
        targets: Array of target matrices to cluster
        reference: Reference matrix (defaults to mean of targets)
        DRdim: Dimension for feature reduction
        embeddingdim: Dimension for spectral embedding
        gamma: RBF kernel parameter
        random_state: Random seed for reproducibility
        
    Returns:
        labels: Cluster assignments
        clusterdim: Estimated number of clusters
        indicator_vectors: Spectral embedding vectors
    """
    # Extract SPADE features
    features = spadefeatures(targets, reference, DRdim)
    features_corr = np.corrcoef(features)
    
    # Create similarity matrix
    A = rbf_kernel(features_corr, gamma=gamma)
    
    # Get spectral embedding
    indicator_vectors = spectral_embedding(A, embeddingdim)
    
    # Estimate optimal number of clusters
    clusterdim = clusterdim_estimate(features)
    
    # Perform clustering
    kmeans = KMeans(
        n_clusters=clusterdim,
        init='k-means++',
        n_init=10,
        random_state=random_state
    )
    labels = kmeans.fit_predict(indicator_vectors)
    
    return labels, clusterdim, indicator_vectors

def spadefeatures(targets, reference=None, DRdim=2):
    """
    Extracts SPADE features from target matrices.
    
    Parameters:
        targets: Array of target matrices
        reference: Reference matrix (defaults to mean of targets)
        DRdim: Dimension for feature reduction
        
    Returns:
        allfeatures: Matrix of extracted features
    """
    if reference is None:
        reference = np.mean(targets, 0)

    allfeatures = None
    for i in range(len(targets)):
        csp_eigvecs, _, _ = basic_csp(reference, targets[i])
        features = features_csp(csp_eigvecs, targets)
        features2 = select_columns(features, DRdim)
        
        if allfeatures is None:
            allfeatures = features2
        else:
            allfeatures = np.vstack([allfeatures, features2])
            
    return allfeatures

def clusterdim_estimate(X, plot=False):
    """
    Estimates optimal number of clusters using PCA and knee detection.
    
    Parameters:
        X: Input data matrix
        
    Returns:
        knee: Estimated number of clusters (minimum 2)
    """
    pca = PCA()  
    pca.fit(X)
    
    # Find knee point in explained variance
    y_knee = pca.explained_variance_ratio_
    x_knee = np.arange(len(y_knee))
    
    kneedle = KneeLocator(x_knee, y_knee, curve='convex', direction='decreasing')
    knee = max(kneedle.knee, 2)  # Ensure minimum of 2 clusters
    
    if plot:
        plot_PCA(y_knee, 'PCA Explained Variance')
        
    return int(knee)

def select_columns(matrix, n):
    """
    Selects columns from matrix alternating between start and end.
    
    Parameters:
        matrix: Input matrix
        n: Number of columns to select
        
    Returns:
        selected_matrix: Matrix with selected columns, flattened
    """
    num_columns = matrix.shape[1]
    selected_columns = []
    
    for i in range(n):
        if i % 2 == 0:
            selected_columns.append(i // 2)  # From start
        else:
            selected_columns.append(num_columns - 1 - (i // 2))  # From end
    
    selected_matrix = matrix[:, selected_columns]
    return selected_matrix.reshape(-1, order='F')

def spectral_embedding(similarity_matrix, num_dims):
    """
    Performs spectral embedding on similarity matrix.
    
    Parameters:
        similarity_matrix: Square symmetric similarity matrix
        num_dims: Number of dimensions for embedding
        
    Returns:
        indicator_vectors: Eigenvectors for embedding
    """
    # Validate input
    assert similarity_matrix.shape[0] == similarity_matrix.shape[1], "Matrix must be square"
    assert np.allclose(similarity_matrix, similarity_matrix.T, atol=1e-8), "Matrix must be symmetric"

    # Compute normalized Laplacian
    degree_matrix = np.diag(similarity_matrix.sum(axis=1))
    laplacian_matrix = degree_matrix - similarity_matrix
    
    with np.errstate(divide='ignore'):
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
        D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
    
    L_sym = D_inv_sqrt @ laplacian_matrix @ D_inv_sqrt
    L_sym = (L_sym + L_sym.T) / 2  # Ensure symmetry

    # Get eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
    idx = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Select vectors for smallest non-zero eigenvalues
    epsilon = 1e-8
    nonzero_indices = np.where(eigenvalues > epsilon)[0]
    indicator_vectors = eigenvectors[:, nonzero_indices[:num_dims]]

    return indicator_vectors

def basic_csp(C1, C2):
    """
    Computes Common Spatial Patterns between two covariance matrices.
    
    Parameters:
        C1, C2: Covariance matrices of same size
        
    Returns:
        csp_eigvecs: CSP eigenvectors
        projected_eigvals1, projected_eigvals2: Projected eigenvalues
    """
    eigvals, csp_eigvecs = eigh(C1, C1 + C2)
    projected_eigvals1 = np.diag(csp_eigvecs.T @ C1 @ csp_eigvecs)
    projected_eigvals2 = np.diag(csp_eigvecs.T @ C2 @ csp_eigvecs)
    return csp_eigvecs, projected_eigvals1, projected_eigvals2

def features_csp(csp_eigvecs, covariances):
    """
    Extracts CSP features from covariance matrices.
    
    Parameters:
        csp_eigvecs: CSP eigenvectors from basic_csp
        covariances: 3D array of covariance matrices
        
    Returns:
        features: Log-variance of projections
    """
    features = np.empty((covariances.shape[0], covariances.shape[1]))
    for i in range(covariances.shape[0]):
        features[i,:]= np.log(np.diag(np.dot(np.dot(np.transpose(csp_eigvecs),np.squeeze(covariances[i,:,:])),csp_eigvecs)))
    return features
