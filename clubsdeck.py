#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 14:32:38 2024

Cluster learning underpinned by SPADE (CLUBS) toolkit

@author: triloo
"""

import numpy as np
from scipy.linalg import eigh
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from kneed import KneeLocator
from visualization import plot_multiscatter, plot_PCA

class CLUBS:
    """
    Cluster Learning Underpinned by SPADE (CLUBS)
    
    A class for clustering symmetric positive definite matrices using
    Common Spatial Patterns (CSP) and spectral clustering.
    """
    
    def __init__(self, dr_dim=2, embedding_dim=4, gamma=0.1, random_state=None):
        """
        Initialize CLUBS algorithm.
        
        Parameters:
            dr_dim: Dimension for feature reduction
            embedding_dim: Dimension for spectral embedding
            gamma: RBF kernel parameter
            random_state: Random seed for reproducibility
        """
        self.dr_dim = dr_dim
        self.embedding_dim = embedding_dim
        self.gamma = gamma
        self.random_state = random_state
        
        # Results storage
        self.embedding_ = None
        self.labels_ = None
        self.n_clusters_ = None
        self.features_ = None
        
    def fit(self, matrices, reference=None):
        """
        Fit the CLUBS model to the input matrices.
        
        Parameters:
            matrices: Array of symmetric positive definite matrices
            reference: Reference matrix (defaults to mean of matrices)
        
        Returns:
            self: The fitted model
        """
        # Extract SPADE features
        self.features_ = self._extract_spade_features(matrices, reference)
        features_corr = np.corrcoef(self.features_)
        
        # Create similarity matrix using scikit-learn's RBF kernel
        A = rbf_kernel(features_corr, gamma=self.gamma)
        
        # Get spectral embedding
        self.embedding_ = self._spectral_embedding(A)
        
        # Estimate optimal number of clusters
        self.n_clusters_ = self._estimate_n_clusters(self.features_)
        
        # Perform clustering
        self.labels_ = self._cluster_embedding()
        
        return self
    
    def _extract_spade_features(self, matrices, reference=None):
        """Extract SPADE features from matrices."""
        if reference is None:
            reference = np.mean(matrices, 0)

        n_matrices = len(matrices)
        features = np.zeros((n_matrices, self.dr_dim * n_matrices))

        for i in range(n_matrices):
            csp_eigvecs = self._compute_csp(reference, matrices[i])
            matrix_features = self._extract_csp_features(csp_eigvecs, matrices)
            features[i,:] = self._select_features(matrix_features)
            
        return features
    
    def _compute_csp(self, C1, C2):
        """Compute Common Spatial Patterns between two matrices."""
        eigvals, csp_eigvecs = eigh(C1, C1 + C2)
        return csp_eigvecs
    
    def _extract_csp_features(self, csp_eigvecs, matrices):
        """Extract CSP features from matrices."""
        features = np.empty((matrices.shape[0], matrices.shape[1]))
        for i in range(matrices.shape[0]):
            features[i,:] = np.log(np.diag(
                csp_eigvecs.T @ np.squeeze(matrices[i,:,:]) @ csp_eigvecs
            ))
        return features
    
    def _select_features(self, matrix):
        """Select features alternating between start and end."""
        num_columns = matrix.shape[1]
        selected_columns = [
            i // 2 if i % 2 == 0 else num_columns - 1 - (i // 2)
            for i in range(self.dr_dim)
        ]
        selected = matrix[:, selected_columns]
        return selected.reshape(-1, order='F')
    

    def _spectral_embedding(self, similarity_matrix):
        """Compute spectral embedding."""
        # Validate input
        assert similarity_matrix.shape[0] == similarity_matrix.shape[1]
        assert np.allclose(similarity_matrix, similarity_matrix.T, atol=1e-8)

        # Compute normalized Laplacian
        degree_matrix = np.diag(similarity_matrix.sum(axis=1))
        laplacian_matrix = degree_matrix - similarity_matrix
        
        with np.errstate(divide='ignore'):
            D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(degree_matrix)))
            D_inv_sqrt[np.isinf(D_inv_sqrt)] = 0
        
        L_sym = D_inv_sqrt @ laplacian_matrix @ D_inv_sqrt
        L_sym = (L_sym + L_sym.T) / 2

        # Get eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(L_sym)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select vectors
        epsilon = 1e-8
        nonzero_indices = np.where(eigenvalues > epsilon)[0]
        return eigenvectors[:, nonzero_indices[:self.embedding_dim]]
    
    def _estimate_n_clusters(self, X, plot=False):
        """Estimate optimal number of clusters."""
        pca = PCA(n_components=10)
        pca.fit(X)
        
        y_knee = pca.explained_variance_ratio_
        x_knee = np.arange(len(y_knee))
        
        kneedle = KneeLocator(x_knee, y_knee, curve='convex', direction='decreasing')
        n_clusters = max(kneedle.knee, 2)
        
        if plot:
            plot_PCA(y_knee, 'PCA Explained Variance')
            
        return int(n_clusters)
    
    def _cluster_embedding(self):
        """Perform clustering on the embedding."""
        kmeans = KMeans(
            n_clusters=self.n_clusters_,
            init='k-means++',
            n_init=10,
            random_state=self.random_state
        )
        return kmeans.fit_predict(self.embedding_)

