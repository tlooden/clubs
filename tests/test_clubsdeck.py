import numpy as np
import pytest
from clubsdeck import (
    clubs, spadefeatures, clusterdim_estimate, select_columns,
    spectral_embedding, basic_csp, features_csp
)

def test_select_columns():
    """Test the select_columns function."""
    # Create a test matrix
    matrix = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])
    
    # Test with n=2
    result = select_columns(matrix, 2)
    assert result.shape == (6,)  # 3 rows * 2 columns
    assert np.array_equal(result, np.array([1, 5, 9, 4, 8, 12]))
    
    # Test with n=3
    result = select_columns(matrix, 3)
    assert result.shape == (9,)  # 3 rows * 3 columns
    assert np.array_equal(result, np.array([1, 5, 9, 4, 8, 12, 2, 6, 10]))

def test_basic_csp():
    """Test the basic_csp function."""
    # Create test covariance matrices
    C1 = np.array([[1, 0.5], [0.5, 1]])
    C2 = np.array([[1, -0.5], [-0.5, 1]])
    
    # Test the function
    csp_eigvecs, proj_eigvals1, proj_eigvals2 = basic_csp(C1, C2)
    
    # Check output shapes
    assert csp_eigvecs.shape == (2, 2)
    assert proj_eigvals1.shape == (2,)
    assert proj_eigvals2.shape == (2,)
    
    # Check orthogonality
    assert np.allclose(csp_eigvecs.T @ csp_eigvecs, np.eye(2))

def test_features_csp():
    """Test the features_csp function."""
    # Create test data
    csp_eigvecs = np.array([[1, 0], [0, 1]])
    covariances = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    
    # Test the function
    features = features_csp(csp_eigvecs, covariances)
    
    # Check output shape
    assert features.shape == (2, 2)
    
    # Check values
    expected = np.array([[0, 0], [np.log(2), np.log(2)]])
    assert np.allclose(features, expected)

def test_spectral_embedding():
    """Test the spectral_embedding function."""
    # Create a test similarity matrix
    similarity = np.array([[1, 0.5, 0.2],
                          [0.5, 1, 0.3],
                          [0.2, 0.3, 1]])
    
    # Test the function
    embedding = spectral_embedding(similarity, num_dims=2)
    
    # Check output shape
    assert embedding.shape == (3, 2)
    
    # Check orthogonality
    assert np.allclose(embedding.T @ embedding, np.eye(2), atol=1e-10)
    
    # Test with invalid input
    with pytest.raises(AssertionError):
        spectral_embedding(np.array([[1, 2], [3, 4], [5, 6]]))  # Non-square
    with pytest.raises(AssertionError):
        spectral_embedding(np.array([[1, 2], [3, 4]]))  # Non-symmetric

def test_clusterdim_estimate():
    """Test the clusterdim_estimate function."""
    # Create test data
    X = np.random.randn(100, 10)
    
    # Test the function
    n_clusters = clusterdim_estimate(X)
    
    # Check output
    assert isinstance(n_clusters, int)
    assert n_clusters >= 2
    
    # Test with plot option
    n_clusters_plot = clusterdim_estimate(X, plot=True)
    assert n_clusters_plot >= 2

def test_clubs():
    """Test the clubs function."""
    # Create test data
    targets = np.random.randn(100, 20, 20)
    for i in range(len(targets)):
        targets[i] = targets[i].T @ targets[i]  # Make positive definite
    
    # Test the function
    labels, clusters, embedding = clubs(targets)
    
    # Check outputs
    assert len(labels) == 100
    assert clusters >= 2
    assert embedding.shape[0] == 100
    
    # Test with custom parameters
    labels2, clusters2, embedding2 = clubs(
        targets,
        DRdim=5,
        embeddingdim=3,
        gamma=0.2,
        random_state=42
    )
    assert len(labels2) == 100
    assert clusters2 >= 2
    assert embedding2.shape[1] == 3 