import numpy as np
import pytest
from clubsdeck import CLUBS


def test_select_features():
    """Test the feature selection logic."""
    model = CLUBS(dr_dim=2)
    matrix = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])
    
    result = model._select_features(matrix)
    assert result.shape == (6,)  # 3 rows * 2 columns
    assert np.array_equal(result, np.array([1, 5, 9, 4, 8, 12]))
    
    # Test with different dr_dim
    model = CLUBS(dr_dim=3)
    result = model._select_features(matrix)
    assert result.shape == (9,)  # 3 rows * 3 columns
    assert np.array_equal(result, np.array([1, 5, 9, 4, 8, 12, 2, 6, 10]))

def test_compute_csp():
    """Test the CSP computation."""
    model = CLUBS()
    # Create test covariance matrices
    C1 = np.array([[1, 0.5], [0.5, 1]])
    C2 = np.array([[1, -0.5], [-0.5, 1]])
    
    # Test the function
    csp_eigvecs = model._compute_csp(C1, C2)
    
    # Check output shape
    assert csp_eigvecs.shape == (2, 2)

def test_extract_csp_features():
    """Test CSP feature extraction."""
    model = CLUBS()
    csp_eigvecs = np.array([[1, 0], [0, 1]])
    covariances = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    
    # Test the function
    features = model._extract_csp_features(csp_eigvecs, covariances)
    
    # Check output shape
    assert features.shape == (2, 2)
    
    # Check values
    expected = np.array([[0, 0], [np.log(2), np.log(2)]])
    assert np.allclose(features, expected)

def test_spectral_embedding():
    """Test the spectral embedding computation."""
    model = CLUBS(embedding_dim=2)
    # Create a test similarity matrix
    similarity = np.array([[1, 0.5, 0.2],
                          [0.5, 1, 0.3],
                          [0.2, 0.3, 1]])
    
    # Test the function
    embedding = model._spectral_embedding(similarity)
    
    # Check output shape
    assert embedding.shape == (3, 2)
    
    # Check orthogonality
    assert np.allclose(embedding.T @ embedding, np.eye(2), atol=1e-10)
    
    # Test with invalid input
    with pytest.raises(AssertionError):
        model._spectral_embedding(np.array([[1, 2], [3, 4], [5, 6]]))  # Non-square
    with pytest.raises(AssertionError):
        model._spectral_embedding(np.array([[1, 2], [3, 4]]))  # Non-symmetric

def test_estimate_n_clusters(sample_matrices):
    """Test the cluster number estimation."""
    model = CLUBS()
    n_samples = sample_matrices.shape[0]
    X = sample_matrices.reshape(n_samples, -1)
    
    n_clusters = model._estimate_n_clusters(X)
    assert isinstance(n_clusters, int)
    assert n_clusters >= 2

def test_clubs_fit(sample_matrices):
    """Test the full CLUBS pipeline."""
    # Test with custom parameters
    model = CLUBS(dr_dim=5, embedding_dim=3, gamma=0.2, random_state=42)
    model.fit(sample_matrices)
    
    assert model.labels_ is not None
    assert len(model.labels_) == len(sample_matrices)
    assert model.n_clusters_ >= 2
    assert model.embedding_.shape[1] == 3

    # Test with defaults
    model = CLUBS()
    model.fit(sample_matrices)
    assert model.labels_ is not None
    assert len(model.labels_) == len(sample_matrices)
    assert model.n_clusters_ >= 2
    assert model.embedding_.shape[1] == 4  # default embedding_dim

def test_clubs_initialization():
    """Test CLUBS class initialization."""
    model = CLUBS(dr_dim=3, embedding_dim=2, gamma=0.1)
    assert model.dr_dim == 3
    assert model.embedding_dim == 2
    assert model.gamma == 0.1
    assert model.random_state is None
    
    model = CLUBS()  # Test defaults
    assert model.dr_dim == 2
    assert model.embedding_dim == 4
    assert model.gamma == 0.1 