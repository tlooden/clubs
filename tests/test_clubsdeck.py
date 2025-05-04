import numpy as np
import pytest
from clubsdeck import CLUBS

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
        spectral_embedding(np.array([[1, 2], [3, 4], [5, 6]]), num_dims=2)  # Non-square
    with pytest.raises(AssertionError):
        spectral_embedding(np.array([[1, 2], [3, 4]]), num_dims=2)  # Non-symmetric

def test_clusterdim_estimate(sample_matrices):
    """Test the clusterdim_estimate function."""
    # Reshape sample_matrices to 2D for PCA
    n_samples = sample_matrices.shape[0]
    X = sample_matrices.reshape(n_samples, -1)
    
    # Test the function
    n_clusters = clusterdim_estimate(X)
    
    # Check output
    assert isinstance(n_clusters, int)
    assert n_clusters >= 2
    

def test_clubs(sample_matrices):
    """Test the clubs function."""
    # sample_matrices is automatically created by pytest
    labels, clusters, embedding = clubs(sample_matrices)
    assert len(labels) == 50  # matches n_samples in fixture
    
    # Create test data
    targets = np.random.randn(100, 20, 20)
    for i in range(len(targets)):
        targets[i] = targets[i].T @ targets[i]  # Make positive definite
    
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

def test_clubs_fit(sample_matrices):
    """Test the full CLUBS pipeline."""
    model = CLUBS(dr_dim=3, embedding_dim=2, random_state=42)
    model.fit(sample_matrices)
    
    # Check results exist and have correct shapes
    assert model.labels_ is not None
    assert model.embedding_ is not None
    assert model.n_clusters_ >= 2
    assert len(model.labels_) == len(sample_matrices)
    assert model.embedding_.shape[1] == 2

def test_select_features():
    """Test the feature selection logic."""
    model = CLUBS(dr_dim=2)
    matrix = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])
    
    result = model._select_features(matrix)
    assert result.shape == (6,)  # 3 rows * 2 columns
    assert np.array_equal(result, np.array([1, 5, 9, 4, 8, 12]))

def test_spectral_embedding():
    """Test the spectral embedding computation."""
    model = CLUBS(embedding_dim=2)
    similarity = np.array([[1, 0.5, 0.2],
                         [0.5, 1, 0.3],
                         [0.2, 0.3, 1]])
    
    embedding = model._spectral_embedding(similarity)
    assert embedding.shape == (3, 2)
    
    # Test invalid inputs
    with pytest.raises(AssertionError):
        model._spectral_embedding(np.array([[1, 2], [3, 4], [5, 6]]))  # Non-square
    with pytest.raises(AssertionError):
        model._spectral_embedding(np.array([[1, 2], [3, 4]]))  # Non-symmetric

def test_estimate_n_clusters(sample_matrices):
    """Test cluster number estimation."""
    model = CLUBS()
    n_samples = sample_matrices.shape[0]
    X = sample_matrices.reshape(n_samples, -1)
    
    n_clusters = model._estimate_n_clusters(X)
    assert isinstance(n_clusters, int)
    assert n_clusters >= 2

def test_extract_csp_features():
    """Test CSP feature extraction."""
    model = CLUBS()
    csp_eigvecs = np.array([[1, 0], [0, 1]])
    covariances = np.array([[[1, 0], [0, 1]], [[2, 0], [0, 2]]])
    
    features = model._extract_csp_features(csp_eigvecs, covariances)
    assert features.shape == (2, 2)
    
    expected = np.array([[0, 0], [np.log(2), np.log(2)]])
    assert np.allclose(features, expected)

def test_compute_csp():
    """Test CSP computation."""
    model = CLUBS()
    C1 = np.array([[1, 0.5], [0.5, 1]])
    C2 = np.array([[1, -0.5], [-0.5, 1]])
    
    csp_eigvecs = model._compute_csp(C1, C2)
    assert csp_eigvecs.shape == (2, 2) 