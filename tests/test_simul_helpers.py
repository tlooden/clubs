import numpy as np
import pytest
from simul_helpers import generate_symmetric_matrices, minimal_epsilon_graph, scatter3D

def test_generate_symmetric_matrices():
    """Test the generate_symmetric_matrices function."""
    # Test with default parameters
    matrices, labels = generate_symmetric_matrices()
    assert matrices.shape[0] == 400  # 100 samples * 4 classes
    assert matrices.shape[1] == 20   # matrix_size
    assert matrices.shape[2] == 20   # matrix_size
    assert len(labels) == 400
    assert len(np.unique(labels)) == 4
    
    # Test matrix properties
    for matrix in matrices:
        # Check symmetry
        assert np.allclose(matrix, matrix.T)
        # Check positive definiteness (eigenvalues > 0)
        eigenvalues = np.linalg.eigvals(matrix)
        assert np.all(eigenvalues > 0)
    
    # Test with custom parameters
    matrices, labels = generate_symmetric_matrices(
        num_samples=50,
        matrix_size=10,
        num_classes=3,
        signal_strength=0.5,
        noise_level=0.5,
        random_state=42
    )
    assert matrices.shape == (150, 10, 10)  # 50 samples * 3 classes
    assert len(np.unique(labels)) == 3
    
    # Test error handling
    with pytest.raises(ValueError):
        generate_symmetric_matrices(matrix_size=-1)
    with pytest.raises(ValueError):
        generate_symmetric_matrices(num_samples=0)
    with pytest.raises(ValueError):
        generate_symmetric_matrices(num_classes=0)

def test_minimal_epsilon_graph():
    """Test the minimal_epsilon_graph function."""
    # Create a simple 2D dataset
    X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    
    # Test the function
    adjacency, epsilon = minimal_epsilon_graph(X)
    
    # Check output types and shapes
    assert isinstance(adjacency, np.ndarray)
    assert adjacency.shape == (4, 4)  # 4 points
    assert isinstance(epsilon, float)
    assert epsilon > 0
    
    # Check adjacency matrix properties
    assert np.allclose(adjacency, adjacency.T)  # Symmetric
    assert np.all(np.diag(adjacency) == 0)      # No self-loops
    assert np.all(adjacency >= 0)               # Non-negative
    
    # Test with different data
    X2 = np.random.randn(10, 3)  # 10 points in 3D
    adjacency2, epsilon2 = minimal_epsilon_graph(X2)
    assert adjacency2.shape == (10, 10)
    assert epsilon2 > 0

def test_scatter3D():
    """Test the scatter3D function."""
    # Create test data
    data = np.random.randn(100, 3)
    labels = np.random.randint(0, 3, 100)
    
    # Test the function
    try:
        scatter3D(data, labels)
    except Exception as e:
        pytest.fail(f"scatter3D raised {type(e)} unexpectedly!")
    
    # Test with default labels
    try:
        scatter3D(data)
    except Exception as e:
        pytest.fail(f"scatter3D raised {type(e)} unexpectedly!")
    
    # Test with invalid data shape
    with pytest.raises(ValueError):
        scatter3D(np.random.randn(100, 2))  # 2D data
    with pytest.raises(ValueError):
        scatter3D(np.random.randn(100, 4))  # 4D data 