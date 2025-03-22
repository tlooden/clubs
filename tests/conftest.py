import pytest
import numpy as np

@pytest.fixture
def random_state():
    """Fixture for random state."""
    return 42

@pytest.fixture
def sample_matrices(random_state):
    """Fixture for sample symmetric matrices."""
    rng = np.random.RandomState(random_state)
    n_samples = 50
    matrix_size = 10
    
    matrices = np.zeros((n_samples, matrix_size, matrix_size))
    for i in range(n_samples):
        A = rng.randn(matrix_size, matrix_size)
        matrices[i] = A.T @ A  # Make positive definite
    
    return matrices

@pytest.fixture
def sample_labels():
    """Fixture for sample labels."""
    return np.repeat(np.arange(4), 12)  # 4 classes, 12 samples each 