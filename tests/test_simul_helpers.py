import numpy as np
import pytest
from simul_helpers import generate_symmetric_matrices

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


