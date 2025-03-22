import numpy as np

#%%
def generate_symmetric_matrices(num_samples=100, matrix_size=20, num_classes=4,
                              signal_strength=1.0, noise_level=1.0, random_state=None):
    """
    Generate simulated positive definite matrices with class-specific signals.
    
    Each generated matrix is a sum of:
    1. A class-specific signal matrix (signal_strength * A^T A)
    2. A random noise matrix (noise_level * B^T B)
    Both components are positive definite by construction.
    
    Parameters
    ----------
    num_samples : int, default=100
        Number of samples per class
    matrix_size : int, default=20 
        Size of the square matrices
    num_classes : int, default=4
        Number of distinct classes
    signal_strength : float, default=1.0
        Scaling factor for class-specific signal
    noise_level : float, default=1.0
        Scaling factor for random noise
    random_state : int or None, default=None
        Random seed for reproducibility
    
    Returns
    -------
    matrices : ndarray of shape (num_samples * num_classes, matrix_size, matrix_size)
        Generated symmetric positive definite matrices
    labels : ndarray of shape (num_samples * num_classes,)
        Class labels corresponding to each matrix
    """
    if matrix_size <= 0 or num_samples <= 0 or num_classes <= 0:
        raise ValueError("Matrix size, number of samples, and number of classes must be positive")
    
    rng = np.random.RandomState(random_state)
    
    # Pre-allocate arrays
    total_samples = num_samples * num_classes
    matrices = np.zeros((total_samples, matrix_size, matrix_size))
    labels = np.repeat(np.arange(num_classes), num_samples)
    
    # Generate class-specific signal matrices
    class_signals = []
    for _ in range(num_classes):
        A = rng.randn(matrix_size, matrix_size)
        S = signal_strength * A.T @ A  # Using @ for matrix multiplication
        class_signals.append(S)
    
    # Generate matrices for each class
    for i in range(total_samples):
        # Add class signal
        class_idx = labels[i]
        matrices[i] = class_signals[class_idx]
        
        # Add noise
        B = rng.randn(matrix_size, matrix_size)
        N = noise_level * B.T @ B
        matrices[i] += N
    
    return matrices, labels

