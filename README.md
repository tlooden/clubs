# CLUBS (Cluster Learning Underpinned by SPADE)

A Python implementation of the CLUBS algorithm for clustering symmetric positive definite matrices. This implementation uses Common Spatial Patterns (CSP) for feature extraction and spectral clustering for the final clustering step.

## Overview

CLUBS is designed to cluster symmetric positive definite matrices, which are common in various fields.

The algorithm combines:
1. CSP-based feature extraction
2. Spectral embedding
3. Automatic cluster number estimation
4. K-means++ clustering

## Installation

### Using pip
```bash
pip install -r requirements.txt
```

### Using conda
```bash
conda env create -f environment.yml
conda activate clubs_demo
```

## Usage

### Basic Example
```python
from clubsdeck import CLUBS
import numpy as np

# Generate some sample symmetric positive definite matrices
n_samples = 100
matrix_size = 20
matrices = np.random.randn(n_samples, matrix_size, matrix_size)
matrices = np.einsum('nij,nkj->nik', matrices, matrices)  # Make positive definite

# Create and fit CLUBS model
model = CLUBS(dr_dim=8, embedding_dim=4, gamma=0.1)
model.fit(matrices)

# Access results
labels = model.labels_  # Cluster assignments
embedding = model.embedding_  # Spectral embedding
n_clusters = model.n_clusters_  # Estimated number of clusters
```

### Command Line Interface
The repository includes a demo script that generates synthetic data and runs the clustering:

```bash
# Basic usage with default parameters
python clubs_demo.py

# Generate more samples with custom parameters
python clubs_demo.py --samples 200 --size 30 --classes 4

# Save results and visualization
python clubs_demo.py --save-dir ./results --output ./results/visualization.png
```

### Available Parameters

#### CLUBS Model Parameters
- `dr_dim`: Dimension for feature reduction (default: 2)
- `embedding_dim`: Dimension for spectral embedding (default: 4)
- `gamma`: RBF kernel parameter (default: 0.1)
- `random_state`: Random seed for reproducibility

#### Data Generation Parameters (Demo Script)
- `--samples`: Number of samples per class (default: 100)
- `--size`: Size of the square matrices (default: 20)
- `--classes`: Number of distinct classes (default: 4)
- `--signal`: Scaling factor for class-specific signal (default: 0.3)
- `--noise`: Scaling factor for random noise (default: 1.0)

#### General Parameters (Demo Script)
- `--seed`: Random seed for reproducibility (default: 77)
- `--output`: Path to save the visualization
- `--save-dir`: Directory to save results and metrics

## Output

When using the `--save-dir` option, the following files are saved:
- `parameters.json`: All command-line arguments used
- `metrics.json`: Performance metrics (ARI, silhouette score, etc.)
- `labels_gt.npy`: Ground truth labels
- `labels_pred.npy`: Predicted labels
- `embedding.npy`: Spectral embedding
- `confusion_matrix.npy`: Confusion matrix
- `visualization.png`: Plot of results (if --output specified)

## Testing

Run the test suite with:
```bash
pytest tests/
```

For coverage report:
```bash
pytest --cov=clubsdeck tests/
```

## Visualization

The package includes several visualization functions:
- `plot_multiscatter`: Creates scatter plot matrix with histograms
- `plot_PCA`: Plots PCA explained variance



