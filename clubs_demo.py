#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:58:16 2024

@author: triloo
"""

import clubsdeck as cd
import simul_helpers as simhelp
from sklearn.metrics import adjusted_rand_score

#%% generate toy matrices

# Generate synthetic data for testing
toymats = simhelp.generate_symmetric_matrices(
    num_samples=100,
    matrix_size=20, 
    num_classes=4,
    signal_strength=0.3,
    noise_level=1.0,
    random_state=77
)

# Extract matrices and ground truth labels
mats = toymats[0]
labels_gt = toymats[1]

# Run CLUBS clustering
labels, clusters, embedding = cd.clubs(
    mats,
    DRdim=10  # Consider trying different dimensions
)

# Evaluate clustering performance
ari = adjusted_rand_score(labels, labels_gt)
print(f"Adjusted Rand Index: {ari:.3f}")

# Visualize results
cd.plot_multiscatter(embedding, labels)
