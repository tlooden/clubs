#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 09:24:57 2025

@author: tristan
"""

from clubsdeck import CLUBS
import simul_helpers as sh
from visualization import plot_multiscatter
from sklearn.metrics import adjusted_rand_score, silhouette_score


# Generate matrices
matrices, labels_gt = sh.generate_symmetric_matrices(
    num_samples=200,
    matrix_size=20,
    num_classes=4,
    signal_strength=0.4,
    noise_level=1.0,
    random_state=1
)

# Create and fit CLUBS model
model = CLUBS(dr_dim=8, embedding_dim=4, gamma=0.1, random_state=1)
model.fit(matrices)

# Calculate metrics
ari = adjusted_rand_score(model.labels_, labels_gt)
silhouette = silhouette_score(model.embedding_, model.labels_)

# Plot results
plot_multiscatter(model.embedding_, model.labels_)


#%%

