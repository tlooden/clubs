#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 09:24:57 2025

@author: tristan
"""

import clubsdeck as cd
import simul_helpers as sh
from visualization import plot_multiscatter
from sklearn.metrics import adjusted_rand_score, silhouette_score


#generate matrices
matrices, labels_gt = sh.generate_symmetric_matrices(num_samples=200, matrix_size=20, num_classes=4,
                              signal_strength=0.4, noise_level=1.0, random_state=1)

# Run CLUBS clustering
labels, n_clusters, embedding = cd.clubs(matrices, DRdim=8, embeddingdim=4, gamma=0.1, random_state=1)



#%%


# Calculate metrics
ari = adjusted_rand_score(labels, labels_gt)
silhouette = silhouette_score(embedding, labels)

    

#%%

plot_multiscatter(embedding, labels)


#%%

