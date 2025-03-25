#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 09:24:57 2025

@author: tristan
"""

import clubsdeck as cd
import simul_helpers as sh


#generate matrices
matrices = sh.generate_symmetric_matrices(num_samples=100, matrix_size=20, num_classes=4,
                              signal_strength=0.3, noise_level=1.0)[0]

# Run CLUBS clustering
labels, n_clusters, embedding = cd.clubs(matrices)