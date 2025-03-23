#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization functions for CLUBS clustering results.
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_multiscatter(data, labels=None, saveloc=None):
    """
    Creates scatter plot matrix with histograms for multidimensional data.
    
    Parameters:
        data: 2D array of shape (n_samples, n_dimensions)
        labels: Optional cluster labels for coloring
        saveloc: Optional path to save figure
    """
    n_samples, n_dimensions = data.shape
    labels = np.zeros(n_samples) if labels is None else labels
    
    # Setup colors
    unique_labels = np.unique(labels)
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(unique_labels)))
    label_colors = {label: colors[i] for i, label in enumerate(unique_labels)}

    # Create plot grid
    fig, axes = plt.subplots(n_dimensions, n_dimensions, figsize=(15, 15))
    handles, plot_labels = [], []

    for i in range(n_dimensions):
        for j in range(n_dimensions):
            ax = axes[i, j]
            
            if i == j:  # Diagonal: histogram
                sns.histplot(data[:, i], ax=ax, kde=True, color="gray")
                ax.set_title(f'Dimension {i + 1} Distribution')
            else:  # Off-diagonal: scatter
                for label in unique_labels:
                    points = data[labels == label]
                    sc = ax.scatter(points[:, j], points[:, i], 
                                  label=f'Label {label}',
                                  color=label_colors[label], 
                                  alpha=0.7)
                    
                    if (i == 0) and (j == 1):
                        handles.append(sc)
                        plot_labels.append(f'Label {label}')
                        
                ax.set_xlabel(f'Dim {j + 1}')
                ax.set_ylabel(f'Dim {i + 1}')

    fig.legend(handles, plot_labels, 
              loc='upper right',
              bbox_to_anchor=(1.1, 0.9), 
              fontsize=30)
    
    plt.suptitle('Scatter Plot Matrix with Diagonal Histograms')
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    if saveloc:
        plt.savefig(saveloc)
    plt.close()  # Close the figure to free memory

def plot_PCA(y, title='Line Plot'):
    """Plots PCA explained variance."""
    sns.set_style('whitegrid')
    
    x = np.arange(1, len(y) + 1)
    fig_width = max(6, min(12, len(y) / 10))
    
    plt.figure(figsize=(fig_width, 6))
    sns.lineplot(x=x, y=y, marker='o')
    
    plt.title(title, fontsize=16)
    plt.xlabel('Index', fontsize=12)
    plt.ylabel('Eigenvalue', fontsize=12)
    plt.tight_layout()
    plt.close()  # Close the figure to free memory 