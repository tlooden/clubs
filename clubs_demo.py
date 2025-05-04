#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLUBS Demo Script
----------------
This script demonstrates the CLUBS (Cluster Learning Underpinned by SPADE) algorithm
on synthetic data. It generates symmetric matrices with class-specific signals,
applies CLUBS clustering, and visualizes the results.

Usage:
    python clubs_demo.py [--samples SAMPLES] [--size SIZE] [--classes CLASSES]
                        [--signal SIGNAL] [--noise NOISE] [--dim DIM] [--seed SEED]
                        [--output OUTPUT] [--save-dir SAVE_DIR]
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import json
from datetime import datetime
from sklearn.metrics import adjusted_rand_score, silhouette_score, confusion_matrix

from clubsdeck import CLUBS
import simul_helpers as simhelp
from visualization import plot_multiscatter

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CLUBS clustering demonstration')
    
    # Data generation parameters
    data_group = parser.add_argument_group('Data Generation')
    data_group.add_argument('--samples', type=int, default=100,
                           help='Number of samples per class')
    data_group.add_argument('--size', type=int, default=20,
                           help='Size of the square matrices')
    data_group.add_argument('--classes', type=int, default=4,
                           help='Number of distinct classes')
    data_group.add_argument('--signal', type=float, default=0.3,
                           help='Scaling factor for class-specific signal')
    data_group.add_argument('--noise', type=float, default=1.0,
                           help='Scaling factor for random noise')
    
    # CLUBS parameters
    clubs_group = parser.add_argument_group('CLUBS Algorithm')
    clubs_group.add_argument('--drdim', type=int, default=10,
                           help='Dimension for feature reduction')
    clubs_group.add_argument('--embeddingdim', type=int, default=4,
                           help='Dimension for spectral embedding')
    clubs_group.add_argument('--gamma', type=float, default=0.1,
                           help='RBF kernel parameter')
    
    # General parameters
    parser.add_argument('--seed', type=int, default=77,
                       help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save the visualization')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save results and metrics')
    
    return parser.parse_args()

def save_results(save_dir, metrics, data, params):
    """Save results and metrics to files."""
    # Create timestamp for unique directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(save_dir) / f"results_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save parameters
    with open(results_dir / "parameters.json", 'w') as f:
        json.dump(params, f, indent=4)
    
    # Save metrics
    with open(results_dir / "metrics.json", 'w') as f:
        json.dump(metrics, f, indent=4)
    
    # Save data
    
    np.save(results_dir / "labels_gt.npy", data['labels_gt'])
    np.save(results_dir / "labels_pred.npy", data['labels_pred'])
    np.save(results_dir / "embedding.npy", data['embedding'])
    np.save(results_dir / "confusion_matrix.npy", data['confusion_matrix'])
    
    return results_dir

def main():
    """Main function to run the CLUBS demonstration."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    params = vars(args)
    
    # Generate synthetic data
    logger.info("Generating synthetic data...")
    mats, labels_gt = simhelp.generate_symmetric_matrices(
        num_samples=args.samples,
        matrix_size=args.size,
        num_classes=args.classes,
        signal_strength=args.signal,
        noise_level=args.noise,
        random_state=args.seed
    )

    
    logger.info(f"Generated {len(mats)} matrices of size {mats.shape[1]}x{mats.shape[2]}")
    logger.info(f"Number of classes: {args.classes}")

    # Run CLUBS clustering
    logger.info("Running CLUBS clustering...")
    model = CLUBS(
        dr_dim=args.drdim,
        embedding_dim=args.embeddingdim,
        gamma=args.gamma,
        random_state=args.seed
    )
    model.fit(mats)
    
    # Calculate metrics
    ari = adjusted_rand_score(model.labels_, labels_gt)
    silhouette = silhouette_score(model.embedding_, model.labels_)
    conf_matrix = confusion_matrix(labels_gt, model.labels_)
    
    # Store metrics
    metrics = {
        'adjusted_rand_index': float(ari),
        'silhouette_score': float(silhouette),
        'estimated_clusters': int(model.n_clusters_)
    }
    
    # Store data
    data = {
        'labels_gt': labels_gt,
        'labels_pred': model.labels_,
        'embedding': model.embedding_,
        'confusion_matrix': conf_matrix
    }
    
    logger.info(f"Adjusted Rand Index: {ari:.3f}")
    logger.info(f"Silhouette Score: {silhouette:.3f}")
    logger.info(f"Estimated number of clusters: {model.n_clusters_}")
    
    # Save results if directory specified
    if args.save_dir:
        results_dir = save_results(args.save_dir, metrics, data, params)
        logger.info(f"Saved results to {results_dir}")
        
        # Only save visualization if explicitly requested
        if args.output:
            output_path = results_dir / "visualization.png"
        else:
            output_path = None
    else:
        output_path = args.output

    # Generate visualization only if save location specified
    if output_path:
        logger.info("Generating visualization...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plot_multiscatter(model.embedding_, model.labels_, saveloc=str(output_path))
        logger.info(f"Saved visualization to {output_path}")
    else:
        logger.info("No visualization save location specified, skipping visualization generation")

if __name__ == '__main__':
    main()
