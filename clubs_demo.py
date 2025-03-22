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
                        [--output OUTPUT]
"""

import argparse
import logging
from pathlib import Path
import numpy as np

import clubsdeck as cd
import simul_helpers as simhelp
from sklearn.metrics import adjusted_rand_score

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CLUBS clustering demonstration')
    parser.add_argument('--samples', type=int, default=100,
                      help='Number of samples per class')
    parser.add_argument('--size', type=int, default=20,
                      help='Size of the square matrices')
    parser.add_argument('--classes', type=int, default=4,
                      help='Number of distinct classes')
    parser.add_argument('--signal', type=float, default=0.3,
                      help='Scaling factor for class-specific signal')
    parser.add_argument('--noise', type=float, default=1.0,
                      help='Scaling factor for random noise')
    parser.add_argument('--dim', type=int, default=10,
                      help='Dimension for feature reduction')
    parser.add_argument('--seed', type=int, default=77,
                      help='Random seed for reproducibility')
    parser.add_argument('--output', type=str, default=None,
                      help='Path to save the visualization')
    
    return parser.parse_args()

def main():
    """Main function to run the CLUBS demonstration."""
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    logger.info("Generating synthetic data...")
    toymats = simhelp.generate_symmetric_matrices(
        num_samples=args.samples,
        matrix_size=args.size,
        num_classes=args.classes,
        signal_strength=args.signal,
        noise_level=args.noise,
        random_state=args.seed
    )

    # Extract matrices and ground truth labels
    mats = toymats[0]
    labels_gt = toymats[1]
    
    logger.info(f"Generated {len(mats)} matrices of size {mats.shape[1]}x{mats.shape[2]}")
    logger.info(f"Number of classes: {args.classes}")

    # Run CLUBS clustering
    logger.info("Running CLUBS clustering...")
    labels, clusters, embedding = cd.clubs(
        mats,
        DRdim=args.dim
    )

    # Evaluate clustering performance
    ari = adjusted_rand_score(labels, labels_gt)
    logger.info(f"Adjusted Rand Index: {ari:.3f}")
    logger.info(f"Estimated number of clusters: {clusters}")

    # Visualize results
    logger.info("Generating visualization...")
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cd.plot_multiscatter(embedding, labels, saveloc=str(output_path))
        logger.info(f"Saved visualization to {output_path}")
    else:
        cd.plot_multiscatter(embedding, labels)

if __name__ == '__main__':
    main()
