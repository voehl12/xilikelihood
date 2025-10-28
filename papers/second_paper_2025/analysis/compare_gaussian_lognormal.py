"""
Compare Gaussian and Lognormal simulation distributions.

This script loads correlation function measurements from both Gaussian and lognormal
simulations and creates corner plots to show that their distributions are similar
(i.e., the non-Gaussianity in lognormal fields doesn't significantly affect the
measured correlation functions).
"""

import numpy as np
import os
import sys

# Add package to path
sys.path.insert(0, os.path.abspath('../../..'))

from xilikelihood.plotting import plot_corner_comparison, print_comparison_statistics

# Configuration
GAUSSIAN_PATH = "/cluster/scratch/veoehl/xi_sims_gaussian/"
LOGNORMAL_PATH = "/cluster/scratch/veoehl/xi_sims_lognormal/"

# Adjust these paths based on your actual simulation output
# The script will look for files like "croco_KiDS_setup_*_job_1.npz" etc.

# Simulation parameters
NJOBS = 1000  # Number of job files to load
LMAX = 20     # Maximum multipole used in simulations

# Which correlations to plot
# Redshift bin combinations (auto and cross correlations)
REDSHIFT_INDICES = [0, 1, 2]  # e.g., bin pairs 0-0, 1-1, 0-1

# Angular bins to include
ANGULAR_INDICES = [2, 3]  # e.g., two specific angular separation bins

# Output
SAVE_PATH = "comparison_gaussian_vs_lognormal_sims.png"


def main():
    """Main comparison workflow."""
    
    print("="*70)
    print("Gaussian vs Lognormal Simulation Comparison")
    print("="*70)
    print(f"\nGaussian sims:  {GAUSSIAN_PATH}")
    print(f"Lognormal sims: {LOGNORMAL_PATH}")
    print(f"Loading {NJOBS} jobs...")
    print(f"Redshift pairs: {REDSHIFT_INDICES}")
    print(f"Angular bins:   {ANGULAR_INDICES}")
    
    # Check if paths exist
    if not os.path.exists(GAUSSIAN_PATH):
        print(f"\nERROR: Gaussian simulation path not found: {GAUSSIAN_PATH}")
        print("Please update GAUSSIAN_PATH in this script.")
        return
        
    if not os.path.exists(LOGNORMAL_PATH):
        print(f"\nERROR: Lognormal simulation path not found: {LOGNORMAL_PATH}")
        print("Please update LOGNORMAL_PATH in this script.")
        return
    
    # Print statistical comparison
    print("\n" + "="*70)
    print("Computing statistical tests...")
    print("="*70)
    
    try:
        print_comparison_statistics(
            simspath_1=GAUSSIAN_PATH,
            simspath_2=LOGNORMAL_PATH,
            label_1="Gaussian",
            label_2="Lognormal",
            njobs=NJOBS,
            lmax=LMAX,
            redshift_indices=REDSHIFT_INDICES,
            angular_indices=ANGULAR_INDICES
        )
    except Exception as e:
        print(f"Warning: Statistical comparison failed: {e}")
    
    # Create corner plot
    print("\n" + "="*70)
    print("Creating corner plot comparison...")
    print("="*70)
    
    try:
        fig = plot_corner_comparison(
            simspath_1=GAUSSIAN_PATH,
            simspath_2=LOGNORMAL_PATH,
            label_1="Gaussian",
            label_2="Lognormal",
            njobs=NJOBS,
            lmax=LMAX,
            redshift_indices=REDSHIFT_INDICES,
            angular_indices=ANGULAR_INDICES,
            nbins=50,
            alpha=0.6,
            save_path=SAVE_PATH
        )
        
        print(f"\n✓ Comparison plot saved: {SAVE_PATH}")
        print("\nInterpretation:")
        print("  - Diagonal: 1D histograms showing marginal distributions")
        print("  - Lower triangle: 2D contours (solid=Gaussian, dashed=Lognormal)")
        print("  - Upper triangle: Density difference (Lognormal - Gaussian)")
        print("\nIf the distributions are similar, the contours should overlap")
        print("and the difference maps should show small values.")
        
    except Exception as e:
        print(f"ERROR: Failed to create comparison plot: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*70)
    print("Comparison complete!")
    print("="*70)


if __name__ == "__main__":
    main()
