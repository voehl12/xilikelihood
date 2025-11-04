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
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

# Add package to path
sys.path.insert(0, os.path.abspath('../../..'))

from xilikelihood.plotting import plot_corner_comparison, print_comparison_statistics

# Configuration
GAUSSIAN_PATH = "/cluster/work/refregier/veoehl/xi_sims/croco_KiDS_setup_circ10000smoothl30_nonoise_llim_20_gaussian/"

LOGNORMAL_PATH = "/cluster/work/refregier/veoehl/xi_sims/croco_KiDS_setup_circ10000smoothl30_nonoise_llim_20_lognormal/"

# Adjust these paths based on your actual simulation output
# The script will look for files like "croco_KiDS_setup_*_job_1.npz" etc.

# Simulation parameters
NJOBS = 1000  # Number of job files to load
LMAX = 20     # Maximum multipole used in simulations

# Which correlations to plot
# Redshift bin combinations (auto and cross correlations)
REDSHIFT_INDICES = [3,10,12]  # e.g., bin pairs 0-0, 1-1, 0-1

# Angular bins to include
ANGULAR_INDICES = [2, 3]  # e.g., two specific angular separation bins

# Output
SAVE_PATH = "comparison_gaussian_vs_lognormal_sims.png"


def main():
    """Main comparison workflow."""
    
    logger.info("="*70)
    logger.info("Gaussian vs Lognormal Simulation Comparison")
    logger.info("="*70)
    logger.info(f"\nGaussian sims:  {GAUSSIAN_PATH}")
    logger.info(f"Lognormal sims: {LOGNORMAL_PATH}")
    logger.info(f"Loading {NJOBS} jobs...")
    logger.info(f"Redshift pairs: {REDSHIFT_INDICES}")
    logger.info(f"Angular bins:   {ANGULAR_INDICES}")
    
    # Check if paths exist
    if not os.path.exists(GAUSSIAN_PATH):
        logger.error(f"\nERROR: Gaussian simulation path not found: {GAUSSIAN_PATH}")
        logger.error("Please update GAUSSIAN_PATH in this script.")
        return
        
    if not os.path.exists(LOGNORMAL_PATH):
        logger.error(f"\nERROR: Lognormal simulation path not found: {LOGNORMAL_PATH}")
        logger.error("Please update LOGNORMAL_PATH in this script.")
        return
    
    """ # Print statistical comparison
    logger.info("\n" + "="*70)
    logger.info("Computing statistical tests...")
    logger.info("="*70)
    
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
        logger.warning(f"Warning: Statistical comparison failed: {e}") """
    
    # Create corner plot
    logger.info("\n" + "="*70)
    logger.info("Creating corner plot comparison...")
    logger.info("="*70)
    
    try:
        fig = plot_corner_comparison(
            simspath_1=GAUSSIAN_PATH,
            simspath_2=LOGNORMAL_PATH,
            label_1=r"Gaussian",
            label_2=r"Lognormal",
            njobs=NJOBS,
            lmax=LMAX,
            redshift_indices=REDSHIFT_INDICES,
            angular_indices=ANGULAR_INDICES,
            nbins=256,
            alpha=0.8,
            save_path=SAVE_PATH
        )
        
        logger.info(f"\n✓ Comparison plot saved: {SAVE_PATH}")
        logger.info("\nInterpretation:")
        logger.info("  - Diagonal: 1D histograms showing marginal distributions")
        logger.info("  - Lower triangle: 2D contours (solid=Gaussian, dashed=Lognormal)")
        logger.info("  - Upper triangle: Density difference (Lognormal - Gaussian)")
        logger.info("\nIf the distributions are similar, the contours should overlap")
        logger.info("and the difference maps should show small values.")
        
    except Exception as e:
        logger.error(f"ERROR: Failed to create comparison plot: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("\n" + "="*70)
    logger.info("Comparison complete!")
    logger.info("="*70)


if __name__ == "__main__":
    main()
