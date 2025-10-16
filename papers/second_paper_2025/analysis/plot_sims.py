"""
This script generates and analyzes 2D likelihoods for weak lensing correlation functions and compares them to 2D subsets of simulations.

Copula marginal recovery test:
For systematic testing of Student-t and Gaussian copula marginal recovery with realistic covariances and marginals, see:
    test_realistic_covariance_marginals.py

You can run it directly or import its functions for benchmarking your pipeline.
"""

# Reference: import for optional use
# from xilikelihood.test_realistic_covariance_marginals import main as run_copula_marginal_recovery_test

import numpy as np
import os
from itertools import product, combinations
import xilikelihood as xlh
from xilikelihood.plotting import plot_corner
from config import (
    EXACT_LMAX,
    FIDUCIAL_COSMO,
    MASK_CONFIG,
    PACKAGE_DIR
)
import logging


# Set up logging to see xilikelihood package output
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logging.getLogger('xilikelihood').setLevel(logging.INFO)

# Optionally, reduce verbosity from other noisy libraries
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('jax').setLevel(logging.WARNING)


mask = xlh.SphereMask(
    spins=MASK_CONFIG['spins'], 
    circmaskattr=MASK_CONFIG['circmaskattr'], 
    exact_lmax=EXACT_LMAX, 
    l_smooth=MASK_CONFIG['l_smooth'],
    working_dir=PACKAGE_DIR  # Use package root for shared arrays
)

redshift_bins, ang_bins_in_deg = xlh.fiducial_dataspace()

rs = np.array([2,4])
ab = np.array([2, 3])
rs_selection = [redshift_bins[i] for i in rs]
ab_selection = [ang_bins_in_deg[i] for i in ab]

likelihood = xlh.XiLikelihood(
        mask=mask, redshift_bins=rs_selection, ang_bins_in_deg=ab_selection,noise=None,include_ximinus=False)
likelihood.setup_likelihood()
likelihood._prepare_likelihood_components(FIDUCIAL_COSMO,highell=True)
#xs,pdfs = likelihood._xs,likelihood._pdfs
data_subset = list(product(np.arange(3), np.arange(2)))


subset_pairs = list(combinations(data_subset, 2))
print(subset_pairs)

x_results = {}

# --- User options ---
clear_cache = True  # Set this flag to True to clear the cache
use_student_t = True  # Set to True for Student-t copula, False for Gaussian copula
student_t_dof = 15    # Degrees of freedom for Student-t copula (if used)

cache_dir = "/cluster/home/veoehl/xilikelihood/papers/second_paper_2025/data/postrefactor"

# Adjust cache file prefix and output file names based on copula type
copula_tag = f"studentt_df{student_t_dof}" if use_student_t else "gauss"

if clear_cache:
    for file in os.listdir(cache_dir):
        if file.startswith(f"likelihood_2d_cache_{copula_tag}"):
            file_path = os.path.join(cache_dir, file)
            print(f"Deleting cached file: {file_path}")
            os.remove(file_path)

likelihood_2d_results = {}
gauss_likelihood_results = {}
x_results = {}

for pair in subset_pairs:
    print(f"Processing pair: {pair}")
    cache_file = f"likelihood_2d_cache_{copula_tag}_{pair[0][0]}_{pair[0][1]}_{pair[1][0]}_{pair[1][1]}.npz"
    cache_file = os.path.join(cache_dir, cache_file)

    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}")
        data = np.load(cache_file)
        likelihood_2d_results[pair] = np.exp(data["likelihood_2d"])
        gauss_likelihood_results[pair] = np.exp(data["gauss_loglikelihood"])
        x_results[pair] = data["x"]
    else:
        print(f"Calculating likelihood for pair: {pair}")
        
        # Set up coupling-specific parameters
        if use_student_t:
            # Student-t copula parameters
            likelihood.config.student_t_dof = student_t_dof
            copula_kwargs = {'use_student_t': True}
        else:
            # Gaussian copula parameters
            copula_kwargs = {'use_student_t': False}
        
        x, loglikelihood_2d, gauss_loglikelihood = likelihood.likelihood_function_2d(
            data_subset=pair,
            gausscompare=True,
            **copula_kwargs
        )
        likelihood_2d_results[pair] = np.exp(loglikelihood_2d)
        gauss_likelihood_results[pair] = np.exp(gauss_loglikelihood)
        x_results[pair] = x
        # Save the results to a file, always include dof if Student-t
        if use_student_t:
            np.savez(cache_file, x=x, likelihood_2d=loglikelihood_2d, gauss_loglikelihood=gauss_loglikelihood, student_t_dof=student_t_dof)
        else:
            np.savez(cache_file, x=x, likelihood_2d=loglikelihood_2d, gauss_loglikelihood=gauss_loglikelihood)
        del loglikelihood_2d, gauss_loglikelihood
        

# Example: plot with copula type and dof in output filename
filepath = "/cluster/work/refregier/veoehl/xi_sims/croco_KiDS_setup_circ10000smoothl30_nonoise_llim_767"
correlations = [3, 10, 12]
plot_name = f"comparison_to_sims_newclipping_{copula_tag}.png"
plot_corner(
    simspath=filepath,
    lmax=mask.lmax,
    njobs=1000,
    save_path=plot_name,
    redshift_indices=correlations,
    angular_indices=ab,
    nbins=256,
    copula_tag=copula_tag,
    cache_dir=cache_dir
)
