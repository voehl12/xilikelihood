import numpy as np
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'
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

EXACT_LMAX=30
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
print(ab_selection)
likelihood = xlh.XiLikelihood(
        mask=mask, redshift_bins=rs_selection, ang_bins_in_deg=ab_selection,noise=None,include_ximinus=False,highell=True)
likelihood.setup_likelihood()
likelihood._prepare_likelihood_components(FIDUCIAL_COSMO,highell=True)
#xs,pdfs = likelihood._xs,likelihood._pdfs
data_subset = list(product(np.arange(3), np.arange(2)))
redshift_combs_indices = [3,10,12]

subset_pairs = list(combinations(data_subset, 2))
print(subset_pairs)

clear_cache = True  # Set this flag to True to clear the cache

cache_dir = "/cluster/home/veoehl/xilikelihood/papers/second_paper_2025/data/postrefactor"

if clear_cache:
    
    for file in os.listdir(cache_dir):
        if file.startswith("likelihood_2d_cache_"):
            file_path = os.path.join(cache_dir, file)
            print(f"Deleting cached file: {file_path}")
            os.remove(file_path)

likelihood_2d_results = {}
gauss_likelihood_results = {}
x_results = {}

for pair in subset_pairs:
    full_rs_i = redshift_combs_indices[pair[0][0]]  # Maps calculated 0,1,2 to full dataset 2,4,...
    full_ang_i = ab[pair[0][1]]
    full_rs_j = redshift_combs_indices[pair[1][0]]
    full_ang_j = ab[pair[1][1]]
    print(f"Processing pair: {pair}")
    cache_file = f"likelihood_2d_cache_{full_rs_i}_{full_ang_i}_{full_rs_j}_{full_ang_j}.npz"
    cache_file = os.path.join(cache_dir, cache_file)

    if os.path.exists(cache_file):
        print(f"Loading cached results from {cache_file}")
        data = np.load(cache_file)
        likelihood_2d_results[pair] = np.exp(data["likelihood_2d"])
        gauss_likelihood_results[pair] = np.exp(data["gauss_loglikelihood"])
        x_results[pair] = data["x"]
    else:
        print(f"Calculating likelihood for pair: {pair}")
        x,loglikelihood_2d,gauss_loglikelihood = likelihood.likelihood_function_2d(data_subset=pair,gausscompare=True)
        likelihood_2d_results[pair] = np.exp(loglikelihood_2d)
        gauss_likelihood_results[pair] = np.exp(gauss_loglikelihood)
        x_results[pair] = x
        
        # Save the results to a file
        np.savez(cache_file, x=x, likelihood_2d=loglikelihood_2d, gauss_loglikelihood=gauss_loglikelihood)
        del loglikelihood_2d, gauss_loglikelihood
        
filepath = "/cluster/work/refregier/veoehl/xi_sims/croco_KiDS_setup_circ10000smoothl30_nonoise_llim_767"
angular_bins_to_plot = [2,3]
correlations_to_plot = [10,12]
plot_corner(simspath=filepath, likelihoodpath=cache_dir, lmax=767, njobs=1000,save_path="comparison_to_sims_with_bootstrap_subplot_newcolor_gaussian.png",redshift_indices=correlations_to_plot,angular_indices=angular_bins_to_plot,nbins=256,use_gaussian=True)
