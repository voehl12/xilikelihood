from plotting import plot_corner
import numpy as np
from likelihood import XiLikelihood, fiducial_dataspace
from mask_props import SphereMask
import os
from itertools import product, combinations
import shutil  # Add this import for deleting files

exact_lmax = 30
fiducial_cosmo = {
    
    "omega_m": 0.31,  # Matter density parameter
    "s8": 0.8,  # Amplitude of matter fluctuations
}

mask = SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=exact_lmax, l_smooth=30)


redshift_bins, ang_bins_in_deg = fiducial_dataspace()

rs = np.array([2,4])
ab = np.array([2, 3])
rs_selection = [redshift_bins[i] for i in rs]
ab_selection = [ang_bins_in_deg[i] for i in ab]
print(ab_selection)
likelihood = XiLikelihood(
        mask=mask, redshift_bins=rs_selection, ang_bins_in_deg=ab_selection,noise=None)
likelihood.initiate_mask_specific()
likelihood.precompute_combination_matrices()
likelihood._prepare_likelihood_components(fiducial_cosmo,highell=True)
#xs,pdfs = likelihood._xs,likelihood._pdfs
data_subset = list(product(np.arange(3), np.arange(2)))


subset_pairs = list(combinations(data_subset, 2))
print(subset_pairs)

clear_cache = True  # Set this flag to True to clear the cache

if clear_cache:
    cache_dir = "/cluster/work/refregier/veoehl/"
    for file in os.listdir(cache_dir):
        if file.startswith("likelihood_2d_cache_"):
            file_path = os.path.join(cache_dir, file)
            print(f"Deleting cached file: {file_path}")
            os.remove(file_path)

likelihood_2d_results = {}
gauss_likelihood_results = {}
x_results = {}

for pair in subset_pairs:
    print(f"Processing pair: {pair}")
    cache_file = f"/cluster/work/refregier/veoehl/likelihood_2d_cache_{pair[0][0]}_{pair[0][1]}_{pair[1][0]}_{pair[1][1]}.npz"
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
        
filepath = "/cluster/work/refregier/veoehl/xi_sims/croco_KiDS_setup_circ10000smoothl30_nonoise_llim_767"
correlations = [3, 10, 12]
plot_corner(simspath=filepath, lmax=mask.lmax, njobs=1000,save_path="comparison_to_sims_with_bootstrap_studentt.png",redshift_indices=correlations,angular_indices=ab,nbins=256)
