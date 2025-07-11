import numpy as np
from likelihood import XiLikelihood, fiducial_dataspace
from mask_props import SphereMask
import time
import sys, os
os.export("JAX_PLATFORMS", "cpu") # Set JAX to use CPU
jobnumber = int(sys.argv[1]) - 1

exact_lmax = 30
mask = SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=exact_lmax, l_smooth=30)


redshift_bins, ang_bins_in_deg = fiducial_dataspace()
ang_bins_in_deg = ang_bins_in_deg[:-1]



mock_data = np.load("fiducial_data_1000sqd.npz")["data"]
gaussian_covariance = np.load("gaussian_covariance_1000sqd.npz")["cov"]

likelihood = XiLikelihood(
        mask=mask, redshift_bins=redshift_bins, ang_bins_in_deg=ang_bins_in_deg)
likelihood.initiate_mask_specific()
likelihood.precompute_combination_matrices()

likelihood.gaussian_covariance = gaussian_covariance



omega_m_prior = np.linspace(0.1, 0.5, 100)
s8_prior = np.linspace(0.5, 1.1, 100)
prior_pairs = np.meshgrid(omega_m_prior, s8_prior)
prior_pairs = np.vstack([prior_pairs[0].ravel(), prior_pairs[1].ravel()]).T

split_prior_pairs = np.array_split(prior_pairs, 500)

# Get the subset for the current job
subset_pairs = split_prior_pairs[jobnumber]
results_dtype = np.dtype([
    ("exact_post", np.float64),
    ("gauss_post", np.float64),
    ("s8", np.float64),
    ("omega_m", np.float64)
])

results = np.empty(len(subset_pairs), dtype=results_dtype)
for i, (omega_m, s8) in enumerate(subset_pairs):
    cosmology = {"omega_m": omega_m, "s8": s8}
    post, gauss_post = likelihood.loglikelihood(mock_data, cosmology, gausscompare=True)
    results[i] = (post, gauss_post, s8, omega_m)

np.save(f"/cluster/scratch/veoehl/posteriors/posterior_{jobnumber}.npy", results)





