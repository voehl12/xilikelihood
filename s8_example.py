# %%
import numpy as np
import scipy
from likelihood import XiLikelihood, fiducial_dataspace
from grf_classes import SphereMask, RedshiftBin
import theory_cl
import os
import matplotlib.pyplot as plt
import re
from simulate import xi_sim_nD
from scipy.integrate import cumtrapz


s8 = np.linspace(0.7, 0.9, 5)
exact_lmax = 30
fiducial_cosmo = {
    "H0": 70.0,  # Hubble constant
    "Omega_m": 0.3,  # Matter density parameter
    "Omega_Lambda": 0.7,  # Dark energy density parameter
    "s8": 0.8,  # Amplitude of matter fluctuations
}


mask = SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=exact_lmax, l_smooth=30)


redshift_bins, ang_bins_in_deg = fiducial_dataspace()


likelihood = XiLikelihood(
    mask=mask, redshift_bins=redshift_bins[2:], ang_bins_in_deg=ang_bins_in_deg[:-1]
)

data_shape = likelihood.prep_data_array()
theory_cls = likelihood.initiate_theory_cl(fiducial_cosmo)
sim = xi_sim_nD(
    theory_cls,
    [mask],
    42,
    ang_bins_in_deg,
    lmin=0,
    lmax=mask.lmax,
    plot=False,
    save_pcl=True,
    ximode="namaster",
    batchsize=1,
)
mock_data = sim[0, :, 0, :]
assert mock_data.shape == data_shape.shape, "Mock data shape does not match data shape"
likelihood.initiate_mask_specific()
likelihood.precompute_combination_matrices()


posterior, gauss_posterior = [], []
for s in s8:
    cosmology = fiducial_cosmo.copy()
    cosmology["s8"] = s
    post, gauss_post = likelihood.likelihood(mock_data, cosmology, gausscompare=True)
    posterior.append(post)
    gauss_posterior.append(gauss_post)
posterior, gauss_posterior = np.array(posterior), np.array(gauss_posterior)
integral_post = cumtrapz(posterior, s8, initial=0)[-1]
integral_gauss_post = cumtrapz(gauss_posterior, s8, initial=0)[-1]

# Normalize the posteriors
normalized_post = posterior / integral_post
normalized_gauss_post = gauss_posterior / integral_gauss_post


plt.plot(s8, normalized_post, color="blue", label="Posterior")
plt.plot(s8, normalized_gauss_post, color="red", label="Gaussian Posterior")
plt.xlabel("s8")
plt.ylabel("Posterior")
plt.legend()
plt.savefig("s8_posterior.png")


# %%
