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
from scipy.integrate import cumulative_trapezoid as cumtrapz
import sys
from random import randint
from time import time, sleep

sleep(randint(1, 5))

jobnumber = int(sys.argv[1]) - 1
s8 = np.linspace(0.5, 1.0, 200)
exact_lmax = 30
fiducial_cosmo = {
    "H0": 70.0,  # Hubble constant
    "Omega_m": 0.3,  # Matter density parameter
    "Omega_Lambda": 0.7,  # Dark energy density parameter
    "s8": 0.8,  # Amplitude of matter fluctuations
}


mask = SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=exact_lmax, l_smooth=30)


redshift_bins, ang_bins_in_deg = fiducial_dataspace()


likelihood = XiLikelihood(
    mask=mask, redshift_bins=redshift_bins, ang_bins_in_deg=ang_bins_in_deg[:-1]
)

data_shape = likelihood.prep_data_array()
theory_cls = likelihood.initiate_theory_cl(fiducial_cosmo)
likelihood.initiate_mask_specific()
likelihood.precompute_combination_matrices()
""" likelihood._prepare_matrix_products()

likelihood.get_covariance_matrix_lowell()
likelihood.get_covariance_matrix_highell()

gaussian_covariance = likelihood._cov_lowell + likelihood._cov_highell
np.savez('gaussian_covariance.npz',cov=gaussian_covariance,s8=fiducial_cosmo['s8']) """

""" sim = xi_sim_nD(
    theory_cls,
    [mask],
    42,
    ang_bins_in_deg[:-1],
    lmin=0,
    lmax=mask.lmax,
    plot=False,
    save_pcl=True,
    ximode="namaster",
    batchsize=1,
)
mock_data = sim[0, :, 0, :]
exit() """
mock_data = np.load("mock_data_1000sqd.npz")["data"]
gaussian_covariance = np.load("gaussian_covariance.npz")["cov"]
likelihood.gaussian_covariance = gaussian_covariance
assert mock_data.shape == data_shape.shape, (mock_data.shape, data_shape.shape)


cosmology = fiducial_cosmo.copy()
cosmology["s8"] = s8[jobnumber]
post, gauss_post = likelihood.loglikelihood(mock_data, cosmology, gausscompare=True)
np.savez(
    "/cluster/home/veoehl/2ptlikelihood/s8post_1000sqd_{:d}.npz".format(jobnumber),
    exact=post,
    gauss=gauss_post,
    s8=s8[jobnumber],
)
