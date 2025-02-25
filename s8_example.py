# %%
import numpy as np
import scipy
from likelihood import XiLikelihood
from grf_classes import SphereMask, RedshiftBin
import theory_cl
import os
import matplotlib.pyplot as plt
import re

# %%
s8 = np.linspace(0.6, 1.0, 10)
exact_lmax = 30
cosmo = {
    "H0": 70.0,  # Hubble constant
    "Omega_m": 0.3,  # Matter density parameter
    "Omega_Lambda": 0.7,  # Dark energy density parameter
    "s8": 0.8,  # Amplitude of matter fluctuations
}
rs_directory = "redshift_bins/KiDS/"
redshift_filepaths = os.listdir(rs_directory)
pattern = re.compile(r"TOMO(\d+)")
nbins = [int(pattern.search(f).group(1)) for f in redshift_filepaths]
redshift_bins = [
    RedshiftBin(nbin=i, filepath=rs_directory + f) for i, f in zip(nbins, redshift_filepaths)
]
redshift_bins_sorted = sorted(redshift_bins, key=lambda x: x.nbin)


# %%
mask = SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=exact_lmax, l_smooth=30)


# %%
# Define the initial log-spaced array between 0.5 and 300 arcminutes
initial_ang_bins_in_arcmin = np.logspace(np.log10(0.5), np.log10(300), num=9, endpoint=True)
# Filter out bins smaller than 15 arcminutes
filtered_ang_bins_in_arcmin = initial_ang_bins_in_arcmin[initial_ang_bins_in_arcmin >= 15]
# Add one more bin on the larger side according to the same pattern
last_bin_ratio = filtered_ang_bins_in_arcmin[-1] / filtered_ang_bins_in_arcmin[-2]
new_bin = filtered_ang_bins_in_arcmin[-1] * last_bin_ratio
extended_ang_bins_in_arcmin = np.append(filtered_ang_bins_in_arcmin, new_bin)
# Convert to degrees
ang_bins_in_deg = extended_ang_bins_in_arcmin / 60

# Create tuples representing the bin edges
ang_bins_in_deg = [
    (ang_bins_in_deg[i], ang_bins_in_deg[i + 1]) for i in range(len(ang_bins_in_deg) - 2)
]


# %%
likelihood = XiLikelihood(
    mask=mask, redshift_bins=redshift_bins_sorted, ang_bins_in_deg=ang_bins_in_deg
)


# %%
data_shape = likelihood.prep_data_array()
mock_data = np.ones_like(data_shape) * 10**-7
likelihood.initiate_mask_specific()
likelihood.precompute_combination_matrices()

# %%
posterior = []
for s in s8:
    cosmology = cosmo.copy()
    cosmology["s8"] = s
    likelihood = likelihood.likelihood(mock_data, cosmology)
    posterior.append(likelihood)

plt.plot(s8, posterior)
plt.savefig("s8_posterior.png")


# %%
