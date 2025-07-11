import numpy as np
from nautilus import Prior, Sampler
import xilikelihood as xili
from functools import partial


exact_lmax = 30
mask = xili.SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=exact_lmax, l_smooth=30)


redshift_bins, ang_bins_in_deg = xili.fiducial_dataspace()
ang_bins_in_deg = ang_bins_in_deg[:-1]



mock_data = np.load("../data/fiducial_data_10000sqd.npz")["data"]
gaussian_covariance = np.load("../data/gaussian_covariance_10000sqd.npz")["cov"]

xilikelihood = xili.XiLikelihood(
        mask=mask, redshift_bins=redshift_bins, ang_bins_in_deg=ang_bins_in_deg)
xilikelihood.setup_likelihood()

xilikelihood.gaussian_covariance = gaussian_covariance

omega_m_prior = np.linspace(0.1, 0.5, 100)
s8_prior = np.linspace(0.5, 1.1, 100)
params = ["omega_m", "s8"]
priors = [(omega_m_prior[0], omega_m_prior[-1]), (s8_prior[0], s8_prior[-1])]
cosmology = {key: None for key in params}
prior = Prior()
for param, prior_values in zip(params, priors):
    prior.add_parameter(param, dist=prior_values)




def likelihood(cosmology):

    return xilikelihood.loglikelihood(mock_data, cosmology)



sampler = Sampler(prior, likelihood, n_live=10)
sampler.run(verbose=True)