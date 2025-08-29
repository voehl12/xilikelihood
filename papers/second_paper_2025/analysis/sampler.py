import xilikelihood as xili
import numpy as np
import sys
import logging
from nautilus import Prior, Sampler
import matplotlib.pyplot as plt
try:
    import corner
except ImportError:
    corner = None
    print("Warning: 'corner' package not found. Plots will not be generated.")
from config import (
    EXACT_LMAX,
    MASK_CONFIG,
    DATA_FILES,
    DATA_DIR,
    BASE_DIR,
    PACKAGE_DIR,
    FIDUCIAL_COSMO
)
import tempfile

jobnumber = int(sys.argv[1]) - 1
# Robust logging setup
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler(f'sampler_{jobnumber}.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)
logger.propagate = False  # Prevent double logging

# Enable ALL xilikelihood logging (catch any submodule loggers)
xili_logger = logging.getLogger('xilikelihood')
xili_logger.setLevel(logging.DEBUG)

logger.info("Starting sampler setup...")
logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Creating mask with exact_lmax={EXACT_LMAX}")
logger.info(f"Package directory: {PACKAGE_DIR}")
mask = xili.SphereMask(
    spins=MASK_CONFIG['spins'],
    circmaskattr=MASK_CONFIG['circmaskattr'],
    exact_lmax=EXACT_LMAX,
    l_smooth=MASK_CONFIG['l_smooth'],
    working_dir=PACKAGE_DIR
)

logger.info("Loading fiducial dataspace...")
redshift_bins, ang_bins_in_deg = xili.fiducial_dataspace(min_ang_cutoff_in_arcmin=0)
#redshift_bins = redshift_bins[-2:]
logger.info(f"Redshift bins: {len(redshift_bins)}, Angular bins: {len(ang_bins_in_deg)}")
include_ximinus = False
logger.info(f"Include xi-: {include_ximinus}")

logger.info("Initializing XiLikelihood...")
xilikelihood = xili.XiLikelihood(
    mask=mask,
    redshift_bins=redshift_bins,
    ang_bins_in_deg=ang_bins_in_deg,
    include_ximinus=include_ximinus,
    exact_lmax=EXACT_LMAX,
    large_angle_threshold=1.0,
)

logger.info("Creating mock data and covariance on the fly...")
with tempfile.TemporaryDirectory() as tmpdir:
    mock_data_path = f"{tmpdir}/mock_data.npz"
    cov_path = f"{tmpdir}/mock_cov.npz"
    mock_data, gaussian_covariance = xili.mock_data.create_mock_data(
        xilikelihood,
        mock_data_path=mock_data_path,
        gaussian_covariance_path=cov_path,
        fiducial_cosmo=FIDUCIAL_COSMO,
        random=None
    )
logger.info(f"Mock data shape: {mock_data.shape}")
logger.info(f"Covariance shape: {gaussian_covariance.shape}")

logger.info("Setting up likelihood...")
xilikelihood.setup_likelihood()
xilikelihood.gaussian_covariance = gaussian_covariance
logger.info("Gaussian covariance set successfully")

logger.info("Setting up priors...")
omega_m_prior = np.linspace(0.1, 0.5, 100)
s8_prior = np.linspace(0.5, 1.1, 100)
params = ["omega_m", "s8"]
priors = [(omega_m_prior[0], omega_m_prior[-1]), (s8_prior[0], s8_prior[-1])]
logger.info(f"Parameters: {params}")
logger.info(f"Prior ranges: {priors}")

prior = Prior()
for param, prior_range in zip(params, priors):
    prior.add_parameter(param, dist=(prior_range[0], prior_range[1]))
    logger.info(f"Added prior for {param}: [{prior_range[0]:.3f}, {prior_range[1]:.3f}]")

def likelihood(cosmology):
    logger.debug(f"Evaluating likelihood for cosmology: {cosmology}")
    try:
        logL = xilikelihood.loglikelihood(mock_data, cosmology)
        logger.debug(f"Likelihood evaluated: logL = {logL}")
        return logL
    except Exception as e:
        logger.error(f"Likelihood evaluation failed for {cosmology}: {e}")
        exit()

# --- Sampler selection ---
USE_NESTED = False  # Set to False to use emcee MCMC instead

def run_nested_sampler(prior, likelihood, jobnumber):
    results_path = f'sampler_results_{jobnumber}_l20_kidsplus.h5'
    logger.info("Setting up Nautilus sampler...")
    sampler = Sampler(prior, likelihood, n_live=16, filepath=results_path)
    logger.info("Starting nested sampling...")
    sampler.run(verbose=True)
    logger.info("Nested sampling completed!")
    points, log_w, log_l = sampler.posterior()
    corner.corner(
        points, weights=np.exp(log_w), bins=20, labels=prior.keys, color='purple',
        plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)))
    plt.savefig(f'corner_plot_{jobnumber}_l20_kidsplus_nested.png')
    logger.info(f"Corner plot saved: corner_plot_{jobnumber}_l20_kidsplus_nested.png")
    return points, log_w, log_l


def run_emcee_sampler(prior, likelihood, jobnumber, n_walkers=4, n_steps=2000):
    import emcee
    logger.info("Setting up emcee MCMC sampler...")
    # Use the priors list of tuples for bounds
    bounds = priors  # List of (low, high) tuples
    logger.info(f"Prior bounds: {bounds}")
    ndim = len(bounds)
    # Initial positions: random in prior
    p0 = [np.array([np.random.uniform(low, high) for (low, high) in bounds]) for _ in range(n_walkers)]
    logger.info(f"Initial positions (p0): {p0}")
    def log_prob(theta):
        # Flat prior using bounds
        for i, (low, high) in enumerate(bounds):
            if not (low <= theta[i] <= high):
                return -np.inf
        param_dict = dict(zip(prior.keys, theta))
        logl = likelihood(param_dict)
        if np.isnan(logl):
            return -np.inf
        return logl
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
    logger.info("Running emcee...")
    state = sampler.run_mcmc(p0, 100)
    sample_sample = sampler.get_chain(flat=True)
    corner.corner(
        sample_sample, bins=20, labels=prior.keys, color='green',
        plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)))
    plt.savefig(f'corner_plot_{jobnumber}_l20_kidsplus_all_rs_emcee_test.png')
    logger.info(f"Corner plot saved: corner_plot_{jobnumber}_l20_kidsplus_all_rs_emcee_test.png")
    # Reset sampler and run for more steps
    sampler.reset()
    sampler.run_mcmc(state, 1000, progress=True)
    logger.info("emcee sampling completed!")
    # Flatten chain, discard burn-in
    flat_samples = sampler.get_chain(discard=n_steps//2, thin=10, flat=True)
    corner.corner(
        flat_samples, bins=20, labels=prior.keys, color='green',
        plot_datapoints=False, range=np.repeat(0.999, len(prior.keys)))
    plt.savefig(f'corner_plot_{jobnumber}_l20_kidsplus_all_rs_emcee.png')
    logger.info(f"Corner plot saved: corner_plot_{jobnumber}_l20_kidsplus_all_rs_emcee.png")
    return flat_samples

if __name__ == "__main__":
    if USE_NESTED:
        run_nested_sampler(prior, likelihood, jobnumber)
    else:
        run_emcee_sampler(prior, likelihood, jobnumber)

