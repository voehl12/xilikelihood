
import xilikelihood as xili

# 1) stack/2024-06   2) gcc/12.2.0   3) swig/4.1.1-ipvpwcc   4) python/3.11.6   5) cuda/12.1.1   6) nccl/2.18.3-1   7) openblas/0.3.24   8) python_cuda/3.11.6   9) cmake/3.27.7  10) cudnn/9.2.0
# module load stack/2024-06 gcc/12.2.0 swig/4.1.1-ipvpwcc python/3.11.6 cuda/12.1.1 nccl/2.18.3-1 openblas/0.3.24 python_cuda/3.11.6 cmake/3.27.7 cudnn/9.2.0
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie
import numpy as np
import logging
from nautilus import Prior, Sampler
from xilikelihood.copula_funcs import data_subset, cov_subset
from xilikelihood.theory_cl import BinCombinationMapper


from config import (
    EXACT_LMAX,
    MASK_CONFIG,
    DATA_FILES,
    DATA_DIR,
    BASE_DIR
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Uncomment for less verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('sampler.log')  # File output
    ]
)
logger = logging.getLogger(__name__)

# Enable ALL xilikelihood logging (catch any submodule loggers)
xili_logger = logging.getLogger('xilikelihood')
xili_logger.setLevel(logging.INFO)



# Set root logger to capture everything
logging.getLogger().setLevel(logging.INFO)

logger.info("Starting MCMC sampler setup...")

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Creating mask with exact_lmax={EXACT_LMAX}")
# Use package-level directory for shared WPM arrays
package_dir = BASE_DIR.parent.parent  # Go up from analysis to package root
logger.info(f"Package directory: {package_dir}")
mask = xili.SphereMask(
    spins=MASK_CONFIG['spins'], 
    circmaskattr=MASK_CONFIG['circmaskattr'], 
    exact_lmax=EXACT_LMAX, 
    l_smooth=30,
    working_dir=str(package_dir)  # Use package root for shared arrays
)


logger.info("Loading fiducial dataspace...")
redshift_bins, ang_bins_in_deg = xili.fiducial_dataspace()
logger.info(f"Original: {len(redshift_bins)} redshift bins, {len(ang_bins_in_deg)} angular bins")

mapper = BinCombinationMapper(len(redshift_bins))
combs = [(3,3),(4,4),(4,3)]
rs_indices = [mapper.get_index(comb) for comb in combs]
ang_indices = [2]
subset = [(rs_index,ang_indices[0]) for rs_index in rs_indices]

ang_bins_in_deg = ang_bins_in_deg[2:-1]
redshift_bins = redshift_bins[3:]
logger.info(f"Reduced: {len(redshift_bins)} redshift bins, {len(ang_bins_in_deg)} angular bins")

logger.info("Loading mock data and covariance...")
mock_data = np.load(DATA_DIR / DATA_FILES['10000sqd']['mock_data'])["data"]
mock_data = data_subset(mock_data, subset)
gaussian_covariance = np.load(DATA_DIR / DATA_FILES['10000sqd']['covariance'])["cov"]
gaussian_covariance = cov_subset(gaussian_covariance, subset, 1)
logger.info(f"Mock data shape: {mock_data.shape}")
logger.info(f"Covariance shape: {gaussian_covariance.shape}")

logger.info("Initializing XiLikelihood...")
xilikelihood = xili.XiLikelihood(
        mask=mask, redshift_bins=redshift_bins, ang_bins_in_deg=ang_bins_in_deg,include_ximinus=False)


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
    
    # Add any fixed cosmological parameters needed by xilikelihood
    # You may need to add other parameters like h, omega_b, etc.
    # cosmology['h'] = 0.7  # example
    # cosmology['omega_b'] = 0.049  # example
    
    try:
        logL = xilikelihood.loglikelihood(mock_data, cosmology)
        logger.debug(f"Likelihood evaluated: logL = {logL}")
        return logL
    except Exception as e:
        logger.error(f"Likelihood evaluation failed for {cosmology}: {e}")
        exit() 


logger.info("Setting up Nautilus sampler...")
sampler = Sampler(prior, likelihood, n_live=10)
logger.info("Starting sampling...")
sampler.run(verbose=True)
logger.info("Sampling completed!")