import xilikelihood as xili

# 1) stack/2024-06   2) gcc/12.2.0   3) swig/4.1.1-ipvpwcc   4) python/3.11.6   5) cuda/12.1.1   6) nccl/2.18.3-1   7) openblas/0.3.24   8) python_cuda/3.11.6   9) cmake/3.27.7  10) cudnn/9.2.0
# module load stack/2024-06 gcc/12.2.0 swig/4.1.1-ipvpwcc python/3.11.6 cuda/12.1.1 nccl/2.18.3-1 openblas/0.3.24 python_cuda/3.11.6 cmake/3.27.7 cudnn/9.2.0
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie
import numpy as np
import sys
import logging
from nautilus import Prior, Sampler
from xilikelihood.copula_funcs import data_subset, cov_subset, expand_subset_for_ximinus
from xilikelihood.theory_cl import BinCombinationMapper
from mock_data_generation import create_mock_data


from config import (
    EXACT_LMAX,
    MASK_CONFIG,
    DATA_FILES,
    DATA_DIR,
    BASE_DIR,
    PACKAGE_DIR
)
jobnumber = int(sys.argv[1]) - 1
# Set up logging
logging.basicConfig(
    level=logging.INFO,  # Uncomment for less verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'sampler_{jobnumber}.log'),  # File output
        #logging.StreamHandler()  # Console output
    ]
)
logger = logging.getLogger(__name__)

# Enable ALL xilikelihood logging (catch any submodule loggers)
xili_logger = logging.getLogger('xilikelihood')
xili_logger.setLevel(logging.DEBUG)


# definitely run with exact_lmax=20!
# Set root logger to capture everything
logging.getLogger().setLevel(logging.INFO)

logger.info("Starting MCMC sampler setup...")

logger.info(f"Base directory: {BASE_DIR}")
logger.info(f"Creating mask with exact_lmax={EXACT_LMAX}")
# Use package-level directory for shared WPM arrays
logger.info(f"Package directory: {PACKAGE_DIR}")
mask = xili.SphereMask(
    spins=MASK_CONFIG['spins'], 
    circmaskattr=MASK_CONFIG['circmaskattr'], 
    exact_lmax=EXACT_LMAX, 
    l_smooth=MASK_CONFIG['l_smooth'],
    working_dir=PACKAGE_DIR  # Use package root for shared arrays
)


logger.info("Loading fiducial dataspace...")
redshift_bins, ang_bins_in_deg = xili.fiducial_dataspace()
logger.info(f"Original: {len(redshift_bins)} redshift bins, {len(ang_bins_in_deg)} angular bins")
n_angbins = len(ang_bins_in_deg)
include_ximinus = False
logger.info(f"Include xi-: {include_ximinus}")

mapper = BinCombinationMapper(len(redshift_bins))
combs = [(2,2),(4,4),(4,2)] # careful, does not match with line 77, but currently, this is not used anyway.
rs_indices = [mapper.get_index(comb) for comb in combs]
ang_indices = [2]
subset = [(rs_index,ang_indices[0]) for rs_index in rs_indices]


if include_ximinus:
    subset = expand_subset_for_ximinus(subset, len(ang_bins_in_deg))
    n_angbins *= 2
ang_bins_in_deg = ang_bins_in_deg[2:-1]
redshift_bins = redshift_bins[3:]
logger.info(f"Reduced: {len(redshift_bins)} redshift bins, {len(ang_bins_in_deg)} angular bins")



logger.info("Initializing XiLikelihood...")
xilikelihood = xili.XiLikelihood(
        mask=mask, redshift_bins=redshift_bins, ang_bins_in_deg=ang_bins_in_deg,validate_means=True,include_ximinus=include_ximinus)

logger.info("Loading mock data and covariance...")
# need to update mockdata including xi_minus, can also check values here, which one is correct
mock_data_path = DATA_DIR / DATA_FILES['10000sqd_random']['mock_data']
gaussian_covariance_path = DATA_DIR / DATA_FILES['10000sqd_random']['covariance']

#create_mock_data(xilikelihood, mock_data_path, gaussian_covariance_path,random=None)
mock_data = np.load(mock_data_path)["data"]
#mock_data = data_subset(mock_data, subset,full_grid=True)

gaussian_covariance = np.load(gaussian_covariance_path)["cov"]
#gaussian_covariance = cov_subset(gaussian_covariance, subset, n_angbins)
logger.info(f"Mock data shape: {mock_data.shape}")
logger.info(f"Mock data content: {mock_data[:5]}")  # Log first 5 entries for verification
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

results_path = f'sampler_results_{jobnumber}_l20.h5'
logger.info("Setting up Nautilus sampler...")
sampler = Sampler(prior, likelihood, n_live=16,filepath=results_path)
logger.info("Starting sampling...")
sampler.run(verbose=True)
logger.info("Sampling completed!")