# Nautilus nested sampling for xilikelihood posterior analysis.
# Run with: python sampler_nested.py <job_number>
# Note: set USE_NESTED=True in config or pass --nested flag to use this sampler.

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import xilikelihood as xili
import numpy as np
import sys
import logging
import time
from pathlib import Path
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    print("Warning: 'corner' package not found. Corner plots will not be generated.")

try:
    from nautilus import Sampler
    HAS_NAUTILUS = True
except ImportError:
    HAS_NAUTILUS = False
    print("Warning: 'nautilus' package not found. Nested sampling will not work.")

from config import (
    EXACT_LMAX,
    MASK_CONFIG_MEDRES_STAGE3 as MASK_CONFIG,
    DATA_FILES,
    DATA_DIR,
    BASE_DIR,
    PACKAGE_DIR,
    FIDUCIAL_COSMO,
    PRIOR_CONFIG
)

# Output directory
OUTPUT_DIR = Path("sampler_output_gaussian")
OUTPUT_DIR.mkdir(exist_ok=True)


def _resolve_log_dir():
    """Choose a writable log directory, preferring scratch over home."""
    override = os.environ.get("XILI_LOG_DIR")
    if override:
        log_dir = Path(override)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir

    scratch_user = Path("/cluster/scratch") / os.environ.get("USER", "unknown")
    scratch_log_dir = scratch_user / "xilikelihood" / "sampler_logs_gaussian"
    try:
        scratch_log_dir.mkdir(parents=True, exist_ok=True)
        return scratch_log_dir
    except Exception:
        return OUTPUT_DIR


LOG_DIR = _resolve_log_dir()

# Sampler configuration
LIKELIHOOD_TYPE = "gaussian"
NAUTILUS_CONFIG = {
    "n_live": 16
}

# Module-level logger
logger = logging.getLogger(__name__)

# Module-level variables (initialized in main)
mask = None
xilikelihood = None
mock_data = None
gaussian_covariance = None
params = None
uniform_priors = None
mvn_config = None
mvn_param = None
mvn_mean = None
mvn_cov = None


def setup_logging(jobnumber, rank, pid):
    """Set up logging for a specific rank and pid."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    logfile = LOG_DIR / f'sampler_nested_{jobnumber}_r{rank}_p{pid}.log'
    file_handler = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.propagate = False
    
    xili_logger = logging.getLogger('xilikelihood')
    xili_logger.setLevel(logging.DEBUG)
    xili_logger.addHandler(file_handler)
    
    return logger, file_handler


def likelihood(cosmology):
    """Evaluate the likelihood for Nautilus nested sampler."""
    if not logger.handlers:
        jobnumber = int(sys.argv[1]) - 1
        rank = 0
        try:
            rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0")))
        except Exception:
            rank = 0
        pid = os.getpid()
        setup_logging(jobnumber, rank, pid)
    
    logger.info(f"Evaluating likelihood for cosmology: {cosmology}")
    try:
        logL = xilikelihood.loglikelihood(
            mock_data, 
            cosmology, 
            likelihood_type=LIKELIHOOD_TYPE,
            allow_diagnostic=False
        )
        logger.info(f"Likelihood result: {logL}")
        return logL
    except Exception as e:
        logger.error(f"Likelihood evaluation failed: {e}")
        return -np.inf


def save_corner_plot(samples, labels, filename, weights=None):
    """Save a corner plot and trace plots if corner package is available."""
    if not HAS_CORNER:
        logger.warning("Corner package not available, skipping plot")
        return
    
    if samples.ndim == 3:
        n_steps, n_walkers, ndim = samples.shape
        flat_samples = samples.reshape(-1, ndim)
        logger.info(f"Flattened samples from {samples.shape} to {flat_samples.shape} for corner plot")
    else:
        flat_samples = samples
    
    fig = corner.corner(
        flat_samples, 
        weights=weights,
        bins=20, 
        labels=labels,
        plot_datapoints=False, 
        range=np.repeat(0.999, len(labels))
    )
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Corner plot saved: {filepath}")


def run_nautilus_nested_sampler(jobnumber):
    """Run Nautilus nested sampler."""
    if not HAS_NAUTILUS:
        raise ImportError(
            "Nautilus package is not installed. "
            "Install it with: pip install nautilus-sampler"
        )

    results_path = OUTPUT_DIR / f'nautilus_results_{jobnumber}_{RUN_NAME}_{LIKELIHOOD_TYPE}.h5'
    logger.info("Setting up Nautilus nested sampler...")
    
    # Build prior dict from params and priors
    prior_dict = {}
    for param in params:
        if param in uniform_priors:
            low, high = uniform_priors[param]
            prior_dict[param] = (low, high)
        elif param == mvn_param:
            # For multivariate parameters, use independent uniform bounds based on mean +/- 3*std
            stds = np.sqrt(np.diag(mvn_cov))
            for i, std in enumerate(stds):
                param_name = f"{param}_{i}"
                mean_val = mvn_mean[i]
                low = mean_val - 3 * std
                high = mean_val + 3 * std
                prior_dict[param_name] = (low, high)
        else:
            logger.warning(f"No prior defined for parameter {param}")
    
    sampler = Sampler(
        prior_dict,
        likelihood,
        n_live=NAUTILUS_CONFIG["n_live"],
        filepath=str(results_path)
    )
    
    logger.info("Starting nested sampling...")
    sampler.run(verbose=True)
    logger.info("Nested sampling completed!")
    
    # Get posterior samples
    points, log_w, log_l = sampler.posterior()
    
    # Save corner plot
    save_corner_plot(
        points, 
        labels=list(prior_dict.keys()),
        filename=f'corner_{jobnumber}_{RUN_NAME}_{LIKELIHOOD_TYPE}_nested.png',
        weights=np.exp(log_w)
    )
    
    logger.info(f"Nested sampling results saved to {results_path}")
    return points, log_w, log_l


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python sampler_nested.py <job_number>")
        sys.exit(1)

    jobnumber = int(sys.argv[1]) - 1
    RUN_NAME = f"exact_lmax{EXACT_LMAX}_n1024_minang5"

    rank = 0
    try:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0")))
    except Exception:
        rank = 0

    pid = os.getpid()
    logger, file_handler = setup_logging(jobnumber, rank, pid)

    logger.info("Starting nested sampler setup...")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Creating mask with exact_lmax={EXACT_LMAX}")
    
    mask = xili.SphereMask(
        spins=MASK_CONFIG['spins'],
        circmaskattr=MASK_CONFIG['circmaskattr'],
        exact_lmax=EXACT_LMAX,
        l_smooth=MASK_CONFIG['l_smooth'],
        working_dir=PACKAGE_DIR
    )

    logger.info("Loading fiducial dataspace...")
    redshift_bins, ang_bins_in_deg = xili.fiducial_dataspace()
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
        large_angle_threshold=1/3,
    )

    logger.info("Preparing mock data and covariance...")
    shared_mock_data_path = OUTPUT_DIR / f"mock_data_{jobnumber}_{RUN_NAME}_{LIKELIHOOD_TYPE}.npz"
    shared_cov_path = OUTPUT_DIR / f"mock_cov_{jobnumber}_{RUN_NAME}_{LIKELIHOOD_TYPE}.npz"

    if rank == 0:
        tmp_mock_path = shared_mock_data_path.with_suffix('.tmp.npz')
        tmp_cov_path = shared_cov_path.with_suffix('.tmp.npz')

        mock_data, gaussian_covariance = xili.mock_data.create_mock_data(
            xilikelihood,
            mock_data_path=str(tmp_mock_path),
            gaussian_covariance_path=str(tmp_cov_path),
            fiducial_cosmo=FIDUCIAL_COSMO,
            random=None
        )

        tmp_mock_path.replace(shared_mock_data_path)
        tmp_cov_path.replace(shared_cov_path)
    else:
        mock_data, _ = xili.mock_data.load_mock_data(str(shared_mock_data_path))
        gaussian_covariance, _ = xili.mock_data.load_gaussian_covariance(str(shared_cov_path))

    logger.info(f"Mock data shape: {mock_data.shape}")
    logger.info(f"Covariance shape: {gaussian_covariance.shape}")

    logger.info("Setting up likelihood...")
    xilikelihood.setup_likelihood()
    xilikelihood.gaussian_covariance = gaussian_covariance
    logger.info("Gaussian covariance set successfully")

    logger.info("Setting up priors from config...")
    params = PRIOR_CONFIG["parameters"]
    uniform_priors = PRIOR_CONFIG["uniform"]
    mvn_config = PRIOR_CONFIG.get("multivariate_gaussian")
    mvn_param = mvn_config["parameter"]
    mvn_mean = np.array(mvn_config["mean"])
    mvn_corr_matrix = np.array(mvn_config["cov"])
    mvn_cov = xili.copula_funcs.correlation_to_covariance(mvn_corr_matrix)

    logger.info(f"Parameters: {params}")
    logger.info(f"Uniform prior ranges: {uniform_priors}")
    logger.info(f"Standard deviations: {np.sqrt(np.diag(mvn_cov))}")

    logger.info("Starting Nautilus nested sampling...")
    results = run_nautilus_nested_sampler(jobnumber)
    logger.info("Nested sampling complete")
    
    logger.info("All done!")
