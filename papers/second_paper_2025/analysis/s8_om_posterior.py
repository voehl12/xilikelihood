import numpy as np
import time
import sys, os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'posterior_job_{sys.argv[1] if len(sys.argv) > 1 else "unknown"}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set JAX to use CPU before importing xilikelihood
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_PLATFORM_NAME"] = "cpu" 
logger.info("Set JAX to use CPU")

try:
    import xilikelihood as xlh
    logger.info("Successfully imported xilikelihood")
except ImportError as e:
    logger.error(f"Failed to import xilikelihood: {e}")
    sys.exit(1)

# Validate command line arguments
if len(sys.argv) != 2:
    logger.error("Usage: python s8_om_posterior.py <job_number>")
    sys.exit(1)

try:
    jobnumber = int(sys.argv[1]) - 1
    if jobnumber < 0:
        raise ValueError("Job number must be positive")
    logger.info(f"Starting job {jobnumber + 1}")
except ValueError as e:
    logger.error(f"Invalid job number: {e}")
    sys.exit(1)

# Setup analysis parameters
EXACT_LMAX = 30
logger.info(f"Setting up analysis with exact_lmax={EXACT_LMAX}")

try:
    mask = xlh.SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=EXACT_LMAX, l_smooth=30)
    logger.info("Created spherical mask")
    
    redshift_bins, ang_bins_in_deg = xlh.fiducial_dataspace()
    ang_bins_in_deg = ang_bins_in_deg[:-1]  # Remove last bin
    logger.info(f"Set up {len(redshift_bins)} redshift bins and {len(ang_bins_in_deg)} angular bins")
    
except Exception as e:
    logger.error(f"Failed to set up mask or dataspace: {e}")
    sys.exit(1)

# Load data files with error checking
try:
    if not Path("fiducial_data_1000sqd.npz").exists():
        raise FileNotFoundError("fiducial_data_1000sqd.npz not found")
    if not Path("gaussian_covariance_1000sqd.npz").exists():
        raise FileNotFoundError("gaussian_covariance_1000sqd.npz not found")
        
    mock_data = np.load("fiducial_data_1000sqd.npz")["data"]
    gaussian_covariance = np.load("gaussian_covariance_1000sqd.npz")["cov"]
    logger.info(f"Loaded data: shape {mock_data.shape}, covariance: shape {gaussian_covariance.shape}")
    
except Exception as e:
    logger.error(f"Failed to load data files: {e}")
    sys.exit(1)

# Set up likelihood
try:
    likelihood = xlh.XiLikelihood(
        mask=mask, redshift_bins=redshift_bins, ang_bins_in_deg=ang_bins_in_deg)
    likelihood.setup_likelihood()
    likelihood.gaussian_covariance = gaussian_covariance
    logger.info("Likelihood setup completed")
    
except Exception as e:
    logger.error(f"Failed to set up likelihood: {e}")
    sys.exit(1)


# Set up parameter grid
omega_m_prior = np.linspace(0.1, 0.5, 100)
s8_prior = np.linspace(0.5, 1.1, 100)
prior_pairs = np.meshgrid(omega_m_prior, s8_prior)
prior_pairs = np.vstack([prior_pairs[0].ravel(), prior_pairs[1].ravel()]).T

n_jobs = 500
split_prior_pairs = np.array_split(prior_pairs, n_jobs)

# Validate job number
if jobnumber >= len(split_prior_pairs):
    logger.error(f"Job number {jobnumber + 1} exceeds maximum jobs ({len(split_prior_pairs)})")
    sys.exit(1)

# Get the subset for the current job
subset_pairs = split_prior_pairs[jobnumber]
logger.info(f"Processing {len(subset_pairs)} parameter combinations for job {jobnumber + 1}/{n_jobs}")

# Set up output directory
output_dir = Path("/cluster/scratch/veoehl/posteriors")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"posterior_{jobnumber}.npy"

results_dtype = np.dtype([
    ("exact_post", np.float64),
    ("gauss_post", np.float64),
    ("s8", np.float64),
    ("omega_m", np.float64)
])

results = np.empty(len(subset_pairs), dtype=results_dtype)
start_time = time.time()

# Main computation loop with progress tracking and error handling
failed_computations = 0
for i, (omega_m, s8) in enumerate(subset_pairs):
    try:
        cosmology = {"omega_m": omega_m, "s8": s8}
        post, gauss_post = likelihood.loglikelihood(mock_data, cosmology, gausscompare=True)
        results[i] = (post, gauss_post, s8, omega_m)
        
        # Progress logging every 10% or every 10 iterations, whichever is less frequent
        progress_interval = max(1, len(subset_pairs) // 10)
        if (i + 1) % progress_interval == 0:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (len(subset_pairs) - i - 1) / rate
            logger.info(f"Progress: {i+1}/{len(subset_pairs)} ({100*(i+1)/len(subset_pairs):.1f}%) "
                       f"- Rate: {rate:.2f} iter/s - ETA: {eta:.0f}s")
            
    except Exception as e:
        logger.warning(f"Failed computation at Ωₘ={omega_m:.3f}, S₈={s8:.3f}: {e}")
        # Fill with NaN for failed computations
        results[i] = (np.nan, np.nan, s8, omega_m)
        failed_computations += 1
        
        # If too many failures, stop
        if failed_computations > len(subset_pairs) * 0.1:  # More than 10% failures
            logger.error(f"Too many failed computations ({failed_computations}). Stopping.")
            sys.exit(1)

# Save results
try:
    np.save(output_file, results)
    total_time = time.time() - start_time
    logger.info(f"Job completed successfully in {total_time:.1f}s")
    logger.info(f"Results saved to {output_file}")
    if failed_computations > 0:
        logger.warning(f"Had {failed_computations} failed computations out of {len(subset_pairs)}")
        
except Exception as e:
    logger.error(f"Failed to save results: {e}")
    sys.exit(1)





