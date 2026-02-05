# at current setup needs a lot of memory:  srun --pty  --mem-per-cpu=4096 --cpus-per-task=32 --time=4:00:00 bash

import os
os.environ["OMP_NUM_THREADS"] = "1"

import xilikelihood as xili
import numpy as np
import sys
import logging
from pathlib import Path
from scipy.stats import multivariate_normal
from schwimmbad import MPIPool
import matplotlib.pyplot as plt
try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    print("Warning: 'corner' package not found. Corner plots will not be generated.")

from config import (
    EXACT_LMAX,
    MASK_CONFIG_MEDRES_STAGE4 as MASK_CONFIG,
    DATA_FILES,
    DATA_DIR,
    BASE_DIR,
    PACKAGE_DIR,
    FIDUCIAL_COSMO,
    PRIOR_CONFIG
)
import tempfile

# Output directory
OUTPUT_DIR = Path("sampler_output_test")
OUTPUT_DIR.mkdir(exist_ok=True)

# Sampler configuration
USE_NESTED = False  # Set to True for Nautilus nested sampling, False for emcee MCMC
EMCEE_CONFIG = {
    "n_walkers": 24,  # Matches ntasks 
    "n_burn": 100,
    "n_steps": 1000,
    "thin": 1,
    "checkpoint_interval": 100  # Save every N steps (set to None to disable)
}
NAUTILUS_CONFIG = {
    "n_live": 16
}

# Module-level logger (safe for workers)
logger = logging.getLogger(__name__)

# Module-level variables (initialized in main, available to workers after spawn)
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
    
    logfile = OUTPUT_DIR / f'sampler_{jobnumber}_r{rank}_p{pid}.log'
    file_handler = logging.FileHandler(logfile, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.propagate = False
    
    # Also route xilikelihood logs
    xili_logger = logging.getLogger('xilikelihood')
    xili_logger.setLevel(logging.DEBUG)
    xili_logger.addHandler(file_handler)
    
    return logger, file_handler

def likelihood(cosmology):
    """Evaluate the likelihood (for Nautilus, prior is handled separately)."""
    # Ensure logging is set up for this process (workers may not have it)
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
        logL = xilikelihood.loglikelihood(mock_data, cosmology)
        logger.info(f"Likelihood result: {logL}")
        return logL
    except Exception as e:
        logger.error(f"Likelihood evaluation failed: {e}")
        return -np.inf

def log_prior_mvn(theta_dict):
    """Compute log prior for multivariate Gaussian component (used by emcee)."""
    if mvn_param is None:
        return 0.0
    
    param_value = theta_dict.get(mvn_param)
    if param_value is None:
        return 0.0
    
    param_array = np.atleast_1d(param_value)
    log_prior = multivariate_normal.logpdf(param_array, mean=mvn_mean, cov=mvn_cov)
    return log_prior

def save_corner_plot(samples, labels, filename, weights=None):
    """Save a corner plot and trace plots if corner package is available.
    
    Parameters
    ----------
    samples : array
        Either 2D (n_samples, ndim) or 3D (n_steps, n_walkers, ndim)
    labels : list
        Parameter names
    filename : str
        Output filename for corner plot
    weights : array, optional
        Sample weights (for nested sampling)
    """
    if not HAS_CORNER:
        logger.warning("Corner package not available, skipping plot")
        return
    
    # Handle 3D samples (n_steps, n_walkers, ndim) from emcee
    if samples.ndim == 3:
        n_steps, n_walkers, ndim = samples.shape
        
        # Save trace plots showing walker chains
        trace_filename = filename.replace('.png', '_traces.png')
        fig_traces, axes = plt.subplots(ndim, figsize=(10, 2.5 * ndim), sharex=True)
        if ndim == 1:
            axes = [axes]
        
        for i in range(ndim):
            ax = axes[i]
            for j in range(n_walkers):
                ax.plot(samples[:, j, i], alpha=0.3, lw=0.5)
            ax.set_ylabel(labels[i] if i < len(labels) else f"param {i}")
            ax.set_xlim(0, n_steps)
        
        axes[-1].set_xlabel("Step")
        fig_traces.tight_layout()
        trace_filepath = OUTPUT_DIR / trace_filename
        plt.savefig(trace_filepath, dpi=150, bbox_inches='tight')
        plt.close(fig_traces)
        logger.info(f"Trace plot saved: {trace_filepath}")
        
        # Flatten for corner plot
        flat_samples = samples.reshape(-1, ndim)
        logger.info(f"Flattened samples from {samples.shape} to {flat_samples.shape} for corner plot")
    else:
        flat_samples = samples
    
    # Corner plot
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

# Module-level log_prob function (must be picklable for multiprocessing)
def _log_prob_fn(theta, params, mvn_param, mvn_mean, uniform_priors, likelihood_fn, log_prior_mvn_fn):
    """Log posterior = log prior + log likelihood."""
    # Build parameter dictionary, handling multi-dimensional parameters
    param_dict = {}
    theta_idx = 0
    for param in params:
        if param == mvn_param:
            # Multi-dimensional parameter (e.g., delta_z with 5 values)
            n_dims = len(mvn_mean)
            param_dict[param] = theta[theta_idx:theta_idx + n_dims]
            theta_idx += n_dims
        else:
            # Scalar parameter
            param_dict[param] = theta[theta_idx]
            theta_idx += 1
    
    # Check uniform prior bounds (log prior = 0 if in bounds, -inf if out)
    for param, value in param_dict.items():
        if param in uniform_priors:
            low, high = uniform_priors[param]
            if not (low <= value <= high):
                return -np.inf
    
    # Add Gaussian prior contribution (if any)
    log_prior = log_prior_mvn_fn(param_dict)
    if not np.isfinite(log_prior):
        return -np.inf
    
    # Evaluate likelihood
    logl = likelihood_fn(param_dict)
    if np.isnan(logl) or not np.isfinite(logl):
        return -np.inf
    
    # Return log posterior
    return log_prior + logl

# --- Sampler functions ---

def run_nested_sampler(prior, likelihood, jobnumber):
    """Run Nautilus nested sampler."""
    results_path = OUTPUT_DIR / f'nautilus_results_{jobnumber}_{RUN_NAME}.h5'
    logger.info("Setting up Nautilus sampler...")
    
    sampler = Sampler(
        prior, 
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
        labels=list(prior.keys),
        filename=f'corner_{jobnumber}_{RUN_NAME}_nested.png',
        weights=np.exp(log_w)
    )
    
    return points, log_w, log_l


def run_emcee_sampler(likelihood, jobnumber, n_walkers=None, n_steps=None):
    """Run emcee MCMC sampler. Continues from previous chain if checkpoint exists."""
    import emcee
    import glob
    
    # Calculate total dimensionality first (accounting for multi-dimensional parameters)
    ndim = sum(len(mvn_mean) if param == mvn_param else 1 for param in params)
    
    # Use config defaults if not specified
    n_walkers = n_walkers or EMCEE_CONFIG["n_walkers"]
    n_burn = EMCEE_CONFIG["n_burn"]
    n_steps = n_steps or EMCEE_CONFIG["n_steps"]
    
    # Check for existing chains to continue from
    existing_files = sorted(glob.glob(str(OUTPUT_DIR / f'emcee_samples_{jobnumber}_{RUN_NAME}*.npz')))
    checkpoint_file = OUTPUT_DIR / f'emcee_checkpoint_{jobnumber}_{RUN_NAME}.npz'
    
    continue_from = None
    chain_num = 0
    
    if existing_files:
        # Find highest chain number
        for f in existing_files:
            if '_chain' in f:
                try:
                    num = int(f.split('_chain')[-1].replace('.npz', ''))
                    chain_num = max(chain_num, num + 1)
                except ValueError:
                    pass
            else:
                chain_num = max(chain_num, 1)
        continue_from = existing_files[-1]
    elif checkpoint_file.exists():
        continue_from = checkpoint_file
        chain_num = 1
    
    logger.info("Setting up emcee MCMC sampler...")
    logger.info(f"Parameters: {len(params)} ({ndim} dimensions), Walkers: {n_walkers}, Steps: {n_steps}")
    
    # Create partial function with all the required variables
    from functools import partial
    log_prob = partial(_log_prob_fn, 
                       params=params, 
                       mvn_param=mvn_param, 
                       mvn_mean=mvn_mean,
                       uniform_priors=uniform_priors, 
                       likelihood_fn=likelihood, 
                       log_prior_mvn_fn=log_prior_mvn)
    
    # Set up MPI pool
    with MPIPool() as pool:
        if not pool.is_master():
            pool.wait()
            sys.exit(0)
        
        logger.info("MPI pool initialized")
        file_handler.flush()
        
        # Initial positions: from previous chain or sample from priors
        if continue_from:
            logger.info(f"Continuing from previous chain: {continue_from}")
            prev_data = np.load(continue_from)
            prev_samples = prev_data['samples']  # (n_steps, n_walkers, ndim)
            # Select half of walkers with highest variance (not stuck)
            walker_var = np.var(prev_samples, axis=0).sum(axis=1)  # variance per walker
            good_walkers = np.argsort(walker_var)[-n_walkers//2:]
            # Sample 2 positions from last 100 steps of each good walker
            last_samples = prev_samples[-100:, good_walkers, :]  # (100, n_good, ndim)
            flat_good = last_samples.reshape(-1, ndim)
            idx = np.random.choice(len(flat_good), size=n_walkers, replace=False)
            p0 = flat_good[idx]
            logger.info(f"Sampled {n_walkers} positions from {len(good_walkers)} best walkers")
            skip_burnin = True
        else:
            logger.info(f"Starting fresh chain with burn-in ({n_burn} steps)")
            p0 = []
            for _ in range(n_walkers):
                pos = []
                for param in params:
                    if param in uniform_priors:
                        low, high = uniform_priors[param]
                        pos.append(np.random.uniform(low, high))
                    elif mvn_param is not None and param == mvn_param:
                        # Sample from Gaussian prior (handles multi-dimensional parameters)
                        pos.extend(np.random.multivariate_normal(mvn_mean, mvn_cov))
                    else:
                        logger.error(f"No prior defined for parameter {param}")
                        raise ValueError(f"No prior defined for parameter {param}")
                p0.append(np.array(pos))
            skip_burnin = False
        
        logger.info(f"Initialized {n_walkers} walkers")
        file_handler.flush()
        
        # Set up sampler with MPI pool
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob, pool=pool)
        
        # Burn-in (skip if continuing)
        if not skip_burnin:
            logger.info(f"Running burn-in ({n_burn} steps)...")
            file_handler.flush()
            state = sampler.run_mcmc(p0, n_burn)
            sampler.reset()
        else:
            state = p0
            logger.info("Skipping burn-in (continuing from previous chain)")
        
        # Production run with periodic checkpointing
        checkpoint_interval = EMCEE_CONFIG.get("checkpoint_interval", 100)
        checkpoint_file = OUTPUT_DIR / f'emcee_checkpoint_{jobnumber}_{RUN_NAME}.npz'
        
        logger.info(f"Running production ({n_steps} steps, checkpointing every {checkpoint_interval} steps)...")
        file_handler.flush()
        
        steps_completed = 0
        while steps_completed < n_steps:
            # Run for checkpoint_interval steps (or remaining steps)
            steps_this_batch = min(checkpoint_interval, n_steps - steps_completed)
            state = sampler.run_mcmc(state, steps_this_batch, progress=False)
            steps_completed += steps_this_batch
            
            # Save checkpoint
            checkpoint_samples = sampler.get_chain()  # All samples so far
            np.savez(
                checkpoint_file,
                samples=checkpoint_samples,
                params=np.array(params, dtype=str),
                n_walkers=n_walkers,
                steps_completed=steps_completed,
                n_steps_target=n_steps,
            )
            logger.info(f"Checkpoint saved at step {steps_completed}/{n_steps}: {checkpoint_file}")
            file_handler.flush()
        
        logger.info("emcee sampling completed!")
        file_handler.flush()
    
    # Get samples
    samples = sampler.get_chain()  # Shape: (n_steps, n_walkers, ndim)
    logger.info(f"Final samples shape: {samples.shape}")
    
    # Save samples (with chain number if continuing)
    if chain_num > 0:
        output_file = OUTPUT_DIR / f'emcee_samples_{jobnumber}_{RUN_NAME}_chain{chain_num}.npz'
    else:
        output_file = OUTPUT_DIR / f'emcee_samples_{jobnumber}_{RUN_NAME}.npz'
    
    np.savez(
        output_file,
        samples=samples,
        params=np.array(params, dtype=str),
        n_walkers=n_walkers,
        n_steps=n_steps
    )
    logger.info(f"Samples saved: {output_file}")

    # Save corner plot (handles flattening and trace plots internally)
    save_corner_plot(
        samples,
        labels=params,
        filename=f'corner_{jobnumber}_{RUN_NAME}_emcee.png'
    )
    
    return samples

if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python sampler.py <job_number>")
        sys.exit(1)

    jobnumber = int(sys.argv[1]) - 1

    # Configuration for this run
    RUN_NAME = f"exact_lmax{EXACT_LMAX}_n1024_minang5"

    # Robust logging setup
    # Determine MPI rank/identity so each MPI task writes to its own log file.
    rank = 0
 
    try:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", os.environ.get("PMI_RANK", "0")))
    except Exception:
        rank = 0

    pid = os.getpid()
    
    logger, file_handler = setup_logging(jobnumber, rank, pid)

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
    redshift_bins, ang_bins_in_deg = xili.fiducial_dataspace(min_ang_cutoff_in_arcmin=5.0)
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

    logger.info("Setting up priors from config...")

    # Extract parameters and priors from config
    params = PRIOR_CONFIG["parameters"]
    uniform_priors = PRIOR_CONFIG["uniform"]
    mvn_config = PRIOR_CONFIG.get("multivariate_gaussian")
    mvn_param = mvn_config["parameter"]
    mvn_mean = np.array(mvn_config["mean"])
    mvn_corr_matrix = np.array(mvn_config["cov"])

    # Convert correlation matrix (with variances on diagonal) to covariance matrix
    mvn_cov = xili.copula_funcs.correlation_to_covariance(mvn_corr_matrix)

    logger.info(f"Parameters: {params}")
    logger.info(f"Uniform prior ranges: {uniform_priors}")
    logger.info(f"Converted correlation matrix to covariance matrix")
    logger.info(f"Standard deviations: {np.sqrt(np.diag(mvn_cov))}")

    logger.info(f"Starting sampler (USE_NESTED={USE_NESTED})")
    
    if USE_NESTED:
        results = run_nested_sampler(prior, likelihood, jobnumber)
        logger.info("Nested sampling complete")
    else:
        results = run_emcee_sampler(likelihood, jobnumber)
        logger.info("MCMC sampling complete")
    
    logger.info("All done!")
