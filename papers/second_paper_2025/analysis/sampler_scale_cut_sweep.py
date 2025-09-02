"""
sampler_scale_cut_sweep.py

Run the sampler for a range of scale cuts, saving chains and logging setup for each run.
"""
import xilikelihood as xili
import numpy as np
import logging
import sys
import os
from config import (
    EXACT_LMAX,
    MASK_CONFIG,
    BASE_DIR,
    PACKAGE_DIR,
    FIDUCIAL_COSMO
)
import tempfile
import json
import glob


def load_chains_for_init(chain_dir, pattern="chain_scale_cut_0.8333333333333334*.npz"):
    """Load existing chains for initialization, similar to plot_scale_cut_posteriors.py"""
    files = sorted(glob.glob(os.path.join(chain_dir, pattern)))
    chains = []
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            if 'samples' in data:
                samples = data['samples']
            elif 'points' in data:
                samples = data['points']
            else:
                continue
            param_names = data['param_names'] if 'param_names' in data else [f"param{i}" for i in range(samples.shape[1])]
            scale_cut = None
            for part in f.split('_'):
                try:
                    scale_cut = float(part)
                    break
                except ValueError:
                    continue
            chains.append({'samples': samples, 'param_names': param_names, 'file': f, 'scale_cut': scale_cut*60.})
        except Exception as e:
            print(f"Warning: Could not load chain from {f}: {e}")
            continue
    return chains


def initialize_walkers_from_chains(chains, n_walkers, param_names, priors, logger):
    """Initialize walkers from existing chain samples"""
    if not chains:
        logger.warning("No existing chains found, using random initialization")
        return None
    
    # Find the chain with the most samples
    best_chain = max(chains, key=lambda c: c['samples'].shape[0])
    logger.info(f"Using chain from {best_chain['file']} for initialization ({best_chain['samples'].shape[0]} samples)")
    
    # Check parameter compatibility
    chain_params = list(best_chain['param_names'])
    if chain_params != param_names:
        logger.warning(f"Parameter mismatch: chain has {chain_params}, need {param_names}")
        return None
    
    # Sample n_walkers points from the chain
    samples = best_chain['samples']
    n_samples = samples.shape[0]
    
    if n_samples < n_walkers:
        logger.warning(f"Chain has only {n_samples} samples but need {n_walkers} walkers, using random initialization")
        return None
    
    # Randomly select n_walkers samples from the chain
    indices = np.random.choice(n_samples, size=n_walkers, replace=False)
    init_positions = samples[indices]
    
    # Verify all positions are within prior bounds
    bounds = priors
    valid_positions = []
    for pos in init_positions:
        is_valid = True
        for i, (low, high) in enumerate(bounds):
            if not (low <= pos[i] <= high):
                is_valid = False
                break
        if is_valid:
            valid_positions.append(pos)
    
    if len(valid_positions) < n_walkers:
        logger.warning(f"Only {len(valid_positions)} valid positions found from chain, using random initialization")
        return None
    
    logger.info(f"Successfully initialized {len(valid_positions)} walkers from existing chain")
    return valid_positions[:n_walkers]
import glob


def run_sampler_for_scale_cut(scale_cut_deg, jobnumber=0, n_redshift_bins=2, n_walkers=6, n_steps=2000, use_nested=False, output_dir="/cluster/scratch/veoehl/scale_cut_chains", init_from_chains=True, use_fixed_covariance=False):
    # Adjust output directory and filenames based on covariance mode
    cov_mode = "fixed_cov" if use_fixed_covariance else "std_cov"
    output_dir = os.path.join(output_dir, cov_mode)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get parent directory if needed
    parent_output_dir = os.path.dirname(output_dir)
    # Or if you want the grandparent: os.path.dirname(os.path.dirname(output_dir))
    
    # Setup logging
    log_path = os.path.join(output_dir, f'sampler_scale_cut_job{jobnumber}_{cov_mode}.log')
    logger = logging.getLogger(f'sampler_scale_cut_job{jobnumber}')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.propagate = False

    logger.info(f"Starting sampler for scale cut {scale_cut_deg} deg, job {jobnumber}")
    logger.info(f"Base directory: {BASE_DIR}")
    logger.info(f"Creating mask with exact_lmax={EXACT_LMAX}, size/nside {MASK_CONFIG['circmaskattr']} and smoothing {MASK_CONFIG['l_smooth']}")
    mask = xili.SphereMask(
        spins=MASK_CONFIG['spins'],
        circmaskattr=MASK_CONFIG['circmaskattr'],
        exact_lmax=EXACT_LMAX,
        l_smooth=MASK_CONFIG['l_smooth'],
        working_dir=PACKAGE_DIR
    )
    logger.info(f"Loading fiducial dataspace (no scale cut applied here)...")
    redshift_bins, ang_bins_in_deg = xili.fiducial_dataspace(min_ang_cutoff_in_arcmin=0)
    redshift_bins = redshift_bins[-n_redshift_bins:]  # Only use last two bins
    logger.info(f"Redshift bins: {len(redshift_bins)}, Angular bins: {len(ang_bins_in_deg)}")
    logger.info(f"Angular bins in degrees: {ang_bins_in_deg}")
    include_ximinus = False
    logger.info(f"Include xi-: {include_ximinus}")
    xilikelihood = xili.XiLikelihood(
        mask=mask,
        redshift_bins=redshift_bins,
        ang_bins_in_deg=ang_bins_in_deg,
        include_ximinus=include_ximinus,
        exact_lmax=EXACT_LMAX,
        large_angle_threshold=scale_cut_deg,  # This is the only place scale_cut_deg is used
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
    logger.info(f"Setting up likelihood with scale cut {scale_cut_deg} deg")
    logger.info(f"Angular bins {np.arange(xilikelihood._n_ang_bins)[~xilikelihood._is_large_angle]} are treated with Gaussian marginals")
    xilikelihood.setup_likelihood()
    xilikelihood.gaussian_covariance = gaussian_covariance
    if use_fixed_covariance:
        xilikelihood.enable_fixed_covariance(True)
        logger.info("Fixed covariance mode enabled - covariance will be cosmology-independent")
    else:
        logger.info("Standard mode - covariance will be cosmology-dependent")
    logger.info("Gaussian covariance set successfully")
    # Priors
    omega_m_prior = np.linspace(0.1, 0.5, 100)
    s8_prior = np.linspace(0.5, 1.1, 100)
    params = ["omega_m", "s8"]
    priors = [(omega_m_prior[0], omega_m_prior[-1]), (s8_prior[0], s8_prior[-1])]
    logger.info(f"Parameters: {params}")
    logger.info(f"Prior ranges: {priors}")
    from nautilus import Prior, Sampler
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
            return -1e30
    # Run sampler
    if use_nested:
        results_path = os.path.join(output_dir, f'sampler_results_scale_cut_{scale_cut_deg}_job{jobnumber}_{cov_mode}_nested.h5')
        logger.info("Setting up Nautilus sampler...")
        sampler = Sampler(prior, likelihood, n_live=16, filepath=results_path)
        logger.info("Starting nested sampling...")
        sampler.run(verbose=True)
        logger.info("Nested sampling completed!")
        points, log_w, log_l = sampler.posterior()
        np.savez(os.path.join(output_dir, f'chain_scale_cut_{scale_cut_deg}_job{jobnumber}_{cov_mode}_nested.npz'), points=points, log_w=log_w, log_l=log_l, log_path=log_path, param_names=params)
        logger.info(f"Chain saved: chain_scale_cut_{scale_cut_deg}_job{jobnumber}_{cov_mode}_nested.npz")
        return points, log_w, log_l, log_path
    else:
        import emcee
        bounds = priors
        ndim = len(bounds)
        
        # Try to initialize from existing chains
        p0 = None
        if init_from_chains:
            logger.info("Attempting to initialize walkers from existing chains...")
            # Search in parent directory to find chains from any covariance mode
            chains = load_chains_for_init(parent_output_dir)
            p0_from_chains = initialize_walkers_from_chains(chains, n_walkers, params, bounds, logger)
            if p0_from_chains is not None:
                p0 = p0_from_chains
        
        # Fall back to random initialization if chain initialization failed
        if p0 is None:
            logger.info("Using random initialization for walkers")
            p0 = [np.array([np.random.uniform(low, high) for (low, high) in bounds]) for _ in range(n_walkers)]
        
        # Log initial positions
        logger.info("Initial walker positions:")
        for i, pos in enumerate(p0):
            logger.info(f"  Walker {i}: {dict(zip(params, pos))}")
        
        def log_prob(theta):
            for i, (low, high) in enumerate(bounds):
                if not (low <= theta[i] <= high):
                    return -np.inf
            param_dict = dict(zip(prior.keys, theta))
            logl = likelihood(param_dict)
            if np.isnan(logl):
                return -np.inf
            return logl
        
        # Test initial log probabilities
        logger.info("Testing initial log probabilities:")
        for i, pos in enumerate(p0):
            logp = log_prob(pos)
            logger.info(f"  Walker {i} initial logprob: {logp}")
            if logp == -np.inf:
                logger.warning(f"  Walker {i} has invalid initial position!")
        
        sampler = emcee.EnsembleSampler(n_walkers, ndim, log_prob)
        logger.info("Running emcee...")
        sampler.run_mcmc(p0, n_steps, progress=True)
        logger.info("emcee sampling completed!")
        flat_samples = sampler.get_chain(discard=0, thin=1, flat=False)
        np.savez(os.path.join(output_dir, f'chain_scale_cut_{scale_cut_deg}_job{jobnumber}_{cov_mode}_emcee.npz'), samples=flat_samples, log_path=log_path, param_names=params)
        logger.info(f"Chain saved: chain_scale_cut_{scale_cut_deg}_job{jobnumber}_{cov_mode}_emcee.npz")
        return flat_samples, log_path

if __name__ == "__main__":
    # Define the array of scale cuts to sweep over
    scale_cuts_deg = np.array([10, 20, 50, 100, 300, 1200]) / 60.  
    if len(sys.argv) < 2:
        print("Usage: python sampler_scale_cut_sweep.py <jobnumber> [--nested] [--random-init] [--fixed-cov]")
        print(f"Available job numbers: 1 to {len(scale_cuts_deg)}")
        print("  --nested: Use nested sampling instead of emcee")
        print("  --random-init: Use random initialization instead of loading from existing chains")
        print("  --fixed-cov: Use fixed covariance (cosmology-independent)")
        sys.exit(1)
    jobnumber = int(sys.argv[1]) - 1
    if not (0 <= jobnumber < len(scale_cuts_deg)):
        print(f"Error: jobnumber must be between 1 and {len(scale_cuts_deg)}")
        sys.exit(1)
    scale_cut = scale_cuts_deg[jobnumber]
    use_nested = "--nested" in sys.argv
    use_random_init = "--random-init" in sys.argv
    use_fixed_cov = "--fixed-cov" in sys.argv
    run_sampler_for_scale_cut(scale_cut, jobnumber=jobnumber, use_nested=use_nested, init_from_chains=not use_random_init, use_fixed_covariance=use_fixed_cov)
