"""
s8_om_posterior.py

Systematic 2D posterior exploration for omega_m and s8 parameters.
More efficient than MCMC for 2D parameter spaces.
"""
import xilikelihood as xili
import numpy as np
import time
import random
import sys
import os
import logging
from pathlib import Path
import tempfile
from config import (
    EXACT_LMAX,
    MASK_CONFIG,
    BASE_DIR,
    PACKAGE_DIR,
    FIDUCIAL_COSMO,
    PARAM_GRIDS
)

def run_2d_posterior_exploration(scale_cut_deg, jobnumber=0, n_redshift_bins=2, 
                                omega_m_range=None, s8_range=None,
                                n_jobs=10, output_dir="/cluster/scratch/veoehl/posteriors_2d", 
                                use_fixed_covariance=False):
    """
    Run systematic 2D posterior exploration for omega_m and s8.
    
    Parameters:
    -----------
    scale_cut_deg : float
        Scale cut in degrees (large_angle_threshold)
    jobnumber : int
        Job array index (0-based)
    n_redshift_bins : int
        Number of redshift bins to use (from the end)
    omega_m_range : tuple, optional
        (min, max, n_points) for omega_m grid. If None, uses PARAM_GRIDS from config.
    s8_range : tuple, optional
        (min, max, n_points) for s8 grid. If None, uses PARAM_GRIDS from config.
    n_jobs : int
        Total number of job array jobs
    output_dir : str
        Base output directory
    use_fixed_covariance : bool
        Whether to use fixed covariance mode
    """
    # Use config defaults if not specified
    if omega_m_range is None:
        omega_m_range = PARAM_GRIDS["omega_m"]
    if s8_range is None:
        s8_range = PARAM_GRIDS["s8"]
    # Adjust output directory and filenames based on covariance mode
    cov_mode = "fixed_cov" if use_fixed_covariance else "std_cov"
    output_dir = os.path.join(output_dir, cov_mode)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_path = os.path.join(output_dir, f'posterior_2d_scale_cut_{scale_cut_deg}_job{jobnumber}_{cov_mode}.log')
    logger = logging.getLogger(f'posterior_2d_job{jobnumber}')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.propagate = False

    logger.info(f"Starting 2D posterior exploration for scale cut {scale_cut_deg} deg, job {jobnumber}")
    logger.info(f"Base directory: {BASE_DIR}")
    
    # Create mask
    logger.info(f"Creating mask with exact_lmax={EXACT_LMAX}")
    mask = xili.SphereMask(
        spins=MASK_CONFIG['spins'],
        circmaskattr=MASK_CONFIG['circmaskattr'],
        exact_lmax=EXACT_LMAX,
        l_smooth=MASK_CONFIG['l_smooth'],
        working_dir=PACKAGE_DIR
    )
    
    # Setup dataspace
    logger.info(f"Loading fiducial dataspace...")
    redshift_bins, ang_bins_in_deg = xili.fiducial_dataspace(min_ang_cutoff_in_arcmin=0)
    redshift_bins = redshift_bins[-n_redshift_bins:]  # Only use last bins
    logger.info(f"Redshift bins: {len(redshift_bins)}, Angular bins: {len(ang_bins_in_deg)}")
    
    # Create likelihood
    include_ximinus = False
    logger.info(f"Include xi-: {include_ximinus}")
    xilikelihood = xili.XiLikelihood(
        mask=mask,
        redshift_bins=redshift_bins,
        ang_bins_in_deg=ang_bins_in_deg,
        include_ximinus=include_ximinus,
        exact_lmax=EXACT_LMAX,
        large_angle_threshold=scale_cut_deg,  # Apply scale cut
    )
    
    # Create mock data and covariance
    logger.info("Creating mock data and covariance...")
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
    
    # Setup likelihood
    logger.info(f"Setting up likelihood with scale cut {scale_cut_deg} deg")
    xilikelihood.setup_likelihood()
    xilikelihood.gaussian_covariance = gaussian_covariance
    
    if use_fixed_covariance:
        xilikelihood.enable_fixed_covariance(True)
        logger.info("Fixed covariance mode enabled - covariance will be cosmology-independent")
    else:
        logger.info("Standard mode - covariance will be cosmology-dependent")
    
    # Create parameter grid
    omega_m_min, omega_m_max, omega_m_points = omega_m_range
    s8_min, s8_max, s8_points = s8_range
    
    omega_m_grid = np.linspace(omega_m_min, omega_m_max, omega_m_points)
    s8_grid = np.linspace(s8_min, s8_max, s8_points)
    
    # Create all parameter combinations
    omega_m_mesh, s8_mesh = np.meshgrid(omega_m_grid, s8_grid)
    param_pairs = np.vstack([omega_m_mesh.ravel(), s8_mesh.ravel()]).T
    
    # Split work across jobs
    split_pairs = np.array_split(param_pairs, n_jobs)
    
    # Validate job number
    if jobnumber >= len(split_pairs):
        logger.error(f"Job number {jobnumber} exceeds maximum jobs ({len(split_pairs)})")
        return
    
    # Get subset for this job
    subset_pairs = split_pairs[jobnumber]
    logger.info(f"Processing {len(subset_pairs)} parameter combinations for job {jobnumber}/{n_jobs}")
    logger.info(f"Parameter ranges: omega_m=[{omega_m_min:.3f}, {omega_m_max:.3f}], s8=[{s8_min:.3f}, {s8_max:.3f}]")
    
    # Setup results array
    results_dtype = np.dtype([
        ("exact_logL", np.float64),
        ("gauss_logL", np.float64),
        ("omega_m", np.float64),
        ("s8", np.float64)
    ])
    results = np.empty(len(subset_pairs), dtype=results_dtype)
    
    # Main computation loop
    logger.info("Starting likelihood evaluations...")
    start_time = time.time()
    failed_computations = 0
    
    for i, (omega_m, s8) in enumerate(subset_pairs):
        try:
            cosmology = {"omega_m": omega_m, "s8": s8}
            exact_logL, gauss_logL = xilikelihood.loglikelihood(mock_data, cosmology, gausscompare=True)
            results[i] = (exact_logL, gauss_logL, omega_m, s8)
            
            # Progress logging
            if (i + 1) % max(1, len(subset_pairs) // 10) == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(subset_pairs) - i - 1) / rate
                logger.info(f"Progress: {i+1}/{len(subset_pairs)} ({100*(i+1)/len(subset_pairs):.1f}%) "
                           f"- Rate: {rate:.2f} iter/s - ETA: {eta:.0f}s")
                           
        except Exception as e:
            logger.warning(f"Failed computation at Ωₘ={omega_m:.3f}, S₈={s8:.3f}: {e}")
            results[i] = (np.nan, np.nan, omega_m, s8)
            failed_computations += 1
            
            if failed_computations > len(subset_pairs) * 0.1:  # >10% failures
                logger.error(f"Too many failed computations ({failed_computations}). Stopping.")
                return
    
    # Save results with metadata
    output_file = os.path.join(output_dir, f'posterior_2d_scale_cut_{scale_cut_deg}_job{jobnumber}_{cov_mode}.npz')
    
    # Save as structured array with metadata
    metadata = {
        'scale_cut_deg': scale_cut_deg,
        'scale_cut_arcmin': scale_cut_deg * 60,
        'covariance_mode': cov_mode,
        'use_fixed_covariance': use_fixed_covariance,
        'jobnumber': jobnumber,
        'n_redshift_bins': n_redshift_bins,
        'omega_m_range': omega_m_range,
        's8_range': s8_range,
        'total_combinations': len(param_pairs),
        'job_combinations': len(subset_pairs),
        'n_jobs': n_jobs,
        'include_ximinus': include_ximinus,
        'exact_lmax': EXACT_LMAX,
        # Mask parameters
        'mask_nside': mask.nside,
        'mask_area_sq_deg': mask.eff_area,
        'mask_circmaskattr': MASK_CONFIG['circmaskattr'],
        'mask_l_smooth': MASK_CONFIG['l_smooth'],
        'field_descriptions': {
            'exact_logL': 'Log-likelihood from exact/copula method',
            'gauss_logL': 'Log-likelihood from Gaussian approximation',
            'omega_m': 'Matter density parameter Omega_m',
            's8': 'Structure growth parameter S8'
        }
    }
    
    np.savez(output_file, 
             results=results,
             **metadata)
    
    total_time = time.time() - start_time
    logger.info(f"Job completed successfully in {total_time:.1f}s")
    logger.info(f"Results saved to {output_file}")
    logger.info(f"Structured array fields: {list(results.dtype.names)}")
    logger.info(f"Metadata keys: {list(metadata.keys())}")
    if failed_computations > 0:
        logger.warning(f"Had {failed_computations} failed computations out of {len(subset_pairs)}")
    
    return results, log_path


if __name__ == "__main__":
    # Scale cuts to explore (same as sampler_scale_cut_sweep.py)
    scale_cuts_deg = np.array([10, 20, 50, 100, 300, 1200]) / 60.
    
    if len(sys.argv) < 2:
        print("Usage: python s8_om_posterior.py <jobnumber> [--scale-cut-index=<idx>] [--n-jobs=<n>] [--fixed-cov]")
        print(f"Available job numbers: 0 to N-1 (where N is specified by --n-jobs)")  
        print(f"Available scale cut indices: 0 to {len(scale_cuts_deg)-1} ({scale_cuts_deg*60} arcmin)")
        print("  --scale-cut-index=<idx>: Which scale cut to use (default: 0)")
        print("  --n-jobs=<n>: Total number of jobs for grid subdivision (default: 10)")
        print("  --fixed-cov: Use fixed covariance (cosmology-independent)")
        sys.exit(1)
    
    # Parse arguments
    jobnumber = int(sys.argv[1])
    
    # Parse optional arguments
    scale_cut_index = 0  # default
    n_jobs = 10  # default
    use_fixed_cov = False
    
    for arg in sys.argv[2:]:
        if arg.startswith('--scale-cut-index='):
            scale_cut_index = int(arg.split('=')[1])
            if not (0 <= scale_cut_index < len(scale_cuts_deg)):
                print(f"Error: scale-cut-index must be between 0 and {len(scale_cuts_deg)-1}")
                sys.exit(1)
        elif arg.startswith('--n-jobs='):
            n_jobs = int(arg.split('=')[1])
            if n_jobs <= 0:
                print(f"Error: n-jobs must be positive")
                sys.exit(1)
        elif arg == '--fixed-cov':
            use_fixed_cov = True
    
    # Validate job number against n_jobs
    if not (0 <= jobnumber < n_jobs):
        print(f"Error: jobnumber must be between 0 and {n_jobs-1} (for {n_jobs} total jobs)")
        sys.exit(1)
    
    scale_cut = scale_cuts_deg[scale_cut_index]
    
    print(f"Running 2D posterior exploration:")
    print(f"  Job number: {jobnumber} (of {n_jobs} total jobs)")
    print(f"  Scale cut: {scale_cut:.4f} deg ({scale_cut*60:.1f} arcmin)")
    print(f"  Covariance mode: {'fixed' if use_fixed_cov else 'standard'}")
    
    run_2d_posterior_exploration(
        scale_cut_deg=scale_cut, 
        jobnumber=jobnumber,
        n_jobs=n_jobs,
        use_fixed_covariance=use_fixed_cov
    )





