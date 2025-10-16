#!/usr/bin/env python3
"""
S8 posterior comparison: Gaussian vs Student-t coupling with different numbers of datapoints.

This script computes S8 posteriors using either Gaussian or Student-t copula coupling,
and compares the results across different numbers of datapoints to show the impact
of copula choice on cosmological parameter inference.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from time import time
from pathlib import Path
import argparse
import xilikelihood as xlh
import logging
from xilikelihood.core_utils import computation_phase, logging_context
from config import (
    EXACT_LMAX, 
    FIDUCIAL_COSMO, 
    DATA_DIR, 
    OUTPUT_DIR, 
    MASK_CONFIG,
    S8_GRIDS,
    REDSHIFT_BINS_PATH,
    DATA_FILES,
    ANG_BINS,
    PACKAGE_DIR
)

# Add package root to path
sys.path.insert(0, str(PACKAGE_DIR))

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
s8_output_dir = OUTPUT_DIR / "s8_copula_comparison"
s8_output_dir.mkdir(parents=True, exist_ok=True)


def setup_likelihood_with_n_datapoints(n_datapoints, correlation_type="all", student_t_dof=5.0, n_angular_bins=None):
    """
    Set up likelihood configuration with specified number of datapoints.
    
    Parameters:
    -----------
    n_datapoints : int
        Target number of datapoints (will be achieved by limiting redshift and angular bins)
    correlation_type : str
        Type of correlations to include: "auto", "cross", "all"
    student_t_dof : float
        Degrees of freedom for Student-t copula (for config setup)
    n_angular_bins : int, optional
        Number of angular bins to use. If None, will be optimized to reach target n_datapoints.
        If specified, will use this many angular bins and adjust redshift bins accordingly.
    """
   
    logger = logging.getLogger('s8_copula_comparison')  # Use the same logger name as main
    
    logger.info(f"Setting up likelihood for {n_datapoints} datapoints")
    logger.info(f"Correlation type: {correlation_type}")
    logger.info(f"Student-t DoF: {student_t_dof}")
    if n_angular_bins is not None:
        logger.info(f"Fixed angular bins: {n_angular_bins}")
        
    # Use fiducial dataspace for proper setup
    redshift_bins_full, ang_bins_in_deg_full = xlh.fiducial_dataspace(min_ang_cutoff_in_arcmin=0)
    
    # Handle fixed angular bins case
    if n_angular_bins is not None:
        # User specified the number of angular bins - optimize redshift bins accordingly
        n_angular_bins_to_use = min(n_angular_bins, len(ang_bins_in_deg_full))
        ang_bins_in_deg = ang_bins_in_deg_full[:n_angular_bins_to_use]
        
        # Find minimum number of redshift bins needed
        target_correlations = max(1, n_datapoints // n_angular_bins_to_use)
        
        # Find the minimum number of redshift bins that gives us enough correlations
        for n_rs_bins in range(1, len(redshift_bins_full) + 1):
            if correlation_type == "auto":
                n_correlations = n_rs_bins
            elif correlation_type == "cross":
                n_correlations = n_rs_bins * (n_rs_bins - 1) // 2
                if n_rs_bins < 2:  # Need at least 2 bins for cross-correlations
                    continue
            else:  # "all"
                n_correlations = n_rs_bins * (n_rs_bins + 1) // 2
                
            if n_correlations >= target_correlations:
                break
        
        # Select redshift bins from highest z
        redshift_bins = redshift_bins_full[-n_rs_bins:]
        actual_n_datapoints = n_correlations * n_angular_bins_to_use
        
        logger.info(f"Configuration: {n_rs_bins} redshift bins, {n_angular_bins_to_use} angular bins → {actual_n_datapoints} datapoints")
        logger.info(f"Selected redshift bins: {redshift_bins}")
        logger.info(f"Selected angular bins (deg): {ang_bins_in_deg}")
        
    else:
        # Original optimization mode - minimize memory usage by prioritizing fewer redshift bins
        max_redshift_bins = len(redshift_bins_full)
        max_angular_bins = len(ang_bins_in_deg_full)
        
        # Strategy: Try to use minimum redshift bins (memory expensive) and maximum angular bins (memory cheap)
        best_config = None
        
        # Try redshift bins in ascending order (prefer fewer redshift bins for memory efficiency)
        for n_rs_bins in range(1, max_redshift_bins + 1):
            if correlation_type == "auto":
                n_correlations = n_rs_bins
            elif correlation_type == "cross":
                n_correlations = n_rs_bins * (n_rs_bins - 1) // 2
                if n_rs_bins < 2:  # Need at least 2 bins for cross-correlations
                    continue
            else:  # "all"
                n_correlations = n_rs_bins * (n_rs_bins + 1) // 2
            
            # Calculate required angular bins to reach target datapoints
            if n_correlations == 0:
                continue
                
            n_angular_bins_needed = max(1, (n_datapoints + n_correlations - 1) // n_correlations)  # Ceiling division
            
            # Check if this is feasible with available angular bins
            if n_angular_bins_needed <= max_angular_bins:
                # This configuration works! Use it and stop (prefer fewer redshift bins)
                n_angular_bins_to_use = min(n_angular_bins_needed, max_angular_bins)
                actual_n_datapoints = n_correlations * n_angular_bins_to_use
                best_config = (n_rs_bins, n_angular_bins_to_use, n_correlations, actual_n_datapoints)
                break  # Found first feasible solution with minimum redshift bins
                break  # Found first feasible solution with minimum redshift bins
        
        if best_config is None:
            # Fallback: use minimum viable configuration
            if correlation_type == "cross":
                n_rs_bins = 2  # Minimum for cross-correlations
                n_correlations = 1  # 2 bins = 1 cross-correlation
            else:
                n_rs_bins = 1
                n_correlations = 1
                
            n_angular_bins_to_use = min(n_datapoints, max_angular_bins)  # Use as many angular bins as needed/possible
            actual_n_datapoints = n_correlations * n_angular_bins_to_use
            best_config = (n_rs_bins, n_angular_bins_to_use, n_correlations, actual_n_datapoints)
        
        n_rs_bins, n_angular_bins_to_use, n_correlations, actual_n_datapoints = best_config
        
        # Limit redshift and angular bins
        # Select redshift bins starting from highest redshift (bin 4, then 3, then 2, etc.)
        redshift_bins = redshift_bins_full[-n_rs_bins:]  # Take the last n_rs_bins (highest z first)
        ang_bins_in_deg = ang_bins_in_deg_full[-n_angular_bins_to_use:]
        
        logger.info(f"Configuration: {n_rs_bins} redshift bins, {n_angular_bins_to_use} angular bins → {actual_n_datapoints} datapoints")
        logger.info(f"Selected redshift bins: {redshift_bins}")
        logger.info(f"Selected angular bins (deg): {ang_bins_in_deg}")

    # Common final setup code
    # Create mask
    mask = xlh.SphereMask(
        spins=MASK_CONFIG["spins"], 
        circmaskattr=MASK_CONFIG["circmaskattr"], 
        exact_lmax=EXACT_LMAX, 
        l_smooth=MASK_CONFIG["l_smooth"],
        working_dir=PACKAGE_DIR
    )

    # Create likelihood configuration with Student-t setup
    config = xlh.LikelihoodConfig(student_t_dof=student_t_dof)

    # Create likelihood
    with computation_phase("Likelihood setup"):
        likelihood = xlh.XiLikelihood(
            mask=mask, redshift_bins=redshift_bins,
            ang_bins_in_deg=ang_bins_in_deg, noise='default',
            config=config, include_ximinus=False
        )
        logger.info(f"Created likelihood with {len(redshift_bins)} redshift bins, {len(ang_bins_in_deg)} angular bins, ximinus=False")
    
    # Always create fresh mock data
    mock_data_path = DATA_DIR / f"mock_data_{correlation_type}_{actual_n_datapoints}dp.npz"
    gaussian_covariance_path = DATA_DIR / f"gaussian_covariance_{correlation_type}_{actual_n_datapoints}dp.npz"
    data_paths = mock_data_path, gaussian_covariance_path
    
    # Always create data to ensure consistency
    with computation_phase("Mock data generation"):
        logger.info(f"Generating mock data: {mock_data_path}")
        logger.info(f"Using random=None (fiducial mean)")
        
        
        xlh.mock_data.create_mock_data(
            likelihood=likelihood,
            mock_data_path=mock_data_path,
            gaussian_covariance_path=gaussian_covariance_path,
            fiducial_cosmo=FIDUCIAL_COSMO,
            random=None,
            exact_lmax=EXACT_LMAX
        )
        
    
            
        logger.info("Mock data generation completed")

    return likelihood, data_paths, actual_n_datapoints, correlation_type


def create_correlation_subset(likelihood, correlation_type):
    """
    Create subset list for data based on correlation type.
    
    Parameters:
    -----------
    likelihood : XiLikelihood
        Configured likelihood object
    correlation_type : str
        Type of correlations: "auto", "cross", "all"
        
    Returns:
    --------
    list : List of (rs_comb, ang_bin) tuples for data subsetting
    """
    likelihood.setup_likelihood()  # Ensure _n_to_bin_comb_mapper is set
    
    # Get total number of redshift bin combinations
    n_redshift_bin_combs = likelihood._n_redshift_bin_combs
    n_angular_bins = len(likelihood.ang_bins_in_deg)
    
    # Get auto/cross classification
    is_cross = likelihood._is_cov_cross
    
    subset = []
    
    for rs_comb in range(n_redshift_bin_combs):
        if correlation_type == "auto" and is_cross[rs_comb]:
            continue  # Skip cross-correlations
        elif correlation_type == "cross" and not is_cross[rs_comb]:
            continue  # Skip auto-correlations
        # For "all", include everything
        
        # Add all angular bins for this redshift combination
        for ang_bin in range(n_angular_bins):
            subset.append((rs_comb, ang_bin))
    
    return subset


def compute_s8_posterior_comparison(n_datapoints, s8_grid="medium", student_t_dof=5.0, correlation_type="all", n_angular_bins=None, job_mode=False):
    """
    Compare S8 posteriors using Gaussian vs Student-t coupling for given number of datapoints.
    
    Parameters:
    -----------
    n_datapoints : int
        Number of datapoints to use
    s8_grid : str
        S8 grid resolution ("narrow", "medium", "wide")
    student_t_dof : float
        Degrees of freedom for Student-t copula
    correlation_type : str
        Type of correlations to include: "auto", "cross", "all"
    n_angular_bins : int, optional
        Number of angular bins to use. If None, optimized automatically.
    job_mode : bool
        If True, suppress console output for job runs
        
    Returns:
    --------
    dict : Results containing S8 grid, posteriors, and statistics
    """
    import logging
    logger = logging.getLogger('s8_copula_comparison')  # Use the same logger name as main
    
    # Create separate progress logger that doesn't get cluttered with xilikelihood logging
    progress_logger = logging.getLogger('s8_progress')
    progress_logger.setLevel(logging.INFO)
    
    # Only add console handler if not in job mode and no handlers exist yet
    if not job_mode and not progress_logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')  # Clean format for progress
        console_handler.setFormatter(formatter)
        progress_logger.addHandler(console_handler)
        progress_logger.propagate = False  # Don't send to parent loggers
    
    logger.info(f"Starting S8 posterior comparison with {n_datapoints} datapoints")
    logger.info(f"Configuration: {correlation_type} correlations, {s8_grid} S8 grid, DoF={student_t_dof}")
    
    # Use progress logger for interactive output
    progress_logger.info(f"{'='*60}")
    progress_logger.info(f"Computing S8 posteriors with {n_datapoints} datapoints")
    progress_logger.info(f"Correlation type: {correlation_type}")
    if n_angular_bins is not None:
        progress_logger.info(f"Fixed angular bins: {n_angular_bins}")
    progress_logger.info(f"S8 grid: {s8_grid}, Student-t DoF: {student_t_dof}")
    progress_logger.info(f"{'='*60}")
    
    # Setup likelihood
    likelihood, data_paths, actual_n_datapoints, _ = setup_likelihood_with_n_datapoints(
        n_datapoints, correlation_type, student_t_dof, n_angular_bins)
    mock_data_path, gaussian_covariance_path = data_paths
    
    # Load data
    with computation_phase("Data loading"):
        logger.info(f"Loading mock data from: {mock_data_path}")
        logger.info(f"Loading covariance from: {gaussian_covariance_path}")
        
        # Check if files exist before loading
        if not mock_data_path.exists():
            logger.error(f"Mock data file does not exist: {mock_data_path}")
            raise FileNotFoundError(f"Mock data file not found: {mock_data_path}")
        if not gaussian_covariance_path.exists():
            logger.error(f"Covariance file does not exist: {gaussian_covariance_path}")
            raise FileNotFoundError(f"Covariance file not found: {gaussian_covariance_path}")
        
        mock_data = xlh.load_arrays(mock_data_path, ["data"])["data"]
        gaussian_covariance = xlh.load_arrays(gaussian_covariance_path, ["cov"])["cov"]
        logger.info(f"Loaded mock data: {mock_data.shape}")
        logger.info(f"Loaded covariance: {gaussian_covariance.shape}")
    
    # Setup likelihood
    with computation_phase("Likelihood configuration"):
        likelihood.setup_likelihood()
        logger.info("Likelihood setup completed")
    
    # Create subset indices based on correlation type
    with computation_phase("Correlation subset creation"):
        subset = create_correlation_subset(likelihood, correlation_type)
        logger.info(f"Created subset with {len(subset)} (rs_comb, ang_bin) pairs")
    
    # For data_subset mode, we pass the full data and let likelihood handle subsetting
    # But we still need to subset the covariance matrix
    with computation_phase("Covariance subsetting"):
        cov = xlh.copula_funcs.cov_subset(gaussian_covariance, subset, len(likelihood.ang_bins_in_deg))
        logger.info(f"Subsetted covariance to shape: {cov.shape}")
    
    # Use the original mock data (likelihood will subset it internally)
    mockdata = mock_data
    
    # Set covariance matrix
    likelihood.gaussian_covariance = cov
    
    logger.info(f"Using {len(subset)} datapoints from {correlation_type} correlations")
    
    # Setup S8 grid
    s8_min, s8_max, s8_points = S8_GRIDS[s8_grid]
    s8 = np.linspace(s8_min, s8_max, s8_points)
    logger.info(f"S8 grid: {s8_points} points from {s8_min} to {s8_max}")
    
    # Setup cosmology
    cosmology = FIDUCIAL_COSMO.copy()
    logger.info(f"Base cosmology: {cosmology}")
    
    # Compute posteriors for both copula types
    results = {}
    
    for copula_name, use_student_t in [("gaussian", False), ("student_t", True)]:
        logger.info(f"Computing {copula_name} copula posteriors...")
        progress_logger.info(f"\nComputing {copula_name} copula posteriors...")
        
        # Quick test evaluation at fiducial cosmology
        test_cosmology = FIDUCIAL_COSMO.copy()
        logger.info(f"Testing {copula_name} copula with fiducial cosmology...")
        try:
            test_post = likelihood.loglikelihood(mockdata, test_cosmology, 
                                              use_student_t=use_student_t, 
                                              data_subset=subset)
            logger.info(f"Test evaluation successful: {test_post:.6f}")
            progress_logger.info(f"  Test evaluation at fiducial: {test_post:.6f}")
        except Exception as e:
            logger.error(f"Test evaluation FAILED for {copula_name}: {e}")
            progress_logger.error(f"  TEST FAILED for {copula_name}: {e}")
            import traceback
            logger.error(f"  Test traceback: {traceback.format_exc()}")
        
        posts = []
        start_total = time()
        
        with computation_phase(f"{copula_name.capitalize()} posterior computation"):
            logger.info(f"Starting {copula_name} copula evaluation over {len(s8)} S8 points")
            progress_logger.info(f"=== {copula_name.upper()} COPULA EVALUATION ===")
            progress_logger.info(f"Evaluating {len(s8)} S8 points from {s8[0]:.3f} to {s8[-1]:.3f}")
            
            for i, s in enumerate(s8):
                start_time = time()
                cosmology["s8"] = s
                
                try:
                    # Use the use_student_t parameter to control copula type
                    # Pass the subset to tell likelihood which correlations to use
                    post = likelihood.loglikelihood(mockdata, cosmology, 
                                                  use_student_t=use_student_t, 
                                                  data_subset=subset)
                    
                    # Check for NaN result
                    if np.isnan(post):
                        error_msg = f"  NaN result at s8={s:.3f} with {copula_name} copula"
                        progress_logger.error(error_msg)
                        logger.error(error_msg)
                        posts.append(-np.inf)
                        continue
                    
                    posts.append(post)
                    
                    iteration_time = time() - start_time
                    
                    # Progress logging - cleaner output for tracking
                    progress_pct = 100 * (i + 1) / len(s8)
                    elapsed_total = time() - start_total
                    eta_total = elapsed_total * len(s8) / (i + 1) if i > 0 else 0
                    eta_remaining = eta_total - elapsed_total
                    
                    if i % 10 == 0 or i == len(s8) - 1:  # Every 10th iteration or last
                        progress_msg = f"Progress: {i+1:3d}/{len(s8)} ({progress_pct:5.1f}%) | s8={s:.3f} → {post:8.3f} | {iteration_time:5.2f}s | ETA: {eta_remaining/60:4.1f}min"
                        progress_logger.info(progress_msg)
                        # Also log to main logger but at debug level to avoid clutter
                        logger.debug(progress_msg)
                        
                except Exception as e:
                    iteration_time = time() - start_time
                    progress_pct = 100 * (i + 1) / len(s8)
                    error_msg = f"ERROR [{progress_pct:.1f}%]: s8={s:.3f} failed after {iteration_time:.2f}s: {e}"
                    progress_logger.error(error_msg)
                    logger.error(error_msg)
                    # Print more detailed traceback for debugging
                    import traceback
                    logger.error(f"  Full traceback: {traceback.format_exc()}")
                    posts.append(-np.inf)
        
        total_time = time() - start_total
        logger.info(f"Completed {copula_name} posterior computation in {total_time:.1f}s")
        progress_logger.info(f"=== {copula_name.upper()} COMPLETED in {total_time:.1f}s ===")
        
        # Normalize posterior
        posts = np.array(posts)
        
        # Check for valid results
        if np.all(posts == -np.inf):
            warning_msg = f"  WARNING: All {copula_name} posteriors are -inf!"
            progress_logger.warning(warning_msg)
            logger.warning(warning_msg)
            results[copula_name] = None
            continue
            
        # Normalize (convert log to linear, normalize)
        with computation_phase(f"{copula_name.capitalize()} posterior normalization"):
            posts_norm = posts - np.max(posts)  # Prevent overflow
            posts_linear = np.exp(posts_norm)
            posts_linear = posts_linear / np.trapz(posts_linear, s8)  # Normalize
            logger.debug(f"Normalized {copula_name} posterior")
        
        # Compute statistics
        mean = np.trapz(s8 * posts_linear, s8)
        max_idx = np.argmax(posts_linear)
        maximum = s8[max_idx]
        
        # Compute credible intervals
        cumulative = np.cumsum(posts_linear) * (s8[1] - s8[0])
        idx_16 = np.searchsorted(cumulative, 0.16)
        idx_84 = np.searchsorted(cumulative, 0.84)
        ci_16 = s8[idx_16] if idx_16 < len(s8) else s8[0]
        ci_84 = s8[idx_84] if idx_84 < len(s8) else s8[-1]
        
        results[copula_name] = {
            'posterior': posts_linear,
            'log_posterior': posts,
            'mean': mean,
            'maximum': maximum,
            'ci_16': ci_16,
            'ci_84': ci_84,
            'sigma': (ci_84 - ci_16) / 2,
        }
        
        logger.info(f"{copula_name.capitalize()} results: Mean={mean:.4f}, Mode={maximum:.4f}, σ={(ci_84 - ci_16) / 2:.4f}")
        progress_logger.info(f"  Mean: {mean:.4f}")
        progress_logger.info(f"  Mode: {maximum:.4f}")
        progress_logger.info(f"  68% CI: [{ci_16:.4f}, {ci_84:.4f}]")
        progress_logger.info(f"  σ: {(ci_84 - ci_16) / 2:.4f}")
    
    # Compare results
    if results["gaussian"] is not None and results["student_t"] is not None:
        mean_diff = results["student_t"]["mean"] - results["gaussian"]["mean"]
        mode_diff = results["student_t"]["maximum"] - results["gaussian"]["maximum"]
        sigma_diff = results["student_t"]["sigma"] - results["gaussian"]["sigma"]
        
        progress_logger.info(f"\nCOPULA COMPARISON:")
        progress_logger.info(f"  Mean difference (St - G): {mean_diff:+.4f}")
        progress_logger.info(f"  Mode difference (St - G): {mode_diff:+.4f}")
        progress_logger.info(f"  σ difference (St - G): {sigma_diff:+.4f}")
        progress_logger.info(f"  Relative mean shift: {100*mean_diff/results['gaussian']['mean']:+.1f}%")
        
        logger.info(f"COPULA COMPARISON: Mean diff={mean_diff:+.4f}, Mode diff={mode_diff:+.4f}, σ diff={sigma_diff:+.4f}")
        logger.info(f"Relative mean shift: {100*mean_diff/results['gaussian']['mean']:+.1f}%")
    
    logger.info(f"S8 posterior comparison completed for {len(subset)} datapoints")
    
    # Log bin configuration details
    logger.info(f"Bin configuration used:")
    logger.info(f"  Redshift bins: {likelihood.redshift_bins}")
    logger.info(f"  Angular bins (deg): {likelihood.ang_bins_in_deg}")
    logger.info(f"  Total redshift bins: {len(likelihood.redshift_bins)}")
    logger.info(f"  Total angular bins: {len(likelihood.ang_bins_in_deg)}")
    
    return {
        's8': s8,
        'n_datapoints': len(subset),
        'correlation_type': correlation_type,
        'student_t_dof': student_t_dof,
        'results': results,
        'fiducial_s8': FIDUCIAL_COSMO['s8'],
        # Store bin configuration for reproducibility
        'redshift_bins': likelihood.redshift_bins,
        'ang_bins_in_deg': likelihood.ang_bins_in_deg,
        'n_redshift_bins': len(likelihood.redshift_bins),
        'n_angular_bins': len(likelihood.ang_bins_in_deg),
        'subset_indices': subset
    }


def plot_s8_comparison(all_results, save_path=None):
    """
    Plot S8 posterior comparison across different numbers of datapoints.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data for plotting
    n_datapoints_list = sorted([r['n_datapoints'] for r in all_results])
    colors = plt.cm.viridis(np.linspace(0, 1, len(n_datapoints_list)))
    
    # Get correlation type from first result for title
    correlation_type = all_results[0].get('correlation_type', 'all') if all_results else 'all'
    
    # Plot 1: Gaussian copula posteriors
    ax = axes[0, 0]
    for i, result in enumerate(all_results):
        n_data = result['n_datapoints']
        if result['results']['gaussian'] is not None:
            s8 = result['s8']
            posterior = result['results']['gaussian']['posterior']
            ax.plot(s8, posterior, color=colors[i], label=f'{n_data} datapoints', linewidth=2)
    
    ax.axvline(FIDUCIAL_COSMO['s8'], color='red', linestyle='--', alpha=0.7, label='Fiducial')
    ax.set_xlabel('S8')
    ax.set_ylabel('Posterior Density')
    ax.set_title(f'Gaussian Copula ({correlation_type} correlations)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Student-t copula posteriors
    ax = axes[0, 1]
    for i, result in enumerate(all_results):
        n_data = result['n_datapoints']
        if result['results']['student_t'] is not None:
            s8 = result['s8']
            posterior = result['results']['student_t']['posterior']
            ax.plot(s8, posterior, color=colors[i], label=f'{n_data} datapoints', linewidth=2)
    
    ax.axvline(FIDUCIAL_COSMO['s8'], color='red', linestyle='--', alpha=0.7, label='Fiducial')
    ax.set_xlabel('S8')
    ax.set_ylabel('Posterior Density')
    student_t_dof = all_results[0].get('student_t_dof', 5.0) if all_results else 5.0
    ax.set_title(f'Student-t Copula (ν={student_t_dof}, {correlation_type} correlations)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Direct comparison (highest n_datapoints)
    ax = axes[1, 0]
    max_n_result = max(all_results, key=lambda x: x['n_datapoints'])
    
    if max_n_result['results']['gaussian'] is not None:
        s8 = max_n_result['s8']
        gauss_post = max_n_result['results']['gaussian']['posterior']
        ax.plot(s8, gauss_post, 'orange', label='Gaussian Copula')
    
    
    if max_n_result['results']['student_t'] is not None:
        s8 = max_n_result['s8']
        st_post = max_n_result['results']['student_t']['posterior']
        ax.plot(s8, st_post, 'blue', linestyle='--', label='Student-t Copula')
    
    ax.axvline(FIDUCIAL_COSMO['s8'], color='red', linestyle='--', alpha=0.7, label='Fiducial')
    ax.set_xlabel('S8')
    ax.set_ylabel('Posterior Density')
    ax.set_title(f'Direct Comparison ({max_n_result["n_datapoints"]} datapoints)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Mean and sigma evolution
    ax = axes[1, 1]
    n_points = []
    gauss_means = []
    st_means = []
    gauss_sigmas = []
    st_sigmas = []
    
    for result in all_results:
        n_points.append(result['n_datapoints'])
        
        if result['results']['gaussian'] is not None:
            gauss_means.append(result['results']['gaussian']['mean'])
            gauss_sigmas.append(result['results']['gaussian']['sigma'])
        else:
            gauss_means.append(np.nan)
            gauss_sigmas.append(np.nan)
            
        if result['results']['student_t'] is not None:
            st_means.append(result['results']['student_t']['mean'])
            st_sigmas.append(result['results']['student_t']['sigma'])
        else:
            st_means.append(np.nan)
            st_sigmas.append(np.nan)
    
    ax.plot(n_points, gauss_means, 'o-', color='orange', linewidth=2, markersize=6, label='Gaussian Mean')
    ax.plot(n_points, st_means, 's--', color='blue', linewidth=2, markersize=6, label='Student-t Mean')
    ax.axhline(FIDUCIAL_COSMO['s8'], color='red', linestyle='--', alpha=0.7, label='Fiducial')
    
    ax.set_xlabel('Number of Datapoints')
    ax.set_ylabel('S8')
    ax.set_title('Posterior Mean vs Data Dimensionality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    #ax.set_xscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved: {save_path}")
    
    return fig


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Compare S8 posteriors with Gaussian vs Student-t coupling")
    parser.add_argument("--n-datapoints", nargs="+", type=int, default=[5, 10, 20, 50], 
                        help="List of datapoint numbers to test")
    parser.add_argument("--job-index", type=int, default=None,
                        help="Job array index (0-based) to select single datapoint configuration")
    parser.add_argument("--correlation-type", choices=["auto", "cross", "all"], default="all",
                        help="Type of correlations to include in analysis")
    parser.add_argument("--n-angular-bins", type=int, default=None,
                        help="Number of angular bins to use (default: optimize automatically)")
    parser.add_argument("--s8-grid", choices=["narrow", "medium", "wide", "test"], default="medium",
                        help="S8 grid resolution")
    parser.add_argument("--student-t-dof", type=float, default=5.0,
                        help="Degrees of freedom for Student-t copula")
    parser.add_argument("--output-dir", type=str, default=str(s8_output_dir),
                        help="Output directory for results")
    parser.add_argument("--save-individual", action="store_true",
                        help="Save individual results as .npz files")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--no-log-file", action="store_true",
                        help="Disable file logging")
    parser.add_argument("--with-tail-analysis", action="store_true",
                        help="Include tail dependence analysis for copula validation")
    parser.add_argument("--tail-realizations", type=int, default=500,
                        help="Number of realizations for tail dependence analysis")

    args = parser.parse_args()
    
    # Handle job array mode
    if args.job_index is not None:
        # Job array mode: select single datapoint configuration
        if args.job_index < 0 or args.job_index >= len(args.n_datapoints):
            print(f"ERROR: Job index {args.job_index} is out of range [0, {len(args.n_datapoints)-1}]")
            print(f"Available datapoint configurations: {args.n_datapoints}")
            return 1
        
        # Select single configuration for this job
        selected_n_datapoints = [args.n_datapoints[args.job_index]]
        job_mode = True
    else:
        # Sequential mode: process all configurations
        selected_n_datapoints = args.n_datapoints
        job_mode = False
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine logging configuration (include job index in log file if in job array mode)
    log_level = "DEBUG" if args.verbose else "INFO"
    if args.no_log_file:
        log_file = None
    elif job_mode:
        log_file = output_dir / f"s8_copula_comparison_job{args.job_index}.log"
    else:
        log_file = output_dir / "s8_copula_comparison.log"
    
    # ENTIRE ANALYSIS with logging context - clean and simple!
    with logging_context(log_file=log_file, level=log_level, console_output=True) as logger:
        
        # Configure xilikelihood logging to separate file while keeping main log clean
        xlh_logger = logging.getLogger('xilikelihood')
        xlh_logger.setLevel(logging.INFO)  # Allow INFO level for file logging
        
        # Remove any existing handlers to avoid duplicates
        xlh_logger.handlers = []
        
        # Add file handler for detailed xilikelihood logs
        xlh_log_file = log_file.parent / f"xilikelihood_{log_file.name}"
        xlh_file_handler = logging.FileHandler(xlh_log_file)
        xlh_file_handler.setLevel(logging.INFO)
        xlh_file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        xlh_file_handler.setFormatter(xlh_file_formatter)
        xlh_logger.addHandler(xlh_file_handler)
        
        # Add console/main log handler that only shows WARNING+
        xlh_main_handler = logging.StreamHandler()
        xlh_main_handler.setLevel(logging.WARNING)
        xlh_main_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        xlh_main_handler.setFormatter(xlh_main_formatter)
        xlh_logger.addHandler(xlh_main_handler)
        
        # Don't propagate to root logger to avoid duplication
        xlh_logger.propagate = False
        
        logger.info(f"xilikelihood INFO+ logs → {xlh_log_file}")
        logger.info(f"xilikelihood WARNING+ logs → main log")
        
        # Also reduce other noisy loggers
        for noisy_logger_name in ['jax', 'jaxlib', 'matplotlib', 'numpy']:
            noisy_logger = logging.getLogger(noisy_logger_name)
            noisy_logger.setLevel(logging.WARNING)
        
        if job_mode:
            logger.info(f"=== S8 Copula Comparison (Job {args.job_index}) ===")
            logger.info(f"Processing {selected_n_datapoints[0]} datapoints")
        else:
            logger.info("=== S8 Copula Comparison Analysis ===")
            logger.info(f"Processing {len(selected_n_datapoints)} configurations: {selected_n_datapoints}")
            
        logger.info(f"Correlation type: {args.correlation_type}")
        logger.info(f"Student-t DoF: {args.student_t_dof}, S8 grid: {args.s8_grid}")
        
        # Run tail dependence analysis if requested
        tail_results = None
        if args.with_tail_analysis:
            logger.info("Running tail dependence analysis for copula validation...")
            try:
                # Use functional approach - simpler than class
                from papers.second_paper_2025.analysis.tail_dependence_functional import analyze_tail_dependence
                
                tail_results = {}
                for n_datapoints in selected_n_datapoints:
                    logger.info(f"Tail dependence analysis for {n_datapoints} datapoints...")
                    
                    result = analyze_tail_dependence(
                        n_realizations=args.tail_realizations,
                        n_datapoints=n_datapoints,
                        correlation_type=args.correlation_type,
                        n_angular_bins=args.n_angular_bins,
                        output_dir=Path(args.output_dir) / "tail_dependence",
                        verbose=args.verbose
                    )
                    
                    tail_results[n_datapoints] = result
                    recommendation = result['recommended_copula']
                    tail_coeffs = result['tail_coefficients']
                    
                    logger.info(f"  Recommended copula: {recommendation}")
                    logger.info(f"  Tail dependence - Upper: {tail_coeffs['upper_tail_avg']:.3f}, Lower: {tail_coeffs['lower_tail_avg']:.3f}")
            
            except ImportError:
                logger.warning("Could not import tail_dependence_functional module. Skipping tail analysis.")
            except Exception as e:
                logger.error(f"Error in tail dependence analysis: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Compute posteriors for each number of datapoints
        all_results = []
        
        analysis_name = f"Job {args.job_index}" if job_mode else "S8 Analysis"
        with computation_phase(analysis_name):
            for i, n_data in enumerate(selected_n_datapoints):
                config_label = f"Job {args.job_index}" if job_mode else f"{i+1}/{len(selected_n_datapoints)}"
                logger.info(f"Processing configuration {config_label}: {n_data} datapoints")
                
                try:
                    with computation_phase(f"S8 Analysis ({n_data} datapoints)"):
                        result = compute_s8_posterior_comparison(
                            n_datapoints=n_data,
                            s8_grid=args.s8_grid,
                            student_t_dof=args.student_t_dof,
                            correlation_type=args.correlation_type,
                            n_angular_bins=args.n_angular_bins,
                            job_mode=job_mode
                        )
                        all_results.append(result)
                        logger.info(f"Successfully completed analysis for {n_data} datapoints")
                    
                    # Save individual result if requested
                    if args.save_individual:
                        job_suffix = f"_job{args.job_index}" if job_mode else ""
                        filename = output_dir / f"s8_comparison_{args.correlation_type}_{n_data}datapoints{job_suffix}.npz"
                        np.savez(filename, **result)
                        logger.info(f"Individual result saved: {filename}")
                        
                except Exception as e:
                    logger.error(f"ERROR with {n_data} datapoints: {e}")
                    print(f"ERROR with {n_data} datapoints: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    traceback.print_exc()
                    continue
        
        if len(all_results) == 0:
            logger.error("No valid results obtained!")
            print("No valid results obtained!")
            return

        # Create plots and save results (skip plotting in job array mode)
        if not job_mode:
            with computation_phase("Plotting results"):
                plot_path = output_dir / f"s8_copula_comparison_{args.correlation_type}.png"
                plot_s8_comparison(all_results, save_path=plot_path)
                logger.info(f"Plot saved: {plot_path}")

        # Save combined results
        job_suffix = f"_job{args.job_index}" if job_mode else "_all"
        combined_path = output_dir / f"s8_copula_comparison_{args.correlation_type}{job_suffix}.npz"
        np.savez(combined_path, results=all_results)
        logger.info(f"Results saved: {combined_path}")

        # Print summary statistics
        logger.info(f"SUMMARY STATISTICS ({args.correlation_type} correlations)")

        for result in all_results:
            n_data = result['n_datapoints']
            
            # Log bin configuration used
            logger.info(f"{n_data}dp Configuration:")
            logger.info(f"  Redshift bins ({result['n_redshift_bins']}): {result['redshift_bins']}")
            logger.info(f"  Angular bins ({result['n_angular_bins']}): {result['ang_bins_in_deg']}")
            
            if result['results']['gaussian'] is not None:
                gauss = result['results']['gaussian']
                logger.info(f"{n_data}dp Gaussian: Mean={gauss['mean']:.4f}, σ={gauss['sigma']:.4f}")
            
            if result['results']['student_t'] is not None:
                studt = result['results']['student_t']
                logger.info(f"{n_data}dp Student-t: Mean={studt['mean']:.4f}, σ={studt['sigma']:.4f}")
                
            if result['results']['gaussian'] is not None and result['results']['student_t'] is not None:
                mean_diff = studt['mean'] - gauss['mean']
                sigma_diff = studt['sigma'] - gauss['sigma']
                logger.info(f"{n_data}dp Difference: ΔMean={mean_diff:+.4f}, Δσ={sigma_diff:+.4f}")
            
            logger.info("")  # Add blank line between configurations

        logger.info("✓ Analysis completed successfully")


if __name__ == "__main__":
    main()
