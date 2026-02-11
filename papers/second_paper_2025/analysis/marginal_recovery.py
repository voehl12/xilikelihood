"""
Test Student-t copula marginal recovery using 2D likelihood evaluation.

This test evaluates how well Student-t copulas with different degrees of freedom
can recover marginal distributions using the same 2D likelihood evaluation approach
as used in plot_sims.py for actual likelihood plotting.

The test:
1. Creates a realistic xilikelihood setup  
2. Tests 2D likelihood marginal recovery using likelihood_function_2d()
3. Determines safe degrees of freedom values for copula usage
"""

import numpy as np
import logging
import time
from itertools import combinations
from pathlib import Path
from config import (
    EXACT_LMAX,
    FIDUCIAL_COSMO,
    MASK_CONFIG,
    PACKAGE_DIR
)

# Set JAX to be quiet about compilation
import os
os.environ['JAX_LOG_COMPILES'] = '0'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Turn off JAX debug logging to reduce clutter
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('jax._src').setLevel(logging.WARNING)
logging.getLogger('jax.interpreters').setLevel(logging.WARNING)
logging.getLogger('jaxlib').setLevel(logging.WARNING)

# Turn off matplotlib font debugging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

# Set matplotlib to use simple backend and suppress font warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans'],  # Use standard font
    'figure.dpi': 100,
    'savefig.dpi': 150
})

try:
    import xilikelihood as xlh
except ImportError as e:
    logger.error(f"Failed to import xilikelihood: {e}")
    raise

def create_realistic_likelihood_setup():
    """Create a realistic xilikelihood setup for testing."""
    logger.info("Setting up realistic likelihood configuration...")
    
    mask = xlh.SphereMask(
    spins=MASK_CONFIG['spins'], 
    circmaskattr=MASK_CONFIG['circmaskattr'], 
    exact_lmax=EXACT_LMAX, 
    l_smooth=MASK_CONFIG['l_smooth'],
    working_dir=PACKAGE_DIR  # Use package root for shared arrays
    )

    redshift_bins, ang_bins_in_deg = xlh.fiducial_dataspace()

    rs = np.array([2,4])
    ab = np.array([2, 3])
    rs_selection = [redshift_bins[i] for i in rs]
    ab_selection = [ang_bins_in_deg[i] for i in ab]

    likelihood = xlh.XiLikelihood(
            mask=mask, redshift_bins=rs_selection, ang_bins_in_deg=ab_selection,noise=None,include_ximinus=False)
    likelihood.setup_likelihood()
    likelihood._prepare_likelihood_components(xlh.fiducial_cosmo(),highell=True)
    #xs,pdfs = likelihood._xs,likelihood._pdfs
    
    
    logger.info(f"Created likelihood with {len(rs_selection)} redshift bins and "
                f"{len(ab_selection)} angular bins")
    
    return likelihood

def test_2d_likelihood_marginal_consistency(likelihood, coupling_types=['gaussian', 'student_t'], df_values=[5, 10, 15, 20], exact_marginals=True, max_pairs=6):
    """
    Test 2D likelihood marginal consistency for different coupling types.
    
    This tests both Gaussian and Student-t coupling to isolate whether issues are 
    coupling-specific or in the overall setup.
    
    Parameters:
    -----------
    likelihood : XiLikelihood
        The likelihood object
    coupling_types : list, default ['gaussian', 'student_t']
        Types of coupling to test: 'gaussian' and/or 'student_t'
    df_values : list, default [3, 5, 10]
        Degrees of freedom values to test (only used for student_t coupling)
    exact_marginals : bool, default True
        If True, uses exact marginal computation (slower but accurate)
        If False, uses quick gamma approximation marginals
    max_pairs : int, default 6
        Maximum number of data pairs to test
        
    Returns:
    --------
    dict : Results for each coupling type and df value tested
    """
    logger.info(f"Testing 2D likelihood marginal consistency with coupling types: {coupling_types}")
    if 'student_t' in coupling_types:
        logger.info(f"Student-t df values: {df_values}")
    logger.info(f"Using {'exact' if exact_marginals else 'quick'} marginal computation")
    
    # Get data dimensions from the likelihood object
    n_redshift_bins = likelihood._n_redshift_bin_combs
    n_data_per_bin = likelihood.n_data_points_per_rs_comb  # Includes ximinus if chosen
    
    logger.info(f"Likelihood has {n_redshift_bins} redshift bin combinations, "
                f"{n_data_per_bin} data points per combination")
    logger.info(f"Actual _xs array shape: {likelihood._xs.shape if hasattr(likelihood, '_xs') else 'Not available'}")
    
    # Create representative data subset pairs (same format as plot_sims.py)
    # Use all available dimensions since likelihood is properly set up
    from itertools import product
    data_subset = list(product(np.arange(3), np.arange(2)))


    subset_pairs = list(combinations(data_subset, 2))
    
    logger.info(f"Created {len(data_subset)} data points from {n_redshift_bins} × {n_data_per_bin} dimensions")
    logger.info(f"Created {len(data_subset)} data points: {data_subset}")
    logger.info(f"Testing {len(subset_pairs)} representative pairs: {subset_pairs[:3]}...")
    
    results = {}
    
    for coupling_type in coupling_types:
        if coupling_type == 'gaussian':
            logger.info("="*50)
            logger.info("TESTING GAUSSIAN COUPLING")
            logger.info("="*50)
            
            # Test Gaussian coupling (no df needed)
            results['gaussian'] = test_coupling_marginals(
                likelihood, coupling_type='gaussian', df=None, 
                subset_pairs=subset_pairs, exact_marginals=exact_marginals
            )
            
        elif coupling_type == 'student_t':
            logger.info("="*50) 
            logger.info("TESTING STUDENT-T COUPLING")
            logger.info("="*50)
            
            # Test each df value for Student-t
            for df in df_values:
                results[f'student_t_df_{df}'] = test_coupling_marginals(
                    likelihood, coupling_type='student_t', df=df,
                    subset_pairs=subset_pairs, exact_marginals=exact_marginals
                )
    
    return results

def test_coupling_marginals(likelihood, coupling_type, df, subset_pairs, exact_marginals=True):
    """Test marginal recovery for a specific coupling type."""
    if coupling_type == 'gaussian':
        logger.info("Testing Gaussian coupling")
        test_label = 'Gaussian'
        use_student_t = False
    else:
        logger.info(f"Testing Student-t coupling with df = {df}")
        test_label = f'Student-t (df={df})'
        use_student_t = True
        likelihood.config.student_t_dof = df
    
    pair_results = []
    
    for pair_idx, pair in enumerate(subset_pairs):
        logger.debug(f"Processing pair {pair_idx + 1}/{len(subset_pairs)}: {pair}")
        
        try:
            # Use XiLikelihood's 2D likelihood evaluation
            x_grid, loglik_2d = likelihood.likelihood_function_2d(
                data_subset=pair,
                gausscompare=False,
                use_student_t=use_student_t
            )
            
            # Convert to probability densities
            prob_2d = np.exp(loglik_2d)
            
            # Check 2D normalization
            dx1 = x_grid[0][1] - x_grid[0][0] if len(x_grid[0]) > 1 else 1.0
            dx2 = x_grid[1][1] - x_grid[1][0] if len(x_grid[1]) > 1 else 1.0
            total_2d = np.sum(prob_2d) * dx1 * dx2
            
            # Recover marginals using corrected integration
            marginal_1_recovered = np.trapz(prob_2d, x=x_grid[1], axis=0)
            marginal_2_recovered = np.trapz(prob_2d, x=x_grid[0], axis=1)
            
            # Check marginal normalization  
            norm_1 = np.trapz(marginal_1_recovered, x=x_grid[0])
            norm_2 = np.trapz(marginal_2_recovered, x=x_grid[1])
            
            # Get true marginals
            dim1_tuple, dim2_tuple = pair
            redshift_bin_1, ang_bin_1 = dim1_tuple
            redshift_bin_2, ang_bin_2 = dim2_tuple
            
            x1_true = likelihood._xs[redshift_bin_1, ang_bin_1]
            pdf1_true = likelihood._pdfs[redshift_bin_1, ang_bin_1]
            x2_true = likelihood._xs[redshift_bin_2, ang_bin_2]
            pdf2_true = likelihood._pdfs[redshift_bin_2, ang_bin_2]
            
            # Compute KL divergences
            pdf1_safe = np.maximum(pdf1_true, 1e-15)
            marg1_safe = np.maximum(marginal_1_recovered, 1e-15)
            kl_div_1 = np.sum(pdf1_safe * np.log(pdf1_safe / marg1_safe)) * (x1_true[1] - x1_true[0])
            
            pdf2_safe = np.maximum(pdf2_true, 1e-15)
            marg2_safe = np.maximum(marginal_2_recovered, 1e-15)
            kl_div_2 = np.sum(pdf2_safe * np.log(pdf2_safe / marg2_safe)) * (x2_true[1] - x2_true[0])
            
            mean_kl_div = (kl_div_1 + kl_div_2) / 2
            
            # PLOT ORIGINAL AND RECOVERED MARGINALS FOR VISUAL COMPARISON
            try:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot marginal 1 comparison
                ax1.plot(x1_true, pdf1_true, 'b-', linewidth=2, label='True Marginal 1', alpha=0.8)
                ax1.plot(x_grid[0], marginal_1_recovered, 'r--', linewidth=2, label='Recovered Marginal 1', alpha=0.8)
                ax1.set_xlabel('Value')
                ax1.set_ylabel('Probability Density')
                ax1.set_title(f'Marginal 1: {test_label}\nPair {pair}, KL={kl_div_1:.4f}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot marginal 2 comparison
                ax2.plot(x2_true, pdf2_true, 'b-', linewidth=2, label='True Marginal 2', alpha=0.8)
                ax2.plot(x_grid[1], marginal_2_recovered, 'r--', linewidth=2, label='Recovered Marginal 2', alpha=0.8)
                ax2.set_xlabel('Value')
                ax2.set_ylabel('Probability Density')
                ax2.set_title(f'Marginal 2: {test_label}\nPair {pair}, KL={kl_div_2:.4f}')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot with descriptive filename
                df_str = f"_df{df}" if coupling_type == 'student_t' else ""
                pair_str = f"{pair[0][0]}_{pair[0][1]}_{pair[1][0]}_{pair[1][1]}"
                filename = f"marginal_comparison_{coupling_type}{df_str}_pair_{pair_str}_pinv.png"
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"    Saved marginal comparison plot: {filename}")
                
            except Exception as plot_e:
                logger.warning(f"Failed to create marginal comparison plot: {plot_e}")
            
            # Success criteria
            pair_success = (mean_kl_div < 0.1 and 
                           abs(norm_1 - 1.0) < 0.01 and 
                           abs(norm_2 - 1.0) < 0.01 and
                           abs(total_2d - 1.0) < 0.01)
            
            pair_results.append({
                'pair': pair,
                'coupling_type': coupling_type,
                'df': df if coupling_type == 'student_t' else None,
                'kl_divergence': mean_kl_div,
                'marginal_1_kl': kl_div_1,
                'marginal_2_kl': kl_div_2,
                'norm_2d': total_2d,
                'norm_1': norm_1,
                'norm_2': norm_2,
                'success': pair_success,
                'x_grids': x_grid,
                'marginal_1_recovered': marginal_1_recovered,
                'marginal_2_recovered': marginal_2_recovered,
                'x1_true': x1_true,
                'pdf1_true': pdf1_true,
                'x2_true': x2_true,
                'pdf2_true': pdf2_true
            })
            
            status = "✓" if pair_success else "✗"
            logger.info(f"  {test_label} Pair {pair}: {status} KL={mean_kl_div:.4f}, 2D_norm={total_2d:.3f}, marg_norms=({norm_1:.3f}, {norm_2:.3f})")
            
        except Exception as e:
            logger.warning(f"Failed to process pair {pair} for {test_label}: {e}")
            pair_results.append({
                'pair': pair,
                'coupling_type': coupling_type,
                'df': df if coupling_type == 'student_t' else None,
                'success': False,
                'error': str(e)
            })
    
    return pair_results

def main(exact_marginals=True, max_pairs=6):
    """Main test function comparing Gaussian vs Student-t coupling."""
    logger.info("="*80)
    logger.info("STARTING GAUSSIAN vs STUDENT-T COUPLING COMPARISON TEST")
    logger.info("="*80)
    
    # Create likelihood setup
    likelihood = create_realistic_likelihood_setup()
    
    # Test both coupling types for comparison
    results = test_2d_likelihood_marginal_consistency(
        likelihood=likelihood,
        coupling_types=['gaussian', 'student_t'],  
        exact_marginals=exact_marginals,
        max_pairs=max_pairs
    )
    
    # Print summary
    logger.info("="*80)
    logger.info("TEST SUMMARY")
    logger.info("="*80)
    
    for test_key, result in results.items():
        if isinstance(result, list):  # pair_results list
            valid_pairs = [r for r in result if r.get('success', False)]
            total_pairs = len(result)
            success_rate = len(valid_pairs) / total_pairs if total_pairs > 0 else 0
            
            if len(valid_pairs) > 0:
                mean_kl = np.mean([r['kl_divergence'] for r in valid_pairs])
                mean_2d_norm = np.mean([r['norm_2d'] for r in valid_pairs])
                status = "✓ PASS" if success_rate >= 0.7 else "✗ FAIL"
                logger.info(f"{test_key}: {status} - {len(valid_pairs)}/{total_pairs} pairs ({success_rate:.1%}), "
                           f"KL={mean_kl:.4f}, 2D_norm={mean_2d_norm:.3f}")
            else:
                logger.info(f"{test_key}: ✗ FAIL - No successful pairs")
    
    return results

if __name__ == "__main__":
    # Run the test with limited pairs for speed
    results = main(exact_marginals=True, max_pairs=6)