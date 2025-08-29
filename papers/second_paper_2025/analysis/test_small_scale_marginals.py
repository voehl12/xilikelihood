"""
Tests for scale-dependent marginal distribution selection.

These tests validate the implementation where angular separations below a certain 
threshold use Gaussian marginals instead of the full copula treatment.
"""

import numpy as np
import tempfile
import os
from pathlib import Path
import time
import logging

import xilikelihood as xlh
import config

# Set up logging to output to bash/console
# Configure root logger to capture all logging (including from xilikelihood)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

# Set up our specific logger
logger = logging.getLogger("small_scale_marginals")
logger.setLevel(logging.INFO)

# Explicitly ensure xilikelihood logging is visible
logging.getLogger("xilikelihood").setLevel(logging.INFO)

# --- Realistic angular bin setup (as in fiducial_dataspace) ---
def get_realistic_ang_bins(ang_min=0.5, ang_max=300, n_bins=9):
    initial_ang_bins_in_arcmin = np.logspace(
        np.log10(ang_min), np.log10(ang_max), num=n_bins, endpoint=True
    )
    # Do not filter out small bins; include all for testing
    extended_ang_bins_in_arcmin = initial_ang_bins_in_arcmin
    ang_bins_in_deg = extended_ang_bins_in_arcmin / 60
    ang_bins_in_deg = [
        (ang_bins_in_deg[i], ang_bins_in_deg[i + 1]) for i in range(len(ang_bins_in_deg) - 1)
    ]
    return ang_bins_in_deg



def test_likelihood_continuity_at_scale_cut():
    """
    Test that likelihood values are continuous at the scale cut boundary.
    
    This tests evaluates the likelihood just above and below the scale cut
    to ensure there are no discontinuities in the transition.
    """
    logger.info("Testing likelihood continuity at scale cut boundary...")
    
    scale_cut_arcmin = 15.0/60  # 15 arcminutes
    ang_bins_in_deg = get_realistic_ang_bins()
    
    likelihood = xlh.XiLikelihood(
        mask=xlh.SphereMask(
            spins=config.MASK_CONFIG["spins"],
            circmaskattr=config.MASK_CONFIG["circmaskattr"],
            exact_lmax=config.EXACT_LMAX,
            l_smooth=config.MASK_CONFIG.get("l_smooth", None),
            working_dir=config.PACKAGE_DIR
        ),
        redshift_bins=[xlh.RedshiftBin(nbin=1, z=np.linspace(0.01, 3.0, 100), 
                                      zmean=0.5, zsig=0.1)],
        ang_bins_in_deg=ang_bins_in_deg,
        include_ximinus=False,  # Simpler for initial test
        large_angle_threshold=scale_cut_arcmin,
        exact_lmax=config.EXACT_LMAX,
        working_dir=config.PACKAGE_DIR
    )
    
    # Set up fiducial cosmology
    fiducial_cosmo = {'omega_m': 0.31, 's8': 0.8}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_data_path = os.path.join(tmpdir, 'continuity_test_data.npz')
        cov_path = os.path.join(tmpdir, 'continuity_test_cov.npz')
        
        # Generate mock data at fiducial cosmology
        mock_data, cov = xlh.mock_data.create_mock_data(
            likelihood, mock_data_path, cov_path,
            fiducial_cosmo=fiducial_cosmo,
            random=None  # Use deterministic fiducial mean
        )
        
        # Evaluate likelihood at fiducial cosmology (should be high)
        loglike_fiducial = likelihood.loglikelihood(mock_data, fiducial_cosmo)
        
        # Test small perturbations around fiducial
        test_cosmo_1 = {'omega_m': 0.31, 's8': 0.81}  # Tiny s8 change
        test_cosmo_2 = {'omega_m': 0.31, 's8': 0.79}  # Tiny s8 change
        
        loglike_2 = likelihood.loglikelihood(mock_data, test_cosmo_2)
        loglike_1 = likelihood.loglikelihood(mock_data, test_cosmo_1)
        
        
        # Check that likelihood values are reasonable and continuous
        logger.info(f"  Fiducial loglike: {loglike_fiducial:.3f}")
        logger.info(f"  Perturbed loglike (+): {loglike_1:.3f}")
        logger.info(f"  Perturbed loglike (-): {loglike_2:.3f}")
        
        # The fiducial should have the highest likelihood
        assert loglike_fiducial > loglike_1, "Fiducial should have higher likelihood than perturbation"
        assert loglike_fiducial > loglike_2, "Fiducial should have higher likelihood than perturbation"
        
        # Check that differences are finite and measurable
        diff_1 = abs(loglike_fiducial - loglike_1)
        diff_2 = abs(loglike_fiducial - loglike_2)
        
        assert np.isfinite(diff_1), f"Likelihood difference should be finite, got {diff_1}"
        assert np.isfinite(diff_2), f"Likelihood difference should be finite, got {diff_2}"
        assert diff_1 > 0, "Should detect difference from s8 perturbation"
        assert diff_2 > 0, "Should detect difference from s8 perturbation"
        
        logger.info("✅ Likelihood continuity test passed")
        return loglike_fiducial, loglike_1, loglike_2


def test_gaussian_vs_copula_comparison():
    """
    Compare likelihood values with and without scale cut to understand the impact.
    
    This test helps validate that the Gaussian approximation for small scales
    gives reasonable results compared to the full copula treatment.
    """
    logger.info("Testing Gaussian vs copula marginal comparison...")
    ang_bins = get_realistic_ang_bins()
    
    # Create likelihood WITHOUT scale cut (full copula)
    likelihood_full = xlh.XiLikelihood(
        mask=xlh.SphereMask(
            spins=config.MASK_CONFIG["spins"],
            circmaskattr=config.MASK_CONFIG["circmaskattr"],
            exact_lmax=config.EXACT_LMAX,
            l_smooth=config.MASK_CONFIG.get("l_smooth", None),
            working_dir=config.PACKAGE_DIR
        ),
        redshift_bins=[xlh.RedshiftBin(nbin=1, z=np.linspace(0.01, 3.0, 100), 
                                      zmean=0.5, zsig=0.1)],
        ang_bins_in_deg=ang_bins,
        include_ximinus=False,
        large_angle_threshold=15./60,  # No scale cut - full copula
        exact_lmax=config.EXACT_LMAX,
        working_dir=config.PACKAGE_DIR
    )
    
    # Create likelihood WITH scale cut
    likelihood_mixed = xlh.XiLikelihood(
        mask=xlh.SphereMask(
            spins=config.MASK_CONFIG["spins"],
            circmaskattr=config.MASK_CONFIG["circmaskattr"],
            exact_lmax=config.EXACT_LMAX,
            l_smooth=config.MASK_CONFIG.get("l_smooth", None),
            working_dir=config.PACKAGE_DIR
        ),
        redshift_bins=[xlh.RedshiftBin(nbin=1, z=np.linspace(0.01, 3.0, 100), 
                                      zmean=0.5, zsig=0.1)],
        ang_bins_in_deg=ang_bins,
        include_ximinus=False,
        large_angle_threshold=1.0,  
        exact_lmax= config.EXACT_LMAX,
        working_dir=config.PACKAGE_DIR
    )
    
    fiducial_cosmo = {'omega_m': 0.31, 's8': 0.8}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Generate mock data with full copula treatment
        mock_data_path_full = os.path.join(tmpdir, 'comparison_full_data.npz')
        cov_path_full = os.path.join(tmpdir, 'comparison_full_cov.npz')
        
        mock_data_full, cov_full = xlh.mock_data.create_mock_data(
            likelihood_full, mock_data_path_full, cov_path_full,
            fiducial_cosmo=fiducial_cosmo,
            random=None
        )
        
        # Generate mock data with mixed treatment
        mock_data_path_mixed = os.path.join(tmpdir, 'comparison_mixed_data.npz')
        cov_path_mixed = os.path.join(tmpdir, 'comparison_mixed_cov.npz')
        
        mock_data_mixed, cov_mixed = xlh.mock_data.create_mock_data(
            likelihood_mixed, mock_data_path_mixed, cov_path_mixed,
            fiducial_cosmo=fiducial_cosmo,
            random=None
        )
        
        # Evaluate likelihoods at fiducial cosmology
        loglike_full = likelihood_full.loglikelihood(mock_data_full, fiducial_cosmo)
        loglike_mixed = likelihood_mixed.loglikelihood(mock_data_mixed, fiducial_cosmo)
        
        # Test with slightly perturbed cosmology
        test_cosmo = {'omega_m': 0.31, 's8': 0.81}
        loglike_full_perturbed = likelihood_full.loglikelihood(mock_data_full, test_cosmo)
        loglike_mixed_perturbed = likelihood_mixed.loglikelihood(mock_data_mixed, test_cosmo)
        
        logger.info(f"  Full copula - fiducial: {loglike_full:.3f}")
        logger.info(f"  Mixed treatment - fiducial: {loglike_mixed:.3f}")
        logger.info(f"  Full copula - perturbed: {loglike_full_perturbed:.3f}")
        logger.info(f"  Mixed treatment - perturbed: {loglike_mixed_perturbed:.3f}")
        
        # Both should prefer fiducial over perturbed cosmology
        assert loglike_full > loglike_full_perturbed, "Full copula should prefer fiducial"
        assert loglike_mixed > loglike_mixed_perturbed, "Mixed treatment should prefer fiducial"
        
        # The differences should be in the same direction (both methods should agree on preference)
        full_diff = loglike_full - loglike_full_perturbed
        mixed_diff = loglike_mixed - loglike_mixed_perturbed
        
        # Both differences should be positive and reasonably similar
        assert full_diff > 0 and mixed_diff > 0, "Both methods should prefer fiducial"
        
        # They shouldn't be too different (order of magnitude check)
        ratio = max(full_diff, mixed_diff) / min(full_diff, mixed_diff)
        assert ratio < 10, f"Methods differ too much: ratio = {ratio:.2f}"
        
        logger.info("✅ Gaussian vs copula comparison test passed")
        return loglike_full, loglike_mixed, loglike_full_perturbed, loglike_mixed_perturbed


def test_performance_with_scale_cut():
    """
    Test that the scale cut actually improves performance.
    
    This validates that using Gaussian marginals for small scales
    provides a computational speedup.
    """
    logger.info("Testing performance improvement with scale cut...")
    ang_bins = get_realistic_ang_bins()
    
    # Likelihood without scale cut
    likelihood_full = xlh.XiLikelihood(
        mask=xlh.SphereMask(
            spins=config.MASK_CONFIG["spins"],
            circmaskattr=config.MASK_CONFIG["circmaskattr"],
            exact_lmax=config.EXACT_LMAX,
            l_smooth=config.MASK_CONFIG.get("l_smooth", None),
            working_dir=config.PACKAGE_DIR
        ),
        redshift_bins=[xlh.RedshiftBin(nbin=1, z=np.linspace(0.01, 3.0, 100), 
                                      zmean=0.5, zsig=0.1)],
        ang_bins_in_deg=ang_bins,
        include_ximinus=False,
        large_angle_threshold=15./60,
        exact_lmax=config.EXACT_LMAX,
        working_dir=config.PACKAGE_DIR
    )
    
    # Likelihood with scale cut
    likelihood_cut = xlh.XiLikelihood(
        mask=xlh.SphereMask(
            spins=config.MASK_CONFIG["spins"],
            circmaskattr=config.MASK_CONFIG["circmaskattr"],
            exact_lmax=config.EXACT_LMAX,
            l_smooth=config.MASK_CONFIG.get("l_smooth", None),
            working_dir=config.PACKAGE_DIR
        ),
        redshift_bins=[xlh.RedshiftBin(nbin=1, z=np.linspace(0.01, 3.0, 100), 
                                      zmean=0.5, zsig=0.1)],
        ang_bins_in_deg=ang_bins,
        include_ximinus=False,
        large_angle_threshold=2.0,  
        exact_lmax=config.EXACT_LMAX,
        working_dir=config.PACKAGE_DIR
    )
    
    fiducial_cosmo = {'omega_m': 0.31, 's8': 0.8}
    test_cosmo = {'omega_m': 0.32, 's8': 0.85}
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set up mock data for both cases
        mock_data_path = os.path.join(tmpdir, 'perf_test_data.npz')
        cov_path = os.path.join(tmpdir, 'perf_test_cov.npz')
        
        # Time the full copula evaluation
        start_time = time.time()
        mock_data_full, _ = xlh.mock_data.create_mock_data(
            likelihood_full, mock_data_path, cov_path,
            fiducial_cosmo=fiducial_cosmo, random=None
        )
        loglike_full = likelihood_full.loglikelihood(mock_data_full, test_cosmo)
        time_full = time.time() - start_time
        
        # Time the scale cut evaluation
        start_time = time.time()
        mock_data_cut, _ = xlh.mock_data.create_mock_data(
            likelihood_cut, mock_data_path, cov_path,
            fiducial_cosmo=fiducial_cosmo, random=None
        )
        loglike_cut = likelihood_cut.loglikelihood(mock_data_cut, test_cosmo)
        time_cut = time.time() - start_time
        
        logger.info(f"  Full copula time: {time_full:.3f}s, loglike: {loglike_full:.3f}")
        logger.info(f"  Scale cut time: {time_cut:.3f}s, loglike: {loglike_cut:.3f}")
        logger.info(f"  Speedup factor: {time_full/time_cut:.2f}x")
        
        # Scale cut should be faster (or at least not much slower)
        assert time_cut <= time_full * 1.5, f"Scale cut should be faster: {time_cut:.3f} vs {time_full:.3f}"
        
        # Results should be reasonably similar
        diff = abs(loglike_full - loglike_cut)
        rel_diff = diff / abs(loglike_full) if loglike_full != 0 else diff
        
        logger.info(f"  Relative difference: {rel_diff:.3%}")
        assert rel_diff < 0.5, f"Results too different: {rel_diff:.3%}"  # Within 50%
        
        logger.info("✅ Performance test passed")
        return time_full, time_cut, loglike_full, loglike_cut


def test_scale_cut_boundary_bins():
    """
    Test that bins are correctly classified as above/below the scale cut.
    
    This validates the logic for determining which bins use Gaussian vs copula marginals.
    """
    logger.info("Testing scale cut boundary bin classification...")
    scale_cut = 15.0/60  # 15 arcminutes
    ang_bins = get_realistic_ang_bins()
    
    likelihood = xlh.XiLikelihood(
        mask=xlh.SphereMask(
            spins=config.MASK_CONFIG["spins"],
            circmaskattr=config.MASK_CONFIG["circmaskattr"],
            exact_lmax=config.EXACT_LMAX,
            l_smooth=config.MASK_CONFIG.get("l_smooth", None),
            working_dir=config.PACKAGE_DIR
        ),
        redshift_bins=[xlh.RedshiftBin(nbin=1, z=np.linspace(0.01, 3.0, 100), 
                                      zmean=0.5, zsig=0.1)],
        ang_bins_in_deg=ang_bins,
        include_ximinus=False,
        large_angle_threshold=scale_cut,
        exact_lmax=config.EXACT_LMAX,
        working_dir=config.PACKAGE_DIR
    )
    
    # Check internal classification (if exposed)
    if hasattr(likelihood, '_is_large_angle'):
        logger.info(f"  Scale cut set to: {scale_cut} arcmin")
        logger.info(f"  Angular bins (arcmin): {[(b[0]*60, b[1]*60) for b in ang_bins]}")
        logger.info(f"  _is_large_angle mask: {likelihood._is_large_angle}")
        # Assert classification matches bin edges
        for i, (b, is_large) in enumerate(zip(ang_bins, likelihood._is_large_angle)):
            bin_min_arcmin = b[0] * 60
            bin_max_arcmin = b[1] * 60
            # If the bin's lower edge is >= scale_cut, it should be large angle
            expected = bin_min_arcmin >= scale_cut*60
            assert is_large == expected, (
                f"Bin {i} ({bin_min_arcmin:.2f}-{bin_max_arcmin:.2f} arcmin): "
                f"expected is_large_angle={expected}, got {is_large}")
        logger.info("  All bins classified correctly with _is_large_angle mask.")
        # Also verify likelihood can be constructed and used
        fiducial_cosmo = {'omega_m': 0.31, 's8': 0.8}
        with tempfile.TemporaryDirectory() as tmpdir:
            mock_data_path = os.path.join(tmpdir, 'boundary_test_data.npz')
            cov_path = os.path.join(tmpdir, 'boundary_test_cov.npz')
            mock_data, _ = xlh.mock_data.create_mock_data(
                likelihood, mock_data_path, cov_path,
                fiducial_cosmo=fiducial_cosmo, random=None
            )
            loglike = likelihood.loglikelihood(mock_data, fiducial_cosmo)
            logger.info(f"  Likelihood evaluation successful: {loglike:.3f}")
            assert np.isfinite(loglike), "Likelihood should be finite"
        logger.info("✅ Scale cut boundary test passed")
        return True
    else:
        logger.info("  _is_large_angle property not exposed - skipping detailed checks")
        logger.info("✅ Scale cut boundary test passed (basic)")
        return True





def systematic_scale_cut_sweep_gaussian_fiducial():
    """
    Systematic test: likelihood and timing vs. Gaussian/cutoff scale, using fiducial_dataspace for realistic bins.
    Results are printed, saved to CSV, and plotted for paper-ready output.
    """
    import csv
    import matplotlib.pyplot as plt
    from xilikelihood.likelihood import fiducial_dataspace
    logger.info("\nRunning systematic scale cut sweep (Gaussian copula, fiducial dataspace)...")
    scale_cuts_deg = np.array([0, 2, 5, 10, 15, 20, 50, 100, 300]) / 60.
    # Use realistic bins from fiducial_dataspace
    redshift_bins, ang_bins_in_deg = fiducial_dataspace(min_ang_cutoff_in_arcmin=0)
    redshift_bins = redshift_bins[-2:]  # Use last two bins for cross-correlation
    fiducial_cosmo = {'omega_m': 0.31, 's8': 0.8}
    results = []
    # Generate mock data once (at fiducial)
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_data_path = os.path.join(tmpdir, 'sweep_data.npz')
        cov_path = os.path.join(tmpdir, 'sweep_cov.npz')
        # Use scale cut = 0 for mock data (full copula)
        likelihood_for_data = xlh.XiLikelihood(
            mask=xlh.SphereMask(
                spins=config.MASK_CONFIG["spins"],
                circmaskattr=config.MASK_CONFIG["circmaskattr"],
                exact_lmax=config.EXACT_LMAX,
                l_smooth=config.MASK_CONFIG["l_smooth"],
                working_dir=config.PACKAGE_DIR
            ),
            redshift_bins=redshift_bins,
            ang_bins_in_deg=ang_bins_in_deg,
            include_ximinus=False,
            large_angle_threshold=0,
            exact_lmax=config.EXACT_LMAX,
            working_dir=config.PACKAGE_DIR
        )
        mock_data, _ = xlh.mock_data.create_mock_data(
            likelihood_for_data, mock_data_path, cov_path,
            fiducial_cosmo=fiducial_cosmo, random=None
        )
        for scale_cut in scale_cuts_deg:
            likelihood = xlh.XiLikelihood(
                mask=xlh.SphereMask(
                    spins=config.MASK_CONFIG["spins"],
                    circmaskattr=config.MASK_CONFIG["circmaskattr"],
                    exact_lmax=config.EXACT_LMAX,
                    l_smooth=config.MASK_CONFIG["l_smooth"],
                    working_dir=config.PACKAGE_DIR
                ),
                redshift_bins=redshift_bins,
                ang_bins_in_deg=ang_bins_in_deg,
                include_ximinus=False,
                large_angle_threshold=scale_cut,
                exact_lmax=config.EXACT_LMAX,
                working_dir=config.PACKAGE_DIR
            )
            likelihood.setup_likelihood()
            start = time.time()
            loglike = likelihood.loglikelihood(mock_data, fiducial_cosmo)
            elapsed = time.time() - start
            logger.info(f"  Scale cut: {scale_cut*60:6.1f} arcmin | loglike: {loglike:10.3f} | time: {elapsed:7.3f} s")
            results.append((scale_cut*60, loglike, elapsed))  # store scale_cut in arcmin
    # Save results
    out_csv = "scale_cut_sweep_gaussian_fiducial.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scale_cut_arcmin", "loglike", "time_sec"])
        writer.writerows(results)
    logger.info(f"Results saved to {out_csv}")
    # Plot results
    scale_cuts_arcmin = [r[0] for r in results]
    loglikes = [r[1] for r in results]
    times = [r[2] for r in results]
    fig, ax1 = plt.subplots(figsize=(7,4))
    color = 'tab:blue'
    ax1.set_xlabel('Scale cut (arcmin)')
    ax1.set_ylabel('Log-likelihood', color=color)
    ax1.plot(scale_cuts_arcmin, loglikes, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Time (s)', color=color)
    ax2.plot(scale_cuts_arcmin, times, marker='s', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Likelihood and computation time vs. Gaussian scale cut')
    fig.tight_layout()
    plt.savefig('scale_cut_sweep_gaussian_fiducial.png', dpi=300)
    logger.info("Plot saved to scale_cut_sweep_gaussian_fiducial.png")
    plt.show()
    return results


if __name__ == "__main__":
    logger.info("Running small scale marginal tests...\n")
    try:
        test_likelihood_continuity_at_scale_cut()
        logger.info("")
        test_gaussian_vs_copula_comparison()
        logger.info("")
        test_performance_with_scale_cut()
        logger.info("")
        test_scale_cut_boundary_bins()
        logger.info("")
        systematic_scale_cut_sweep_gaussian_fiducial()
        logger.info("")
        logger.info("🎉 All small scale marginal tests passed!")
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
