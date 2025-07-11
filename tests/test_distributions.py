import numpy as np
import pytest


def test_cf2pdf():
    import scipy.stats as stats
    from xilikelihood.distributions import gaussian_cf, cf_to_pdf_1d
    

    mu = 0
    sigma = 1
    val_max = mu + 10 * sigma
    dt = 0.45 * 2 * np.pi / val_max
    steps = 2048
    t0 = -0.5 * dt * (steps - 1)
    t = np.linspace(t0, -t0, steps - 1)
    cf = gaussian_cf(t, mu, sigma)
    x, pdf_from_cf = cf_to_pdf_1d(t, cf)
    pdf = stats.norm.pdf(x, mu, sigma)
    assert np.allclose(pdf, pdf_from_cf), pdf_from_cf


def test_cov_xi_gaussian_nD_snapshot_xiplus(covariance_test_setup, snapshot, regtest):
    """Snapshot test for cov_xi_gaussian_nD with include_ximinus=False."""
    import xilikelihood as xlh
    from xilikelihood.distributions import cov_xi_gaussian_nD
    
    # Get test setup
    setup = covariance_test_setup
    mask = setup['mask']
    redshift_bins = setup['redshift_bins'] 
    angular_bins_in_deg = setup['angular_bins_in_deg']
    cosmo_params = setup['cosmo_params']
    theory_lmax = setup['theory_lmax']
    
    # Prepare theory inputs
    numerical_combinations, redshift_bin_combinations, is_cov_cross, shot_noise, mapper = xlh.prepare_theory_cl_inputs(redshift_bins)
    
    # Generate theory power spectra
    theory_cls = xlh.generate_theory_cl(
        theory_lmax,
        redshift_bin_combinations,
        shot_noise,
        cosmo=cosmo_params
    )
    
    # Compute covariance matrix for xi_plus only
    cov_matrix = cov_xi_gaussian_nD(
        theory_cls,
        numerical_combinations, 
        angular_bins_in_deg,
        mask.eff_area,
        lmin=0,
        lmax=theory_lmax,
        include_ximinus=False
    )
    
    # Write key properties to regtest
    regtest.write(f"=== XIPLUS TEST ===\n")
    regtest.write(f"Shape: {cov_matrix.shape}\n")
    regtest.write(f"Is symmetric: {np.allclose(cov_matrix, cov_matrix.T)}\n")
    regtest.write(f"Is positive semidefinite: {np.all(np.linalg.eigvals(cov_matrix) >= -1e-10)}\n")
    regtest.write(f"Condition number: {np.linalg.cond(cov_matrix):.6e}\n")
    regtest.write(f"Min eigenvalue: {np.min(np.linalg.eigvals(cov_matrix)):.6e}\n")
    regtest.write(f"Max diagonal: {np.max(np.diag(cov_matrix)):.6e}\n")
    regtest.write(f"Min diagonal: {np.min(np.diag(cov_matrix)):.6e}\n")
    regtest.write("\n")
    
    # Snapshot test for the entire covariance matrix
    snapshot.check(cov_matrix)


def test_cov_xi_gaussian_nD_snapshot_both(covariance_test_setup, snapshot, regtest):
    """Snapshot test for cov_xi_gaussian_nD with include_ximinus=True."""
    import xilikelihood as xlh
    from xilikelihood.distributions import cov_xi_gaussian_nD
    
    # Get test setup
    setup = covariance_test_setup
    mask = setup['mask']
    redshift_bins = setup['redshift_bins'] 
    angular_bins_in_deg = setup['angular_bins_in_deg']
    cosmo_params = setup['cosmo_params']
    theory_lmax = setup['theory_lmax']
    
    # Prepare theory inputs
    numerical_combinations, redshift_bin_combinations, is_cov_cross, shot_noise, mapper = xlh.prepare_theory_cl_inputs(redshift_bins)
    
    # Generate theory power spectra
    theory_cls = xlh.generate_theory_cl(
        theory_lmax,
        redshift_bin_combinations,
        shot_noise,
        cosmo=cosmo_params
    )
    
    # Compute covariance matrix for xi_plus and xi_minus
    cov_matrix = cov_xi_gaussian_nD(
        theory_cls,
        numerical_combinations, 
        angular_bins_in_deg,
        mask.eff_area,
        lmin=0,
        lmax=theory_lmax,
        include_ximinus=True
    )
    
    # Write key properties to regtest
    regtest.write(f"=== XIMINUS TEST ===\n")
    regtest.write(f"Shape: {cov_matrix.shape}\n")
    regtest.write(f"Is symmetric: {np.allclose(cov_matrix, cov_matrix.T)}\n")
    regtest.write(f"Is positive semidefinite: {np.all(np.linalg.eigvals(cov_matrix) >= -1e-10)}\n")
    regtest.write(f"Condition number: {np.linalg.cond(cov_matrix):.6e}\n")
    regtest.write(f"Min eigenvalue: {np.min(np.linalg.eigvals(cov_matrix)):.6e}\n")
    regtest.write(f"Max diagonal: {np.max(np.diag(cov_matrix)):.6e}\n")
    regtest.write(f"Min diagonal: {np.min(np.diag(cov_matrix)):.6e}\n")
    regtest.write("\n")
    
    # Snapshot test for the entire covariance matrix
    snapshot.check(cov_matrix)


def test_cov_xi_gaussian_nD_properties(covariance_test_setup):
    """Test mathematical properties of the covariance matrix."""
    import xilikelihood as xlh
    from xilikelihood.distributions import cov_xi_gaussian_nD
    
    # Get test setup
    setup = covariance_test_setup
    mask = setup['mask']
    redshift_bins = setup['redshift_bins']
    angular_bins_in_deg = setup['angular_bins_in_deg']
    cosmo_params = setup['cosmo_params'] 
    theory_lmax = setup['theory_lmax']
    
    # Prepare inputs
    numerical_combinations, redshift_bin_combinations, is_cov_cross, shot_noise, mapper = xlh.prepare_theory_cl_inputs(redshift_bins)
    theory_cls = xlh.generate_theory_cl(theory_lmax, redshift_bin_combinations, shot_noise, cosmo=cosmo_params)
    
    # Test both cases
    for include_ximinus in [False, True]:
        cov_matrix = cov_xi_gaussian_nD(
            theory_cls,
            numerical_combinations,
            angular_bins_in_deg, 
            mask.eff_area,
            lmin=0,
            lmax=theory_lmax,
            include_ximinus=include_ximinus
        )
        
        # Test mathematical properties
        assert cov_matrix.ndim == 2, "Covariance matrix should be 2D"
        assert cov_matrix.shape[0] == cov_matrix.shape[1], "Covariance matrix should be square"
        assert np.allclose(cov_matrix, cov_matrix.T, rtol=1e-14), "Covariance matrix should be symmetric"
        assert np.all(np.linalg.eigvals(cov_matrix) >= -1e-10), "Covariance matrix should be positive semidefinite"
        assert np.all(np.diag(cov_matrix) > 0), "Diagonal elements should be positive (variances)"
        
        # Test expected size relationships
        n_redshift_combs = len(redshift_bin_combinations)
        n_angbins = len(angular_bins_in_deg)
        if include_ximinus:
            expected_size = n_redshift_combs * 2 * n_angbins
        else:
            expected_size = n_redshift_combs * n_angbins
        
        assert cov_matrix.shape == (expected_size, expected_size), \
            f"Expected shape {(expected_size, expected_size)}, got {cov_matrix.shape}"