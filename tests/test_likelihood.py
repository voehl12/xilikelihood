import os
# Force JAX to use CPU for testing - must be before any imports that use JAX
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

# Also try to set JAX config directly if it's already imported
try:
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    jax.config.update('jax_platforms', 'cpu')
except:
    pass

import pytest
import numpy as np

import xilikelihood as xlh


@pytest.fixture
def minimal_redshift_bins():
    """Create minimal redshift bins for testing."""
    z = np.linspace(0.01, 3.0, 50)
    return [
        xlh.RedshiftBin(nbin=1, z=z, zmean=0.5, zsig=0.1),
    ]

@pytest.fixture
def multiple_redshift_bins():
    """Create multiple redshift bins for testing."""
    z = np.linspace(0.01, 3.0, 100)
    return [
        xlh.RedshiftBin(nbin=1, z=z, zmean=0.5, zsig=0.1),
        xlh.RedshiftBin(nbin=2, z=z, zmean=1.0, zsig=0.1)
    ]

@pytest.fixture
def small_mask():
    """Create a small mask for testing."""
    return xlh.SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=30, l_smooth=30)

@pytest.fixture
def medium_mask():
    """Create a medium mask for testing."""
    return xlh.SphereMask(spins=[2], circmaskattr=(1000, 256), l_smooth=30, exact_lmax=30)

@pytest.fixture
def angular_bins():
    """Angular bins for testing."""
    return [(1.0, 2.0)]

@pytest.fixture
def multiple_angular_bins():
    """Multiple angular bins for testing."""
    return [(1.0, 2.0), (2.0, 4.0)]

@pytest.fixture
def likelihood_instance():
    # Set up a minimal instance of XiLikelihood
    mask = xlh.SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=30, l_smooth=30)

    # Create a simple redshift bin instead of using file
    z = np.linspace(0.01, 3.0, 100)
    redshift_bin = xlh.RedshiftBin(nbin=1, z=z, zmean=0.5, zsig=0.1)

    ang_bins_in_deg = [(0.5, 1.0), (4, 6)]

    likelihood = xlh.XiLikelihood(
        mask=mask,
        redshift_bins=[redshift_bin],
        ang_bins_in_deg=ang_bins_in_deg,
        noise='default',
    )
    likelihood.setup_likelihood()
    likelihood.initiate_theory_cl({'omega_m': 0.31, 's8': 0.8})
    likelihood._prepare_matrix_products()
    return likelihood

def test_get_cfs_1d_lowell(likelihood_instance, snapshot):
    # Ensure JAX uses CPU
    import os
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    os.environ["JAX_PLATFORMS"] = "cpu"
    
    # Call the method
    likelihood_instance.get_covariance_matrix_lowell()
    likelihood_instance._get_cfs_1d_lowell()

    # Capture outputs
    variances = np.array(likelihood_instance._variances_lowell.real)
    eigvals = np.array(likelihood_instance._eigvals.real)
    t_lowell = np.array(likelihood_instance._t_lowell.real)
    cfs_lowell = np.array(likelihood_instance._cfs_lowell.real)
    ximax = np.array(likelihood_instance._ximax.real)
    ximin = np.array(likelihood_instance._ximin.real)
    
    # Write outputs to the regression test snapshot
    snapshot.check(variances, rtol=1e-10)
    snapshot.check(eigvals, rtol=1e-8)
    snapshot.check(t_lowell, rtol=1e-10)
    snapshot.check(cfs_lowell, rtol=1e-10)
    snapshot.check(ximax, rtol=1e-10)
    snapshot.check(ximin, rtol=1e-10)

def test_initiate_theory_cl(likelihood_instance, snapshot, regtest):
    # Define cosmological parameters
    cosmo_params = {'omega_m': 0.31, 's8': 0.8}

    # The likelihood_instance fixture already calls initiate_theory_cl, so let's test it
    # Call the method again to test it
    likelihood_instance.initiate_theory_cl(cosmo_params)

    # Capture outputs
    theory_cl = likelihood_instance._theory_cl

    # Assert that theory_cl is a list of objects with expected attributes
    assert isinstance(theory_cl, list), "theory_cl should be a list"
    assert len(theory_cl) > 0, "theory_cl should not be empty"
    for cl in theory_cl:
        assert hasattr(cl, 'lmax'), "Each theory_cl object should have an 'lmax' attribute"
        assert hasattr(cl, 'cosmo'), "Each theory_cl object should have a 'cosmo' attribute"
        assert cl.cosmo == cosmo_params, "Cosmological parameters should match the input"
        assert hasattr(cl, 'ee'), "Each theory_cl object should have an 'ee' attribute"
        assert hasattr(cl, 'sigma_e'), "Each theory_cl object should have a 'sigma_e' attribute"
        assert cl.sigma_e == likelihood_instance.noise, "sigma_e should match the noise value"

        # Write sigma_e to the regression test snapshot
        regtest.write(f"sigma_e for TheoryCl instance:\n{cl.sigma_e}\n")

        # Snapshot test for the 'ee' array
        snapshot.check(cl.ee, rtol=1e-10)

class TestXiMinusShapes:
    """Test xi-minus shape handling in likelihood."""
    
    @pytest.mark.parametrize("include_ximinus", [False, True])
    def test_data_array_shapes(self, small_mask, minimal_redshift_bins, angular_bins, include_ximinus):
        """Test that data array shapes are correct for both xi+ only and xi+/xi- modes."""

        likelihood = xlh.XiLikelihood(
            small_mask, minimal_redshift_bins, angular_bins,
            include_ximinus=include_ximinus
        )
        
        # Test data array shape
        data_array = likelihood.prep_data_array()
        n_redshift_combs = 1  # Only 1 redshift bin = 1 auto-correlation
        n_correlation_types = 2 if include_ximinus else 1
        expected_shape = (n_redshift_combs, n_correlation_types * len(angular_bins))
        
        assert data_array.shape == expected_shape, (
            f"Data array shape {data_array.shape} does not match expected {expected_shape} "
            f"for include_ximinus={include_ximinus}"
        )
    
    @pytest.mark.parametrize("include_ximinus", [False, True])
    def test_combination_matrices(self, small_mask, minimal_redshift_bins, angular_bins, include_ximinus):
        """Test that combination matrices are set up correctly for both xi+ only and xi+/xi- modes."""

        likelihood = xlh.XiLikelihood(
            small_mask, minimal_redshift_bins, angular_bins,
            include_ximinus=include_ximinus
        )
        
        # Test combination matrices
        likelihood.precompute_combination_matrices()
        
        if include_ximinus:
            # Should have both xi_plus and xi_minus matrices
            assert hasattr(likelihood, '_m_xiplus'), "Should have _m_xiplus matrix"
            assert hasattr(likelihood, '_m_ximinus'), "Should have _m_ximinus matrix"
            assert hasattr(likelihood, '_m_combined'), "Should have _m_combined matrix"
            
            # Combined should be concatenation of xi+ and xi-
            expected_combined_shape = (2 * len(angular_bins), likelihood._m_xiplus.shape[1])
            assert likelihood._m_combined.shape == expected_combined_shape, (
                f"Combined matrix shape {likelihood._m_combined.shape} does not match "
                f"expected {expected_combined_shape}"
            )
            
        else:
            # Should only have xi_plus matrices
            assert hasattr(likelihood, '_m_xiplus'), "Should have _m_xiplus matrix"
            assert hasattr(likelihood, '_m_combined'), "Should have _m_combined matrix"
            
            # Combined should be the same as xi_plus
            assert np.array_equal(likelihood._m_combined, likelihood._m_xiplus), (
                "Combined matrix should equal xi_plus matrix when include_ximinus=False"
            )

class TestXiMinusLikelihood:
    """Test xi-minus likelihood functionality."""
    
    @pytest.mark.parametrize("include_ximinus", [False, True])
    def test_likelihood_setup_with_ximinus(self, medium_mask, multiple_redshift_bins, 
                                          multiple_angular_bins, include_ximinus):
        """Test that likelihood can be set up correctly with xi-minus support."""

        likelihood = xlh.XiLikelihood(
            medium_mask, multiple_redshift_bins, multiple_angular_bins,
            include_ximinus=include_ximinus,
            exact_lmax=medium_mask.exact_lmax,
            lmax=medium_mask.lmax
        )
        
        # Setup likelihood
        likelihood.setup_likelihood()
        
        # Check data array shape
        data_array = likelihood.prep_data_array()
        n_redshift_combs = len(multiple_redshift_bins) * (len(multiple_redshift_bins) + 1) // 2
        n_correlation_types = 2 if include_ximinus else 1
        expected_shape = (n_redshift_combs, n_correlation_types * len(multiple_angular_bins))
        
        assert data_array.shape == expected_shape, (
            f"Data array shape {data_array.shape} does not match expected {expected_shape} "
            f"for include_ximinus={include_ximinus}"
        )
    
    @pytest.mark.parametrize("include_ximinus", [False, True])
    def test_covariance_matrix_with_ximinus(self, medium_mask, multiple_redshift_bins, 
                                           multiple_angular_bins, include_ximinus,snapshot):
        """Test that covariance matrices have correct shapes with xi-minus support."""

        likelihood = xlh.XiLikelihood(
            medium_mask, multiple_redshift_bins, multiple_angular_bins,
            include_ximinus=include_ximinus,
            exact_lmax=medium_mask.exact_lmax,
            lmax=medium_mask.lmax
        )
        
        # Setup likelihood and theory
        likelihood.setup_likelihood()
        likelihood.initiate_theory_cl({'omega_m': 0.31, 's8': 0.8})
        
        # Test high-ell covariance
        likelihood.get_covariance_matrix_highell()
        
        n_redshift_combs = len(multiple_redshift_bins) * (len(multiple_redshift_bins) + 1) // 2
        n_correlation_types = 2 if include_ximinus else 1
        expected_cov_size = n_redshift_combs * n_correlation_types * len(multiple_angular_bins)
        
        assert likelihood._cov_highell.shape == (expected_cov_size, expected_cov_size), (
            f"High-ell covariance shape {likelihood._cov_highell.shape} does not match "
            f"expected {(expected_cov_size, expected_cov_size)} for include_ximinus={include_ximinus}"
        )
        
        # Check that covariance matrix is symmetric and positive semi-definite
        cov = likelihood._cov_highell
        snapshot.check(cov, rtol=1e-10)
        assert np.allclose(cov, cov.T), "Covariance matrix should be symmetric"
        
        # Check eigenvalues are non-negative (allowing for numerical errors)
        eigvals = np.linalg.eigvals(cov)
        assert np.all(eigvals >= -1e-10), f"Covariance matrix should be positive semi-definite, got min eigenvalue {np.min(eigvals)}"
    
    def test_ximinus_mean_calculation(self, small_mask, minimal_redshift_bins, angular_bins, snapshot):
        """Test that mean calculation works correctly with xi-minus."""

        likelihood = xlh.XiLikelihood(
            small_mask, minimal_redshift_bins, angular_bins,
            include_ximinus=True
        )
        
        likelihood.setup_likelihood()
        likelihood.initiate_theory_cl({'omega_m': 0.31, 's8': 0.8})
        likelihood._prepare_matrix_products()
        
        # The means are calculated in _prepare_matrix_products and stored
        # Check that we have the correct mean shapes after setup
        assert hasattr(likelihood, '_means_lowell'), "Should have _means_lowell after setup"
        
        # Check shapes
        n_redshift_combs = 1  # Only 1 redshift bin = 1 auto-correlation
        n_angular_bins = len(angular_bins)
        n_correlation_types = 2  # xi+ and xi- 
        expected_shape = (n_redshift_combs, n_correlation_types * n_angular_bins)
        
        assert likelihood._means_lowell.shape == expected_shape, (
            f"Mean shape incorrect: {likelihood._means_lowell.shape}, expected: {expected_shape}"
        )
        
        # Test that we can also compute means separately for comparison
        from xilikelihood.distributions import mean_xi_gaussian_nD
        
        # Get the stored pseudo_cl and prefactors
        pseudo_cl = likelihood.pseudo_cl
        prefactors = likelihood._prefactors
        
        # Calculate means separately for xi+ and xi-
        means_both = mean_xi_gaussian_nD(
            prefactors, pseudo_cl, lmin=0, lmax=likelihood._exact_lmax, kind="both"
        )
        means_xiplus, means_ximinus = means_both
        snapshot.check(means_xiplus, rtol=1e-10)
        snapshot.check(means_ximinus, rtol=1e-10)
        # Check that stored means match the separately computed ones
        expected_means = np.concatenate([means_xiplus, means_ximinus], axis=1)
        assert np.allclose(likelihood._means_lowell, expected_means), (
            "Stored means should match separately computed means"
        )
        
        # Verify individual xi+ and xi- mean shapes
        assert means_xiplus.shape == (n_redshift_combs, n_angular_bins), f"Xi+ mean shape incorrect: {means_xiplus.shape}"
        assert means_ximinus.shape == (n_redshift_combs, n_angular_bins), f"Xi- mean shape incorrect: {means_ximinus.shape}"


