import pytest
import numpy as np

import xilikelihood as xlh


@pytest.fixture
def minimal_redshift_bin(sample_redshift_bins):
    """Create minimal redshift bins for testing."""
    return [sample_redshift_bins[0]]



@pytest.fixture
def angular_bin(sample_angular_bins):
    """Angular bins for testing."""
    return [sample_angular_bins[0]]


@pytest.fixture
def likelihood_instance(sample_mask, sample_redshift_bins, sample_angular_bins,sample_cosmo):
    # Set up a minimal instance of XiLikelihood
   
    likelihood = xlh.XiLikelihood(
        mask=sample_mask,
        redshift_bins=sample_redshift_bins,
        ang_bins_in_deg=sample_angular_bins,
        noise='default',
    )
    likelihood.setup_likelihood()
    likelihood.initiate_theory_cl(sample_cosmo)
    likelihood._prepare_matrix_products()
    return likelihood

def test_get_cfs_1d_lowell(likelihood_instance, snapshot):
        
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

def test_initiate_theory_cl(likelihood_instance, sample_cosmo, snapshot, regtest):
    # Call the method
    likelihood_instance.initiate_theory_cl(sample_cosmo)

    # Capture outputs
    theory_cl = likelihood_instance._theory_cl

    # Assert that theory_cl is a list of objects with expected attributes
    assert isinstance(theory_cl, list), "theory_cl should be a list"
    assert len(theory_cl) > 0, "theory_cl should not be empty"
    for idx, cl in enumerate(theory_cl):
        assert hasattr(cl, 'lmax'), "Each theory_cl object should have an 'lmax' attribute"
        assert hasattr(cl, 'cosmo'), "Each theory_cl object should have a 'cosmo' attribute"
        assert cl.cosmo == sample_cosmo, "Cosmological parameters should match the input"
        assert hasattr(cl, 'ee'), "Each theory_cl object should have an 'ee' attribute"
        assert hasattr(cl, 'sigma_e'), "Each theory_cl object should have a 'sigma_e' attribute"
        expected_sigma_e = None if likelihood_instance._is_cov_cross[idx] else likelihood_instance.noise
        assert cl.sigma_e == expected_sigma_e, "sigma_e should match the noise value"

        # Write sigma_e to the regression test snapshot
        regtest.write(f"sigma_e for TheoryCl instance:\n{cl.sigma_e}\n")

        # Snapshot test for the 'ee' array
        snapshot.check(cl.ee, rtol=1e-10)

class TestXiMinusShapes:
    """Test xi-minus shape handling in likelihood."""
    
    @pytest.mark.parametrize("include_ximinus", [False, True])
    def test_data_array_shapes(self, sample_mask, minimal_redshift_bin, angular_bin, include_ximinus):
        """Test that data array shapes are correct for both xi+ only and xi+/xi- modes."""

        likelihood = xlh.XiLikelihood(
            sample_mask, minimal_redshift_bin, angular_bin,
            include_ximinus=include_ximinus
        )
        
        # Test data array shape
        data_array = likelihood.prep_data_array()
        n_redshift_combs = 1  # Only 1 redshift bin = 1 auto-correlation
        n_correlation_types = 2 if include_ximinus else 1
        expected_shape = (n_redshift_combs, n_correlation_types * len(angular_bin))
        
        assert data_array.shape == expected_shape, (
            f"Data array shape {data_array.shape} does not match expected {expected_shape} "
            f"for include_ximinus={include_ximinus}"
        )
    
    @pytest.mark.parametrize("include_ximinus", [False, True])
    def test_combination_matrices(self, sample_mask, minimal_redshift_bin, angular_bin, include_ximinus):
        """Test that combination matrices are set up correctly for both xi+ only and xi+/xi- modes."""

        likelihood = xlh.XiLikelihood(
            sample_mask, minimal_redshift_bin, angular_bin,
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
            expected_combined_shape = (2 * len(angular_bin), likelihood._m_xiplus.shape[1])
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
    def test_likelihood_setup_with_ximinus(self, sample_mask, sample_redshift_bins, 
                                          sample_angular_bins, include_ximinus):
        """Test that likelihood can be set up correctly with xi-minus support."""
        
        likelihood = xlh.XiLikelihood(
            sample_mask, sample_redshift_bins, sample_angular_bins,
            include_ximinus=include_ximinus,
            exact_lmax=sample_mask.exact_lmax,
            lmax=sample_mask.lmax
        )
        
        # Setup likelihood
        likelihood.setup_likelihood()
        
        # Check data array shape
        data_array = likelihood.prep_data_array()
        n_redshift_combs = len(sample_redshift_bins) * (len(sample_redshift_bins) + 1) // 2
        n_correlation_types = 2 if include_ximinus else 1
        expected_shape = (n_redshift_combs, n_correlation_types * len(sample_angular_bins))
        
        assert data_array.shape == expected_shape, (
            f"Data array shape {data_array.shape} does not match expected {expected_shape} "
            f"for include_ximinus={include_ximinus}"
        )
    
    @pytest.mark.parametrize("include_ximinus", [False, True])
    def test_covariance_matrix_with_ximinus(self, sample_mask, sample_redshift_bins, 
                                           sample_angular_bins, include_ximinus,snapshot):
        """Test that covariance matrices have correct shapes with xi-minus support."""

        likelihood = xlh.XiLikelihood(
            sample_mask, sample_redshift_bins, sample_angular_bins,
            include_ximinus=include_ximinus,
            exact_lmax=sample_mask.exact_lmax,
            lmax=sample_mask.lmax
        )
        
        # Setup likelihood and theory
        likelihood.setup_likelihood()
        likelihood.initiate_theory_cl({'omega_m': 0.31, 's8': 0.8})
        
        # Test high-ell covariance
        likelihood.get_covariance_matrix_highell()
        
        n_redshift_combs = len(sample_redshift_bins) * (len(sample_redshift_bins) + 1) // 2
        n_correlation_types = 2 if include_ximinus else 1
        expected_cov_size = n_redshift_combs * n_correlation_types * len(sample_angular_bins)
        
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
    
    def test_ximinus_mean_calculation(self, sample_mask, minimal_redshift_bin, angular_bin, snapshot):
        """Test that mean calculation works correctly with xi-minus."""

        likelihood = xlh.XiLikelihood(
            sample_mask, minimal_redshift_bin, angular_bin,
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
        n_angular_bins = len(angular_bin)
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


