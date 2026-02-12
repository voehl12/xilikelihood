"""
Tests for mock data generation functionality.
"""

import numpy as np

import xilikelihood as xlh


def test_mock_data_creation(covariance_test_setup, tmp_path):
    """Test basic mock data creation functionality."""
    setup = covariance_test_setup

    likelihood = xlh.XiLikelihood(
        mask=setup["mask"],
        redshift_bins=setup["redshift_bins"],
        ang_bins_in_deg=setup["angular_bins_in_deg"],
        include_ximinus=True,
        exact_lmax=setup["exact_lmax"],
    )

    fiducial_cosmo = setup["cosmo_params"]

    mock_data_path = tmp_path / "test_mock_data.npz"
    cov_path = tmp_path / "test_cov.npz"

    # Test fiducial mean (deterministic)
    mock_data, cov = xlh.mock_data.create_mock_data(
        likelihood,
        mock_data_path,
        cov_path,
        fiducial_cosmo=fiducial_cosmo,
        random=None,
    )

    # Check files were created
    assert mock_data_path.exists()
    assert cov_path.exists()

    # Check data shapes
    assert isinstance(mock_data, np.ndarray)
    assert isinstance(cov, np.ndarray)
    assert cov.shape[0] == cov.shape[1]  # Square matrix

    # Mock data shape: (n_correlations, n_angbins)
    # Covariance shape: (n_correlations*n_angbins, n_correlations*n_angbins)
    expected_cov_size = mock_data.shape[0] * mock_data.shape[1]
    assert cov.shape[0] == expected_cov_size  # Compatible dimensions

    # Test loading functionality
    loaded_data, data_metadata = xlh.mock_data.load_mock_data(mock_data_path)
    loaded_cov, cov_metadata = xlh.mock_data.load_gaussian_covariance(cov_path)

    # Check loaded data matches
    np.testing.assert_array_equal(loaded_data, mock_data)
    np.testing.assert_array_equal(loaded_cov, cov)

    # Check metadata
    assert data_metadata["s8"] == fiducial_cosmo["s8"]
    assert cov_metadata["s8"] == fiducial_cosmo["s8"]
    assert data_metadata["random"] is None
        

def test_gaussian_mock_data(covariance_test_setup, tmp_path):
    """Test Gaussian mock data generation."""
    setup = covariance_test_setup

    likelihood = xlh.XiLikelihood(
        mask=setup["mask"],
        redshift_bins=setup["redshift_bins"],
        ang_bins_in_deg=setup["angular_bins_in_deg"],
        include_ximinus=False,  # Test with only xi_plus
        exact_lmax=setup["exact_lmax"],
    )

    mock_data_path = tmp_path / "test_gaussian_mock.npz"
    cov_path = tmp_path / "test_gaussian_cov.npz"

    # Set random seed for reproducibility
    np.random.seed(42)

    mock_data, cov = xlh.mock_data.create_mock_data(
        likelihood,
        mock_data_path,
        cov_path,
        random="gaussian",
    )

    # Check that Gaussian mock data is different from deterministic
    np.random.seed(42)
    mock_data2, _ = xlh.mock_data.create_mock_data(
        likelihood,
        mock_data_path,
        cov_path,
        random=None,  # Deterministic
    )

    # They should be different (Gaussian vs deterministic)
    assert not np.allclose(mock_data, mock_data2)

    # But shapes should match
    assert mock_data.shape == mock_data2.shape


