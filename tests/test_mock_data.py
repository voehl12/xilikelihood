"""
Tests for mock data generation functionality.
"""

import numpy as np
import tempfile
import os
from pathlib import Path

import xilikelihood as xlh


def test_mock_data_creation():
    """Test basic mock data creation functionality."""
    
    # Create a simple likelihood setup
    likelihood = xlh.XiLikelihood(
        mask=xlh.SphereMask(spins=[2], circmaskattr=(100, 16),exact_lmax=10),  # Small for fast testing
        redshift_bins=[xlh.RedshiftBin(nbin=1, z=np.linspace(0.01, 3.0, 100), 
                                      zmean=0.5, zsig=0.1)],
        ang_bins_in_deg=[(1.0, 2.0)],  # Single bin for speed
        include_ximinus=True,
        exact_lmax=10
    )
    
    fiducial_cosmo = {'omega_m': 0.31, 's8': 0.8}
    
    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_data_path = os.path.join(tmpdir, 'test_mock_data.npz')
        cov_path = os.path.join(tmpdir, 'test_cov.npz')
        
        # Test fiducial mean (deterministic)
        mock_data, cov = xlh.mock_data.create_mock_data(
            likelihood, mock_data_path, cov_path,
            fiducial_cosmo=fiducial_cosmo,
            random=None
        )
        
        # Check files were created
        assert os.path.exists(mock_data_path)
        assert os.path.exists(cov_path)
        
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
        assert data_metadata['s8'] == fiducial_cosmo['s8']
        assert cov_metadata['s8'] == fiducial_cosmo['s8']
        assert data_metadata['random'] is None
        

def test_gaussian_mock_data():
    """Test Gaussian mock data generation."""
    
    likelihood = xlh.XiLikelihood(
        mask=xlh.SphereMask(spins=[2], circmaskattr=(100, 16), exact_lmax=10),
        redshift_bins=[xlh.RedshiftBin(nbin=1, z=np.linspace(0.01, 3.0, 100), 
                                      zmean=0.5, zsig=0.1)],
        ang_bins_in_deg=[(1.0, 2.0)],
        include_ximinus=False,  # Test with only xi_plus
        exact_lmax=10
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_data_path = os.path.join(tmpdir, 'test_gaussian_mock.npz')
        cov_path = os.path.join(tmpdir, 'test_gaussian_cov.npz')
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        mock_data, cov = xlh.mock_data.create_mock_data(
            likelihood, mock_data_path, cov_path,
            random='gaussian'
        )
        
        # Check that Gaussian mock data is different from deterministic
        np.random.seed(42)
        mock_data2, _ = xlh.mock_data.create_mock_data(
            likelihood, mock_data_path, cov_path,
            random=None  # Deterministic
        )
        
        # They should be different (Gaussian vs deterministic)
        assert not np.allclose(mock_data, mock_data2)
        
        # But shapes should match
        assert mock_data.shape == mock_data2.shape


if __name__ == "__main__":
    print("Running mock data tests...")
    test_mock_data_creation()
    print("✅ Basic mock data creation test passed")
    
    test_gaussian_mock_data()
    print("✅ Gaussian mock data test passed")
    
    print("🎉 All mock data tests passed!")
