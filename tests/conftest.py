# tests/conftest.py
import pytest
import numpy as np
import xilikelihood as xlh


@pytest.fixture
def sample_theory_cl():
    """Provide a sample TheoryCl object for testing."""
    return xlh.TheoryCl(30, path="Cl_3x2pt_kids55.txt")

@pytest.fixture
def sample_mask():
    """Provide a sample mask for testing."""
    return xlh.SphereMask([2], circmaskattr=(1000, 256))

@pytest.fixture
def sample_angular_bins():
    """Provide sample angular bins for testing."""
    return [(4, 6), (7, 10)]

@pytest.fixture
def covariance_test_setup():
    """Provide a consistent setup for testing covariance functions."""
    # Use small, deterministic values for reproducible tests
    mask = xlh.SphereMask(spins=[2], circmaskattr=(1000, 32))  # Small lmax for speed

    # Create redshift bins with known parameters
    z = np.linspace(0.01, 3.0, 100)
    redshift_bins = [
        xlh.RedshiftBin(nbin=1, z=z, zmean=0.5, zsig=0.1),
        xlh.RedshiftBin(nbin=2, z=z, zmean=1.0, zsig=0.1)
    ]
    
    # Fixed angular bins for reproducibility
    angular_bins_in_deg = [(1.0, 2.0), (2.0, 4.0)]
    
    # Fixed cosmological parameters
    cosmo_params = {'omega_m': 0.31, 's8': 0.8}
    
    return {
        'mask': mask,
        'redshift_bins': redshift_bins,
        'angular_bins_in_deg': angular_bins_in_deg,
        'cosmo_params': cosmo_params,
        'theory_lmax': 30  # Small lmax for fast computation
    }