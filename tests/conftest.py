# tests/conftest.py
import os

# Ensure JAX runs on CPU for tests (set before importing xilikelihood/jax)
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.1")

import pytest
import numpy as np
import xilikelihood as xlh


@pytest.fixture
def sample_theory_cl(covariance_test_setup):
    """Provide a sample TheoryCl object for testing (computed from cosmology)."""
    setup = covariance_test_setup
    return xlh.TheoryCl(
        setup["theory_lmax"],
        cosmo=setup["cosmo_params"],
        z_bins=setup["redshift_bins"],
        clname="test_cl"
    )

@pytest.fixture
def sample_cosmo(covariance_test_setup):
    """Provide a sample cosmology dictionary for testing."""
    return covariance_test_setup["cosmo_params"]

@pytest.fixture
def sample_mask(covariance_test_setup):
    """Provide a sample mask for testing."""
    return covariance_test_setup["mask"]

@pytest.fixture
def sample_angular_bins(covariance_test_setup):
    """Provide sample angular bins for testing."""
    return covariance_test_setup["angular_bins_in_deg"]

@pytest.fixture
def sample_redshift_bins(covariance_test_setup):
    """Provide sample redshift bins for testing."""
    return covariance_test_setup["redshift_bins"]

@pytest.fixture
def covariance_test_setup():
    """Provide a consistent setup for testing covariance functions."""
    # Use small values for reproducible tests
    exact_lmax = 10
    mask = xlh.SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=exact_lmax,l_smooth=30)  # Small nside/lmax for speed

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
        'theory_lmax': mask.lmax,
        'exact_lmax': exact_lmax
    }