"""
DEPRECATED: Mock data generation - moved to xilikelihood.mock_data module.

This script is kept for backward compatibility but the functionality
has been moved to xilikelihood.mock_data for reusability.
"""

import numpy as np
import xilikelihood as xlh
from config import (
    FIDUCIAL_COSMO,
    EXACT_LMAX
)

# Import the new implementation
from xilikelihood.mock_data import create_mock_data as _create_mock_data


def create_mock_data(likelihood, mock_data_path, gaussian_covariance_path, random=None):
    """
    DEPRECATED: Use xilikelihood.mock_data.create_mock_data instead.
    
    This wrapper is kept for backward compatibility.
    """
    print("WARNING: This function is deprecated. Use xilikelihood.mock_data.create_mock_data instead.")
    
    return _create_mock_data(
        likelihood=likelihood,
        mock_data_path=mock_data_path,
        gaussian_covariance_path=gaussian_covariance_path,
        fiducial_cosmo=FIDUCIAL_COSMO,
        random=random,
        exact_lmax=EXACT_LMAX
    )


if __name__ == "__main__":
    # Example usage with new module
    likelihood = xlh.XiLikelihood(
        mask=xlh.SphereMask(spins=[2], circmaskattr=(1000, 256)),
        redshift_bins=[xlh.RedshiftBin(nbin=1, z=np.linspace(0.01, 3.0, 100), zmean=0.5, zsig=0.1)],
        angular_bins_in_deg=[(1.0, 2.0), (2.0, 4.0)],
        include_ximinus=True,
    )
    
    mock_data_path = 'mock_data.npz'
    gaussian_covariance_path = 'gaussian_covariance.npz'
    
    # Use the new module directly (recommended)
    xlh.mock_data.create_mock_data(
        likelihood, mock_data_path, gaussian_covariance_path, 
        fiducial_cosmo=FIDUCIAL_COSMO,
        random='gaussian'
    )