"""
Mock data generation utilities for xilikelihood.

This module provides functions to generate mock correlation function data
for testing, validation, and analysis workflows.
"""

import numpy as np
from typing import Optional, Dict, Any, Union, Tuple


def create_mock_data(
    likelihood,
    mock_data_path: str,
    gaussian_covariance_path: str,
    fiducial_cosmo: Optional[Dict[str, Any]] = None,
    random: Optional[str] = None,
    exact_lmax: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create mock correlation function data for likelihood analysis.

    This function generates mock data by either using the fiducial mean,
    drawing from a Gaussian distribution, or simulating from maps. It also
    saves the Gaussian covariance matrix for reference.

    Parameters
    ----------
    likelihood : XiLikelihood
        Configured likelihood object with mask, redshift bins, and angular bins.
    mock_data_path : str
        Path to save the mock data (.npz file).
    gaussian_covariance_path : str
        Path to save the Gaussian covariance matrix (.npz file).
    fiducial_cosmo : dict, optional
        Fiducial cosmological parameters. If None, uses default values.
        Should contain keys like 'omega_m', 's8', etc.
    random : {'gaussian', 'frommap'}, optional
        Type of mock data to generate:
        - None: Use fiducial mean (deterministic)
        - 'gaussian': Draw from multivariate Gaussian with fiducial mean and covariance
        - 'frommap': Simulate from actual correlation function maps
    exact_lmax : int, optional
        Maximum multipole for exact calculations. Default is 20.

    Returns
    -------
    mock_data : np.ndarray
        Generated mock data vector.
    gaussian_covariance : np.ndarray
        Gaussian covariance matrix.

    Examples
    --------
    >>> import xilikelihood as xlh
    >>> import numpy as np
    >>> 
    >>> # Set up likelihood
    >>> likelihood = xlh.XiLikelihood(
    ...     mask=xlh.SphereMask(spins=[2], circmaskattr=(1000, 256)),
    ...     redshift_bins=[xlh.RedshiftBin(nbin=1, z=np.linspace(0.01, 3.0, 100), 
    ...                                   zmean=0.5, zsig=0.1)],
    ...     angular_bins_in_deg=[(1.0, 2.0), (2.0, 4.0)],
    ...     include_ximinus=True,
    ... )
    >>> 
    >>> # Generate Gaussian mock data
    >>> mock_data, cov = xlh.mock_data.create_mock_data(
    ...     likelihood, 'mock_data.npz', 'cov.npz', 
    ...     fiducial_cosmo={'omega_m': 0.31, 's8': 0.8},
    ...     random='gaussian'
    ... )
    """
    # Import here to avoid circular imports
    from . import simulate_correlation_functions
    
    # Default fiducial cosmology
    if fiducial_cosmo is None:
        fiducial_cosmo = {"omega_m": 0.31, "s8": 0.8}
    
    # Set up likelihood computation
    theory_cls = likelihood.initiate_theory_cl(fiducial_cosmo)
    likelihood.setup_likelihood()
    likelihood._prepare_matrix_products()

    # Compute covariance and means
    likelihood.get_covariance_matrix_lowell()
    likelihood.get_covariance_matrix_highell()
    likelihood._get_means_highell()

    # Combine low-ell and high-ell contributions
    gaussian_covariance = likelihood._cov_highell + likelihood._cov_lowell
    fiducial_mean = likelihood._means_highell + likelihood._means_lowell

    # Generate mock data based on specified method
    if random is None:
        # Use fiducial mean (deterministic)
        mock_data = fiducial_mean
    elif random == 'gaussian':
        # Draw from multivariate Gaussian (flatten mean for numpy)
        mock_data_flat = np.random.multivariate_normal(
            mean=fiducial_mean.flatten(),
            cov=gaussian_covariance,
            size=1,
        )[0]  # Extract single realization
        # Reshape back to original shape
        mock_data = mock_data_flat.reshape(fiducial_mean.shape)
    elif random == 'frommap':
        # Simulate from correlation function maps
        sim = simulate_correlation_functions(
            theory_cls,
            [likelihood.mask],
            likelihood.ang_bins_in_deg,
            n_batch=1,
        )
        xi_plus, xi_minus = sim['xi_plus'][0], sim['xi_minus'][0]
        if likelihood.include_ximinus:
            mock_data = np.concatenate([xi_plus, xi_minus], axis=1)
        else:
            mock_data = xi_plus
    else:
        raise ValueError(
            f"Invalid random option '{random}'. Choose None, 'gaussian', or 'frommap'."
        )

    # Prepare metadata for saving
    redshift_bin_info = [
        likelihood.redshift_bins[i].nbin for i in range(len(likelihood.redshift_bins))
    ]
    
    # Save covariance matrix
    np.savez(
        gaussian_covariance_path,
        cov=gaussian_covariance,
        s8=fiducial_cosmo['s8'],
        angs=likelihood.ang_bins_in_deg,
        rs_bins=redshift_bin_info,
        random=random,
        fiducial_cosmo=fiducial_cosmo
    )
    
    # Save mock data
    np.savez(
        mock_data_path,
        data=mock_data,
        s8=fiducial_cosmo['s8'],
        angs=likelihood.ang_bins_in_deg,
        rs_bins=redshift_bin_info,
        random=random,
        fiducial_cosmo=fiducial_cosmo
    )
    
    print(f"Mock data and covariance matrix saved to {mock_data_path} and {gaussian_covariance_path}")
    
    return mock_data, gaussian_covariance


def load_mock_data(mock_data_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load mock data from a saved file.

    Parameters
    ----------
    mock_data_path : str
        Path to the mock data .npz file.

    Returns
    -------
    data : np.ndarray
        Mock data vector.
    metadata : dict
        Dictionary containing metadata (s8, angular bins, etc.).
    """
    loaded = np.load(mock_data_path, allow_pickle=True)
    data = loaded['data']
    
    metadata = {}
    for key in loaded.keys():
        if key != 'data':
            metadata[key] = loaded[key].item() if loaded[key].ndim == 0 else loaded[key]
    
    return data, metadata


def load_gaussian_covariance(gaussian_covariance_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load Gaussian covariance matrix from a saved file.

    Parameters
    ----------
    gaussian_covariance_path : str
        Path to the covariance matrix .npz file.

    Returns
    -------
    cov : np.ndarray
        Gaussian covariance matrix.
    metadata : dict
        Dictionary containing metadata (s8, angular bins, etc.).
    """
    loaded = np.load(gaussian_covariance_path, allow_pickle=True)
    cov = loaded['cov']
    
    metadata = {}
    for key in loaded.keys():
        if key != 'cov':
            metadata[key] = loaded[key].item() if loaded[key].ndim == 0 else loaded[key]
    
    return cov, metadata
