"""
Coupling utilities for copula-based likelihood experiments.

This module provides utilities for experimenting with different coupling 
approaches in likelihood computations using copulas and various marginal 
distributions.

⚠️ Note: This is experimental code for research purposes, not part of the 
main package API.
"""

import numpy as np
from scipy.stats import norm, gamma, multivariate_normal
import matplotlib.pyplot as plt

# ============================================================================
# Model Generation Utilities
# ============================================================================


def simple_model(parameter, ndim=10):
    """
    Generate a simple model vector for testing purposes.
    
    Parameters:
    -----------
    parameter : float or array-like
        Model parameter(s)
    ndim : int, default=10
        Number of dimensions
        
    Returns:
    --------
    ndarray
        Model vector(s)
    """
    
    base_vector = 1e-3 * np.linspace(1, ndim, ndim)
    if isinstance(parameter, (int, float)):
        return parameter**2 * base_vector
    else:
        param = parameter**2
        return param[:, None] * base_vector
    
# ============================================================================
# Covariance and Correlation Utilities
# ============================================================================


def covariance_to_correlation(cov_matrix):
    """
    Converts a covariance matrix to a correlation matrix.

    Parameters:
    - cov_matrix: A 2D numpy array representing the covariance matrix

    Returns:
    - corr_matrix: A 2D numpy array representing the correlation matrix
    """
    # Compute the standard deviations
    std_devs = np.sqrt(np.diag(cov_matrix))

    # Outer product of standard deviations
    std_devs_outer = np.outer(std_devs, std_devs)

    # Compute the correlation matrix
    corr_matrix = cov_matrix / std_devs_outer

    # Set the diagonal elements to 1
    np.fill_diagonal(corr_matrix, 1)

    return corr_matrix

def generate_covariance(datavector, correlation):
    ndim = len(datavector)
    base_cov = 0.5*np.outer(datavector / (np.arange(1,ndim+1) + 1),datavector) 
    corr_matrix = np.eye(ndim) * (1 - correlation) + correlation
    cov = base_cov * corr_matrix
    cov += np.eye(ndim) * 1e-6  # Add small diagonal term for numerical stability
    return cov


# ============================================================================
# Copula Density Functions
# ============================================================================


def gaussian_copula_point_density(cdf_point, covariance_matrix):
    z = norm.ppf(cdf_point)
    if covariance_matrix.ndim == 2:
        corr_matrix = covariance_to_correlation(covariance_matrix)

        mean = np.zeros(len(corr_matrix))
        mvn = multivariate_normal(
            mean=mean, cov=corr_matrix
        )  # multivariate normal with right correlation structure

        mvariate_logpdf = mvn.logpdf(z.T)
        pdf = norm.logpdf(z)
        return mvariate_logpdf - np.sum(pdf, axis=0)
    elif covariance_matrix.ndim == 3:
        # Multiple covariance matrices (3D case)
        n_prior_values, n_dim, _ = covariance_matrix.shape

        # Convert covariance matrices to correlation matrices
        std_devs = np.sqrt(np.diagonal(covariance_matrix, axis1=1, axis2=2))  # Shape: (n_prior_values, n_dim)
        corr_matrices = covariance_matrix / (std_devs[:, :, None] * std_devs[:, None, :])  # Shape: (n_prior_values, n_dim, n_dim)
        np.fill_diagonal(corr_matrices, 1, wrap=True)  # Ensure diagonals are exactly 1

        # Compute multivariate normal log PDFs for all prior values
        mean = np.zeros((n_prior_values, n_dim))  # Mean vector for each prior value
        mvn_logpdf = np.array([
            multivariate_normal.logpdf(z[:, i], mean=mean[i], cov=corr_matrices[i])
            for i in range(n_prior_values)
        ])  # Shape: (n_prior_values,)

        # Compute the sum of univariate normal log PDFs
        univariate_logpdf = np.sum(norm.logpdf(z), axis=0)  # Shape: (n_prior_values,)

        # Compute the copula density
        copula_density = mvn_logpdf - univariate_logpdf  # Shape: (n_prior_values,)
        return copula_density

    else:
        raise ValueError("Covariance matrix must be 2D or 3D.")

# ============================================================================
# Likelihood Functions
# ============================================================================


def copula_likelihood(datavector, cov, prior_model,type='gamma'):
    """
    Compute the copula likelihood using 1D marginals for each dimension.
    This function evaluates the log PDFs for all dimensions and prior values simultaneously.
    
    Compute copula likelihood using specified marginal distributions.
    
    Parameters:
    -----------
    datavector : ndarray
        Observed data vector
    cov : ndarray
        Covariance matrix or matrices
    prior_model : ndarray
        Prior model predictions
    marginal_type : str, default='gamma'
        Type of marginal distribution ('gamma' or 'gaussian')
        
    Returns:
    --------
    ndarray
        Log likelihood values
    
    """
    # Extract standard deviations from the covariance matrix
    if not np.all(np.linalg.eigvals(cov) >= 0):  
        print(cov)        
        raise ValueError("Covariance matrix must be positive semi-definite.")
     # Shape: (n_dim,)
    mean = prior_model.T
    if type == 'gamma':
        if cov.ndim == 3:
            
            std_devs = np.sqrt(np.diagonal(cov, axis1=1, axis2=2))  # Shape: (n_prior_values, n_dim)
            a = mean**2 / std_devs.T**2  # Shape: (n_dim, n_prior_values)
            scale = std_devs.T**2 / mean  # Shape: (n_dim, n_prior_values)
        elif cov.ndim == 2:   
            std_devs = np.sqrt(np.diag(cov)) 
            a = mean**2 / (std_devs**2)[:, None]  # Shape: (n_dim, n_prior_values)
            scale = (std_devs**2)[:, None] / mean  # Shape: (n_dim, n_prior_values)
        
        # Compute log PDFs for each dimension and prior value
        # prior_model: (n_dim, n_prior_values), std_devs: (n_dim,)
        log_pdfs = gamma.logpdf(datavector[:, None], a=a, scale=scale)
        # Shape of log_pdfs: (n_dim, n_prior_values)

        # Sum log PDFs across dimensions to get the total log likelihood for each prior value
        total_log_likelihood = np.sum(log_pdfs, axis=0)  # Shape: (n_prior_values,)
        cdf_point = gamma.cdf(datavector[:, None], a=a, scale=scale)
    elif type == 'gaussian':
        # Compute log PDFs for each dimension and prior value
        log_pdfs = norm.logpdf(datavector[:, None], loc=mean, scale=std_devs[:, None])
        # Shape of log_pdfs: (n_dim, n_prior_values)

        # Sum log PDFs across dimensions to get the total log likelihood for each prior value
        total_log_likelihood = np.sum(log_pdfs, axis=0)
        cdf_point = norm.cdf(datavector[:, None], loc=mean, scale=std_devs[:, None])
    else:
        raise ValueError("Unsupported marginal type. Use 'gamma' or 'gaussian'.")
    copula_density = gaussian_copula_point_density(cdf_point, cov)
    # Convert log likelihood to likelihood
    likelihood = total_log_likelihood + copula_density  # Shape: (n_prior_values,)
    return likelihood


