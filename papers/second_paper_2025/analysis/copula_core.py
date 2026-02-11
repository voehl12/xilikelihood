#!/usr/bin/env python3
"""
Core functions for copula impact analysis.

This module contains the essential functions for:
- Simple linear model
- Covariance generation
- Log-likelihood computation
- Posterior normalization and statistics
"""

import numpy as np
import sys
from scipy.stats import multivariate_normal, norm, gamma, t, lognorm

# Add xilikelihood to path
sys.path.insert(0, '/cluster/home/veoehl/xilikelihood')
import xilikelihood as xlh


def simple_linear_model(param, n_data_points=10, slope=1.0, intercept=0.0):
    """
    Simple linear model that maps parameter to data vector.
    """
    x_values = np.linspace(1, 2, n_data_points)
    
    if np.isscalar(param):
        return slope * param**2 * x_values + intercept
    else:
        param = np.asarray(param)
        return slope * param[:, np.newaxis]**2 * x_values[np.newaxis, :] + intercept


def generate_covariance(datavector, correlation, noise_scale=0.5):
    """
    Generate covariance matrix with specified correlation structure.
    Uses data-dependent covariance for stronger copula effects.
    """
    ndim = len(datavector)
    
    # Data-dependent base covariance
    base_cov = noise_scale * np.outer(datavector / (np.arange(1, ndim+1) + 1), datavector) 
    
    # Full correlation matrix
    corr_matrix = np.eye(ndim) * (1 - correlation) + correlation
    
    # Element-wise multiplication
    cov = base_cov * corr_matrix
    
    # Add small diagonal term for numerical stability
    cov += np.eye(ndim) * 1e-6
    
    return cov

def compute_marginal_likelihoods_and_cdfs(prediction, data, covariance, marginal_type='normal'):
    marginal_stds = np.sqrt(np.diag(covariance))
    cdf_vals = []
    marginal_log_densities = []

    for i, (pred, obs, std) in enumerate(zip(prediction, data, marginal_stds)):
        
        if marginal_type == 'normal':
            pdf_val = norm.pdf(obs, loc=pred, scale=std)
            cdf_val = norm.cdf(obs, loc=pred, scale=std)
            log_density = norm.logpdf(obs, loc=pred, scale=std)
            
            
        elif marginal_type == 'lognormal':
            # Log-normal marginals: prediction gives the mean, evaluate at observed data
            if obs <= 0:
                raise ValueError(f"Log-normal marginal failed: observed data {obs:.6f} <= 0 "
                                f"(data must be positive for log-normal)")
            
            # Set up log-normal distribution with mean = pred
            # For log-normal: mean = exp(mu + sigma²/2)
            # We want mean = pred, so: mu = log(pred) - sigma²/2
            sigma = 0.7  # Shape parameter (controls relative variance)
            mu = np.log(max(pred, 1e-10)) - 0.5 * sigma**2  
            if not np.fabs(lognorm.stats(sigma, scale=np.exp(mu))[0] - pred) < 1e-10:
                raise ValueError(f"Log-normal marginal setup failed: mean mismatch "
                                f"(pred={pred}, implied_mean={lognorm.stats(sigma, scale=np.exp(mu))[0]})")
            # Evaluate at observed data point
            cdf_val = lognorm.cdf(obs, s=sigma, scale=np.exp(mu))
            log_density = lognorm.logpdf(obs, s=sigma, scale=np.exp(mu))

            if not np.isfinite(cdf_val) or not np.isfinite(log_density):
                raise ValueError(f"Log-normal marginal produced non-finite values: "
                                f"cdf={cdf_val}, log_density={log_density}")
                
        elif marginal_type == 'student_t':
            # Student-t marginals with df=10
            df_marginal = 10.0
            scaled_resid = (pred-obs) / std
            cdf_val = t.cdf(scaled_resid, df=df_marginal)
            log_density = t.logpdf(scaled_resid, df=df_marginal) - np.log(std)
            
        else:
            raise ValueError(f"Unknown marginal type: {marginal_type}")
        
        # Ensure u is in (0,1) with safe bounds
        cdf_val_clipped = np.clip(cdf_val, 1e-12, 1-1e-12)
        if cdf_val_clipped != cdf_val:
            print(f"Warning: CDF value clipped from {cdf_val} to {cdf_val_clipped}")
        cdf_val = cdf_val_clipped

        cdf_vals.append(cdf_val)
        marginal_log_densities.append(log_density)

    cdf_vals = np.array(cdf_vals)
    total_marginal_log_density = np.sum(marginal_log_densities)
    return cdf_vals, total_marginal_log_density

def compute_log_likelihood(data_pred, data_obs, covariance, copula_type='gaussian', 
                          df=None, marginal_type='normal'):
    """
    Compute log-likelihood using xilikelihood copula functionality.
    """
    if copula_type == 'gaussian' and marginal_type == 'normal':
        # Standard multivariate normal (fastest case)
        return multivariate_normal.logpdf(data_pred, mean=data_obs, cov=covariance)
    
    
    
    
    
    cdf_vals, total_marginal_log_density = compute_marginal_likelihoods_and_cdfs(
        data_pred,data_obs, covariance, marginal_type=marginal_type)
    
    # Evaluate copula density
    try:
        if copula_type == 'gaussian':
            copula_log_density = xlh.copula_funcs.gaussian_copula_point_density(cdf_vals, covariance)
        elif copula_type == 'student_t':
            if df is None:
                raise ValueError("df must be specified for Student-t copula")
            copula_log_density = xlh.copula_funcs.student_t_copula_point_density(cdf_vals, covariance, df)
        else:
            raise ValueError(f"Unknown copula type: {copula_type}")
        
        if not np.isfinite(copula_log_density):
            return -np.inf
            
    except Exception as e:
        return -np.inf
    
    return copula_log_density + total_marginal_log_density


def normalize_posterior(param_grid, log_likelihood):
    """
    Normalize log-likelihood to get proper posterior.
    """
    likelihood = np.exp(log_likelihood - np.max(log_likelihood))
    posterior = likelihood / np.trapz(likelihood, param_grid)
    return posterior


def compute_posterior_stats(param_grid, posterior):
    """
    Compute posterior mean and maximum.
    """
    mean = np.trapz(param_grid * posterior, param_grid)
    max_idx = np.argmax(posterior)
    maximum = param_grid[max_idx]
    return mean, maximum
