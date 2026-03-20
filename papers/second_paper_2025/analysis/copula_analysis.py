#!/usr/bin/env python3
"""
Main analysis functions for copula comparison study.

This module contains the core analysis logic without plotting or testing functions.
"""

import numpy as np
from copula_core import (simple_linear_model, generate_covariance, compute_log_likelihood,
                        normalize_posterior, compute_posterior_stats)


def run_copula_analysis(config):
    """
    Run the main copula impact analysis.
    
    Parameters:
    -----------
    config : dict
        Configuration dictionary with keys:
        - fiducial_param: float
        - param_grid: array
        - correlation_values: list
        - marginal_types: list
        - n_data_points_list: list
        - df: int (for Student-t copula)
    
    Returns:
    --------
    dict : Results dictionary
    """
    print("Starting copula impact analysis...")
    
    # Extract configuration
    fiducial_param = config['fiducial_param']
    param_grid = config['param_grid']
    correlation_values = config['correlation_values']
    marginal_types = config['marginal_types']
    n_data_points_list = config['n_data_points_list']
    df = config.get('df', 10)
    
    # Storage for results
    results = {}
    
    # Loop over number of data points
    for n_data in n_data_points_list:
        print(f"\nProcessing {n_data} data points...")
        results[n_data] = {}
        
        # Generate fiducial data for this dimensionality
        fiducial_data = simple_linear_model(fiducial_param, n_data_points=n_data)
        
        # Loop over marginal types
        for marginal_type in marginal_types:
            print(f"  Processing marginal type: {marginal_type}")
            results[n_data][marginal_type] = {}
            
            # Loop over correlation values
            for corr in correlation_values:
                print(f"    Processing correlation = {corr:.1f}")
                
                # Generate covariance matrix
                cov = generate_covariance(fiducial_data, corr)
                results[n_data][marginal_type][corr] = {}
                
                # Gaussian copula
                gaussian_result = analyze_single_copula(
                    param_grid, fiducial_data, cov, 'gaussian', marginal_type, n_data, df=None
                )
                results[n_data][marginal_type][corr]['gaussian'] = gaussian_result
                
                # Student-t copula
                student_t_result = analyze_single_copula(
                    param_grid, fiducial_data, cov, 'student_t', marginal_type, n_data, df=df
                )
                results[n_data][marginal_type][corr]['student_t_df10'] = student_t_result
                
                # Print summary for this configuration
                if gaussian_result is not None:
                    print(f"      Gaussian: mean={gaussian_result['mean']:.3f}, max={gaussian_result['maximum']:.3f}")
                
                if student_t_result is not None:
                    print(f"      Student-t: mean={student_t_result['mean']:.3f}, max={student_t_result['maximum']:.3f}")
                    
                    # Check for difference
                    if gaussian_result is not None:
                        diff_mean = abs(gaussian_result['mean'] - student_t_result['mean'])
                        diff_max = abs(gaussian_result['maximum'] - student_t_result['maximum'])
                        if diff_mean > 0.1 or diff_max > 0.1:
                            print(f"      *** Significant copula effect! Mean diff: {diff_mean:.3f}, Max diff: {diff_max:.3f}")
                else:
                    print(f"      Student-t: FAILED")
    
    return results


def analyze_single_copula(param_grid, fiducial_data, cov, copula_type, marginal_type, n_data, df=None):
    """
    Analyze a single copula configuration.
    
    Returns:
    --------
    dict or None : Result dictionary with posterior, mean, maximum, or None if failed
    """
    try:
        log_likes = []
        for param in param_grid:
            pred_data = simple_linear_model(param, n_data_points=n_data)
            log_like = compute_log_likelihood(pred_data, fiducial_data, cov, 
                                            copula_type=copula_type, df=df, 
                                            marginal_type=marginal_type)
            log_likes.append(log_like)
        
        # Check if we got valid results
        log_likes_array = np.array(log_likes)
        if np.all(np.isfinite(log_likes_array)) and not np.all(log_likes_array == -np.inf):
            posterior = normalize_posterior(param_grid, log_likes_array)
            mean, maximum = compute_posterior_stats(param_grid, posterior)
            
            return {
                'posterior': posterior,
                'mean': mean,
                'maximum': maximum
            }
        else:
            print(f"      Warning: {copula_type} copula failed for {marginal_type} (all -inf)")
            return None
            
    except Exception as e:
        print(f"      Error with {copula_type} copula for {marginal_type}: {e}")
        return None


def get_default_config():
    """
    Get default configuration for the analysis.
    """
    return {
        'fiducial_param': 5.0,
        'param_grid': np.linspace(0.5, 8.5, 1000),
        'correlation_values': [0.0, 0.5, 0.7, 0.9],  # Removed 1.0 for stability
        'marginal_types': ['normal', 'lognormal', 'student_t'],
        'n_data_points_list': np.arange(1,20),
        'df': 10
    }
        

def get_fast_config():
    """
    Get a fast configuration for testing.
    """
    return {
        'fiducial_param': 5.0,
        'param_grid': np.linspace(0.5, 8.5, 200),  # Lower resolution
        'correlation_values': [0.0, 0.7, 0.9],  # Fewer correlations
        'marginal_types': ['normal', 'lognormal'],  # Fewer marginals
        'n_data_points_list': [5, 10],  # Fewer dimensions
        'df': 10
    }
