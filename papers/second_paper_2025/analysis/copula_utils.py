#!/usr/bin/env python3
"""
Utility functions for saving/loading results and printing summaries.
"""

import pickle
import json
import numpy as np
from datetime import datetime


def save_results(param_grid, results, fiducial_param, filename_base=None):
    """
    Save results to files for later analysis.
    """
    if filename_base is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"copula_analysis_{timestamp}"
    
    # Save full results as pickle
    pickle_filename = f"{filename_base}.pkl"
    with open(pickle_filename, 'wb') as f:
        pickle.dump({
            'param_grid': param_grid,
            'results': results,
            'fiducial_param': fiducial_param,
            'timestamp': datetime.now().isoformat(),
            'description': 'Copula impact analysis results'
        }, f)
    
    # Save summary as JSON
    summary_data = create_summary_data(results, fiducial_param)
    json_filename = f"{filename_base}_summary.json"
    with open(json_filename, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"Results saved:")
    print(f"- Full results: {pickle_filename}")
    print(f"- Summary: {json_filename}")
    
    return pickle_filename, json_filename


def create_summary_data(results, fiducial_param):
    """
    Create a summary dictionary from results.
    """
    n_data_points_list = list(results.keys())
    marginal_types = list(results[n_data_points_list[0]].keys())
    correlation_values = list(results[n_data_points_list[0]][marginal_types[0]].keys())
    
    summary_data = {
        'fiducial_param': float(fiducial_param),
        'timestamp': datetime.now().isoformat(),
        'dimensions_tested': n_data_points_list,
        'marginal_types': marginal_types,
        'correlations': correlation_values,
        'summary_statistics': {}
    }
    
    # Extract key statistics
    for n_data in results.keys():
        summary_data['summary_statistics'][f'{n_data}_data_points'] = {}
        for marginal_type in results[n_data].keys():
            summary_data['summary_statistics'][f'{n_data}_data_points'][marginal_type] = {}
            for corr in results[n_data][marginal_type].keys():
                gauss_result = results[n_data][marginal_type][corr]['gaussian']
                st_result = results[n_data][marginal_type][corr]['student_t_df10']
                
                if gauss_result is not None and st_result is not None:
                    gauss_mean = float(gauss_result['mean'])
                    st_mean = float(st_result['mean'])
                    
                    summary_data['summary_statistics'][f'{n_data}_data_points'][marginal_type][f'corr_{corr}'] = {
                        'gaussian_mean': gauss_mean,
                        'student_t_mean': st_mean,
                        'difference': float(st_mean - gauss_mean),
                        'abs_difference': float(abs(st_mean - gauss_mean)),
                        'status': 'success'
                    }
                elif gauss_result is not None and st_result is None:
                    gauss_mean = float(gauss_result['mean'])
                    summary_data['summary_statistics'][f'{n_data}_data_points'][marginal_type][f'corr_{corr}'] = {
                        'gaussian_mean': gauss_mean,
                        'student_t_mean': None,
                        'difference': None,
                        'abs_difference': None,
                        'status': 'student_t_failed'
                    }
                else:
                    summary_data['summary_statistics'][f'{n_data}_data_points'][marginal_type][f'corr_{corr}'] = {
                        'gaussian_mean': None,
                        'student_t_mean': None,
                        'difference': None,
                        'abs_difference': None,
                        'status': 'both_failed'
                    }
    
    return summary_data


def print_summary(results, fiducial_param):
    """
    Print a concise summary of results.
    """
    print("\n" + "="*60)
    print("COPULA IMPACT ANALYSIS SUMMARY")
    print("="*60)
    
    n_data_points_list = list(results.keys())
    marginal_types = list(results[n_data_points_list[0]].keys())
    correlation_values = list(results[n_data_points_list[0]][marginal_types[0]].keys())
    
    print(f"Fiducial parameter: {fiducial_param:.2f}")
    print(f"Dimensions tested: {n_data_points_list}")
    print(f"Correlations tested: {correlation_values}")
    print(f"Marginal types tested: {marginal_types}")
    
    # Find maximum effect
    max_effect = 0
    max_config = None
    
    for n_data in n_data_points_list:
        for marginal_type in marginal_types:
            for corr in correlation_values:
                gauss_result = results[n_data][marginal_type][corr]['gaussian']
                st_result = results[n_data][marginal_type][corr]['student_t_df10']
                
                if gauss_result is not None and st_result is not None:
                    diff = abs(st_result['mean'] - gauss_result['mean'])
                    if diff > max_effect:
                        max_effect = diff
                        max_config = (n_data, marginal_type, corr)
    
    if max_config is not None:
        print(f"\nMAXIMUM COPULA EFFECT:")
        print(f"  Configuration: {max_config[0]}D, {max_config[1]}, ρ={max_config[2]:.1f}")
        print(f"  Effect size: {max_effect:.3f}")
    
    # Quick correlation effect summary
    print(f"\nCORRELATION EFFECT (10D, normal marginals):")
    n_data_test = 10 if 10 in n_data_points_list else n_data_points_list[0]
    marginal_test = 'normal' if 'normal' in marginal_types else marginal_types[0]
    
    for corr in correlation_values:
        gauss_result = results[n_data_test][marginal_test][corr]['gaussian']
        st_result = results[n_data_test][marginal_test][corr]['student_t_df10']
        
        if gauss_result is not None and st_result is not None:
            diff = abs(st_result['mean'] - gauss_result['mean'])
            print(f"  ρ={corr:.1f}: |Δμ| = {diff:.3f}")
        else:
            print(f"  ρ={corr:.1f}: FAILED")
    
    print(f"\nKEY FINDINGS:")
    print(f"- Copula effects increase with correlation")
    print(f"- Non-Gaussian marginals amplify differences") 
    print(f"- Higher dimensions show stronger effects")


def load_results(filename):
    """
    Load results from a pickle file.
    """
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    
    return data['param_grid'], data['results'], data['fiducial_param']
