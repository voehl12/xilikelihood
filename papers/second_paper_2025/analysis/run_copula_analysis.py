#!/usr/bin/env python3
"""
Clean, focused copula impact analysis script.

This is the main script that runs the copula analysis in a structured way.
Run with --fast for a quick test, or --full for complete analysis.
"""

import sys
import argparse
import os
from copula_analysis import run_copula_analysis, get_default_config, get_fast_config
from copula_plotting import plot_all_results
from copula_utils import save_results, print_summary
from copula_tests import test_maximum_likelihood_data_generation, plot_ml_data_test_results, test_model_consistency


def main():
    """
    Main function to run the copula impact analysis.
    """
    parser = argparse.ArgumentParser(description='Run copula impact analysis')
    parser.add_argument('--mode', choices=['fast', 'full', 'test'], default='full',
                       help='Analysis mode: fast for testing, full for complete analysis, test for ML data tests')
    parser.add_argument('--output', default='.', help='Output directory for plots and results')
    parser.add_argument('--save-results', action='store_true', help='Save results to files')
    parser.add_argument('--test-type', choices=['ml-data', 'consistency', 'both'], default='both',
                       help='Type of test to run (only used with --mode test)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("COPULA IMPACT ANALYSIS")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Handle test mode
    if args.mode == 'test':
        print("Running tests...")
        config = get_fast_config()  # Use fast config for tests
        
        if args.test_type in ['ml-data', 'both']:
            print("\n" + "="*50)
            print("MAXIMUM LIKELIHOOD DATA TEST")
            print("="*50)
            
            # Test with gamma marginals (most interesting case)
            ml_results = test_maximum_likelihood_data_generation(
                fiducial_param=config['fiducial_param'],
                n_data_points=10,
                correlation=0.7,
                marginal_type='gamma',
                param_grid=config['param_grid']
            )
            
            # Plot results
            save_path = os.path.join(args.output, 'ml_data_test.png')
            plot_ml_data_test_results(ml_results, config['fiducial_param'], 
                                    config['param_grid'], save_path)
        
        if args.test_type in ['consistency', 'both']:
            print("\n" + "="*50)
            print("MODEL CONSISTENCY TEST")
            print("="*50)
            
            consistency_results = test_model_consistency(config, n_realizations=20)
        
        print(f"\n✓ Tests completed!")
        print(f"📊 Test outputs saved to: {args.output}")
        return
    
    # Get configuration
    if args.mode == 'fast':
        config = get_fast_config()
        print("Using fast configuration (for testing)")
    else:
        config = get_default_config()
        print("Using full configuration")
    
    print(f"Fiducial parameter: {config['fiducial_param']}")
    print(f"Parameter grid: {len(config['param_grid'])} points")
    print(f"Correlations: {config['correlation_values']}")
    print(f"Marginals: {config['marginal_types']}")
    print(f"Dimensions: {config['n_data_points_list']}")
    
    # Run analysis
    print("\nRunning analysis...")
    results = run_copula_analysis(config)
    
    # Save results if requested
    if args.save_results:
        print("\nSaving results...")
        save_results(config['param_grid'], results, config['fiducial_param'])
    
    # Create plots
    print("\nCreating plots...")
    figures = plot_all_results(config['param_grid'], results, config['fiducial_param'], 
                             output_dir=args.output)
    
    # Print summary
    print_summary(results, config['fiducial_param'])
    
    print(f"\n✓ Analysis completed successfully!")
    
    if args.save_results:
        print(f"- copula_analysis_*.pkl (full results)")
        print(f"- copula_analysis_*_summary.json (summary data)")


if __name__ == "__main__":
    main()
