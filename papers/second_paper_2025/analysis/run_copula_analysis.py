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


def main():
    """
    Main function to run the copula impact analysis.
    """
    parser = argparse.ArgumentParser(description='Run copula impact analysis')
    parser.add_argument('--mode', choices=['fast', 'full'], default='full',
                       help='Analysis mode: fast for testing, full for complete analysis')
    parser.add_argument('--output', default='.', help='Output directory for plots and results')
    parser.add_argument('--save-results', action='store_true', help='Save results to files')
  
    
    args = parser.parse_args()
    
    print("="*60)
    print("COPULA IMPACT ANALYSIS")
    print("="*60)
    print(f"Mode: {args.mode}")
    print(f"Output directory: {args.output}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
       
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
