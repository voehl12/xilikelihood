#!/usr/bin/env python3
"""
Collect and plot results from job array runs of s8_copula_comparison.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def collect_job_results(output_dir, correlation_type, n_jobs, exclude_jobs=None):
    """Collect results from individual job files."""
    output_dir = Path(output_dir)
    all_results = []
    
    if exclude_jobs is None:
        exclude_jobs = []
    
    for job_id in range(n_jobs):
        if job_id in exclude_jobs:
            print(f"Skipping job {job_id} (excluded)")
            continue
            
        job_file = output_dir / f"s8_copula_comparison_{correlation_type}_job{job_id}.npz"
        
        if job_file.exists():
            print(f"Loading results from job {job_id}: {job_file}")
            data = np.load(job_file, allow_pickle=True)
            
            # Debug: print what keys are available
            print(f"  Available keys: {list(data.keys())}")
            
            # Check if 'results' key exists (from combined files) or individual result fields
            if 'results' in data:
                # This is a combined results file
                results = data['results'].item()  # Convert back from numpy array
                if isinstance(results, list):
                    all_results.extend(results)
                else:
                    all_results.append(results)
            else:
                # This is an individual result file saved with **result
                # Reconstruct the result dictionary from the saved fields
                result = {}
                for key in data.keys():
                    result[key] = data[key].item() if data[key].ndim == 0 and data[key].dtype == object else data[key]
                all_results.append(result)
                
        else:
            print(f"Warning: Job {job_id} file not found: {job_file}")
    
    if not all_results:
        print("No job results found!")
        return None
    
    print(f"Debug: First result keys: {list(all_results[0].keys()) if all_results else 'None'}")
    print(f"Debug: First result type: {type(all_results[0]) if all_results else 'None'}")
    
    # Sort by number of datapoints
    all_results.sort(key=lambda x: x['n_datapoints'])
    
    print(f"Collected {len(all_results)} results")
    for i, result in enumerate(all_results):
        print(f"  Result {i}: {result['n_datapoints']} datapoints, {result['correlation_type']} correlations")
        # Also print bin configuration
        if 'n_redshift_bins' in result and 'n_angular_bins' in result:
            print(f"    Bins: {result['n_redshift_bins']} redshift × {result['n_angular_bins']} angular")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description="Collect and plot job array results")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory containing job results")
    parser.add_argument("--correlation-type", type=str, default="all",
                        help="Correlation type used in jobs")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="Number of jobs in the array")
    parser.add_argument("--exclude-jobs", nargs="+", type=int, default=[1],
                        help="Job indices to exclude (default: [1])")
    
    args = parser.parse_args()
    
    print(f"Excluding jobs: {args.exclude_jobs}")
    
    # Collect results
    all_results = collect_job_results(args.output_dir, args.correlation_type, args.n_jobs, args.exclude_jobs)
    
    if all_results is None:
        return 1
    
    # Import plotting function from main script
    import sys
    sys.path.insert(0, '.')
    from s8_copula_comparison import plot_s8_comparison
    
    # Create combined plot
    output_dir = Path(args.output_dir)
    plot_path = output_dir / f"s8_copula_comparison_{args.correlation_type}_combined.png"
    plot_s8_comparison(all_results, save_path=plot_path)
    
    # Save combined results
    combined_path = output_dir / f"s8_copula_comparison_{args.correlation_type}_combined.npz"
    np.savez(combined_path, results=all_results)
    
    print(f"✓ Combined plot saved: {plot_path}")
    print(f"✓ Combined results saved: {combined_path}")
    
    # Print summary
    print(f"\nSUMMARY:")
    for result in all_results:
        n_data = result['n_datapoints']
        
        # Print bin configuration
        if 'redshift_bins' in result and 'ang_bins_in_deg' in result:
            print(f"{n_data}dp Configuration:")
            print(f"  Redshift bins ({result['n_redshift_bins']}): {result['redshift_bins']}")
            print(f"  Angular bins ({result['n_angular_bins']}): {result['ang_bins_in_deg']}")
        
        if result['results']['gaussian'] is not None:
            gauss = result['results']['gaussian']
            print(f"{n_data}dp Gaussian: Mean={gauss['mean']:.4f}, σ={gauss['sigma']:.4f}")
        
        if result['results']['student_t'] is not None:
            studt = result['results']['student_t']
            print(f"{n_data}dp Student-t: Mean={studt['mean']:.4f}, σ={studt['sigma']:.4f}")
            
        if result['results']['gaussian'] is not None and result['results']['student_t'] is not None:
            mean_diff = studt['mean'] - gauss['mean']
            sigma_diff = studt['sigma'] - gauss['sigma']
            print(f"{n_data}dp Difference: ΔMean={mean_diff:+.4f}, Δσ={sigma_diff:+.4f}")
        print()

if __name__ == "__main__":
    main()
