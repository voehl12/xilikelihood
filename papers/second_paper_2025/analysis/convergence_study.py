#!/usr/bin/env python3
"""
Convergence study for tail dependence analysis.
Systematically test different numbers of realizations to find convergence.
"""

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import xilikelihood.file_handling as fh

# Load simulation data
filepath = "/cluster/work/refregier/veoehl/xi_sims/croco_KiDS_setup_circ10000smoothl30_nonoise_llim_767"
sims, angles = fh.read_sims_nd(filepath, 20, 767)

def generate_correlation_ensemble(n_realizations, n_datapoints, seed=None):
    """Generate ensemble with fixed seed for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
    
    total_realizations, n_cross, n_angular_bins = sims.shape
    total_datapoints = n_cross * n_angular_bins
    
    n_realizations = min(n_realizations, total_realizations)
    n_datapoints = min(n_datapoints, total_datapoints)
    
    # Random selection
    realization_indices = np.random.choice(total_realizations, n_realizations, replace=False)
    
    # Flatten and subsample datapoints
    ensemble = sims[realization_indices].reshape(n_realizations, total_datapoints)
    if n_datapoints < total_datapoints:
        datapoint_indices = np.random.choice(total_datapoints, n_datapoints, replace=False)
        ensemble = ensemble[:, datapoint_indices]
    
    return ensemble

def transform_to_uniform_gaussian(data):
    """Transform data to uniform using Gaussian transformation."""
    standardized = stats.zscore(data, axis=0)
    uniform_data = stats.norm.cdf(standardized)
    return uniform_data, standardized

def compute_tail_dependence_coefficient_fixed(u, v, sigma_threshold=2.0):
    """Compute tail dependence coefficient using standard deviation based extremes."""
    # Convert uniform back to Gaussian for extreme detection
    z_u = stats.norm.ppf(u)
    z_v = stats.norm.ppf(v)
    
    # Define extremes based on standard deviations
    extreme_mask = (z_u > sigma_threshold) & (z_v > sigma_threshold)
    n_extreme = np.sum(extreme_mask)
    
    if n_extreme < 5:  # Need minimum samples
        return 0.0, n_extreme, n_extreme / len(u)
    
    # Among observations where U is extreme, what fraction has V also extreme?
    u_extreme_mask = z_u > sigma_threshold
    n_u_extreme = np.sum(u_extreme_mask)
    
    if n_u_extreme == 0:
        return 0.0, n_extreme, n_extreme / len(u)
    
    # Tail dependence coefficient
    lambda_u = np.sum(z_v[u_extreme_mask] > sigma_threshold) / n_u_extreme
    
    return lambda_u, n_extreme, n_extreme / len(u)

def estimate_nu_from_tail_dependence(lambda_u, rho, method='bisection'):
    """Estimate degrees of freedom ν from empirical upper tail dependence coefficient."""
    from scipy.stats import t
    from scipy.optimize import brentq, minimize_scalar
    
    if lambda_u <= 0 or lambda_u >= 1:
        return np.nan
    
    def tail_dependence_formula(nu, rho):
        """Theoretical upper tail dependence for Student-t copula."""
        if nu <= 0:
            return 0
        try:
            arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
            return 2 * t.cdf(arg, df=nu + 1)
        except:
            return 0
    
    def objective(nu):
        theoretical = tail_dependence_formula(nu, rho)
        return (theoretical - lambda_u) ** 2
    
    try:
        result = minimize_scalar(objective, bounds=(1.0, 50.0), method='bounded')
        return result.x if result.success else np.nan
    except:
        return np.nan

def analyze_single_sample(n_realizations, sigma_threshold=2.0, max_pairs=50, seed=None):
    """Analyze tail dependence for a single sample size."""
    
    # Generate data
    ensemble = generate_correlation_ensemble(n_realizations, n_datapoints=200, seed=seed)
    uniform_data, _ = transform_to_uniform_gaussian(ensemble)
    
    n_variables = uniform_data.shape[1]
    
    # Limit pairwise comparisons
    if n_variables > max_pairs:
        np.random.seed(seed)  # For reproducible pair selection
        pairs = []
        for _ in range(max_pairs):
            i, j = np.random.choice(n_variables, 2, replace=False)
            pairs.append((i, j))
    else:
        pairs = [(i, j) for i in range(n_variables) for j in range(i+1, n_variables)]
    
    lambda_coeffs = []
    extreme_counts = []
    extreme_fractions = []
    correlations = []
    nu_estimates = []
    
    for i, j in pairs:
        u, v = uniform_data[:, i], uniform_data[:, j]
        
        # Compute correlation
        rho = np.corrcoef(u, v)[0, 1]
        
        # Compute tail dependence
        lambda_u, n_extreme, frac_extreme = compute_tail_dependence_coefficient_fixed(
            u, v, sigma_threshold
        )
        
        # Estimate ν if possible
        if lambda_u > 0 and not np.isnan(rho):
            nu_est = estimate_nu_from_tail_dependence(lambda_u, rho)
        else:
            nu_est = np.nan
        
        lambda_coeffs.append(lambda_u)
        extreme_counts.append(n_extreme)
        extreme_fractions.append(frac_extreme)
        correlations.append(rho)
        nu_estimates.append(nu_est)
    
    # Compute summary statistics
    results = {
        'n_realizations': n_realizations,
        'lambda_mean': np.mean(lambda_coeffs),
        'lambda_std': np.std(lambda_coeffs),
        'extreme_count_mean': np.mean(extreme_counts),
        'extreme_fraction_mean': np.mean(extreme_fractions),
        'nu_estimates': np.array(nu_estimates),
        'nu_mean': np.nanmean(nu_estimates),
        'nu_std': np.nanstd(nu_estimates),
        'nu_valid_fraction': np.sum(~np.isnan(nu_estimates)) / len(nu_estimates),
        'n_pairs': len(pairs)
    }
    
    return results

def convergence_study(sample_sizes=None, n_bootstrap=5, sigma_threshold=2.0):
    """
    Run convergence study across different sample sizes.
    
    Parameters:
    -----------
    sample_sizes : list
        Sample sizes to test
    n_bootstrap : int
        Number of bootstrap samples for each size
    sigma_threshold : float
        Standard deviation threshold for extremes
    """
    
    if sample_sizes is None:
        # Start small and go up to available data
        total_available = sims.shape[0]
        sample_sizes = [100, 200, 500, 1000, 2000, 5000, min(10000, total_available)]
        # Remove sizes larger than available
        sample_sizes = [s for s in sample_sizes if s <= total_available]
    
    print(f"=== CONVERGENCE STUDY ===")
    print(f"Testing sample sizes: {sample_sizes}")
    print(f"Bootstrap samples per size: {n_bootstrap}")
    print(f"Extreme threshold: {sigma_threshold}σ")
    
    convergence_results = []
    
    for n_real in sample_sizes:
        print(f"\nTesting {n_real} realizations...")
        
        bootstrap_results = []
        for boot in range(n_bootstrap):
            seed = 42 + boot  # Reproducible but different seeds
            result = analyze_single_sample(n_real, sigma_threshold, max_pairs=50, seed=seed)
            bootstrap_results.append(result)
        
        # Aggregate bootstrap results
        lambda_means = [r['lambda_mean'] for r in bootstrap_results]
        nu_means = [r['nu_mean'] for r in bootstrap_results if not np.isnan(r['nu_mean'])]
        nu_valid_fractions = [r['nu_valid_fraction'] for r in bootstrap_results]
        extreme_fractions = [r['extreme_fraction_mean'] for r in bootstrap_results]
        
        summary = {
            'n_realizations': n_real,
            'lambda_mean': np.mean(lambda_means),
            'lambda_std': np.std(lambda_means),
            'lambda_bootstrap_std': np.std(lambda_means),  # Uncertainty from finite sample
            'nu_mean': np.nanmean(nu_means) if nu_means else np.nan,
            'nu_std': np.nanstd(nu_means) if nu_means else np.nan,
            'nu_bootstrap_std': np.nanstd(nu_means) if nu_means else np.nan,
            'nu_valid_fraction': np.mean(nu_valid_fractions),
            'extreme_fraction_mean': np.mean(extreme_fractions),
            'n_bootstrap_with_nu': len(nu_means)
        }
        
        convergence_results.append(summary)
        
        print(f"  λ_U: {summary['lambda_mean']:.3f} ± {summary['lambda_bootstrap_std']:.3f}")
        if not np.isnan(summary['nu_mean']):
            print(f"  ν: {summary['nu_mean']:.1f} ± {summary['nu_bootstrap_std']:.1f}")
        print(f"  ν success rate: {summary['nu_valid_fraction']:.1%}")
        print(f"  Extreme fraction: {summary['extreme_fraction_mean']:.3f}")
    
    return convergence_results

def assess_convergence(convergence_results):
    """Assess when estimates have converged."""
    print("\n=== CONVERGENCE ASSESSMENT ===")
    
    sample_sizes = [r['n_realizations'] for r in convergence_results]
    lambda_means = [r['lambda_mean'] for r in convergence_results]
    lambda_uncertainties = [r['lambda_bootstrap_std'] for r in convergence_results]
    nu_means = [r['nu_mean'] for r in convergence_results]
    nu_uncertainties = [r['nu_bootstrap_std'] for r in convergence_results]
    
    # Find where uncertainty drops below thresholds
    lambda_converged_idx = None
    nu_converged_idx = None
    
    for i, (n_real, lambda_unc) in enumerate(zip(sample_sizes, lambda_uncertainties)):
        if lambda_unc < 0.05:  # 5% uncertainty in λ
            lambda_converged_idx = i
            break
    
    for i, (n_real, nu_unc) in enumerate(zip(sample_sizes, nu_uncertainties)):
        if not np.isnan(nu_unc) and nu_unc < 1.0:  # 1 unit uncertainty in ν
            nu_converged_idx = i
            break
    
    print("Convergence criteria:")
    print("  λ_U uncertainty < 0.05")
    print("  ν uncertainty < 1.0")
    
    if lambda_converged_idx is not None:
        n_conv = sample_sizes[lambda_converged_idx]
        print(f"✅ λ_U converged at ~{n_conv} realizations")
    else:
        print("❌ λ_U not yet converged - need more data")
    
    if nu_converged_idx is not None:
        n_conv = sample_sizes[nu_converged_idx]
        print(f"✅ ν converged at ~{n_conv} realizations")
    else:
        print("❌ ν not yet converged - need more data")
    
    # Recommendations
    max_available = max(sample_sizes)
    print(f"\nRECOMMENDATIONS:")
    
    if lambda_converged_idx and nu_converged_idx:
        recommended = max(sample_sizes[lambda_converged_idx], sample_sizes[nu_converged_idx])
        print(f"✅ Use {recommended}+ realizations for stable estimates")
        
        final_lambda = lambda_means[lambda_converged_idx]
        final_nu = nu_means[nu_converged_idx] if not np.isnan(nu_means[nu_converged_idx]) else None
        
        if final_nu:
            print(f"✅ Final estimates: λ_U ≈ {final_lambda:.3f}, ν ≈ {final_nu:.1f}")
    else:
        if max_available >= 10000:
            print("❌ Need more than 10k realizations - consider:")
            print("  • Different extreme threshold (try 1.5σ)")
            print("  • More datapoints per realization")
            print("  • Check if tail dependence exists")
        else:
            print(f"📈 Try up to {min(20000, sims.shape[0])} realizations if available")

def plot_convergence(convergence_results):
    """Plot convergence behavior."""
    try:
        sample_sizes = [r['n_realizations'] for r in convergence_results]
        lambda_means = [r['lambda_mean'] for r in convergence_results]
        lambda_uncertainties = [r['lambda_bootstrap_std'] for r in convergence_results]
        nu_means = [r['nu_mean'] for r in convergence_results]
        nu_uncertainties = [r['nu_bootstrap_std'] for r in convergence_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Lambda convergence
        ax = axes[0, 0]
        ax.errorbar(sample_sizes, lambda_means, yerr=lambda_uncertainties, 
                   marker='o', capsize=5)
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Target uncertainty')
        ax.set_xlabel('Number of Realizations')
        ax.set_ylabel('λ_U')
        ax.set_title('Upper Tail Dependence Convergence')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Lambda uncertainty
        ax = axes[0, 1]
        ax.plot(sample_sizes, lambda_uncertainties, marker='s', color='orange')
        ax.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Target < 0.05')
        ax.set_xlabel('Number of Realizations')
        ax.set_ylabel('λ_U Bootstrap Uncertainty')
        ax.set_title('λ_U Uncertainty vs Sample Size')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Nu convergence (skip NaN values)
        ax = axes[1, 0]
        valid_nu = [(s, nu, err) for s, nu, err in zip(sample_sizes, nu_means, nu_uncertainties) 
                   if not np.isnan(nu) and not np.isnan(err)]
        if valid_nu:
            s_valid, nu_valid, err_valid = zip(*valid_nu)
            ax.errorbar(s_valid, nu_valid, yerr=err_valid, 
                       marker='^', capsize=5, color='green')
        ax.set_xlabel('Number of Realizations')
        ax.set_ylabel('ν')
        ax.set_title('Degrees of Freedom Convergence')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Nu uncertainty
        ax = axes[1, 1]
        if valid_nu:
            ax.plot(s_valid, err_valid, marker='d', color='purple')
            ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Target < 1.0')
        ax.set_xlabel('Number of Realizations')
        ax.set_ylabel('ν Bootstrap Uncertainty')
        ax.set_title('ν Uncertainty vs Sample Size')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('tail_dependence_convergence.png', dpi=150, bbox_inches='tight')
        plt.show()
        
    except Exception as e:
        print(f"Plotting failed: {e}")

def main():
    """Run the convergence study."""
    
    # Check available data
    total_realizations = sims.shape[0]
    print(f"Total available realizations: {total_realizations}")
    
    # Run convergence study
    results = convergence_study(
        sample_sizes=[100, 200, 500, 1000, 2000, 5000, min(10000, total_realizations)],
        n_bootstrap=100,  # Reduce for speed
        sigma_threshold=3.0
    )
    
    # Assess convergence
    assess_convergence(results)
    
    # Plot results
    plot_convergence(results)
    
    return results

if __name__ == "__main__":
    convergence_results = main()
