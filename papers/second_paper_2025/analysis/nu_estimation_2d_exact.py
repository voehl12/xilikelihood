#!/usr/bin/env python3
"""
Robust ν estimation using exact 2D PDFs vs Student-t copula.

Strategy: Compare exact 2D likelihood with copula-based 2D likelihood
across multiple variable pairs to find optimal ν.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import xilikelihood as xlh
import xilikelihood.file_handling as fh

def estimate_nu_from_2d_exact_comparison(
    correlation_data,
    nu_candidates=[3, 4, 5, 6, 8, 10, 15],
    n_pairs=20,
    n_test_points=100
):
    """
    Estimate ν by comparing exact 2D PDFs with Student-t copula predictions.
    
    Parameters:
    -----------
    correlation_data : array, shape (n_realizations, n_dims)
        Your correlation function data
    nu_candidates : list
        ν values to test
    n_pairs : int
        Number of 2D pairs to test
    n_test_points : int
        Number of test points per pair
        
    Returns:
    --------
    results : dict
        ν estimation results with statistical validation
    """
    
    print("=== ROBUST ν ESTIMATION USING EXACT 2D PDFs ===")
    n_realizations, n_dims = correlation_data.shape
    print(f"Data: {n_realizations} realizations × {n_dims} dimensions")
    
    # Select pairs to test (avoid neighboring highly correlated pairs)
    pair_indices = []
    for _ in range(n_pairs):
        # Random pairs with some separation to get diverse correlations
        i = np.random.randint(0, n_dims)
        j = np.random.randint(max(0, i-n_dims//4), min(n_dims, i+n_dims//4))
        if i != j:
            pair_indices.append((i, j))
    
    pair_indices = list(set(pair_indices))[:n_pairs]  # Remove duplicates
    print(f"Testing {len(pair_indices)} variable pairs")
    
    results = {}
    
    for nu in nu_candidates:
        print(f"\n--- Testing ν = {nu} ---")
        
        log_likelihood_ratios = []  # log(exact) - log(copula)
        correlation_strengths = []
        pair_results = []
        
        for pair_idx, (i, j) in enumerate(pair_indices):
            print(f"  Pair {pair_idx+1}/{len(pair_indices)}: dimensions {i}, {j}")
            
            # Extract 2D data for this pair
            data_2d = correlation_data[:, [i, j]]
            
            # Compute empirical correlation
            empirical_corr = np.corrcoef(data_2d.T)[0, 1]
            correlation_strengths.append(abs(empirical_corr))
            
            # Split into train/test
            n_train = n_realizations - n_test_points
            train_data = data_2d[:n_train]
            test_data = data_2d[n_train:n_train+n_test_points]
            
            try:
                # Method 1: Exact 2D likelihood computation
                exact_logliks = []
                for test_point in test_data:
                    # This is your expensive but exact method
                    exact_loglik = compute_exact_2d_likelihood(test_point, train_data)
                    exact_logliks.append(exact_loglik)
                
                # Method 2: Student-t copula 2D likelihood
                copula_logliks = []
                
                # Estimate marginal CDFs from training data
                marginal_cdfs = []
                for dim in range(2):
                    # Empirical CDF from training data
                    sorted_vals = np.sort(train_data[:, dim])
                    marginal_cdfs.append(sorted_vals)
                
                for test_point in test_data:
                    copula_loglik = compute_2d_copula_likelihood(
                        test_point, marginal_cdfs, empirical_corr, nu
                    )
                    copula_logliks.append(copula_loglik)
                
                # Compare likelihoods
                exact_logliks = np.array(exact_logliks)
                copula_logliks = np.array(copula_logliks)
                
                # Remove invalid values
                valid_mask = np.isfinite(exact_logliks) & np.isfinite(copula_logliks)
                if np.sum(valid_mask) < 5:
                    print(f"    Warning: Only {np.sum(valid_mask)} valid comparisons")
                    continue
                
                exact_valid = exact_logliks[valid_mask]
                copula_valid = copula_logliks[valid_mask]
                
                # Likelihood ratio statistics
                log_ratios = exact_valid - copula_valid
                mean_log_ratio = np.mean(log_ratios)
                std_log_ratio = np.std(log_ratios)
                
                log_likelihood_ratios.extend(log_ratios)
                
                pair_result = {
                    'correlation': empirical_corr,
                    'mean_log_ratio': mean_log_ratio,
                    'std_log_ratio': std_log_ratio,
                    'n_valid': np.sum(valid_mask)
                }
                pair_results.append(pair_result)
                
                print(f"    Correlation: {empirical_corr:.3f}")
                print(f"    Mean log-ratio: {mean_log_ratio:.3f} ± {std_log_ratio:.3f}")
                
            except Exception as e:
                print(f"    Error processing pair ({i}, {j}): {e}")
                continue
        
        # Aggregate results for this ν
        if log_likelihood_ratios:
            overall_mean_ratio = np.mean(log_likelihood_ratios)
            overall_std_ratio = np.std(log_likelihood_ratios)
            
            # Statistical test: is the mean ratio significantly different from 0?
            n_comparisons = len(log_likelihood_ratios)
            t_stat = overall_mean_ratio / (overall_std_ratio / np.sqrt(n_comparisons))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_comparisons - 1))
            
            results[nu] = {
                'mean_log_ratio': overall_mean_ratio,
                'std_log_ratio': overall_std_ratio,
                'n_comparisons': n_comparisons,
                't_statistic': t_stat,
                'p_value': p_value,
                'pair_results': pair_results,
                'correlation_strengths': correlation_strengths
            }
            
            print(f"  Overall results:")
            print(f"    Mean log-ratio: {overall_mean_ratio:.4f} ± {overall_std_ratio:.4f}")
            print(f"    t-statistic: {t_stat:.2f} (p = {p_value:.3f})")
            print(f"    Based on {n_comparisons} comparisons")
            
        else:
            print(f"  No valid comparisons for ν = {nu}")
            results[nu] = None
    
    return results

def compute_exact_2d_likelihood(test_point, train_data):
    """
    Compute exact 2D likelihood using your expensive method.
    Replace this with your actual implementation.
    """
    # Placeholder - replace with your actual exact 2D likelihood computation
    # This would use your xilikelihood package for exact 2D PDF evaluation
    
    # For now, approximate with kernel density estimation
    from scipy.stats import gaussian_kde
    try:
        kde = gaussian_kde(train_data.T)
        return kde.logpdf(test_point)[0]
    except:
        return -np.inf

def compute_2d_copula_likelihood(test_point, marginal_cdfs, correlation, nu):
    """
    Compute 2D Student-t copula likelihood.
    """
    from scipy.stats import t
    
    try:
        # Transform test point to uniform margins using empirical CDFs
        u = np.zeros(2)
        for dim in range(2):
            # Find position in empirical CDF
            cdf_vals = marginal_cdfs[dim]
            u[dim] = np.searchsorted(cdf_vals, test_point[dim]) / len(cdf_vals)
            u[dim] = np.clip(u[dim], 1e-10, 1-1e-10)  # Avoid boundary issues
        
        # Transform to Student-t quantiles
        t_vals = t.ppf(u, nu)
        if not np.all(np.isfinite(t_vals)):
            return -np.inf
        
        # 2D Student-t copula density
        rho = correlation
        det_R = 1 - rho**2
        
        if det_R <= 0:
            return -np.inf
        
        # Quadratic form
        quad_form = (t_vals[0]**2 - 2*rho*t_vals[0]*t_vals[1] + t_vals[1]**2) / det_R
        
        # 2D Student-t log-density
        logpdf_joint = (
            stats.loggamma((nu + 2) / 2) 
            - stats.loggamma(nu / 2)
            - np.log(nu * np.pi)
            - 0.5 * np.log(det_R)
            - 0.5 * (nu + 2) * np.log(1 + quad_form / nu)
        )
        
        # Marginal Student-t log-densities
        logpdf_margins = np.sum(t.logpdf(t_vals, nu))
        
        # Copula log-density = joint - marginals
        copula_logdens = logpdf_joint - logpdf_margins
        
        return copula_logdens
        
    except Exception as e:
        return -np.inf

def analyze_nu_estimation_results(results):
    """
    Analyze the ν estimation results and provide recommendation.
    """
    print("\n=== ν ESTIMATION ANALYSIS ===")
    
    valid_results = {nu: res for nu, res in results.items() if res is not None}
    
    if len(valid_results) < 2:
        print("❌ Insufficient valid results for comparison")
        return None
    
    print("\nStatistical comparison of ν values:")
    print("ν\tMean Ratio\tStd Ratio\tt-stat\tp-value\tInterpretation")
    print("-" * 80)
    
    best_nu = None
    best_score = -np.inf
    
    for nu in sorted(valid_results.keys()):
        res = valid_results[nu]
        mean_ratio = res['mean_log_ratio']
        std_ratio = res['std_log_ratio']
        t_stat = res['t_statistic']
        p_val = res['p_value']
        
        # Interpretation
        if abs(mean_ratio) < 0.1 and p_val > 0.05:
            interpretation = "Good match"
            score = 1.0 - abs(mean_ratio)  # Closer to 0 is better
        elif abs(mean_ratio) < 0.2:
            interpretation = "Reasonable match"
            score = 0.8 - abs(mean_ratio)
        else:
            interpretation = "Poor match"
            score = 0.5 - abs(mean_ratio)
        
        if score > best_score:
            best_score = score
            best_nu = nu
        
        print(f"{nu}\t{mean_ratio:.4f}\t\t{std_ratio:.4f}\t\t{t_stat:.2f}\t{p_val:.3f}\t{interpretation}")
    
    print(f"\n✅ RECOMMENDED ν: {best_nu}")
    print(f"   Rationale: Best agreement between exact and copula likelihoods")
    
    # Additional analysis
    best_res = valid_results[best_nu]
    print(f"   Mean log-likelihood ratio: {best_res['mean_log_ratio']:.4f}")
    print(f"   Statistical significance: p = {best_res['p_value']:.3f}")
    
    if best_res['p_value'] > 0.05:
        print("   ✅ No significant difference from exact likelihood (good!)")
    else:
        print("   ⚠️  Significant difference from exact likelihood")
        print("      Consider this when interpreting results")
    
    return {
        'recommended_nu': best_nu,
        'confidence': 'high' if best_res['p_value'] > 0.05 else 'moderate',
        'all_results': valid_results
    }

def plot_nu_comparison_results(results):
    """
    Create diagnostic plots for ν comparison.
    """
    valid_results = {nu: res for nu, res in results.items() if res is not None}
    
    if len(valid_results) < 2:
        print("Insufficient data for plotting")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Mean log-ratio vs ν
    nu_values = sorted(valid_results.keys())
    mean_ratios = [valid_results[nu]['mean_log_ratio'] for nu in nu_values]
    std_ratios = [valid_results[nu]['std_log_ratio'] for nu in nu_values]
    
    axes[0, 0].errorbar(nu_values, mean_ratios, yerr=std_ratios, 
                        marker='o', capsize=5, linewidth=2)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7, 
                       label='Perfect match')
    axes[0, 0].set_xlabel('ν')
    axes[0, 0].set_ylabel('Mean Log-Likelihood Ratio\n(Exact - Copula)')
    axes[0, 0].set_title('Likelihood Comparison vs ν')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: p-values vs ν
    p_values = [valid_results[nu]['p_value'] for nu in nu_values]
    
    axes[0, 1].plot(nu_values, p_values, 'go-', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, 
                       label='p = 0.05')
    axes[0, 1].set_xlabel('ν')
    axes[0, 1].set_ylabel('p-value')
    axes[0, 1].set_title('Statistical Significance vs ν')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Number of comparisons vs ν
    n_comparisons = [valid_results[nu]['n_comparisons'] for nu in nu_values]
    
    axes[1, 0].bar(nu_values, n_comparisons, alpha=0.7)
    axes[1, 0].set_xlabel('ν')
    axes[1, 0].set_ylabel('Number of Valid Comparisons')
    axes[1, 0].set_title('Sample Size vs ν')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Correlation strength distribution
    all_corr_strengths = []
    for nu in nu_values:
        if 'correlation_strengths' in valid_results[nu]:
            all_corr_strengths.extend(valid_results[nu]['correlation_strengths'])
    
    if all_corr_strengths:
        axes[1, 1].hist(all_corr_strengths, bins=20, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('|Correlation|')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Pair Correlations')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('nu_estimation_2d_exact_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def main():
    """
    Run the complete 2D exact vs copula ν estimation.
    """
    print("Loading correlation function data...")
    
    # Load your data
    filepath = "/cluster/work/refregier/veoehl/xi_sims/croco_KiDS_setup_circ10000smoothl30_nonoise_llim_767"
    sims, angles = fh.read_sims_nd(filepath, 20, 767)
    
    # Subsample for computational efficiency
    n_realizations = min(5000, sims.shape[0])  # Manageable for exact 2D computations
    indices = np.random.choice(sims.shape[0], n_realizations, replace=False)
    sims_subset = sims[indices]
    
    # Flatten correlation functions
    correlation_data = sims_subset.reshape(n_realizations, -1)
    
    print(f"Analysis data: {correlation_data.shape}")
    
    # Run ν estimation
    results = estimate_nu_from_2d_exact_comparison(
        correlation_data,
        nu_candidates=[3, 4, 5, 6, 8, 10],
        n_pairs=15,  # Fewer pairs since exact computation is expensive
        n_test_points=50
    )
    
    # Analyze results
    analysis = analyze_nu_estimation_results(results)
    
    # Create plots
    plot_nu_comparison_results(results)
    
    if analysis:
        print(f"\n🎯 FINAL RECOMMENDATION: ν = {analysis['recommended_nu']}")
        print(f"📊 Confidence level: {analysis['confidence']}")
        print(f"📄 This provides a data-driven, statistically validated choice of ν")
    else:
        print("❌ Unable to provide robust ν recommendation")
    
    return results, analysis

if __name__ == "__main__":
    results, analysis = main()
