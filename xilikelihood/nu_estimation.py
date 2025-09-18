"""
Student-t copula parameter estimation using exact likelihood comparison.

This module provides functions to estimate the degrees of freedom (ν) parameter
for Student-t copulas by comparing copula-based likelihood with exact 2D likelihood
computations. This is essential for users who want to "get the coupling right"
in their cosmological analyses.
"""

import numpy as np
from scipy import stats
from .exact_2d import Exact2DLikelihood


def estimate_nu_from_exact_2d(xi_likelihood, correlation_data, 
                             nu_candidates=[3, 4, 5, 6, 8, 10], 
                             n_pairs=10, n_test_points=50,
                             cross_validation_folds=3,
                             random_seed=42):
    """
    Estimate Student-t copula ν by comparing with exact 2D likelihood.
    
    This function provides a data-driven approach to selecting the degrees of
    freedom parameter for Student-t copulas by comparing the copula-based
    likelihood with exact 2D likelihood computations on multiple pairs of
    correlation functions.
    
    Parameters:
    -----------
    xi_likelihood : XiLikelihood
        Configured likelihood instance
    correlation_data : array, shape (n_samples, n_dims)
        Correlation function simulation data
    nu_candidates : list, default [3, 4, 5, 6, 8, 10]
        ν values to test
    n_pairs : int, default 10
        Number of most-correlated pairs to test
    n_test_points : int, default 50
        Number of test points per pair for cross-validation
    cross_validation_folds : int, default 3
        Number of cross-validation folds
    random_seed : int, default 42
        Random seed for reproducibility
        
    Returns:
    --------
    best_nu : float
        Best ν estimate based on likelihood comparison
    results : dict
        Detailed comparison results with statistical validation
        
    Examples:
    ---------
    >>> import xilikelihood as xlh
    >>> likelihood = xlh.XiLikelihood(...)  # Configure your likelihood
    >>> best_nu, results = xlh.estimate_nu_from_exact_2d(
    ...     likelihood, your_simulation_data
    ... )
    >>> print(f"Recommended ν: {best_nu}")
    """
    np.random.seed(random_seed)
    
    print("=== STUDENT-T COPULA ν ESTIMATION ===")
    print(f"Testing ν candidates: {nu_candidates}")
    print(f"Using {n_pairs} correlation function pairs")
    print(f"Cross-validation with {n_test_points} test points per pair")
    
    # Validate inputs
    n_samples, n_dims = correlation_data.shape
    if n_pairs > n_dims * (n_dims - 1) // 2:
        n_pairs = n_dims * (n_dims - 1) // 2
        print(f"Reduced n_pairs to {n_pairs} (maximum possible)")
    
    # Create exact 2D likelihood computer
    try:
        exact_2d = Exact2DLikelihood(xi_likelihood)
    except Exception as e:
        raise RuntimeError(f"Failed to create Exact2DLikelihood: {e}")
    
    # Get most correlated pairs for testing
    high_corr_pairs = _get_top_correlated_pairs(correlation_data, n_pairs)
    print(f"Selected {len(high_corr_pairs)} pairs with highest correlations")
    
    results = {}
    
    for nu in nu_candidates:
        print(f"\n--- Testing ν = {nu} ---")
        
        all_differences = []
        pair_results = []
        
        for pair_idx, (i, j) in enumerate(high_corr_pairs):
            print(f"  Pair {pair_idx+1}/{len(high_corr_pairs)}: dims {i}, {j}")
            
            try:
                pair_diffs = _compare_pair_likelihoods(
                    xi_likelihood, exact_2d, correlation_data, 
                    i, j, nu, n_test_points, cross_validation_folds
                )
                
                if pair_diffs:
                    all_differences.extend(pair_diffs)
                    pair_results.append({
                        'indices': (i, j),
                        'n_comparisons': len(pair_diffs),
                        'mean_diff': np.mean(pair_diffs),
                        'std_diff': np.std(pair_diffs)
                    })
                    print(f"    Mean difference: {np.mean(pair_diffs):.4f} ± {np.std(pair_diffs):.4f}")
                else:
                    print(f"    Warning: No valid comparisons for pair ({i}, {j})")
                    
            except Exception as e:
                print(f"    Error processing pair ({i}, {j}): {e}")
                continue
        
        # Aggregate results for this ν
        if all_differences:
            mean_diff = np.mean(all_differences)
            std_diff = np.std(all_differences)
            n_comparisons = len(all_differences)
            
            # Statistical significance test
            if std_diff > 0:
                t_stat = mean_diff / (std_diff / np.sqrt(n_comparisons))
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_comparisons - 1))
            else:
                t_stat = 0
                p_value = 1.0
            
            results[nu] = {
                'mean_difference': mean_diff,
                'std_difference': std_diff,
                'n_comparisons': n_comparisons,
                't_statistic': t_stat,
                'p_value': p_value,
                'pair_results': pair_results,
                'all_differences': all_differences
            }
            
            print(f"  Overall: {mean_diff:.4f} ± {std_diff:.4f} ({n_comparisons} comparisons)")
            print(f"  Statistical test: t = {t_stat:.2f}, p = {p_value:.3f}")
            
        else:
            print(f"  No valid comparisons for ν = {nu}")
            results[nu] = None
    
    # Analyze results and select best ν
    analysis = _analyze_nu_results(results)
    
    if analysis:
        best_nu = analysis['best_nu']
        print(f"\n✅ RECOMMENDED ν: {best_nu}")
        print(f"📊 Confidence: {analysis['confidence']}")
        print(f"📝 Justification: {analysis['justification']}")
        return best_nu, results
    else:
        print("\n❌ Unable to determine optimal ν from available data")
        return None, results


def _compare_pair_likelihoods(xi_likelihood, exact_2d, correlation_data, 
                             i, j, nu, n_test_points, cv_folds):
    """
    Compare exact vs copula likelihood for one correlation function pair.
    
    Uses cross-validation to ensure robust comparison.
    """
    data_2d = correlation_data[:, [i, j]]
    n_samples = len(data_2d)
    
    if n_samples < n_test_points:
        n_test_points = n_samples // 2
    
    differences = []
    
    # Cross-validation to avoid overfitting
    fold_size = n_samples // cv_folds
    
    for fold in range(cv_folds):
        # Split data
        start_idx = fold * fold_size
        end_idx = min((fold + 1) * fold_size, n_samples)
        
        test_indices = list(range(start_idx, end_idx))
        train_indices = list(range(0, start_idx)) + list(range(end_idx, n_samples))
        
        # Sample test points from this fold
        fold_test_points = min(n_test_points // cv_folds, len(test_indices))
        if fold_test_points < 5:  # Need minimum points
            continue
            
        selected_test_indices = np.random.choice(
            test_indices, fold_test_points, replace=False
        )
        
        for idx in selected_test_indices:
            test_point = data_2d[idx]
            
            try:
                # Exact 2D likelihood
                exact_loglik = exact_2d.compute_2d_likelihood(test_point, (i, j))
                
                # Student-t copula likelihood 
                # Note: This assumes your XiLikelihood can be configured with different ν
                # You may need to adjust this based on your actual interface
                copula_loglik = _compute_copula_likelihood(
                    xi_likelihood, test_point, nu, train_indices, data_2d
                )
                
                # Store difference (exact - copula)
                if np.isfinite(exact_loglik) and np.isfinite(copula_loglik):
                    differences.append(exact_loglik - copula_loglik)
                
            except Exception as e:
                # Skip problematic points
                continue
    
    return differences


def _compute_copula_likelihood(xi_likelihood, test_point, nu, train_indices, data_2d):
    """
    Compute Student-t copula likelihood for a test point.
    
    Note: This is a placeholder that needs to be adapted based on your
    actual XiLikelihood interface for Student-t copulas.
    """
    # This function needs to be implemented based on how your xilikelihood
    # handles Student-t copula configuration. Some possibilities:
    
    # Option 1: If you can reconfigure the likelihood
    # xi_likelihood.configure_student_t(nu=nu)
    # return xi_likelihood.loglikelihood(test_point)
    
    # Option 2: If you have a separate method
    # return xi_likelihood.student_t_loglikelihood(test_point, nu=nu)
    
    # Option 3: If you need to create a new instance
    # config = xi_likelihood.get_config()
    # config.update({'copula_type': 'student_t', 'student_t_dof': nu})
    # temp_likelihood = XiLikelihood(config)
    # return temp_likelihood.loglikelihood(test_point)
    
    # For now, return a placeholder
    # You'll need to replace this with the actual implementation
    print(f"Warning: _compute_copula_likelihood needs implementation for your XiLikelihood interface")
    return 0.0


def _get_top_correlated_pairs(correlation_data, n_pairs):
    """
    Get pairs of correlation functions with highest absolute correlation.
    
    Focus on strongly correlated pairs as they're most sensitive to ν choice.
    """
    corr_matrix = np.corrcoef(correlation_data.T)
    n_dims = corr_matrix.shape[0]
    
    # Get all possible pairs with their correlation strengths
    pairs = []
    for i in range(n_dims):
        for j in range(i+1, n_dims):
            corr_strength = abs(corr_matrix[i, j])
            pairs.append((i, j, corr_strength))
    
    # Sort by correlation strength (highest first)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Return top n_pairs
    selected_pairs = [(i, j) for i, j, _ in pairs[:n_pairs]]
    
    print(f"Top correlation strengths: {[corr for _, _, corr in pairs[:min(5, len(pairs))]]}")
    
    return selected_pairs


def _analyze_nu_results(results):
    """
    Analyze ν estimation results and provide recommendation.
    """
    valid_results = {nu: res for nu, res in results.items() if res is not None}
    
    if len(valid_results) < 2:
        print("Insufficient valid results for ν comparison")
        return None
    
    print("\n=== ν COMPARISON ANALYSIS ===")
    print("ν\tMean Diff\tStd Diff\tt-stat\tp-value\tN_comp\tInterpretation")
    print("-" * 85)
    
    best_nu = None
    best_score = np.inf
    
    interpretations = {}
    
    for nu in sorted(valid_results.keys()):
        res = valid_results[nu]
        mean_diff = res['mean_difference']
        std_diff = res['std_difference']
        t_stat = res['t_statistic']
        p_val = res['p_value']
        n_comp = res['n_comparisons']
        
        # Score based on how close to zero the mean difference is
        # and statistical significance
        score = abs(mean_diff)
        if p_val > 0.05:  # Not significantly different from zero
            interpretation = "Good match"
            score *= 0.5  # Bonus for statistical insignificance
        elif abs(mean_diff) < 0.1:
            interpretation = "Reasonable match"
        else:
            interpretation = "Poor match"
        
        interpretations[nu] = interpretation
        
        if score < best_score and n_comp >= 20:  # Need sufficient comparisons
            best_score = score
            best_nu = nu
        
        print(f"{nu}\t{mean_diff:.4f}\t\t{std_diff:.4f}\t\t{t_stat:.2f}\t{p_val:.3f}\t{n_comp}\t{interpretation}")
    
    if best_nu is None:
        return None
    
    best_res = valid_results[best_nu]
    confidence = "high" if best_res['p_value'] > 0.05 else "moderate"
    
    return {
        'best_nu': best_nu,
        'confidence': confidence,
        'justification': f"Best agreement with exact 2D likelihood (mean diff: {best_res['mean_difference']:.4f})",
        'all_interpretations': interpretations
    }


def plot_nu_comparison(results, save_path=None):
    """
    Create diagnostic plots for ν comparison results.
    
    Parameters:
    -----------
    results : dict
        Results from estimate_nu_from_exact_2d
    save_path : str, optional
        Path to save the plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available for plotting")
        return None
    
    valid_results = {nu: res for nu, res in results.items() if res is not None}
    
    if len(valid_results) < 2:
        print("Insufficient data for plotting")
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    nu_values = sorted(valid_results.keys())
    
    # Plot 1: Mean differences vs ν
    mean_diffs = [valid_results[nu]['mean_difference'] for nu in nu_values]
    std_diffs = [valid_results[nu]['std_difference'] for nu in nu_values]
    
    axes[0, 0].errorbar(nu_values, mean_diffs, yerr=std_diffs, 
                        marker='o', capsize=5, linewidth=2, markersize=8)
    axes[0, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7, 
                       label='Perfect match')
    axes[0, 0].set_xlabel('ν (degrees of freedom)')
    axes[0, 0].set_ylabel('Mean Log-Likelihood Difference\n(Exact - Copula)')
    axes[0, 0].set_title('Likelihood Agreement vs ν')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: p-values vs ν
    p_values = [valid_results[nu]['p_value'] for nu in nu_values]
    
    axes[0, 1].plot(nu_values, p_values, 'go-', linewidth=2, markersize=8)
    axes[0, 1].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, 
                       label='p = 0.05')
    axes[0, 1].set_xlabel('ν (degrees of freedom)')
    axes[0, 1].set_ylabel('p-value')
    axes[0, 1].set_title('Statistical Significance vs ν')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # Plot 3: Number of comparisons vs ν
    n_comparisons = [valid_results[nu]['n_comparisons'] for nu in nu_values]
    
    axes[1, 0].bar(nu_values, n_comparisons, alpha=0.7, color='skyblue')
    axes[1, 0].set_xlabel('ν (degrees of freedom)')
    axes[1, 0].set_ylabel('Number of Valid Comparisons')
    axes[1, 0].set_title('Sample Size vs ν')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Distribution of differences for best ν
    best_nu = min(valid_results.keys(), 
                  key=lambda nu: abs(valid_results[nu]['mean_difference']))
    best_diffs = valid_results[best_nu]['all_differences']
    
    axes[1, 1].hist(best_diffs, bins=20, alpha=0.7, edgecolor='black', color='lightcoral')
    axes[1, 1].axvline(x=0, color='red', linestyle='--', alpha=0.7, 
                       label='Perfect match')
    axes[1, 1].set_xlabel('Log-Likelihood Difference (Exact - Copula)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Distribution of Differences (ν = {best_nu})')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()
    return fig


# Convenience function for quick ν estimation
def quick_nu_estimate(xi_likelihood, correlation_data, nu_candidates=[3, 5, 8, 10]):
    """
    Quick ν estimation with default settings for rapid assessment.
    
    Parameters:
    -----------
    xi_likelihood : XiLikelihood
        Configured likelihood instance
    correlation_data : array
        Correlation function simulation data
    nu_candidates : list, default [3, 5, 8, 10]
        Reduced set of ν values for quick testing
        
    Returns:
    --------
    best_nu : float
        Recommended ν value
    """
    print("Running quick ν estimation...")
    best_nu, _ = estimate_nu_from_exact_2d(
        xi_likelihood, 
        correlation_data,
        nu_candidates=nu_candidates,
        n_pairs=5,  # Fewer pairs for speed
        n_test_points=25,  # Fewer test points
        cross_validation_folds=2  # Fewer folds
    )
    return best_nu
