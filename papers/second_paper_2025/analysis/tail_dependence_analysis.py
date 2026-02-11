#!/usr/bin/env python3
"""
Tail Dependence Analysis for Copula Selection in Cosmological Parameter Inference

This module implements tail dependence analysis to provide data-driven guidance
for choosing between Gaussian and Student-t copulas in cosmological correlation
function analysis.

Key concepts:
- Tail dependence measures whether extreme values in one variable tend to occur
  with extreme values in another variable
- Gaussian copulas have zero tail dependence (λ = 0)
- Student-t copulas have symmetric positive tail dependence (λ > 0)

Usage:
    python tail_dependence_analysis.py --config config.yaml --n-realizations 1000
"""

import numpy as np
import logging
from scipy import stats
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
from time import time
import argparse

import xilikelihood as xlh
from xilikelihood.core_utils import computation_phase, logging_context
from config import (
    EXACT_LMAX, 
    FIDUCIAL_COSMO, 
    DATA_DIR, 
    OUTPUT_DIR, 
    MASK_CONFIG,
    REDSHIFT_BINS_PATH,
    PACKAGE_DIR
)

logger = logging.getLogger(__name__)


class TailDependenceAnalyzer:
    """
    Analyze tail dependence structure in correlation function ensembles
    to guide copula selection for cosmological parameter inference.
    """
    
    def __init__(self, likelihood_config=None):
        """
        Initialize analyzer with likelihood configuration.
        
        Parameters:
        -----------
        likelihood_config : dict, optional
            Configuration for likelihood setup (redshift bins, angular bins, etc.)
        """
        self.likelihood_config = likelihood_config or {}
        self.ensemble_data = None
        self.uniform_data = None
        self.tail_results = None
        self.empirical_cov = None
        self.ensemble_config = None
    
    def generate_or_load_ensemble(self, n_realizations=1000, output_dir="tail_analysis_data", 
                                force_regenerate=False, correlation_type="all"):
        """
        Generate or load ensemble of correlation function realizations using
        xilikelihood's mock data generation infrastructure.
        
        Parameters:
        -----------
        n_realizations : int
            Number of correlation function realizations to generate
        output_dir : str or Path
            Directory to save/load ensemble data
        force_regenerate : bool
            If True, regenerate even if cached data exists
        correlation_type : str
            Type of correlations: "auto", "cross", "all"
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        ensemble_file = output_path / f"correlation_ensemble_{correlation_type}_{n_realizations}real.npz"
        
        if ensemble_file.exists() and not force_regenerate:
            logger.info(f"Loading existing ensemble from {ensemble_file}")
            try:
                data = xlh.load_arrays(str(ensemble_file), ["correlations", "covariance", "config"])
                self.ensemble_data = data["correlations"]
                self.empirical_cov = data["covariance"] 
                self.ensemble_config = data["config"]
                logger.info(f"Loaded ensemble: {self.ensemble_data.shape}")
                return self.ensemble_data
            except Exception as e:
                logger.warning(f"Failed to load ensemble: {e}. Regenerating...")
        
        logger.info(f"Generating {n_realizations} correlation function realizations...")
        
        # Use the same likelihood setup as s8_copula_comparison.py
        from s8_copula_comparison import setup_likelihood_with_n_datapoints
        
        # Determine n_datapoints from likelihood config or use default
        n_datapoints = self.likelihood_config.get('n_datapoints', 20)
        student_t_dof = self.likelihood_config.get('student_t_dof', 5.0)
        
        # Setup likelihood (this creates the mock data infrastructure)
        likelihood, data_paths, actual_n_datapoints, _ = setup_likelihood_with_n_datapoints(
            n_datapoints, correlation_type, student_t_dof)
        
        correlation_realizations = []
        failed_realizations = 0
        
        with computation_phase(f"Generating {n_realizations} realizations"):
            for i in range(n_realizations):
                if i % 100 == 0:
                    logger.info(f"Progress: {i}/{n_realizations} ({100*i/n_realizations:.1f}%)")
                
                try:
                    # Generate realistic mock data with noise
                    mock_data_path = DATA_DIR / f"mock_data_realization_{i}.npz"
                    gaussian_covariance_path = DATA_DIR / f"gaussian_covariance_realization_{i}.npz"
                    
                    # Create mock data with random noise (not fiducial mean)
                    xlh.mock_data.create_mock_data(
                        likelihood=likelihood,
                        mock_data_path=mock_data_path,
                        gaussian_covariance_path=gaussian_covariance_path,
                        fiducial_cosmo=FIDUCIAL_COSMO,
                        random=i,  # Use realization number as seed for noise
                        exact_lmax=EXACT_LMAX
                    )
                    
                    # Load the generated data
                    mock_data = xlh.load_arrays(mock_data_path, ["data"])["data"]
                    
                    # Flatten correlation functions into vector
                    correlation_vector = self._flatten_correlation_data(mock_data)
                    correlation_realizations.append(correlation_vector)
                    
                    # Clean up temporary files
                    mock_data_path.unlink(missing_ok=True)
                    gaussian_covariance_path.unlink(missing_ok=True)
                    
                except Exception as e:
                    logger.warning(f"Realization {i} failed: {e}")
                    failed_realizations += 1
                    continue
        
        if len(correlation_realizations) == 0:
            raise RuntimeError("No successful realizations generated!")
        
        self.ensemble_data = np.array(correlation_realizations)
        logger.info(f"Generated {len(correlation_realizations)} realizations "
                   f"({failed_realizations} failed)")
        logger.info(f"Ensemble shape: {self.ensemble_data.shape}")
        
        # Compute empirical covariance
        with computation_phase("Computing empirical covariance"):
            self.empirical_cov = np.cov(self.ensemble_data.T)
            logger.info(f"Empirical covariance shape: {self.empirical_cov.shape}")
            
            # Diagnose covariance conditioning
            eigenvals = np.linalg.eigvals(self.empirical_cov)
            condition_number = np.max(eigenvals) / np.min(eigenvals)
            logger.info(f"Empirical covariance condition number: {condition_number:.1e}")
        
        # Save ensemble configuration
        self.ensemble_config = {
            "n_realizations": len(correlation_realizations),
            "n_failed": failed_realizations,
            "correlation_type": correlation_type,
            "n_datapoints": actual_n_datapoints,
            "student_t_dof": student_t_dof,
            "data_shape": self.ensemble_data.shape,
            "condition_number": float(condition_number)
        }
        
        # Save ensemble for reuse
        save_data = {
            "correlations": self.ensemble_data,
            "covariance": self.empirical_cov,
            "config": self.ensemble_config
        }
        xlh.save_arrays(str(ensemble_file), save_data)
        logger.info(f"Saved ensemble to {ensemble_file}")
        
        return self.ensemble_data
    
    def _flatten_correlation_data(self, correlation_data):
        """
        Flatten correlation function data into vector format.
        Handles the xilikelihood data structure.
        """
        # correlation_data should be shape (n_redshift_combinations, n_angular_bins)
        # Flatten to 1D vector
        return correlation_data.flatten()
    
    def transform_to_uniform_margins(self):
        """
        Transform ensemble to uniform [0,1] margins, isolating copula structure.
        
        This is the key step that separates marginal distributions from
        dependence structure (copula). After this transformation, each
        correlation function type has uniform [0,1] distribution, but
        the dependence between different measurements is preserved.
        """
        if self.ensemble_data is None:
            raise ValueError("Must generate or load ensemble first")
        
        logger.info("Transforming to uniform margins...")
        n_realizations, n_correlations = self.ensemble_data.shape
        self.uniform_data = np.zeros_like(self.ensemble_data)
        
        with computation_phase("Uniform margin transformation"):
            for i in range(n_correlations):
                # Empirical CDF transformation: data → ranks → uniform [0,1]
                values = self.ensemble_data[:, i]
                ranks = stats.rankdata(values, method='average')
                self.uniform_data[:, i] = ranks / (n_realizations + 1)
        
        logger.info(f"Transformed {n_correlations} correlation types to uniform margins")
        return self.uniform_data
    
    def covariance_guided_tail_analysis(self, top_k=50):
        """
        Prioritize tail dependence analysis on pairs with highest linear correlation.
        
        This is a smart approach for high-dimensional data: instead of analyzing
        all n(n-1)/2 pairs, focus on the pairs with strongest linear correlation,
        which are most likely to also show tail dependence.
        
        Parameters:
        -----------
        top_k : int
            Number of top correlated pairs to analyze for tail dependence
        """
        if self.empirical_cov is None:
            raise ValueError("Need empirical covariance matrix")
        if self.uniform_data is None:
            raise ValueError("Must transform to uniform margins first")
        
        logger.info("Performing covariance-guided tail dependence analysis...")
        
        # Extract correlation coefficients from covariance matrix
        with computation_phase("Extracting correlation structure"):
            std_devs = np.sqrt(np.diag(self.empirical_cov))
            correlation_matrix = self.empirical_cov / np.outer(std_devs, std_devs)
            
            # Find pairs with highest absolute correlation
            n_dims = correlation_matrix.shape[0]
            high_corr_pairs = []
            
            for i in range(n_dims):
                for j in range(i+1, n_dims):
                    corr_coeff = abs(correlation_matrix[i, j])
                    high_corr_pairs.append((i, j, corr_coeff))
            
            # Sort by correlation strength and take top k
            high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
            selected_pairs = high_corr_pairs[:top_k]
            
            logger.info(f"Analyzing top {len(selected_pairs)} pairs (highest correlation)")
            logger.info(f"Correlation range: {selected_pairs[-1][2]:.3f} - {selected_pairs[0][2]:.3f}")
        
        # Measure tail dependence for selected pairs
        tail_results = {}
        
        with computation_phase("Tail dependence measurement"):
            for threshold in [0.90, 0.95, 0.99]:
                logger.info(f"Computing tail dependence at {threshold:.0%} threshold...")
                pair_results = {}
                
                for pair_idx, (i, j, corr_coeff) in enumerate(selected_pairs):
                    if pair_idx % 10 == 0:
                        logger.debug(f"Analyzing pair {pair_idx+1}/{len(selected_pairs)}")
                    
                    u_i = self.uniform_data[:, i]
                    u_j = self.uniform_data[:, j]
                    
                    # Upper tail dependence: P(U_j > t | U_i > t)
                    upper_mask_i = u_i > threshold
                    upper_mask_both = upper_mask_i & (u_j > threshold)
                    n_upper_i = np.sum(upper_mask_i)
                    n_upper_both = np.sum(upper_mask_both)
                    lambda_upper = n_upper_both / n_upper_i if n_upper_i > 10 else np.nan
                    
                    # Lower tail dependence: P(U_j ≤ t | U_i ≤ t)  
                    lower_threshold = 1 - threshold
                    lower_mask_i = u_i <= lower_threshold
                    lower_mask_both = lower_mask_i & (u_j <= lower_threshold)
                    n_lower_i = np.sum(lower_mask_i)
                    n_lower_both = np.sum(lower_mask_both)
                    lambda_lower = n_lower_both / n_lower_i if n_lower_i > 10 else np.nan
                    
                    pair_results[f'pair_{i}_{j}'] = {
                        'lambda_upper': lambda_upper,
                        'lambda_lower': lambda_lower,
                        'linear_correlation': corr_coeff,
                        'sample_sizes': (n_upper_i, n_lower_i),
                        'indices': (i, j)
                    }
                
                tail_results[f'threshold_{threshold}'] = pair_results
        
        self.tail_results = tail_results
        logger.info(f"Completed tail dependence analysis for {len(selected_pairs)} pairs")
        return tail_results
    
    def interpret_for_copula_selection(self):
        """
        Interpret tail dependence results for copula model selection.
        
        Key signatures:
        - Gaussian copula: λ_upper ≈ λ_lower ≈ 0 (no tail dependence)
        - Student-t copula: λ_upper ≈ λ_lower > 0 (symmetric tail dependence)
        - Other copulas: asymmetric tail dependence patterns
        
        Returns:
        --------
        recommendation : dict
            Copula recommendation with confidence and evidence
        """
        if self.tail_results is None:
            raise ValueError("Must run tail analysis first")
        
        evidence = {'gaussian': 0, 'student_t': 0, 'mixed': 0}
        threshold_summaries = {}
        
        logger.info("Interpreting tail dependence for copula selection...")
        
        for threshold_key, pairs in self.tail_results.items():
            upper_deps = []
            lower_deps = []
            valid_pairs = 0
            
            for pair_key, results in pairs.items():
                if not np.isnan(results['lambda_upper']) and not np.isnan(results['lambda_lower']):
                    upper_deps.append(results['lambda_upper'])
                    lower_deps.append(results['lambda_lower'])
                    valid_pairs += 1
            
            if len(upper_deps) == 0:
                logger.warning(f"No valid tail dependence measurements for {threshold_key}")
                continue
                
            avg_upper = np.mean(upper_deps)
            avg_lower = np.mean(lower_deps)
            max_upper = np.max(upper_deps)
            max_lower = np.max(lower_deps)
            std_upper = np.std(upper_deps)
            std_lower = np.std(lower_deps)
            
            threshold_summaries[threshold_key] = {
                'avg_upper_tail_dep': avg_upper,
                'avg_lower_tail_dep': avg_lower, 
                'max_upper_tail_dep': max_upper,
                'max_lower_tail_dep': max_lower,
                'std_upper_tail_dep': std_upper,
                'std_lower_tail_dep': std_lower,
                'n_valid_pairs': valid_pairs
            }
            
            # Evidence classification
            if avg_upper < 0.05 and avg_lower < 0.05:
                evidence['gaussian'] += 1
                logger.info(f"{threshold_key}: Low tail dependence (λ̄_u={avg_upper:.3f}, λ̄_l={avg_lower:.3f}) → Gaussian evidence")
            elif avg_upper > 0.15 or avg_lower > 0.15:
                if abs(avg_upper - avg_lower) < 0.1:  # Symmetric tail dependence
                    evidence['student_t'] += 1
                    logger.info(f"{threshold_key}: Symmetric tail dependence (λ̄_u={avg_upper:.3f}, λ̄_l={avg_lower:.3f}) → Student-t evidence")
                else:  # Asymmetric tail dependence
                    evidence['mixed'] += 1
                    logger.info(f"{threshold_key}: Asymmetric tail dependence (λ̄_u={avg_upper:.3f}, λ̄_l={avg_lower:.3f}) → Mixed evidence")
            else:
                logger.info(f"{threshold_key}: Moderate tail dependence (λ̄_u={avg_upper:.3f}, λ̄_l={avg_lower:.3f}) → Inconclusive")
        
        # Overall recommendation
        total_evidence = sum(evidence.values())
        if total_evidence == 0:
            recommendation = "inconclusive"
            confidence = 0.0
        else:
            recommendation = max(evidence.keys(), key=lambda k: evidence[k])
            confidence = evidence[recommendation] / len(self.tail_results)
        
        result = {
            'recommendation': recommendation,
            'confidence': confidence,
            'evidence_scores': evidence,
            'threshold_summaries': threshold_summaries,
            'total_thresholds': len(self.tail_results),
            'interpretation': self._generate_interpretation(recommendation, confidence, evidence)
        }
        
        logger.info(f"RECOMMENDATION: {recommendation} copula (confidence: {confidence:.1%})")
        logger.info(f"Evidence scores: {evidence}")
        
        return result
    
    def _generate_interpretation(self, recommendation, confidence, evidence):
        """Generate human-readable interpretation of results."""
        interpretation = []
        
        if recommendation == "gaussian":
            interpretation.append("Data shows minimal tail dependence, consistent with Gaussian copula assumption.")
            interpretation.append("Extreme values in different correlation measurements tend to be independent.")
        elif recommendation == "student_t":
            interpretation.append("Data shows significant symmetric tail dependence, supporting Student-t copula.")
            interpretation.append("Extreme values tend to occur together across correlation measurements.")
        elif recommendation == "mixed":
            interpretation.append("Data shows asymmetric tail dependence patterns.")
            interpretation.append("Consider more complex copula models (Clayton, Gumbel, etc.).")
        else:
            interpretation.append("Tail dependence analysis is inconclusive.")
            interpretation.append("May need more data or different analysis approach.")
        
        if confidence < 0.5:
            interpretation.append("Low confidence - results may be sensitive to data or methodology.")
        elif confidence > 0.8:
            interpretation.append("High confidence - strong evidence for recommendation.")
        
        return interpretation
    
    def save_results(self, output_file):
        """Save all results for reproducibility and further analysis."""
        save_data = {
            'tail_results': self.tail_results,
            'empirical_covariance': self.empirical_cov,
            'ensemble_config': self.ensemble_config,
            'ensemble_shape': self.ensemble_data.shape if self.ensemble_data is not None else None,
            'uniform_data_sample': self.uniform_data[:100] if self.uniform_data is not None else None  # Save small sample for diagnostics
        }
        xlh.save_arrays(output_file, save_data)
        logger.info(f"Saved tail dependence results to {output_file}")
    
    def plot_tail_dependence_summary(self, output_dir):
        """Create summary plots of tail dependence analysis."""
        if self.tail_results is None:
            logger.warning("No tail results to plot")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Plot 1: Tail dependence vs linear correlation
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for threshold_key, pairs in self.tail_results.items():
            threshold_val = float(threshold_key.split('_')[1])
            
            linear_corrs = []
            upper_deps = []
            lower_deps = []
            
            for pair_results in pairs.values():
                if not np.isnan(pair_results['lambda_upper']):
                    linear_corrs.append(pair_results['linear_correlation'])
                    upper_deps.append(pair_results['lambda_upper'])
                    lower_deps.append(pair_results['lambda_lower'])
            
            if len(linear_corrs) > 0:
                alpha = 0.7 if threshold_val == 0.95 else 0.5
                axes[0].scatter(linear_corrs, upper_deps, alpha=alpha, 
                              label=f'Upper (t={threshold_val:.2f})', s=20)
                axes[1].scatter(linear_corrs, lower_deps, alpha=alpha,
                              label=f'Lower (t={threshold_val:.2f})', s=20)
        
        axes[0].set_xlabel('Linear Correlation')
        axes[0].set_ylabel('Upper Tail Dependence')
        axes[0].set_title('Upper Tail Dependence vs Linear Correlation')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Linear Correlation')
        axes[1].set_ylabel('Lower Tail Dependence')
        axes[1].set_title('Lower Tail Dependence vs Linear Correlation')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "tail_dependence_vs_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Threshold dependence
        fig, ax = plt.subplots(figsize=(10, 6))
        
        thresholds = []
        avg_upper = []
        avg_lower = []
        std_upper = []
        std_lower = []
        
        for threshold_key, pairs in self.tail_results.items():
            threshold_val = float(threshold_key.split('_')[1])
            thresholds.append(threshold_val)
            
            upper_deps = [r['lambda_upper'] for r in pairs.values() if not np.isnan(r['lambda_upper'])]
            lower_deps = [r['lambda_lower'] for r in pairs.values() if not np.isnan(r['lambda_lower'])]
            
            avg_upper.append(np.mean(upper_deps) if upper_deps else 0)
            avg_lower.append(np.mean(lower_deps) if lower_deps else 0)
            std_upper.append(np.std(upper_deps) if upper_deps else 0)
            std_lower.append(np.std(lower_deps) if lower_deps else 0)
        
        ax.errorbar(thresholds, avg_upper, yerr=std_upper, label='Upper Tail Dependence', 
                   marker='o', capsize=5, linewidth=2)
        ax.errorbar(thresholds, avg_lower, yerr=std_lower, label='Lower Tail Dependence', 
                   marker='s', capsize=5, linewidth=2)
        
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Average Tail Dependence')
        ax.set_title('Tail Dependence vs Threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "tail_dependence_vs_threshold.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved tail dependence plots to {output_path}")


def run_tail_dependence_comparison(n_realizations=1000, output_dir="tail_analysis", 
                                 correlation_type="all", n_datapoints=20, force_regenerate=False):
    """
    Complete workflow: generate ensemble → analyze tail dependence → recommend copula.
    
    This is the main function that orchestrates the entire tail dependence analysis.
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_path / "tail_dependence_analysis.log"
    
    with logging_context(log_file=log_file, level="INFO", console_output=True) as logger:
        logger.info("=== TAIL DEPENDENCE ANALYSIS FOR COPULA SELECTION ===")
        logger.info(f"Configuration: {n_realizations} realizations, {correlation_type} correlations, {n_datapoints} datapoints")
        
        # Initialize analyzer
        likelihood_config = {
            'n_datapoints': n_datapoints,
            'student_t_dof': 5.0
        }
        analyzer = TailDependenceAnalyzer(likelihood_config=likelihood_config)
        
        with computation_phase("Complete tail dependence analysis"):
            # Generate/load correlation function ensemble
            ensemble = analyzer.generate_or_load_ensemble(
                n_realizations=n_realizations,
                output_dir=output_dir,
                force_regenerate=force_regenerate,
                correlation_type=correlation_type
            )
            
            # Transform to uniform margins (isolate copula structure)
            uniform_data = analyzer.transform_to_uniform_margins()
            
            # Covariance-guided tail dependence analysis
            top_k = min(50, ensemble.shape[1] * (ensemble.shape[1] - 1) // 4)  # Adaptive top_k
            tail_results = analyzer.covariance_guided_tail_analysis(top_k=top_k)
            
            # Interpret for copula selection
            recommendation = analyzer.interpret_for_copula_selection()
            
            # Create diagnostic plots
            analyzer.plot_tail_dependence_summary(output_dir)
            
            # Save results
            output_file = output_path / "tail_dependence_results.npz"
            analyzer.save_results(str(output_file))
        
        logger.info("=== ANALYSIS COMPLETE ===")
        logger.info(f"Final recommendation: {recommendation['recommendation']} copula")
        logger.info(f"Confidence: {recommendation['confidence']:.1%}")
        logger.info(f"Interpretation: {' '.join(recommendation['interpretation'])}")
        
        return recommendation, analyzer


def compare_with_s8_posterior_analysis(n_datapoints, correlation_type="all", 
                                     n_realizations=500, output_dir=None):
    """
    Integration function: compare tail dependence recommendation with S8 posterior analysis.
    
    This validates whether the data-driven copula choice matches the differences
    observed in cosmological parameter constraints.
    """
    if output_dir is None:
        output_dir = f"tail_vs_posterior_comparison_{n_datapoints}dp"
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with logging_context(log_file=output_path / "comparison_analysis.log", 
                        level="INFO", console_output=True) as logger:
        
        logger.info("=== TAIL DEPENDENCE vs POSTERIOR ANALYSIS COMPARISON ===")
        
        # 1. Run tail dependence analysis
        logger.info("Step 1: Running tail dependence analysis...")
        tail_recommendation, analyzer = run_tail_dependence_comparison(
            n_realizations=n_realizations,
            output_dir=output_path / "tail_analysis",
            correlation_type=correlation_type,
            n_datapoints=n_datapoints
        )
        
        # 2. Run S8 posterior comparison (import from existing analysis)
        logger.info("Step 2: Running S8 posterior comparison...")
        try:
            from s8_copula_comparison import compute_s8_posterior_comparison
            posterior_results = compute_s8_posterior_comparison(
                n_datapoints=n_datapoints,
                s8_grid="medium",
                correlation_type=correlation_type,
                job_mode=True  # Suppress console output
            )
        except ImportError:
            logger.warning("Could not import s8_copula_comparison. Skipping posterior analysis.")
            posterior_results = None
        
        # 3. Compare recommendations
        if posterior_results is not None:
            # Determine which copula gives better (tighter, less biased) constraints
            gaussian_results = posterior_results['results']['gaussian']
            student_t_results = posterior_results['results']['student_t']
            
            if gaussian_results is not None and student_t_results is not None:
                # Preference based on constraint quality
                gaussian_sigma = gaussian_results['sigma']
                student_t_sigma = student_t_results['sigma']
                gaussian_bias = abs(gaussian_results['mean'] - posterior_results['fiducial_s8'])
                student_t_bias = abs(student_t_results['mean'] - posterior_results['fiducial_s8'])
                
                # Simple scoring: prefer tighter constraints and less bias
                gaussian_score = 1/gaussian_sigma - gaussian_bias
                student_t_score = 1/student_t_sigma - student_t_bias
                
                posterior_preference = "student_t" if student_t_score > gaussian_score else "gaussian"
                
                logger.info(f"Posterior analysis results:")
                logger.info(f"  Gaussian:  σ={gaussian_sigma:.4f}, bias={gaussian_bias:.4f}, score={gaussian_score:.2f}")
                logger.info(f"  Student-t: σ={student_t_sigma:.4f}, bias={student_t_bias:.4f}, score={student_t_score:.2f}")
                logger.info(f"  Preference: {posterior_preference}")
            else:
                posterior_preference = "inconclusive"
                logger.warning("Incomplete posterior results")
        else:
            posterior_preference = "unavailable"
        
        tail_preference = tail_recommendation['recommendation']
        
        # 4. Summary comparison
        logger.info("")
        logger.info("="*60)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*60)
        logger.info(f"Tail dependence analysis recommends: {tail_preference}")
        logger.info(f"  Confidence: {tail_recommendation['confidence']:.1%}")
        logger.info(f"  Evidence: {tail_recommendation['evidence_scores']}")
        logger.info("")
        logger.info(f"Posterior constraints favor: {posterior_preference}")
        if posterior_results is not None and 'results' in posterior_results:
            if posterior_results['results']['gaussian'] is not None:
                logger.info(f"  Gaussian σ: {posterior_results['results']['gaussian']['sigma']:.4f}")
            if posterior_results['results']['student_t'] is not None:
                logger.info(f"  Student-t σ: {posterior_results['results']['student_t']['sigma']:.4f}")
        logger.info("")
        
        if tail_preference == posterior_preference:
            logger.info("✅ CONSISTENT: Both methods favor the same copula")
            consistency = "consistent"
        elif tail_preference == "inconclusive" or posterior_preference in ["inconclusive", "unavailable"]:
            logger.info("⚠️  INCONCLUSIVE: One or both methods are inconclusive")
            consistency = "inconclusive"
        else:
            logger.info("❌ INCONSISTENT: Methods disagree - investigate further")
            consistency = "inconsistent"
        
        # Save comparison results
        comparison_results = {
            'tail_recommendation': tail_recommendation,
            'posterior_preference': posterior_preference,
            'consistency': consistency,
            'n_datapoints': n_datapoints,
            'correlation_type': correlation_type
        }
        
        if posterior_results is not None:
            comparison_results['posterior_results'] = posterior_results
        
        comparison_file = output_path / "tail_vs_posterior_comparison.npz"
        xlh.save_arrays(str(comparison_file), comparison_results)
        logger.info(f"Saved comparison results to {comparison_file}")
        
        return comparison_results


def main():
    """Command-line interface for tail dependence analysis."""
    parser = argparse.ArgumentParser(
        description="Tail dependence analysis for copula selection in cosmological parameter inference"
    )
    parser.add_argument("--n-realizations", type=int, default=1000, 
                       help="Number of correlation function realizations to generate")
    parser.add_argument("--output-dir", default="tail_analysis", 
                       help="Output directory for results and plots")
    parser.add_argument("--correlation-type", choices=["auto", "cross", "all"], default="all",
                       help="Type of correlations to include")
    parser.add_argument("--n-datapoints", type=int, default=20,
                       help="Number of datapoints for analysis")
    parser.add_argument("--force-regenerate", action="store_true", 
                       help="Force regenerate ensemble even if cached data exists")
    parser.add_argument("--compare-with-posteriors", action="store_true",
                       help="Also run S8 posterior analysis and compare recommendations")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.compare_with_posteriors:
        # Run full comparison analysis
        results = compare_with_s8_posterior_analysis(
            n_datapoints=args.n_datapoints,
            correlation_type=args.correlation_type,
            n_realizations=args.n_realizations,
            output_dir=args.output_dir
        )
        
        print(f"\nComparison complete!")
        print(f"Tail analysis: {results['tail_recommendation']['recommendation']}")
        print(f"Posterior preference: {results['posterior_preference']}")
        print(f"Consistency: {results['consistency']}")
        
    else:
        # Run just tail dependence analysis
        recommendation, analyzer = run_tail_dependence_comparison(
            n_realizations=args.n_realizations,
            output_dir=args.output_dir,
            correlation_type=args.correlation_type,
            n_datapoints=args.n_datapoints,
            force_regenerate=args.force_regenerate
        )
        
        print(f"\nTail dependence analysis complete!")
        print(f"Recommendation: {recommendation['recommendation']} copula")
        print(f"Confidence: {recommendation['confidence']:.1%}")
        print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
