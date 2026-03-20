#!/usr/bin/env python3
"""
Plotting functions for copula analysis results.

This module contains all visualization functions separated from the main analysis.
Produces publication-quality plots with consistent styling.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cmasher as cmr

# Set up LaTeX rendering
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
})

# Define color scheme using CMasher (consistent with plotting.py)
COLORS = {
    'gaussian': cmr.take_cmap_colors('cmr.gem', 1, cmap_range=(0.2, 0.2), return_fmt='hex')[0],
    'student_t': cmr.take_cmap_colors('cmr.gem', 1, cmap_range=(0.7, 0.7), return_fmt='hex')[0],
    'fiducial': 'black',
    'mean_gaussian': cmr.take_cmap_colors('cmr.gem', 1, cmap_range=(0.3, 0.3), return_fmt='hex')[0],
    'mean_student_t': cmr.take_cmap_colors('cmr.gem', 1, cmap_range=(0.8, 0.8), return_fmt='hex')[0],
}

# Figure size for talks (6 inches for good legibility)
FIGSIZE = (6, 5)


def create_simple_comparison_plot(param_grid, results, fiducial_param, save_path='copula_comparison.png'):
    """
    Create a simple, clean comparison plot showing the main copula effect.
    Shows Gaussian vs Student-t copula posteriors.
    """
    # Find the best configuration to show
    n_data = 10 if 10 in results else list(results.keys())[-1]
    marginal_type = 'lognormal' if 'lognormal' in results[n_data] else list(results[n_data].keys())[0]
    
    # Find highest valid correlation
    available_corrs = sorted(results[n_data][marginal_type].keys())
    corr_low = None
    
    for corr_candidate in available_corrs:
        gauss_result = results[n_data][marginal_type][corr_candidate]['gaussian']
        st_result = results[n_data][marginal_type][corr_candidate]['student_t_df10']
        
        if gauss_result is not None and st_result is not None:
            corr_low = corr_candidate
            break
    
    if corr_low is None:
        print("No valid configuration found for plotting")
        return None
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    
    # Get posteriors
    gauss_result = results[n_data][marginal_type][corr_low]['gaussian']
    st_result = results[n_data][marginal_type][corr_low]['student_t_df10']
    
    # Plot posteriors with consistent colors
    ax.plot(param_grid, gauss_result['posterior'], color=COLORS['gaussian'], 
           linewidth=2, label=r'Gaussian Copula', alpha=0.9)
    ax.plot(param_grid, st_result['posterior'], color=COLORS['student_t'], 
           linewidth=2, label=r'Student-$t$ Copula, $\nu=10$', alpha=0.9)
    
    # Add vertical lines for truth and means
    ax.axvline(fiducial_param, color=COLORS['fiducial'], linestyle='--', 
              linewidth=1.5, label=r'Fiducial', alpha=0.8)
    ax.axvline(gauss_result['mean'], color=COLORS['mean_gaussian'], 
              linestyle=':', linewidth=1.5, alpha=0.7)
    ax.axvline(st_result['mean'], color=COLORS['mean_student_t'], 
              linestyle=':', linewidth=1.5, alpha=0.7)
    
    # Calculate and display difference
    mean_diff = abs(st_result['mean'] - gauss_result['mean'])
    ax.text(0.02, 0.98, rf'$|\Delta\mu| = {mean_diff:.3f}$', 
           transform=ax.transAxes, fontsize=11, 
           verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    ax.set_xlabel(r'Parameter $\theta$')
    ax.set_ylabel(r'Posterior $p(\theta|d)$')
    ax.legend(frameon=True, fancybox=True, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.close(fig)
    
    return save_path


def create_copula_comparison_evolution_correlation(param_grid, results, fiducial_param, 
                                                  marginal_type='normal',
                                                  save_path='copula_comparison_correlation.png'):
    """
    Show how Gaussian vs Student-t posteriors evolve with increasing correlation.
    Shows full posterior shapes, not just means.
    """
    # Use fixed settings
    n_data_fixed = 10 if 10 in results else list(results.keys())[-1]
    
    # Get available marginal types and correlations
    if marginal_type not in results[n_data_fixed]:
        marginal_type = list(results[n_data_fixed].keys())[0]
    
    correlation_values = sorted(list(results[n_data_fixed][marginal_type].keys()))
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    
    # Set up colormap for correlations
    cmap_gauss = cmr.get_sub_cmap('cmr.gem', 0.1, 0.4)
    cmap_student = cmr.get_sub_cmap('cmr.gem', 0.6, 0.9)
    norm = Normalize(vmin=min(correlation_values), vmax=max(correlation_values))
    
    # Plot both Gaussian and Student-t for different correlations
    for corr in correlation_values:
        gauss_result = results[n_data_fixed][marginal_type][corr]['gaussian']
        st_result = results[n_data_fixed][marginal_type][corr]['student_t_df10']
        
        if gauss_result is not None:
            color = cmap_gauss(norm(corr))
            ax.plot(param_grid, gauss_result['posterior'], 
                   color=color, linewidth=2, alpha=0.8, linestyle='-',
                   label=rf'Gauss $\rho={corr:.1f}$')
        
        if st_result is not None:
            color = cmap_student(norm(corr))
            ax.plot(param_grid, st_result['posterior'], 
                   color=color, linewidth=2, alpha=0.8, linestyle='--',
                   label=rf'Student-$t$ $\rho={corr:.1f}$')
    
    # Add fiducial line
    ax.axvline(fiducial_param, color=COLORS['fiducial'], linestyle=':', 
              alpha=0.8, linewidth=1.5, label=r'Fiducial')
    
    ax.set_xlabel(r'Parameter $\theta$')
    ax.set_ylabel(r'Posterior $p(\theta|d)$')
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, ncol=2, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.close(fig)
    
    return save_path


def create_dimensionality_evolution_plot(param_grid, results, fiducial_param, 
                                        correlation=None, marginal_type='normal',
                                        save_path='dimensionality_evolution.png'):
    """
    Show how Gaussian vs Student-t posteriors evolve with increasing dimensionality.
    Shows full posterior shapes, not just means.
    """
    n_data_points_list = sorted(list(results.keys()))
    
    # Get available marginal types and correlations
    if marginal_type not in results[n_data_points_list[0]]:
        marginal_type = list(results[n_data_points_list[0]].keys())[0]
    
    correlation_values = sorted(list(results[n_data_points_list[0]][marginal_type].keys()))
    
    # Use highest correlation if not specified
    if correlation is None:
        correlation = max(correlation_values)
    
    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=FIGSIZE)
    
    # Set up colormap for dimensions
    cmap_gauss = cmr.get_sub_cmap('cmr.gem', 0.1, 0.4)
    cmap_student = cmr.get_sub_cmap('cmr.gem', 0.6, 0.9)
    norm = Normalize(vmin=min(n_data_points_list), vmax=max(n_data_points_list))
    
    # Plot both Gaussian and Student-t for different dimensions
    for n_data in n_data_points_list:
        if marginal_type not in results[n_data] or correlation not in results[n_data][marginal_type]:
            continue
            
        gauss_result = results[n_data][marginal_type][correlation]['gaussian']
        st_result = results[n_data][marginal_type][correlation]['student_t_df10']
        
        if gauss_result is not None:
            color = cmap_gauss(norm(n_data))
            ax.plot(param_grid, gauss_result['posterior'], 
                   color=color, linewidth=2, alpha=0.8, linestyle='-',
                   label=rf'Gauss $n={n_data}$')
        
        if st_result is not None:
            color = cmap_student(norm(n_data))
            ax.plot(param_grid, st_result['posterior'], 
                   color=color, linewidth=2, alpha=0.8, linestyle='--',
                   label=rf'Student-$t$ $n={n_data}$')
    
    # Add fiducial line
    ax.axvline(fiducial_param, color=COLORS['fiducial'], linestyle=':', 
              alpha=0.8, linewidth=1.5, label=r'Fiducial')
    
    ax.set_xlabel(r'Parameter $\theta$')
    ax.set_ylabel(r'Posterior $p(\theta|d)$')
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, ncol=2, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {save_path}")
    plt.close(fig)
    
    return save_path


def plot_all_results(param_grid, results, fiducial_param, output_dir='.'):
    """
    Create all the main plots and save them to the output directory.
    Each plot is saved as a separate figure for talk presentations.
    Shows full posterior evolution to capture shape changes, not just mean shifts.
    """
    import os
    
    print("Creating all plots...")
    
    saved_plots = []
    
    # Main comparison plot (single snapshot)
    saved_plots.append(create_simple_comparison_plot(
        param_grid, results, fiducial_param, 
        save_path=os.path.join(output_dir, 'copula_comparison.png')
    ))
    
    # Copula comparison evolution with correlation - Normal marginals
    saved_plots.append(create_copula_comparison_evolution_correlation(
        param_grid, results, fiducial_param, marginal_type='normal',
        save_path=os.path.join(output_dir, 'copula_comparison_correlation_normal.png')
    ))
    
    # Copula comparison evolution with correlation - Lognormal marginals
    if 'lognormal' in results[list(results.keys())[0]]:
        saved_plots.append(create_copula_comparison_evolution_correlation(
            param_grid, results, fiducial_param, marginal_type='lognormal',
            save_path=os.path.join(output_dir, 'copula_comparison_correlation_lognormal.png')
        ))
    
    # Copula comparison evolution with correlation - Gamma marginals
    if 'gamma' in results[list(results.keys())[0]]:
        saved_plots.append(create_copula_comparison_evolution_correlation(
            param_grid, results, fiducial_param, marginal_type='gamma',
            save_path=os.path.join(output_dir, 'copula_comparison_correlation_gamma.png')
        ))
    
    # Dimensionality evolution - Normal marginals
    saved_plots.append(create_dimensionality_evolution_plot(
        param_grid, results, fiducial_param, marginal_type='normal',
        save_path=os.path.join(output_dir, 'copula_comparison_dimensionality_normal.png')
    ))
    
    # Dimensionality evolution - Lognormal marginals
    if 'lognormal' in results[list(results.keys())[0]]:
        saved_plots.append(create_dimensionality_evolution_plot(
            param_grid, results, fiducial_param, marginal_type='lognormal',
            save_path=os.path.join(output_dir, 'copula_comparison_dimensionality_lognormal.png')
        ))
    
    print(f"All {len(saved_plots)} plots created successfully!")
    return saved_plots
   
