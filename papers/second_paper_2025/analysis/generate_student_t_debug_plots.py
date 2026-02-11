#!/usr/bin/env python3
"""
Generate Student-t copula debug plots.

This script creates diagnostic plots for the Student-t copula implementation:
1. Comparison of true vs recovered marginal distributions
2. Marginal error plots
3. Joint PDF heatmaps

The plots help diagnose whether the copula preserves marginal distributions
and visualize the joint distribution structure.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import t
import os
import sys

# Add the package to the path
sys.path.insert(0, '/cluster/home/veoehl/xilikelihood')
import xilikelihood as xlh


def generate_debug_plots(df_values=[3.0, 5.0, 10.0], 
                        correlations=[0.0, 0.3, 0.5, 0.7, 0.9],
                        n_points=100,
                        x_range_lims=(-4, 4),
                        output_dir=None):
    """
    Generate comprehensive Student-t copula debug plots.
    
    Parameters:
    -----------
    df_values : list
        Degrees of freedom values to test
    correlations : list  
        Correlation values to test
    n_points : int
        Number of points for discretization
    x_range_lims : tuple
        (min, max) for the x-axis range
    output_dir : str, optional
        Directory to save plots. If None, saves to current directory.
    """
    
    if output_dir is None:
        output_dir = '/cluster/home/veoehl/xilikelihood'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    x_range = np.linspace(x_range_lims[0], x_range_lims[1], n_points)
    
    print("Generating Student-t copula debug plots...")
    print(f"Output directory: {output_dir}")
    print(f"Testing {len(df_values)} df values × {len(correlations)} correlations")
    
    for df in df_values:
        print(f"\nProcessing df = {df}")
        
        # True marginal: Student-t with df degrees of freedom
        true_pdf = t.pdf(x_range, df)
        true_cdf = t.cdf(x_range, df)
        
        # Create 2D case with identical marginals
        marginal_pdfs = np.stack([true_pdf, true_pdf])[np.newaxis, :, :]
        marginal_cdfs = np.stack([true_cdf, true_cdf])[np.newaxis, :, :]
        
        for rho in correlations:
            print(f"  ρ = {rho:.1f}", end=" ")
            
            try:
                # Create covariance matrix
                cov_matrix = np.array([[1.0, rho], [rho, 1.0]])
                
                # Compute joint PDF using Student-t copula
                joint_pdf = xlh.copula_funcs.joint_pdf(
                    marginal_cdfs, marginal_pdfs, cov_matrix, 
                    copula_type="student-t", df=df
                )
                
                # Recover marginals by integrating out one dimension
                recovered_marginal_1 = np.trapz(joint_pdf, x_range, axis=1)
                recovered_marginal_2 = np.trapz(joint_pdf, x_range, axis=0)
                
                # Calculate errors
                error_1 = np.mean(np.abs(recovered_marginal_1 - true_pdf))
                error_2 = np.mean(np.abs(recovered_marginal_2 - true_pdf))
                max_error = max(error_1, error_2)
                
                # Create diagnostic plot
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'Student-t Copula Debug: df={df:.1f}, ρ={rho:.1f}\n'
                           f'Max Marginal Error: {max_error:.2e}', fontsize=14)
                
                # Plot 1: Marginal comparison
                ax1 = axes[0, 0]
                ax1.plot(x_range, true_pdf, 'k-', label='True marginal', linewidth=2)
                ax1.plot(x_range, recovered_marginal_1, 'r--', label='Recovered 1', alpha=0.8)
                ax1.plot(x_range, recovered_marginal_2, 'b:', label='Recovered 2', alpha=0.8)
                ax1.set_xlabel('x')
                ax1.set_ylabel('PDF')
                ax1.set_title('Marginal Distributions')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Marginal errors
                ax2 = axes[0, 1]
                ax2.plot(x_range, recovered_marginal_1 - true_pdf, 'r-', label='Error 1', linewidth=2)
                ax2.plot(x_range, recovered_marginal_2 - true_pdf, 'b-', label='Error 2', linewidth=2)
                ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                ax2.set_xlabel('x')
                ax2.set_ylabel('Error')
                ax2.set_title('Marginal Errors')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Joint PDF heatmap
                ax3 = axes[1, 0]
                im = ax3.imshow(joint_pdf, 
                               extent=[x_range[0], x_range[-1], x_range[0], x_range[-1]], 
                               origin='lower', aspect='auto', cmap='viridis')
                plt.colorbar(im, ax=ax3, label='Joint PDF')
                ax3.set_xlabel('x₁')
                ax3.set_ylabel('x₂')
                ax3.set_title('Joint PDF')
                
                # Plot 4: Joint PDF contour
                ax4 = axes[1, 1]
                X, Y = np.meshgrid(x_range, x_range)
                contours = ax4.contour(X, Y, joint_pdf, levels=10, colors='black', alpha=0.6)
                ax4.contourf(X, Y, joint_pdf, levels=20, cmap='viridis', alpha=0.8)
                ax4.set_xlabel('x₁')
                ax4.set_ylabel('x₂')
                ax4.set_title('Joint PDF Contours')
                ax4.set_aspect('equal')
                
                plt.tight_layout()
                
                # Save plot
                filename = f'student_t_copula_debug_rho_{rho:.1f}_df_{df:.0f}.png'
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=150, bbox_inches='tight')
                plt.close()
                
                # Status indicator
                if max_error < 1e-3:
                    print("✅")
                elif max_error < 1e-2:
                    print("⚠️")
                else:
                    print("❌")
                    
            except Exception as e:
                print(f"❌ (Error: {str(e)[:50]}...)")
                continue
    
    print(f"\nDebug plots saved to: {output_dir}")
    print("Files generated:")
    for df in df_values:
        for rho in correlations:
            filename = f'student_t_copula_debug_rho_{rho:.1f}_df_{df:.0f}.png'
            filepath = os.path.join(output_dir, filename)
            if os.path.exists(filepath):
                print(f"  ✓ {filename}")


def generate_summary_plot(df_values=[3.0, 5.0, 10.0], 
                         correlations=[0.0, 0.3, 0.5, 0.7, 0.9],
                         n_points=100,
                         output_dir=None):
    """
    Generate a summary plot showing marginal errors across all parameter combinations.
    """
    
    if output_dir is None:
        output_dir = '/cluster/home/veoehl/xilikelihood'
    
    x_range = np.linspace(-4, 4, n_points)
    
    # Collect error data
    error_matrix = np.zeros((len(df_values), len(correlations)))
    
    for i, df in enumerate(df_values):
        true_pdf = t.pdf(x_range, df)
        true_cdf = t.cdf(x_range, df)
        
        marginal_pdfs = np.stack([true_pdf, true_pdf])[np.newaxis, :, :]
        marginal_cdfs = np.stack([true_cdf, true_cdf])[np.newaxis, :, :]
        
        for j, rho in enumerate(correlations):
            try:
                cov_matrix = np.array([[1.0, rho], [rho, 1.0]])
                
                joint_pdf = xlh.copula_funcs.joint_pdf(
                    marginal_cdfs, marginal_pdfs, cov_matrix, 
                    copula_type="student-t", df=df
                )
                
                recovered_marginal_1 = np.trapz(joint_pdf, x_range, axis=1)
                error = np.mean(np.abs(recovered_marginal_1 - true_pdf))
                error_matrix[i, j] = error
                
            except Exception:
                error_matrix[i, j] = np.nan
    
    # Create summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    im = ax.imshow(error_matrix, aspect='auto', cmap='RdYlBu_r',                    norm=LogNorm(vmin=1e-5, vmax=1e-1))
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Mean Marginal Error')
    
    # Set ticks and labels
    ax.set_xticks(range(len(correlations)))
    ax.set_xticklabels([f'{rho:.1f}' for rho in correlations])
    ax.set_yticks(range(len(df_values)))
    ax.set_yticklabels([f'{df:.1f}' for df in df_values])
    
    ax.set_xlabel('Correlation (ρ)')
    ax.set_ylabel('Degrees of Freedom (df)')
    ax.set_title('Student-t Copula Marginal Preservation Errors')
    
    # Add text annotations
    for i in range(len(df_values)):
        for j in range(len(correlations)):
            if not np.isnan(error_matrix[i, j]):
                text = f'{error_matrix[i, j]:.1e}'
                ax.text(j, i, text, ha="center", va="center", 
                       color="white" if error_matrix[i, j] > 1e-3 else "black",
                       fontsize=8)
    
    plt.tight_layout()
    
    # Save summary plot
    filepath = os.path.join(output_dir, 'student_t_copula_error_summary.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved: {filepath}")


def main():
    """Main function to generate all debug plots."""
    
    print("Student-t Copula Debug Plot Generator")
    print("=" * 50)
    
    # Default parameters
    df_values = [3.0, 5.0, 10.0]
    correlations = [0.0, 0.3, 0.5, 0.7, 0.9]
    
    # Generate individual debug plots
    generate_debug_plots(df_values=df_values, correlations=correlations)
    
    print("\n" + "=" * 50)
    
    # Generate summary plot
    generate_summary_plot(df_values=df_values, correlations=correlations)
    
    print("\n✅ All debug plots generated successfully!")
    print("\nThese plots help diagnose:")
    print("  • Whether marginals are preserved (should be close to true marginals)")
    print("  • Numerical accuracy of the implementation")
    print("  • Visual structure of the joint distribution")
    print("  • Parameter sensitivity (df and correlation effects)")


if __name__ == "__main__":
    main()
