"""
Postprocess 2D posterior grids and create contour plots with marginals.
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator, MultipleLocator
from scipy.ndimage import median_filter
import cmasher as cmr

from xilikelihood.plotting import find_contour_levels_pdf
from config import N_JOBS_2D, PARAM_GRIDS_NARROW as PARAM_GRIDS

# LaTeX rendering
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')

# Constants
LARGE_THRESHOLD = 940
FIDUCIAL_OMEGA_M = 0.31
FIDUCIAL_S8 = 0.8
OMEGA_M_LIM = (0.2, 0.45)  # Match PARAM_GRIDS_NARROW
S8_LIM = (0.7, 0.9)  # Match PARAM_GRIDS_NARROW


# =============================================================================
# Functions
# =============================================================================

def diagnose_posterior(log_posterior, name="posterior"):
    """Print diagnostic info to help find appropriate threshold."""
    finite_vals = log_posterior[np.isfinite(log_posterior)]
    print(f"\n=== Diagnostics for {name} ===")
    print(f"Shape: {log_posterior.shape}")
    print(f"Finite values: {len(finite_vals)} / {log_posterior.size}")
    if len(finite_vals) > 0:
        print(f"Range: [{finite_vals.min():.2f}, {finite_vals.max():.2f}]")
        print(f"Percentiles (50, 90, 95, 99, 99.9):")
        for p in [50, 90, 95, 99, 99.9]:
            print(f"  {p}%: {np.percentile(finite_vals, p):.2f}")
        # Suggest threshold: values beyond 99.9th percentile
        suggested = np.percentile(finite_vals, 99.9)
        print(f"Suggested threshold (99.9th percentile): {suggested:.2f}")
    print()
    return finite_vals


def clean_large_entries(posterior, name="posterior", threshold=LARGE_THRESHOLD):
    """Clean outliers and apply median filter to posterior grid."""
    large_mask = posterior > threshold
    n_large = np.sum(large_mask)
    
    if n_large > 0:
        print(f"Warning: {n_large} large values found in {name} (>{threshold})")
        if n_large < 1000:
            posterior[large_mask] = np.nan
            print(f"Replaced {n_large} large values with NaN")
            return median_filter(posterior, size=3)
    
    return posterior


def load_posteriors(filepath, n_jobs, omega_m_prior, s8_prior):
    """
    Load posterior files and assemble into 2D grids.
    
    Returns
    -------
    exact_posteriors_2d, gauss_posteriors_2d : ndarray
        2D grids of log-posteriors
    missing_files : list
        Job numbers with missing files
    """
    exact_posteriors_2d = np.full((len(s8_prior), len(omega_m_prior)), -np.inf)
    gauss_posteriors_2d = np.full((len(s8_prior), len(omega_m_prior)), -np.inf)
    missing_files = []
    
    for jobnumber in range(n_jobs):
        jobfile = f"{filepath}/posterior_{jobnumber}.npy"
        
        if not os.path.exists(jobfile):
            print(f"Warning: File {jobfile} does not exist.")
            missing_files.append(jobnumber + 1)
            continue
        
        data = np.load(jobfile, allow_pickle=True)
        for entry in data:
            omega_m_idx = np.searchsorted(omega_m_prior, entry["omega_m"], side='left')
            s8_idx = np.searchsorted(s8_prior, entry["s8"], side='left')
            
            if 0 <= omega_m_idx < len(omega_m_prior) and 0 <= s8_idx < len(s8_prior):
                exact_posteriors_2d[s8_idx, omega_m_idx] = entry["exact_post"]
                gauss_posteriors_2d[s8_idx, omega_m_idx] = entry["gauss_post"]
    
    if missing_files:
        print(f"Missing files for job numbers: {missing_files}")
    
    return exact_posteriors_2d, gauss_posteriors_2d, missing_files


def normalize_posterior(log_posterior, dx, dy, name="posterior", threshold=None):
    """Convert log-posterior to normalized PDF.
    
    Parameters
    ----------
    threshold : float, optional
        If provided, values above this threshold are replaced with NaN before processing.
        If None, no outlier cleaning is performed.
    """
    # Diagnostic before any processing
    diagnose_posterior(log_posterior, f"{name} (raw)")
    
    if threshold is not None:
        log_posterior = clean_large_entries(log_posterior, name, threshold=threshold)
    log_posterior = np.nan_to_num(log_posterior, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
    
    finite_mask = np.isfinite(log_posterior)
    if not np.any(finite_mask):
        print(f"ERROR: No finite values in {name} after cleaning!")
        return np.zeros_like(log_posterior)
    
    max_val = np.max(log_posterior[finite_mask])
    posterior = np.exp(log_posterior - max_val)
    
    integral = np.nansum(posterior) * dx * dy
    if integral == 0:
        print(f"ERROR: {name} integral is zero! Check threshold or data.")
        print(f"  max_val = {max_val}, posterior range = [{posterior.min()}, {posterior.max()}]")
        return posterior
    
    posterior /= integral
    posterior[np.isnan(posterior)] = 0
    
    print(f"{name}: max log-posterior = {max_val:.2f}, integral = {integral:.4f}")
    return posterior


def compute_posterior_mean(omega_m_prior, s8_prior, posterior_2d, dx, dy):
    """Compute mean of 2D posterior."""
    mean_omega_m = np.sum(omega_m_prior[np.newaxis, :] * posterior_2d * dx * dy)
    mean_s8 = np.sum(s8_prior[:, np.newaxis] * posterior_2d * dx * dy)
    return mean_omega_m, mean_s8


def create_marginal_plot(ax_main, ax_top, ax_right, omega_m_prior, s8_prior, posteriors_2d, 
                         thresholds, colors, mean_omega_m, mean_s8, linecolor, dx, dy, 
                         alpha=1.0, show_ylabel=True):
    """Create a 2D contour plot with marginals.
    
    Parameters
    ----------
    ax_main, ax_top, ax_right : matplotlib axes
        The three axes for main contour, top marginal, and right marginal
    show_ylabel : bool
        Whether to show y-axis label on main panel (useful for side-by-side plots)
    """
    contour_fn = ax_main.contour if alpha == 1.0 else ax_main.contourf
    contour_fn(omega_m_prior, s8_prior, posteriors_2d, levels=thresholds, colors=colors, alpha=alpha)
    
    # Mean and fiducial markers
    ax_main.scatter(mean_omega_m, mean_s8, color=linecolor, zorder=5)
    ax_main.axvline(FIDUCIAL_OMEGA_M, color="black", linestyle="dashed", linewidth=1)
    ax_main.axhline(FIDUCIAL_S8, color="black", linestyle="dashed", linewidth=1)
    ax_main.xaxis.set_major_locator(MultipleLocator(0.1))
    ax_main.yaxis.set_major_locator(MultipleLocator(0.1))
    ax_main.set_xlabel(r"$\Omega_m$")
    if show_ylabel:
        ax_main.set_ylabel(r"$S_8$")
    else:
        ax_main.tick_params(axis="y", labelleft=False)
    
    # Top marginal: Omega_m
    ax_top.plot(omega_m_prior, posteriors_2d.sum(axis=0) * dy, color=linecolor)
    ax_top.axvline(mean_omega_m, color=linecolor, linestyle="solid")
    ax_top.axvline(FIDUCIAL_OMEGA_M, color="black", linestyle="dashed", linewidth=1)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    
    # Right marginal: S8
    ax_right.plot(posteriors_2d.sum(axis=1) * dx, s8_prior, color=linecolor)
    ax_right.axhline(mean_s8, color=linecolor, linestyle="solid")
    ax_right.axhline(FIDUCIAL_S8, color="black", linestyle="dashed", linewidth=1)
    ax_right.set_xticks([])
    ax_right.set_yticks([])


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Parameter grids
    omega_m_min, omega_m_max, omega_m_points = PARAM_GRIDS["omega_m"]
    s8_min, s8_max, s8_points = PARAM_GRIDS["s8"]
    omega_m_prior = np.linspace(omega_m_min, omega_m_max, omega_m_points)
    s8_prior = np.linspace(s8_min, s8_max, s8_points)
    dx = omega_m_prior[1] - omega_m_prior[0]
    dy = s8_prior[1] - s8_prior[0]
    # Figure 1: from kidsplus_1000sqd with and without small scales (need to rerun with nside=1024)
    # Figure 2: from kidsplus_1000sqd and kidsplus_10000sqd (no small scales) comparing mask effects
    # Load and normalize posteriors
    FILEPATH = '/cluster/scratch/veoehl/posteriors'
    exact_log, gauss_log, missing = load_posteriors(FILEPATH, N_JOBS_2D, omega_m_prior, s8_prior)
    exact_posteriors_2d = normalize_posterior(exact_log, dx, dy, "Exact")
    gauss_posteriors_2d = normalize_posterior(gauss_log, dx, dy, "Gaussian")
    
    # Compute means
    mean_omega_m_exact, mean_s8_exact = compute_posterior_mean(omega_m_prior, s8_prior, exact_posteriors_2d, dx, dy)
    mean_omega_m_gauss, mean_s8_gauss = compute_posterior_mean(omega_m_prior, s8_prior, gauss_posteriors_2d, dx, dy)
    print(f"Exact mean: (Ωm={mean_omega_m_exact:.4f}, S8={mean_s8_exact:.4f})")
    print(f"Gauss mean: (Ωm={mean_omega_m_gauss:.4f}, S8={mean_s8_gauss:.4f})")
    
    # Colors
    sigma_levels = [0.68, 0.95]
    pdf_cm = cmr.torch
    colors_exact = cmr.take_cmap_colors(pdf_cm, len(sigma_levels), cmap_range=(0.3, 0.4), return_fmt='hex')
    colors_gauss = cmr.take_cmap_colors(pdf_cm, len(sigma_levels), cmap_range=(0.65, 0.9), return_fmt='hex')
    linecolor_exact, linecolor_gauss = colors_exact[0], colors_gauss[0]
    
    # Contour thresholds
    exact_thresholds = find_contour_levels_pdf(omega_m_prior, s8_prior, exact_posteriors_2d, sigma_levels)
    gauss_thresholds = find_contour_levels_pdf(omega_m_prior, s8_prior, gauss_posteriors_2d, sigma_levels)
    
    # Create figure
    fig, ax = plt.subplots(2, 2, figsize=(5, 5),
        gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 4], 'wspace': 0.05, 'hspace': 0.05})
    ax[0, 1].axis("off")  # Hide unused corner
    
    # Plot Gaussian (filled)
    create_marginal_plot(ax[1, 0], ax[0, 0], ax[1, 1], omega_m_prior, s8_prior, gauss_posteriors_2d,
        gauss_thresholds, colors_gauss, mean_omega_m_gauss, mean_s8_gauss, linecolor_gauss, dx, dy, alpha=0.5)
    
    # Plot Exact (contour lines)
    create_marginal_plot(ax[1, 0], ax[0, 0], ax[1, 1], omega_m_prior, s8_prior, exact_posteriors_2d,
        exact_thresholds, colors_exact, mean_omega_m_exact, mean_s8_exact, linecolor_exact, dx, dy, alpha=1.0)
    
    # Legend
    ax[1, 0].legend(
        [plt.Line2D([0], [0], color=linecolor_gauss), plt.Line2D([0], [0], color=linecolor_exact)],
        [r"Gaussian Likelihood", r"Exact Likelihood"], loc="upper right", frameon=False)
    
    plt.tight_layout()
    fig.savefig("combined_2d_contours_with_marginals_10000sqd_nonoise.png", dpi=500)
    fig.savefig("combined_2d_contours_with_marginals_10000sqd_nonoise.pdf", dpi=300, bbox_inches='tight')
    plt.close('all')
    print("Done!") 