"""
Create side-by-side 2D posterior comparison plots for different masks or scale cuts.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import cmasher as cmr

from xilikelihood.plotting import find_contour_levels_pdf
from config import N_JOBS_2D, PARAM_GRIDS_NARROW, PARAM_GRIDS_MEDIUM, PARAM_GRIDS_WIDE
from postprocess_2dpost import (
    load_posteriors, 
    normalize_posterior, 
    compute_posterior_mean,
    create_marginal_plot,
    FIDUCIAL_OMEGA_M, 
    FIDUCIAL_S8,
    OMEGA_M_LIM,
    S8_LIM
)

# LaTeX rendering
rc('font', **{'family': 'serif', 'serif': ['Times']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')


# =============================================================================
# Configuration for comparison
# =============================================================================

# Define the two datasets to compare
COMPARISON_CONFIGS = {
    "mask_comparison": {
        "left": {
            "filepath": "/cluster/work/refregier/veoehl/posteriors/2d_post_1000sqd_kidsplus_nosmall_n256",
            "label": r"$1000$ deg$^2$",
            "n_jobs": N_JOBS_2D,
            "cleaning_threshold_exact": 1000,
            "cleaning_threshold_gauss": 1000,
            "omega_m_lim": (0.1, 0.5),
            "s8_lim": (0.5, 1.1),
            "param_grid": PARAM_GRIDS_WIDE,
        },
        "right": {
            "filepath": "/cluster/work/refregier/veoehl/posteriors/2d_post_10000sqd_kidsplus_nosmall_n256",
            "label": r"$10000$ deg$^2$",
            "n_jobs": N_JOBS_2D,
            "cleaning_threshold_exact": 928,
            "cleaning_threshold_gauss": 928,
            "omega_m_lim": (0.2, 0.45),
            "s8_lim": (0.7, 0.9),
            "param_grid": PARAM_GRIDS_NARROW,
        },
        "output_name": "mask_comparison",
    },
    "scale_cut_comparison": {
        "left": {
            "filepath": "/cluster/work/refregier/veoehl/posteriors/2d_post_1000sqd_kidsplus_inclsmall_n1024",
            "label": r"$> 5'$",
            "n_jobs": N_JOBS_2D,
            "cleaning_threshold_exact": 1231,
            "cleaning_threshold_gauss": 1900,
            "omega_m_lim": (0.1, 0.5),
            "s8_lim": (0.5, 1.1),
            "param_grid": PARAM_GRIDS_WIDE,
        },
        "right": {
            "filepath": "/cluster/work/refregier/veoehl/posteriors/2d_post_1000sqd_kidsplus_nosmall_n1024",
            "label": r"$> 15'$",
            "n_jobs": N_JOBS_2D,
            "cleaning_threshold_exact": 864,
            "cleaning_threshold_gauss": 865,
            "omega_m_lim": (0.1, 0.5),
            "s8_lim": (0.5, 1.1),
            "param_grid": PARAM_GRIDS_WIDE,
        },
        "output_name": "scale_cut_comparison",
    },
    "scale_cut_comparison_10000sqd": {
        "left": {
            "filepath": "/cluster/work/refregier/veoehl/posteriors/2d_post_10000sqd_kidsplus_inclsmall_n1024",
            "label": r"$> 5'$",
            "n_jobs": N_JOBS_2D,
            "cleaning_threshold_exact": 2000,
            "cleaning_threshold_gauss": 2000,
            "omega_m_lim": (0.2, 0.45),
            "s8_lim": (0.7, 0.9),
            "param_grid": PARAM_GRIDS_NARROW,
        },
        "right": {
            "filepath": "/cluster/work/refregier/veoehl/posteriors/2d_post_10000sqd_kidsplus_nosmall_n1024",
            "label": r"$> 15'$",
            "n_jobs": N_JOBS_2D,
            "cleaning_threshold_exact": 2000,
            "cleaning_threshold_gauss": 2000,
            "omega_m_lim": (0.2, 0.45),
            "s8_lim": (0.7, 0.9),
            "param_grid": PARAM_GRIDS_NARROW,
        },
        "output_name": "scale_cut_comparison_10000sqd",
    },
}

# Select which comparison to run
ACTIVE_COMPARISON = "scale_cut_comparison_10000sqd"


# =============================================================================
# Functions
# =============================================================================

def load_and_process_dataset(config):
    """Load posteriors and compute derived quantities for one dataset."""
    # Get param grid from config
    param_grid = config["param_grid"]
    omega_m_min, omega_m_max, omega_m_points = param_grid["omega_m"]
    s8_min, s8_max, s8_points = param_grid["s8"]
    omega_m_prior = np.linspace(omega_m_min, omega_m_max, omega_m_points)
    s8_prior = np.linspace(s8_min, s8_max, s8_points)
    dx = omega_m_prior[1] - omega_m_prior[0]
    dy = s8_prior[1] - s8_prior[0]
    
    exact_log, gauss_log, missing = load_posteriors(
        config["filepath"], config["n_jobs"], omega_m_prior, s8_prior
    )
    exact_2d = normalize_posterior(exact_log, dx, dy, f"{config['label']} Exact", threshold=config.get("cleaning_threshold_exact"))
    gauss_2d = normalize_posterior(gauss_log, dx, dy, f"{config['label']} Gaussian", threshold=config.get("cleaning_threshold_gauss"))
    
    mean_exact = compute_posterior_mean(omega_m_prior, s8_prior, exact_2d, dx, dy)
    mean_gauss = compute_posterior_mean(omega_m_prior, s8_prior, gauss_2d, dx, dy)
    
    return {
        "exact_2d": exact_2d,
        "gauss_2d": gauss_2d,
        "mean_exact": mean_exact,
        "mean_gauss": mean_gauss,
        "label": config["label"],
        "omega_m_lim": config.get("omega_m_lim", (0.2, 0.45)),
        "s8_lim": config.get("s8_lim", (0.7, 0.9)),
        "omega_m_prior": omega_m_prior,
        "s8_prior": s8_prior,
        "dx": dx,
        "dy": dy,
    }


def create_comparison_figure(data_left, data_right,
                             sigma_levels=[0.68, 0.95], output_name="comparison"):
    """Create side-by-side comparison figure."""
    
    # Extract priors from data dicts
    omega_m_prior_left = data_left["omega_m_prior"]
    s8_prior_left = data_left["s8_prior"]
    dx_left, dy_left = data_left["dx"], data_left["dy"]
    
    omega_m_prior_right = data_right["omega_m_prior"]
    s8_prior_right = data_right["s8_prior"]
    dx_right, dy_right = data_right["dx"], data_right["dy"]
    
    # Colors
    pdf_cm = cmr.torch
    colors_exact = cmr.take_cmap_colors(pdf_cm, len(sigma_levels), cmap_range=(0.3, 0.4), return_fmt='hex')
    colors_gauss = cmr.take_cmap_colors(pdf_cm, len(sigma_levels), cmap_range=(0.65, 0.9), return_fmt='hex')
    linecolor_exact, linecolor_gauss = colors_exact[0], colors_gauss[0]
    
    # Create figure with GridSpec for two panels with marginals
    # Layout: [top_left, gap, top_right] / [main_left, right_marg_left, main_right, right_marg_right]
    fig = plt.figure(figsize=(10, 5))
    
    # GridSpec: 2 rows (top marginal + main), 5 cols (main1, right1, gap, main2, right2)
    gs = fig.add_gridspec(2, 5, 
                          width_ratios=[4, 1, 0.5, 4, 1],
                          height_ratios=[1, 4],
                          wspace=0.05, hspace=0.05)
    
    # Left panel axes
    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_main_left = fig.add_subplot(gs[1, 0])
    ax_right_left = fig.add_subplot(gs[1, 1])
    
    # Right panel axes  
    ax_top_right = fig.add_subplot(gs[0, 3])
    ax_main_right = fig.add_subplot(gs[1, 3])
    ax_right_right = fig.add_subplot(gs[1, 4])
    
    # Compute contour thresholds
    thresholds_left_exact = find_contour_levels_pdf(omega_m_prior_left, s8_prior_left, data_left["exact_2d"], sigma_levels)
    thresholds_left_gauss = find_contour_levels_pdf(omega_m_prior_left, s8_prior_left, data_left["gauss_2d"], sigma_levels)
    thresholds_right_exact = find_contour_levels_pdf(omega_m_prior_right, s8_prior_right, data_right["exact_2d"], sigma_levels)
    thresholds_right_gauss = find_contour_levels_pdf(omega_m_prior_right, s8_prior_right, data_right["gauss_2d"], sigma_levels)
    
    # Plot left panel - Gaussian (filled)
    create_marginal_plot(ax_main_left, ax_top_left, ax_right_left,
        omega_m_prior_left, s8_prior_left, data_left["gauss_2d"], thresholds_left_gauss,
        colors_gauss, data_left["mean_gauss"][0], data_left["mean_gauss"][1],
        linecolor_gauss, dx_left, dy_left, alpha=0.5, show_ylabel=True)
    
    # Plot left panel - Exact (lines)
    create_marginal_plot(ax_main_left, ax_top_left, ax_right_left,
        omega_m_prior_left, s8_prior_left, data_left["exact_2d"], thresholds_left_exact,
        colors_exact, data_left["mean_exact"][0], data_left["mean_exact"][1],
        linecolor_exact, dx_left, dy_left, alpha=1.0, show_ylabel=True)
    
    # Plot right panel - Gaussian (filled)
    create_marginal_plot(ax_main_right, ax_top_right, ax_right_right,
        omega_m_prior_right, s8_prior_right, data_right["gauss_2d"], thresholds_right_gauss,
        colors_gauss, data_right["mean_gauss"][0], data_right["mean_gauss"][1],
        linecolor_gauss, dx_right, dy_right, alpha=0.5, show_ylabel=False)
    
    # Plot right panel - Exact (lines)
    create_marginal_plot(ax_main_right, ax_top_right, ax_right_right,
        omega_m_prior_right, s8_prior_right, data_right["exact_2d"], thresholds_right_exact,
        colors_exact, data_right["mean_exact"][0], data_right["mean_exact"][1],
        linecolor_exact, dx_right, dy_right, alpha=1.0, show_ylabel=False)
    
    # Set axis limits for each panel
    ax_main_left.set_xlim(*data_left["omega_m_lim"])
    ax_main_left.set_ylim(*data_left["s8_lim"])
    ax_top_left.set_xlim(*data_left["omega_m_lim"])
    ax_right_left.set_ylim(*data_left["s8_lim"])
    ax_main_left.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_main_left.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    ax_main_right.set_xlim(*data_right["omega_m_lim"])
    ax_main_right.set_ylim(*data_right["s8_lim"])
    ax_top_right.set_xlim(*data_right["omega_m_lim"])
    ax_right_right.set_ylim(*data_right["s8_lim"])
    
    # Show y-axis tick labels on right panel (since limits may differ)
    ax_main_right.tick_params(axis="y", labelleft=True)
    ax_main_right.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_main_right.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    # Panel text labels (inside main plot area)
    ax_main_left.text(0.05, 0.95, data_left["label"], transform=ax_main_left.transAxes,
                      fontsize=11, verticalalignment='top', horizontalalignment='left')
    ax_main_right.text(0.05, 0.95, data_right["label"], transform=ax_main_right.transAxes,
                       fontsize=11, verticalalignment='top', horizontalalignment='left')
    
    # Shared legend (on right panel)
    ax_main_right.legend(
        [plt.Line2D([0], [0], color=linecolor_gauss), 
         plt.Line2D([0], [0], color=linecolor_exact)],
        [r"Gaussian", r"Exact"], 
        loc="upper right", frameon=False
    )
    
    plt.tight_layout()
    
    # Save
    fig.savefig(f"{output_name}_2d_comparison.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{output_name}_2d_comparison.pdf", dpi=300, bbox_inches='tight')
    print(f"Saved: {output_name}_2d_comparison.png/pdf")
    
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Get comparison config
    config = COMPARISON_CONFIGS[ACTIVE_COMPARISON]
    print(f"Running comparison: {ACTIVE_COMPARISON}")
    print(f"  Left:  {config['left']['label']} ({config['left']['filepath']})")
    print(f"  Right: {config['right']['label']} ({config['right']['filepath']})")
    
    # Load both datasets (each with its own param grid)
    data_left = load_and_process_dataset(config["left"])
    data_right = load_and_process_dataset(config["right"])
    
    # Create comparison figure
    create_comparison_figure(data_left, data_right,
                             output_name=config["output_name"])
    
    print("Done!")
