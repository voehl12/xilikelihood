import numpy as np 
import os
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from matplotlib import rc
from xilikelihood.plotting import plot_2D, find_contour_levels_pdf
from config import (N_JOBS_2D,
                PARAM_GRIDS_NARROW as PARAM_GRIDS)
import cmasher as cmr
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')

REGULARIZER = 500
LARGE_THRESHOLD = 928

def clean_large_entries(posterior, name="posterior"):
    threshold = np.percentile(posterior[np.isfinite(posterior)], 99.92)
    print(threshold)
    large_mask = posterior > LARGE_THRESHOLD
    n_large = np.sum(large_mask)
    if n_large > 0:
        print(f"Warning: {n_large} large values found in {name} (>{LARGE_THRESHOLD})")
        # Print their indices and values
        indices = np.argwhere(large_mask)
        for idx in indices:
            print(f"  {name}[{tuple(idx)}] = {posterior[tuple(idx)]}")
        # Optionally replace if there are only a few
        if n_large < 100:
            # Replace with the minimum of the non-large entries
            min_val = np.max(posterior[~large_mask])
            posterior[large_mask] = np.nan
            print(f"Replaced {n_large} large values in {name} with {min_val}")
        else:
            print(f"Too many large values in {name}, not replacing automatically.")
    return posterior

# Use configuration values
omega_m_min, omega_m_max, omega_m_points = PARAM_GRIDS["omega_m"]
s8_min, s8_max, s8_points = PARAM_GRIDS["s8"]

omega_m_prior = np.linspace(omega_m_min, omega_m_max, omega_m_points)
s8_prior = np.linspace(s8_min, s8_max, s8_points)
prior_pairs = np.meshgrid(omega_m_prior, s8_prior)
prior_pairs = np.vstack([prior_pairs[0].ravel(), prior_pairs[1].ravel()]).T

split_prior_pairs = np.array_split(prior_pairs, N_JOBS_2D)

# Initialize 2D grids for posteriors
exact_posteriors_2d = np.full((len(s8_prior), len(omega_m_prior)), -np.inf)
gauss_posteriors_2d = np.full((len(s8_prior), len(omega_m_prior)), -np.inf)
missing_files = []
# Load all posterior files
FILEPATH = '/cluster/work/refregier/veoehl/posteriors/kidsplus_10000sqd_2d'
#FILEPATH = '/cluster/scratch/veoehl/posteriors'
for jobnumber in range(N_JOBS_2D):
    filepath = f"{FILEPATH}/posterior_{jobnumber}.npy"
    if os.path.exists(filepath):  # Check if the file exists
        data = np.load(filepath, allow_pickle=True)
        for entry in data:
            omega_m = entry["omega_m"]
            s8 = entry["s8"]
            exact_post = entry["exact_post"]
            gauss_post = entry["gauss_post"]
            print(f"Exact post: {exact_post}, Gaussian post: {gauss_post}")
            
            # Find the indices in the 2D grid
            omega_m_idx = np.searchsorted(omega_m_prior, omega_m, side='left')
            s8_idx = np.searchsorted(s8_prior, s8, side='left')

            # Validate indices
            if 0 <= omega_m_idx < len(omega_m_prior) and 0 <= s8_idx < len(s8_prior):
                exact_posteriors_2d[s8_idx, omega_m_idx] = exact_post
                gauss_posteriors_2d[s8_idx, omega_m_idx] = gauss_post
            else:
                print(f"Warning: Indices out of bounds for omega_m_idx={omega_m_idx}, s8_idx={s8_idx}")
    else:
        print(f"Warning: File {filepath} does not exist.")
        missing_files.append(jobnumber+1)
print(f"Missing files for job numbers: {missing_files}")
# Calculate grid spacing
dx = omega_m_prior[1] - omega_m_prior[0]
dy = s8_prior[1] - s8_prior[0]



exact_posteriors_2d = clean_large_entries(exact_posteriors_2d, "exact_posteriors_2d")
gauss_posteriors_2d = clean_large_entries(gauss_posteriors_2d, "gauss_posteriors_2d")

exact_posteriors_2d = np.nan_to_num(exact_posteriors_2d, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
gauss_posteriors_2d = np.nan_to_num(gauss_posteriors_2d, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

# Find maximum for each posterior separately
max_exact = np.max(exact_posteriors_2d[np.isfinite(exact_posteriors_2d)])
max_gauss = np.max(gauss_posteriors_2d[np.isfinite(gauss_posteriors_2d)])

print(f"Max log-posterior - Exact: {max_exact:.2f}, Gauss: {max_gauss:.2f}")




# Exponentiate with their own maxima
exact_posteriors_2d = np.exp(exact_posteriors_2d - max_exact)
gauss_posteriors_2d = np.exp(gauss_posteriors_2d - max_gauss)


print("Exact posterior integral:", np.sum(exact_posteriors_2d[~np.isnan(exact_posteriors_2d)]) * dx * dy)
print("Gaussian posterior integral:", np.sum(gauss_posteriors_2d[~np.isnan(gauss_posteriors_2d)]) * dx * dy)
# Normalize the posteriors using 2D integral
exact_posteriors_2d /= (np.sum(exact_posteriors_2d[~np.isnan(exact_posteriors_2d)]) * dx * dy)
gauss_posteriors_2d /= (np.sum(gauss_posteriors_2d[~np.isnan(gauss_posteriors_2d)]) * dx * dy)

# replace nans with zeros
exact_posteriors_2d[np.isnan(exact_posteriors_2d)] = 0
gauss_posteriors_2d[np.isnan(gauss_posteriors_2d)] = 0

# Debugging print statements to verify alignment
print("Exact posteriors shape:", exact_posteriors_2d.shape)
print("Exact posterior integral:", np.sum(exact_posteriors_2d[~np.isnan(exact_posteriors_2d)]) * dx * dy)
print("Gaussian posteriors shape:", gauss_posteriors_2d.shape)
print("Gaussian posterior integral:", np.sum(gauss_posteriors_2d[~np.isnan(gauss_posteriors_2d)]) * dx * dy)

# Plot the 2D corner plot
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
# Define sigma levels for contours
sigma_levels = [0.68, 0.95]  # 1-sigma and 2-sigma levels
n_levels = len(sigma_levels)
pdf_cm = cmr.torch
colors_exact = cmr.take_cmap_colors(pdf_cm, n_levels, cmap_range=(0.3, 0.4), return_fmt='hex')
colors_gauss = cmr.take_cmap_colors(pdf_cm, n_levels, cmap_range=(0.65, 0.9), return_fmt='hex')
linecolor_exact = colors_exact[0]
linecolor_gauss = colors_gauss[0]
# Exact posterior
plot_2D(
    fig,
    ax[0],
    omega_m_prior,
    s8_prior,
    exact_posteriors_2d,
    colormap=pdf_cm,  # Adjust based on the range of your data
    log=False
)
ax[0].set_title("Exact Posterior")
ax[0].set_xlabel("Omega_m")
ax[0].set_ylabel("S8")

# Gaussian posterior
plot_2D(
    fig,
    ax[1],
    omega_m_prior,
    s8_prior,
    gauss_posteriors_2d,
    colormap=pdf_cm,
    log=False
)
ax[1].set_title("Gaussian Posterior")
ax[1].set_xlabel("Omega_m")
ax[1].set_ylabel("S8")

plt.tight_layout()
plt.savefig("2d_corner_plot_10000sqd_angbinplus_newcolors.png")

from matplotlib.ticker import MaxNLocator




# Compute thresholds for exact and Gaussian posteriors
exact_thresholds = find_contour_levels_pdf(omega_m_prior, s8_prior, exact_posteriors_2d, sigma_levels)
print("Exact thresholds:", exact_thresholds)
gaussian_thresholds = find_contour_levels_pdf(omega_m_prior, s8_prior, gauss_posteriors_2d, sigma_levels)
print("Gaussian thresholds:", gaussian_thresholds)
# Create a new figure for combined plot
fig, ax = plt.subplots(2, 2, figsize=(5, 5), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 4], 'wspace': 0.05, 'hspace': 0.05})

# Fiducial values
fiducial_omega_m = 0.31
fiducial_s8 = 0.8

# Compute means of the 2D PDFs
mean_omega_m_exact = np.sum(omega_m_prior[np.newaxis, :] * exact_posteriors_2d * dx * dy)
mean_s8_exact = np.sum(s8_prior[:, np.newaxis] * exact_posteriors_2d * dx * dy)
mean_omega_m_gauss = np.sum(omega_m_prior[np.newaxis, :] * gauss_posteriors_2d * dx * dy)
mean_s8_gauss = np.sum(s8_prior[:, np.newaxis] * gauss_posteriors_2d * dx * dy)

print("Exact mean (omega_m, s8):", mean_omega_m_exact, mean_s8_exact)
print("Gaussian mean (omega_m, s8):", mean_omega_m_gauss, mean_s8_gauss)


# Main panel: 2D contours
def create_marginal_plot(fig, ax, omega_m_prior, s8_prior, posteriors_2d, thresholds, colors, 
                         mean_omega_m, mean_s8, linecolor, fiducial_omega_m, fiducial_s8,
                         alpha=1.0):
    """Create a 2D contour plot with marginals."""
    
    # Main panel: 2D contours
    main_ax = ax[1, 0]
    if alpha == 1.0: 
        contourfunction = main_ax.contour
    else: 
        contourfunction = main_ax.contourf
        #thresholds = np.append(thresholds, posteriors_2d.max())
    
    contourfunction(omega_m_prior, s8_prior, posteriors_2d,
                     levels=thresholds, colors=colors, alpha=alpha)
    
    # Add mean
    main_ax.scatter(mean_omega_m, mean_s8, color=linecolor, zorder=5)
    
    # Add fiducial lines
    main_ax.axvline(fiducial_omega_m, color="black", linestyle="dashed", linewidth=1)
    main_ax.axhline(fiducial_s8, color="black", linestyle="dashed", linewidth=1)
    
    main_ax.set_xlabel(r"$\Omega_m$")
    main_ax.set_ylabel(r"$S_8$")
    main_ax.set_xlim(omega_m_prior.min(), omega_m_prior.max())
    main_ax.set_ylim(s8_prior.min(), s8_prior.max())
    
    # Top marginal: Omega_m
    top_ax = ax[0, 0]
    top_ax.plot(omega_m_prior, posteriors_2d.sum(axis=0) * dy, color=linecolor)
    top_ax.axvline(mean_omega_m, color=linecolor, linestyle="solid")
    top_ax.axvline(fiducial_omega_m, color="black", linestyle="dashed", linewidth=1)
    top_ax.set_xlim(omega_m_prior.min(), omega_m_prior.max())
    top_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    top_ax.tick_params(axis="x", labelbottom=False)
    
    # Right marginal: S8
    right_ax = ax[1, 1]
    right_ax.plot(posteriors_2d.sum(axis=1) * dx, s8_prior, color=linecolor)
    right_ax.axhline(mean_s8, color=linecolor, linestyle="solid")
    right_ax.axhline(fiducial_s8, color="black", linestyle="dashed", linewidth=1)
    right_ax.set_ylim(s8_prior.min(), s8_prior.max())
    right_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    right_ax.tick_params(axis="y", labelleft=False)
    
    # Hide unused top-right panel
    ax[0, 1].axis("off")
    
    plt.tight_layout()
    return fig, ax



# Plot Gaussian only
fig_gauss, ax_gauss = create_marginal_plot(fig, ax,
    omega_m_prior, s8_prior, gauss_posteriors_2d, gaussian_thresholds, colors_gauss,
    mean_omega_m_gauss, mean_s8_gauss, linecolor_gauss, fiducial_omega_m, fiducial_s8,
    alpha=0.5
)
ax_gauss[1, 0].legend([plt.Line2D([0], [0], color=linecolor_gauss)], 
                      [r"Gaussian Likelihood"], loc="upper right",frameon=False)
fig_gauss.savefig("gaussian_only_2d_contours_with_marginals_10000sqd.png", dpi=500)


fig, ax = create_marginal_plot(fig_gauss, ax_gauss,omega_m_prior, s8_prior, exact_posteriors_2d, exact_thresholds, colors_exact,
    mean_omega_m_exact, mean_s8_exact, linecolor_exact, fiducial_omega_m, fiducial_s8,
    alpha=1.0)

ax[1, 0].legend([plt.Line2D([0], [0], color=linecolor_gauss),
    plt.Line2D([0], [0], color=linecolor_exact)], 
                      [r"Gaussian Likelihood", r"Exact Likelihood"], loc="upper right",frameon=False)

# Hide unused top-right panel
ax[0, 1].axis("off")

plt.tight_layout()
fig.savefig("combined_2d_contours_with_marginals_10000sqd_angbinplus_pres_nosmallscales.png",dpi=500)

plt.close('all') 