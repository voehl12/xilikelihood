import numpy as np 
import os
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from plotting import plot_2D

njobs = 500
omega_m_prior = np.linspace(0.1, 0.5, 100)
s8_prior = np.linspace(0.5, 1.1, 100)
prior_pairs = np.meshgrid(omega_m_prior, s8_prior)
prior_pairs = np.vstack([prior_pairs[0].ravel(), prior_pairs[1].ravel()]).T

split_prior_pairs = np.array_split(prior_pairs, njobs)

# Initialize 2D grids for posteriors
exact_posteriors_2d = np.full((len(s8_prior), len(omega_m_prior)), -np.inf)
gauss_posteriors_2d = np.full((len(s8_prior), len(omega_m_prior)), -np.inf)

# Load all posterior files
for jobnumber in range(njobs):
    filepath = f"/cluster/scratch/veoehl/posteriors/posterior_{jobnumber}.npy"
    if os.path.exists(filepath):  # Check if the file exists
        data = np.load(filepath, allow_pickle=True)
        for entry in data:
            omega_m = entry["omega_m"]
            s8 = entry["s8"]
            exact_post = entry["exact_post"]
            gauss_post = entry["gauss_post"]
            #print(f"Exact post: {exact_post}, Gaussian post: {gauss_post}")
            
            # Find the indices in the 2D grid
            omega_m_idx = np.searchsorted(omega_m_prior, omega_m, side='left')
            s8_idx = np.searchsorted(s8_prior, s8, side='left')

            # Validate indices
            if 0 <= omega_m_idx < len(omega_m_prior) and 0 <= s8_idx < len(s8_prior):
                exact_posteriors_2d[s8_idx, omega_m_idx] = exact_post
                gauss_posteriors_2d[s8_idx, omega_m_idx] = gauss_post
            else:
                print(f"Warning: Indices out of bounds for omega_m_idx={omega_m_idx}, s8_idx={s8_idx}")

# Calculate grid spacing
dx = omega_m_prior[1] - omega_m_prior[0]
dy = s8_prior[1] - s8_prior[0]

# Exponentiate the posteriors
exact_posteriors_2d = np.exp(exact_posteriors_2d - 700)
gauss_posteriors_2d = np.exp(gauss_posteriors_2d - 700)

# Normalize the posteriors using 2D integral
exact_posteriors_2d /= (np.sum(exact_posteriors_2d[~np.isnan(exact_posteriors_2d)]) * dx * dy)
gauss_posteriors_2d /= (np.sum(gauss_posteriors_2d) * dx * dy)

# replace nans with zeros
exact_posteriors_2d[np.isnan(exact_posteriors_2d)] = 0
gauss_posteriors_2d[np.isnan(gauss_posteriors_2d)] = 0

# Debugging print statements to verify alignment
print("Exact posteriors shape:", exact_posteriors_2d.shape)
print("Exact posterior integral:", np.sum(exact_posteriors_2d[~np.isnan(exact_posteriors_2d)]) * dx * dy)
print("Gaussian posteriors shape:", gauss_posteriors_2d.shape)
print("Gaussian posterior integral:", np.sum(gauss_posteriors_2d) * dx * dy)

# Plot the 2D corner plot
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Exact posterior
plot_2D(
    fig,
    ax[0],
    omega_m_prior,
    s8_prior,
    exact_posteriors_2d,
    colormap="viridis",  # Adjust based on the range of your data
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
    colormap="viridis",
    log=False
)
ax[1].set_title("Gaussian Posterior")
ax[1].set_xlabel("Omega_m")
ax[1].set_ylabel("S8")

plt.tight_layout()
plt.savefig("2d_corner_plot.png")

from matplotlib.ticker import MaxNLocator

# Define sigma levels for contours
sigma_levels = [0.68, 0.95]  # 1-sigma and 2-sigma levels

# Function to compute thresholds using quantiles
def compute_thresholds(posterior, levels):
    # Exclude NaN values
    valid_posterior = posterior[~np.isnan(posterior)]
    sorted_posterior = np.sort(valid_posterior.ravel())[::-1]  # Sort in descending order
    cumsum = np.cumsum(sorted_posterior) / np.sum(sorted_posterior)  # Normalize and compute cumulative sum
    return [sorted_posterior[np.searchsorted(cumsum, level)] for level in levels]

# Compute thresholds for exact and Gaussian posteriors
exact_thresholds = compute_thresholds(exact_posteriors_2d, sigma_levels)
print("Exact thresholds:", exact_thresholds)
gaussian_thresholds = compute_thresholds(gauss_posteriors_2d, sigma_levels)
print("Gaussian thresholds:", gaussian_thresholds)
# Create a new figure for combined plot
fig, ax = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [4, 1], 'height_ratios': [1, 4], 'wspace': 0.05, 'hspace': 0.05})

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
main_ax = ax[1, 0]
gaussian_contour = main_ax.contour(
    omega_m_prior,
    s8_prior,
    gauss_posteriors_2d,
    levels=[gaussian_thresholds[1], gaussian_thresholds[0], gauss_posteriors_2d.max()],
    colors=["#4575b4", "#91bfdb", "#e0f3f8"],
    alpha=1.0,
    label="Gaussian Likelihood"
)
exact_contour = main_ax.contourf(
    omega_m_prior,
    s8_prior,
    exact_posteriors_2d,
    levels=[exact_thresholds[1], exact_thresholds[0], exact_posteriors_2d.max()],
    colors=["#d73027", "#fc8d59", "#fee090"],
    alpha=0.5,
    label="Exact Likelihood"
)

# Add means to the 2D panel
main_ax.scatter(mean_omega_m_exact, mean_s8_exact, color="#d73027", label="Exact Mean", zorder=5)
main_ax.scatter(mean_omega_m_gauss, mean_s8_gauss, color="#4575b4", label="Gaussian Mean", zorder=5)

# Add dashed lines for fiducial values in the 2D panel
main_ax.axvline(fiducial_omega_m, color="black", linestyle="dashed", linewidth=1)
main_ax.axhline(fiducial_s8, color="black", linestyle="dashed", linewidth=1)

main_ax.set_xlabel("$\Omega_m$")
main_ax.set_ylabel("$S_8$")
main_ax.set_xlim(omega_m_prior.min(), omega_m_prior.max())
main_ax.set_ylim(s8_prior.min(), s8_prior.max())

# Add legend for the main panel
handles = [
    plt.Line2D([0], [0], color="#d73027", lw=1, label="Exact Likelihood"),
    plt.Line2D([0], [0], color="#4575b4", lw=1, label="Gaussian Likelihood")
]
main_ax.legend(loc="upper right", handles=handles)

# Top marginal: Omega_m
top_ax = ax[0, 0]
top_ax.plot(
    omega_m_prior,
    exact_posteriors_2d.sum(axis=0) * dy,  # Multiply by grid spacing in y
    color="#d73027",
    label="Exact Marginal"
)
top_ax.plot(
    omega_m_prior,
    gauss_posteriors_2d.sum(axis=0) * dy,  # Multiply by grid spacing in y
    color="#4575b4",
    label="Gaussian Marginal"
)
# Add means to the top marginal
top_ax.axvline(mean_omega_m_exact, color="#d73027", linestyle="solid", linewidth=1, label="Exact Mean")
top_ax.axvline(mean_omega_m_gauss, color="#4575b4", linestyle="solid", linewidth=1, label="Gaussian Mean")

# Add dashed line for fiducial omega_m in the top marginal
top_ax.axvline(fiducial_omega_m, color="black", linestyle="dashed", linewidth=1)

top_ax.set_xlim(omega_m_prior.min(), omega_m_prior.max())
top_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
top_ax.tick_params(axis="x", labelbottom=False)

# Right marginal: S8
right_ax = ax[1, 1]
right_ax.plot(
    exact_posteriors_2d.sum(axis=1) * dx,  # Multiply by grid spacing in x
    s8_prior,
    color="#d73027",
    label="Exact Marginal"
)
right_ax.plot(
    gauss_posteriors_2d.sum(axis=1) * dx,  # Multiply by grid spacing in x
    s8_prior,
    color="#4575b4",
    label="Gaussian Marginal"
)
# Add means to the right marginal
right_ax.axhline(mean_s8_exact, color="#d73027", linestyle="solid", linewidth=1, label="Exact Mean")
right_ax.axhline(mean_s8_gauss, color="#4575b4", linestyle="solid", linewidth=1, label="Gaussian Mean")

# Add dashed line for fiducial s8 in the right marginal
right_ax.axhline(fiducial_s8, color="black", linestyle="dashed", linewidth=1)

right_ax.set_ylim(s8_prior.min(), s8_prior.max())
right_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
right_ax.tick_params(axis="y", labelleft=False)

# Hide unused top-right panel
ax[0, 1].axis("off")

plt.tight_layout()
plt.savefig("combined_2d_contours_with_marginals.png")
