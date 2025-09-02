import numpy as np 
import os
import glob
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from xilikelihood.plotting import plot_2D
from config import PARAM_GRIDS
REGULARIZER = 500
LARGE_THRESHOLD = 1000

def load_posterior_results(results_dir, scale_cut_index=0, covariance_mode="std_cov"):
    """
    Load posterior results from new structured format.
    
    Parameters:
    -----------
    results_dir : str
        Base directory containing results
    scale_cut_index : int
        Which scale cut to load (0-5 for the predefined cuts)
    covariance_mode : str
        Either "std_cov" or "fixed_cov"
    
    Returns:
    --------
    combined_results : structured array
        Combined results from all jobs
    metadata : dict
        Metadata from the first file
    """
    scale_cuts_deg = np.array([10, 20, 50, 100, 300, 1200]) / 60.
    scale_cut = scale_cuts_deg[scale_cut_index]
    
    # Find all files for this scale cut and covariance mode
    pattern = os.path.join(results_dir, covariance_mode, f'posterior_2d_scale_cut_{scale_cut}_job*_{covariance_mode}.npz')
    files = sorted(glob.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files found matching pattern: {pattern}")
    
    print(f"Found {len(files)} result files for scale cut {scale_cut:.4f} deg ({scale_cut*60:.1f} arcmin)")
    
    all_results = []
    metadata = None
    
    for file_path in files:
        try:
            data = np.load(file_path,allow_pickle=True)
            results = data['results']
            
            # Extract metadata from first file
            if metadata is None:
                metadata = {k: data[k].item() if data[k].ndim == 0 else data[k] 
                           for k in data.files if k != 'results'}
                print(f"Loaded metadata: scale_cut={metadata['scale_cut_arcmin']:.1f} arcmin, "
                     f"covariance_mode={metadata['covariance_mode']}")
            
            all_results.append(results)
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
    
    if not all_results:
        raise ValueError("No valid result files found")
    
    # Combine all results
    combined_results = np.concatenate(all_results)
    print(f"Combined {len(combined_results)} parameter evaluations from {len(all_results)} jobs")
    
    return combined_results, metadata

def clean_large_entries(posterior, name="posterior"):
    large_mask = posterior > LARGE_THRESHOLD
    n_large = np.sum(large_mask)
    if n_large > 0:
        print(f"Warning: {n_large} large values found in {name} (>{LARGE_THRESHOLD})")
        # Print their indices and values
        indices = np.argwhere(large_mask)
        for idx in indices:
            print(f"  {name}[{tuple(idx)}] = {posterior[tuple(idx)]}")
        # Optionally replace if there are only a few
        if n_large < 10:
            # Replace with the minimum of the non-large entries
            min_val = np.min(posterior[~large_mask])
            posterior[large_mask] = min_val
            print(f"Replaced {n_large} large values in {name} with {min_val}")
        else:
            print(f"Too many large values in {name}, not replacing automatically.")
    return posterior

def process_posteriors(results_dir="/cluster/scratch/veoehl/posteriors_2d", 
                      scale_cut_index=0, covariance_mode="std_cov"):
    """
    Process 2D posterior results and create plots.
    
    Parameters:
    -----------
    results_dir : str
        Base directory containing results
    scale_cut_index : int
        Which scale cut to process (0-5)
    covariance_mode : str
        Either "std_cov" or "fixed_cov"
    """
    
    # Load results
    combined_results, metadata = load_posterior_results(results_dir, scale_cut_index, covariance_mode)
    
    # Get parameter ranges from metadata or config
    omega_m_range = metadata.get('omega_m_range', PARAM_GRIDS["omega_m"])
    s8_range = metadata.get('s8_range', PARAM_GRIDS["s8"])
    
    print(f"Debug: omega_m_range = {omega_m_range}, type = {type(omega_m_range)}")
    print(f"Debug: s8_range = {s8_range}, type = {type(s8_range)}")
    
    # Ensure ranges are tuples/lists and convert to int for n_points
    try:
        omega_m_min, omega_m_max, omega_m_points = omega_m_range
        omega_m_points = int(omega_m_points)  # Ensure integer
        s8_min, s8_max, s8_points = s8_range  
        s8_points = int(s8_points)  # Ensure integer
        
        print(f"Debug: After conversion - omega_m_points = {omega_m_points} (type: {type(omega_m_points)})")
        print(f"Debug: After conversion - s8_points = {s8_points} (type: {type(s8_points)})")
        
    except Exception as e:
        print(f"Error unpacking parameter ranges: {e}")
        print(f"omega_m_range: {omega_m_range}")
        print(f"s8_range: {s8_range}")
        raise
    
    try:
        omega_m_prior = np.linspace(omega_m_min, omega_m_max, omega_m_points)
        s8_prior = np.linspace(s8_min, s8_max, s8_points)
        print(f"Debug: Created grids - omega_m_prior shape: {omega_m_prior.shape}, s8_prior shape: {s8_prior.shape}")
    except Exception as e:
        print(f"Error creating parameter grids: {e}")
        print(f"Parameters: omega_m_min={omega_m_min}, omega_m_max={omega_m_max}, omega_m_points={omega_m_points}")
        print(f"Parameters: s8_min={s8_min}, s8_max={s8_max}, s8_points={s8_points}")
        raise
    
    print(f"Parameter grids: Ωₘ {omega_m_points} points [{omega_m_min:.3f}, {omega_m_max:.3f}], "
          f"S₈ {s8_points} points [{s8_min:.3f}, {s8_max:.3f}]")
    
    # Initialize 2D grids for posteriors
    try:
        print(f"Debug: Initializing grids with shapes: s8={len(s8_prior)}, omega_m={len(omega_m_prior)}")
        exact_posteriors_2d = np.full((len(s8_prior), len(omega_m_prior)), -np.inf)
        gauss_posteriors_2d = np.full((len(s8_prior), len(omega_m_prior)), -np.inf)
        print(f"Debug: Grid initialization successful")
    except Exception as e:
        print(f"Error initializing posterior grids: {e}")
        print(f"s8_prior length: {len(s8_prior)} (type: {type(len(s8_prior))})")
        print(f"omega_m_prior length: {len(omega_m_prior)} (type: {type(len(omega_m_prior))})")
        raise
    
    # Fill the 2D grids
    print(f"Debug: Starting to fill grids with {len(combined_results)} results")
    for i, entry in enumerate(combined_results):
        try:
            omega_m = entry["omega_m"]
            s8 = entry["s8"]
            exact_logL = entry["exact_logL"]
            gauss_logL = entry["gauss_logL"]
            
            # Skip NaN entries
            if np.isnan(exact_logL) or np.isnan(gauss_logL):
                continue
                
            # Find the indices in the 2D grid
            omega_m_idx = np.searchsorted(omega_m_prior, omega_m, side='left')
            s8_idx = np.searchsorted(s8_prior, s8, side='left')

            # Validate indices
            if 0 <= omega_m_idx < len(omega_m_prior) and 0 <= s8_idx < len(s8_prior):
                exact_posteriors_2d[s8_idx, omega_m_idx] = exact_logL
                gauss_posteriors_2d[s8_idx, omega_m_idx] = gauss_logL
            else:
                print(f"Warning: Indices out of bounds for omega_m_idx={omega_m_idx}, s8_idx={s8_idx}")
                
        except Exception as e:
            print(f"Error processing entry {i}: {e}")
            print(f"Entry: {entry}")
            print(f"omega_m_idx: {omega_m_idx if 'omega_m_idx' in locals() else 'not computed'}")
            print(f"s8_idx: {s8_idx if 's8_idx' in locals() else 'not computed'}")
            raise

    # Calculate grid spacing
    dx = omega_m_prior[1] - omega_m_prior[0]
    dy = s8_prior[1] - s8_prior[0]

    exact_posteriors_2d = clean_large_entries(exact_posteriors_2d, "exact_posteriors_2d")
    
    # Exponentiate the posteriors
    exact_posteriors_2d = np.exp(exact_posteriors_2d - REGULARIZER)
    gauss_posteriors_2d = np.exp(gauss_posteriors_2d - REGULARIZER)

    # Normalize the posteriors using 2D integral
    exact_posteriors_2d /= (np.sum(exact_posteriors_2d[~np.isnan(exact_posteriors_2d)]) * dx * dy)
    gauss_posteriors_2d /= (np.sum(gauss_posteriors_2d) * dx * dy)

    # Replace nans with zeros
    exact_posteriors_2d[np.isnan(exact_posteriors_2d)] = 0
    gauss_posteriors_2d[np.isnan(gauss_posteriors_2d)] = 0

    # Debugging print statements to verify alignment
    print("Exact posteriors shape:", exact_posteriors_2d.shape)
    print("Exact posterior integral:", np.sum(exact_posteriors_2d[~np.isnan(exact_posteriors_2d)]) * dx * dy)
    print("Gaussian posteriors shape:", gauss_posteriors_2d.shape)
    print("Gaussian posterior integral:", np.sum(gauss_posteriors_2d) * dx * dy)

    # Create output filename
    scale_cut_arcmin = metadata['scale_cut_arcmin']
    
    # Plot the 2D corner plot
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Exact posterior
    plot_2D(
        fig,
        ax[0],
        omega_m_prior,
        s8_prior,
        exact_posteriors_2d,
        colormap="viridis",
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
    
    # Create dynamic output filename
    output_file = f"2d_corner_plot_scale_{scale_cut_arcmin:.1f}arcmin_{covariance_mode}.png"
    plt.savefig(output_file)
    print(f"Saved corner plot to {output_file}")

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
    main_ax.set_xlim(0.225, 0.425)
    main_ax.set_ylim(0.77, 0.82)

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

    top_ax.set_xlim(0.225, 0.425)
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

    right_ax.set_ylim(0.77, 0.82)
    right_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    right_ax.tick_params(axis="y", labelleft=False)

    # Hide unused top-right panel
    ax[0, 1].axis("off")

    plt.tight_layout()
    
    # Save combined plot with dynamic filename
    combined_output_file = f"combined_2d_contours_marginals_scale_{scale_cut_arcmin:.1f}arcmin_{covariance_mode}.png"
    plt.savefig(combined_output_file)
    print(f"Saved combined plot to {combined_output_file}")
    
    return {
        'exact_posteriors_2d': exact_posteriors_2d,
        'gauss_posteriors_2d': gauss_posteriors_2d,
        'omega_m_prior': omega_m_prior,
        's8_prior': s8_prior,
        'metadata': metadata,
        'mean_omega_m_exact': mean_omega_m_exact,
        'mean_s8_exact': mean_s8_exact,
        'mean_omega_m_gauss': mean_omega_m_gauss,
        'mean_s8_gauss': mean_s8_gauss
    }


if __name__ == "__main__":
    import sys
    
    # Default values
    results_dir = "/cluster/scratch/veoehl/posteriors_2d"
    scale_cut_index = 0
    covariance_mode = "std_cov"
    
    if len(sys.argv) > 1:
        print("Usage: python postprocess_2dpost.py [--results-dir=<path>] [--scale-cut-index=<idx>] [--covariance-mode=<mode>]")
        print("  --results-dir=<path>: Base results directory (default: /cluster/scratch/veoehl/posteriors_2d)")
        print("  --scale-cut-index=<idx>: Scale cut index 0-5 (default: 0)")
        print("    0: 10 arcmin, 1: 20 arcmin, 2: 50 arcmin, 3: 100 arcmin, 4: 300 arcmin, 5: 1200 arcmin")
        print("  --covariance-mode=<mode>: 'std_cov' or 'fixed_cov' (default: std_cov)")
    
    # Parse arguments
    for arg in sys.argv[1:]:
        if arg.startswith('--results-dir='):
            results_dir = arg.split('=')[1]
        elif arg.startswith('--scale-cut-index='):
            scale_cut_index = int(arg.split('=')[1])
            if not (0 <= scale_cut_index <= 5):
                print(f"Error: scale-cut-index must be between 0 and 5")
                sys.exit(1)
        elif arg.startswith('--covariance-mode='):
            covariance_mode = arg.split('=')[1]
            if covariance_mode not in ['std_cov', 'fixed_cov']:
                print(f"Error: covariance-mode must be 'std_cov' or 'fixed_cov'")
                sys.exit(1)
    
    # Run processing
    scale_cuts_deg = np.array([10, 20, 50, 100, 300, 1200]) / 60.
    scale_cut = scale_cuts_deg[scale_cut_index]
    
    print(f"Processing 2D posterior results:")
    print(f"  Results directory: {results_dir}")
    print(f"  Scale cut: {scale_cut:.4f} deg ({scale_cut*60:.1f} arcmin)")
    print(f"  Covariance mode: {covariance_mode}")
    
    try:
        results = process_posteriors(results_dir, scale_cut_index, covariance_mode)
        print("Processing completed successfully!")
    except Exception as e:
        import traceback
        print(f"Error during processing: {e}")
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)
