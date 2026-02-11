"""
Postprocessing script for emcee MCMC chains from sampler.py

Produces paper-quality plots including:
- Corner plots with posterior distributions
- Trace plots showing walker convergence
- Autocorrelation analysis
- Summary statistics

Usage:
    python postprocess_emcee.py <npz_file> [--burn <burn_in>] [--thin <thin_factor>] [--output <output_dir>]
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    print("Warning: 'corner' package not found. Corner plots will not be generated.")

# Fiducial values from theory_cl.py defaults (used for mock data generation)
# When omega_m=0.31 is given: w_c = (omega_m - w_b/h^2) * h^2
# omega_b = 0.021/0.49 = 0.04286, omega_c = 0.31 - 0.04286 = 0.26714
# w_c = 0.26714 * 0.49 = 0.1309
FIDUCIALS = {
    "s8": 0.8,
    "w_c": 0.1309,
    "w_b": 0.021,
    "h": 0.7,
    "n_s": 0.97,
    "A_IA": 0.0,
    "delta_z_0": 0.0,
    "delta_z_1": 0.0,
    "delta_z_2": 0.0,
    "delta_z_3": 0.0,
    "delta_z_4": 0.0,
}

# Paper-quality plot settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_emcee_samples(filepath):
    """
    Load emcee samples from .npz file produced by sampler.py
    
    Parameters
    ----------
    filepath : str or Path
        Path to the .npz file
        
    Returns
    -------
    samples : ndarray
        Shape (n_steps, n_walkers, ndim)
    params : list
        Parameter names (expanded for multi-dimensional params)
    metadata : dict
        Additional metadata (n_walkers, n_steps, etc.)
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.npz':
        data = np.load(filepath, allow_pickle=True)
        samples = data['samples']
        params = list(data['params'])
        
        metadata = {
            'n_walkers': int(data.get('n_walkers', samples.shape[1])),
            'n_steps': int(data.get('n_steps', samples.shape[0])),
            'filepath': str(filepath),
        }
        
        # Check for checkpoint-specific fields
        if 'steps_completed' in data:
            metadata['steps_completed'] = int(data['steps_completed'])
        if 'n_steps_target' in data:
            metadata['n_steps_target'] = int(data['n_steps_target'])
            
    elif filepath.suffix == '.h5':
        # Fallback for HDF5 files (legacy support)
        import h5py
        with h5py.File(filepath, 'r') as f:
            if 'sampler' in f:
                # Nautilus-style format
                sampler_data = f['sampler']
                points = sampler_data['points_0'][:]
                params = ['omega_m', 's8']  # Default params for old format
                # Reshape to emcee format (treat as 1 walker)
                samples = points.reshape(-1, 1, points.shape[-1])
            else:
                raise ValueError(f"Unknown HDF5 structure in {filepath}")
        
        metadata = {
            'n_walkers': samples.shape[1],
            'n_steps': samples.shape[0],
            'filepath': str(filepath),
        }
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    # Expand delta_z to 5 bins if needed
    ndim = samples.shape[-1]
    if len(params) != ndim and 'delta_z' in params:
        idx = params.index('delta_z')
        params = params[:idx] + [f'delta_z_{i}' for i in range(5)] + params[idx+1:]
    
    print(f"Loaded samples: shape={samples.shape}")
    print(f"Parameters ({len(params)}): {params}")
    print(f"Metadata: {metadata}")
    
    return samples, params, metadata


def inspect_npz_file(filepath):
    """Print the structure and contents of an NPZ file."""
    filepath = Path(filepath)
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath}")
    print(f"{'='*60}")
    
    data = np.load(filepath, allow_pickle=True)
    for key in data.files:
        arr = data[key]
        if isinstance(arr, np.ndarray):
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            if arr.ndim == 0 or arr.size <= 10:
                print(f"    value: {arr}")
        else:
            print(f"  {key}: {type(arr)} = {arr}")
    print(f"{'='*60}\n")


def compute_autocorrelation_time(samples, quiet=False):
    """
    Estimate integrated autocorrelation time for each parameter.
    
    Parameters
    ----------
    samples : ndarray
        Shape (n_steps, n_walkers, ndim)
    quiet : bool
        Suppress warnings
        
    Returns
    -------
    tau : ndarray
        Autocorrelation time for each parameter
    """
    try:
        import emcee
        # Flatten across walkers for autocorrelation estimation
        tau = emcee.autocorr.integrated_time(samples, quiet=quiet)
        return tau
    except Exception as e:
        if not quiet:
            print(f"Warning: Could not compute autocorrelation time: {e}")
        return None


def plot_traces(samples, params, output_path, burn_in=0):
    """
    Create trace plots showing walker chains.
    
    Parameters
    ----------
    samples : ndarray
        Shape (n_steps, n_walkers, ndim)
    params : list
        Parameter names
    output_path : Path
        Output file path
    burn_in : int
        Burn-in period to mark with vertical line
    """
    n_steps, n_walkers, ndim = samples.shape
    
    fig, axes = plt.subplots(ndim, 1, figsize=(12, 2.5 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        for j in range(n_walkers):
            ax.plot(samples[:, j, i], alpha=0.4, lw=0.5)
        
        ax.set_ylabel(params[i] if i < len(params) else f"param {i}")
        ax.set_xlim(0, n_steps)
        
        # Mark burn-in period
        if burn_in > 0:
            ax.axvline(burn_in, color='red', linestyle='--', alpha=0.7, label='burn-in')
        
        # Add median line after burn-in
        if burn_in < n_steps:
            median_val = np.median(samples[burn_in:, :, i])
            ax.axhline(median_val, color='black', linestyle='-', alpha=0.5, lw=1)
    
    axes[-1].set_xlabel("Step")
    axes[0].set_title(f"MCMC Traces ({n_walkers} walkers, {n_steps} steps)")
    
    if burn_in > 0:
        axes[0].legend(loc='upper right')
    
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Trace plot saved: {output_path}")


def plot_corner(samples, params, output_path, burn_in=0, thin=1, 
                truths=None, title=None, quantiles=[0.16, 0.5, 0.84]):
    """
    Create corner plot with posterior distributions.
    
    Parameters
    ----------
    samples : ndarray
        Shape (n_steps, n_walkers, ndim)
    params : list
        Parameter names
    output_path : Path
        Output file path
    burn_in : int
        Number of steps to discard as burn-in
    thin : int
        Thinning factor
    truths : list, optional
        True parameter values to mark
    title : str, optional
        Plot title
    quantiles : list
        Quantiles to show in corner plot
    """
    if not HAS_CORNER:
        print("Corner package not available, skipping corner plot")
        return
    
    # Apply burn-in and thinning
    flat_samples = samples[burn_in::thin, :, :].reshape(-1, samples.shape[-1])
    
    print(f"Corner plot: using {flat_samples.shape[0]} samples "
          f"(burn-in={burn_in}, thin={thin})")
    
    # Create corner plot
    fig = corner.corner(
        flat_samples,
        labels=params,
        quantiles=quantiles,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        truths=truths,
        truth_color='red',
        bins=30,
        smooth=1.0,
        smooth1d=1.0,
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True,
        levels=(0.68, 0.95),
        color='C0',
        range=np.repeat(0.999, len(params)),
    )
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Corner plot saved: {output_path}")


def plot_posterior_scatter(samples, params, output_path, burn_in=0, thin=1):
    """
    Create 2D scatter plot of posterior samples (for 2-parameter case).
    
    Parameters
    ----------
    samples : ndarray
        Shape (n_steps, n_walkers, ndim)
    params : list
        Parameter names
    output_path : Path
        Output file path
    burn_in : int
        Burn-in steps to discard
    thin : int
        Thinning factor
    """
    flat_samples = samples[burn_in::thin, :, :].reshape(-1, samples.shape[-1])
    
    if flat_samples.shape[1] < 2:
        print("Scatter plot requires at least 2 parameters, skipping")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot samples with transparency
    ax.scatter(flat_samples[:, 0], flat_samples[:, 1], 
               s=2, alpha=0.3, c='C0', rasterized=True)
    
    ax.set_xlabel(params[0])
    ax.set_ylabel(params[1])
    ax.set_title(f'Posterior Samples (N={flat_samples.shape[0]})')
    
    # Add contours
    try:
        from scipy import stats
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        
        # Create KDE
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        kernel = stats.gaussian_kde(flat_samples[:, :2].T)
        f = np.reshape(kernel(positions).T, xx.shape)
        
        # Plot contours
        ax.contour(xx, yy, f, levels=4, colors='k', alpha=0.5, linewidths=0.5)
    except Exception as e:
        print(f"Could not add contours: {e}")
    
    fig.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Scatter plot saved: {output_path}")


def compute_summary_statistics(samples, params, burn_in=0, thin=1):
    """
    Compute and print summary statistics for the chain.
    
    Parameters
    ----------
    samples : ndarray
        Shape (n_steps, n_walkers, ndim)
    params : list
        Parameter names
    burn_in : int
        Burn-in steps to discard
    thin : int
        Thinning factor
        
    Returns
    -------
    stats : dict
        Dictionary of summary statistics
    """
    flat_samples = samples[burn_in::thin, :, :].reshape(-1, samples.shape[-1])
    n_eff = flat_samples.shape[0]
    
    print(f"\n{'='*60}")
    print(f"Summary Statistics (burn-in={burn_in}, thin={thin}, N_eff={n_eff})")
    print(f"{'='*60}")
    
    stats = {}
    for i, param in enumerate(params):
        median = np.median(flat_samples[:, i])
        mean = np.mean(flat_samples[:, i])
        std = np.std(flat_samples[:, i])
        q16, q84 = np.percentile(flat_samples[:, i], [16, 84])
        
        stats[param] = {
            'median': median,
            'mean': mean,
            'std': std,
            'q16': q16,
            'q84': q84,
        }
        
        print(f"  {param}:")
        print(f"    median = {median:.6f}")
        print(f"    mean   = {mean:.6f} ± {std:.6f}")
        print(f"    68% CI = [{q16:.6f}, {q84:.6f}]")
    
    # Compute autocorrelation time if possible
    tau = compute_autocorrelation_time(samples[burn_in:], quiet=True)
    if tau is not None:
        print(f"\n  Autocorrelation times:")
        for i, param in enumerate(params):
            print(f"    {param}: τ = {tau[i]:.1f}")
        print(f"  Effective samples: {n_eff / np.max(tau):.0f}")
    
    print(f"{'='*60}\n")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description='Postprocess emcee MCMC chains from sampler.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('filepath', type=str, 
                        help='Path to the .npz file with emcee samples')
    parser.add_argument('--burn', type=int, default=0,
                        help='Number of burn-in steps to discard')
    parser.add_argument('--thin', type=int, default=1,
                        help='Thinning factor')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: same as input file)')
    parser.add_argument('--inspect', action='store_true',
                        help='Only inspect file structure, do not plot')
    parser.add_argument('--truths', type=float, nargs='+', default=None,
                        help='True parameter values for corner plot')
    parser.add_argument('--title', type=str, default=None,
                        help='Title for corner plot')
    
    args = parser.parse_args()
    
    filepath = Path(args.filepath)
    
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return 1
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = filepath.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate base name for output files
    base_name = filepath.stem
    
    # Inspect file
    if filepath.suffix == '.npz':
        inspect_npz_file(filepath)
    
    if args.inspect:
        return 0
    
    # Load samples
    samples, params, metadata = load_emcee_samples(filepath)
    n_steps, n_walkers, ndim = samples.shape
    
    # Auto-estimate burn-in if not specified
    burn_in = args.burn
    if burn_in == 0:
        # Use first 10% as burn-in by default
        burn_in = max(1, n_steps // 10)
        print(f"Auto burn-in: {burn_in} steps (10% of chain)")
    
    # Compute summary statistics
    stats = compute_summary_statistics(samples, params, burn_in=burn_in, thin=args.thin)
    
    # Build truths array from FIDUCIALS dict based on parameter names
    if args.truths:
        truths = args.truths
    else:
        truths = [FIDUCIALS.get(p, None) for p in params]
        print(f"Fiducial values: {dict(zip(params, truths))}")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Trace plot
    trace_path = output_dir / f'{base_name}_traces.png'
    plot_traces(samples, params, trace_path, burn_in=burn_in)
    
    # Corner plot
    corner_path = output_dir / f'{base_name}_corner.png'
    plot_corner(samples, params, corner_path, 
                burn_in=burn_in, thin=args.thin,
                truths=truths, title=args.title)
    
    # Scatter plot (for 2D case)
    if ndim >= 2:
        scatter_path = output_dir / f'{base_name}_scatter.png'
        plot_posterior_scatter(samples, params, scatter_path,
                               burn_in=burn_in, thin=args.thin)
    
    print(f"\nAll plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
