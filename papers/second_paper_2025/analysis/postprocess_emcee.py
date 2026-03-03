"""
Postprocessing script for emcee MCMC chains from sampler.py

Produces paper-quality plots including:
- Corner plots with posterior distributions
- Trace plots showing walker convergence
- Autocorrelation analysis
- Summary statistics

Usage:
    # Single chain
    python postprocess_emcee.py chain.npz [--burn <burn_in>] [--thin <thin_factor>] [--output <output_dir>]
    
    # Multiple chains (explicit paths)
    python postprocess_emcee.py chain1.npz chain2.npz chain3.npz [--burn <burn_in>]
    
    # Multiple chains (auto-discovery from base path)
    python postprocess_emcee.py --base /path/to/chain [--burn <burn_in>]
    # This will automatically find chain1.npz, chain2.npz, etc. or chain_1.npz, chain_2.npz, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import glob
import re

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


def discover_chain_files(base_path):
    """
    Discover chain files from a base path.
    
    Looks for patterns like:
    - {base}1.npz, {base}2.npz, ...
    - {base}_1.npz, {base}_2.npz, ...
    - {base}chain1.npz, {base}chain2.npz, ...
    - {base}_chain1.npz, {base}_chain2.npz, ...
    
    Parameters
    ----------
    base_path : str or Path
        Base path (without the chain number suffix)
        
    Returns
    -------
    chain_files : list of Path
        Sorted list of discovered chain files
    """
    base_path = Path(base_path)
    base_dir = base_path.parent
    base_name = base_path.name
    
    # Try various patterns
    patterns = [
        f"{base_name}[0-9]*.npz",           # chain1.npz, chain2.npz, ...
        f"{base_name}_[0-9]*.npz",          # chain_1.npz, chain_2.npz, ...
        f"{base_name}chain[0-9]*.npz",      # basechainX.npz
        f"{base_name}_chain[0-9]*.npz",     # base_chainX.npz
        f"{base_name}_chain_[0-9]*.npz",    # base_chain_X.npz
    ]
    
    found_files = []
    for pattern in patterns:
        matches = list(base_dir.glob(pattern))
        if matches:
            found_files = matches
            break
    
    if not found_files:
        # Also check if base_path itself exists with .npz
        if base_path.with_suffix('.npz').exists():
            return [base_path.with_suffix('.npz')]
        raise FileNotFoundError(
            f"No chain files found matching base path: {base_path}\n"
            f"Tried patterns: {patterns}"
        )
    
    # Sort by the numeric part
    def extract_number(path):
        # Extract trailing numbers from stem
        match = re.search(r'(\d+)$', path.stem)
        return int(match.group(1)) if match else 0
    
    found_files.sort(key=extract_number)
    
    return found_files


def load_multiple_chains(filepaths, verbose=True):
    """
    Load and concatenate multiple emcee chain files.
    
    Parameters
    ----------
    filepaths : list of str or Path
        List of paths to .npz files
    verbose : bool
        Print information about each chain
        
    Returns
    -------
    combined_samples : ndarray
        Concatenated samples, shape (total_steps, n_walkers, ndim)
    params : list
        Parameter names
    metadata : dict
        Combined metadata
    """
    all_samples = []
    params = None
    combined_metadata = {
        'n_chains': len(filepaths),
        'filepaths': [str(p) for p in filepaths],
        'n_steps_per_chain': [],
        'n_walkers': None,
    }
    
    for i, filepath in enumerate(filepaths):
        filepath = Path(filepath)
        if verbose:
            print(f"Loading chain {i+1}/{len(filepaths)}: {filepath.name}")
        
        samples, chain_params, metadata = load_emcee_samples(filepath)
        
        # Validate consistency across chains
        if params is None:
            params = chain_params
            combined_metadata['n_walkers'] = metadata['n_walkers']
        else:
            if chain_params != params:
                raise ValueError(
                    f"Parameter mismatch in chain {filepath}:\n"
                    f"  Expected: {params}\n"
                    f"  Got: {chain_params}"
                )
            if metadata['n_walkers'] != combined_metadata['n_walkers']:
                raise ValueError(
                    f"Walker count mismatch in chain {filepath}:\n"
                    f"  Expected: {combined_metadata['n_walkers']}\n"
                    f"  Got: {metadata['n_walkers']}"
                )
        
        all_samples.append(samples)
        combined_metadata['n_steps_per_chain'].append(samples.shape[0])
    
    # Concatenate along the steps axis
    combined_samples = np.concatenate(all_samples, axis=0)
    combined_metadata['n_steps'] = combined_samples.shape[0]
    
    if verbose:
        print(f"\nCombined {len(filepaths)} chains:")
        print(f"  Total steps: {combined_metadata['n_steps']}")
        print(f"  Steps per chain: {combined_metadata['n_steps_per_chain']}")
        print(f"  Walkers: {combined_metadata['n_walkers']}")
        print(f"  Parameters: {params}")
    
    return combined_samples, params, combined_metadata


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


def select_parameters(samples, params, selected_params, truths=None):
    """
    Select a subset of parameters from samples (marginalizing over others).
    
    Parameters
    ----------
    samples : ndarray
        Shape (n_steps, n_walkers, ndim) or (n_samples, ndim)
    params : list
        All parameter names
    selected_params : list
        Parameter names to keep
    truths : list, optional
        Truth values for all parameters
        
    Returns
    -------
    selected_samples : ndarray
        Samples with only selected parameters
    selected_params : list
        Selected parameter names
    selected_truths : list or None
        Truth values for selected parameters
    """
    # Find indices of selected parameters
    indices = []
    valid_params = []
    for p in selected_params:
        if p in params:
            indices.append(params.index(p))
            valid_params.append(p)
        else:
            print(f"Warning: Parameter '{p}' not found in chain. Available: {params}")
    
    if not indices:
        raise ValueError(f"No valid parameters found. Available: {params}")
    
    # Select columns
    if samples.ndim == 3:
        selected_samples = samples[:, :, indices]
    else:
        selected_samples = samples[:, indices]
    
    # Select truths if provided
    selected_truths = None
    if truths is not None:
        selected_truths = [truths[i] for i in indices]
    
    return selected_samples, valid_params, selected_truths


def plot_corner(samples, params, output_path, burn_in=0, thin=1, 
                truths=None, title=None, quantiles=[0.16, 0.5, 0.84],
                selected_params=None):
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
    selected_params : list, optional
        Subset of parameters to plot (others are marginalized over)
    """
    if not HAS_CORNER:
        print("Corner package not available, skipping corner plot")
        return
    
    # Apply burn-in and thinning
    flat_samples = samples[burn_in::thin, :, :].reshape(-1, samples.shape[-1])
    
    # Select subset of parameters if specified
    plot_params = params
    plot_truths = truths
    if selected_params is not None:
        flat_samples, plot_params, plot_truths = select_parameters(
            flat_samples, params, selected_params, truths
        )
        print(f"Plotting {len(plot_params)} parameters (marginalized over others): {plot_params}")
    
    print(f"Corner plot: using {flat_samples.shape[0]} samples "
          f"(burn-in={burn_in}, thin={thin})")
    
    # Create corner plot
    fig = corner.corner(
        flat_samples,
        labels=plot_params,
        quantiles=quantiles,
        show_titles=True,
        title_kwargs={"fontsize": 12},
        truths=plot_truths,
        truth_color='red',
        bins=30,
        smooth=1.0,
        smooth1d=1.0,
        plot_datapoints=False,
        plot_density=True,
        fill_contours=True,
        levels=(0.68, 0.95),
        color='C0',
        range=np.repeat(0.999, len(plot_params)),
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
    parser.add_argument('filepaths', type=str, nargs='*',
                        help='Path(s) to the .npz file(s) with emcee samples')
    parser.add_argument('--base', type=str, default=None,
                        help='Base path for auto-discovering chains (e.g., /path/to/chain finds chain1.npz, chain2.npz, etc.)')
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
    parser.add_argument('--params', type=str, nargs='+', default=None,
                        help='Subset of parameters to plot (others marginalized). E.g., --params s8 w_c h')
    
    args = parser.parse_args()
    
    # Determine which files to process
    if args.base:
        # Auto-discover chains from base path
        try:
            filepaths = discover_chain_files(args.base)
            print(f"Discovered {len(filepaths)} chain files from base '{args.base}':")
            for fp in filepaths:
                print(f"  - {fp}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
    elif args.filepaths:
        filepaths = [Path(fp) for fp in args.filepaths]
    else:
        print("Error: Must provide either file path(s) or --base argument")
        parser.print_help()
        return 1
    
    # Validate all files exist
    for fp in filepaths:
        if not fp.exists():
            print(f"Error: File not found: {fp}")
            return 1
    
    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = filepaths[0].parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate base name for output files
    if len(filepaths) == 1:
        base_name = filepaths[0].stem
    else:
        # Use common prefix or first file's stem + "_combined"
        base_name = filepaths[0].stem.rstrip('0123456789').rstrip('_') + "_combined"
    
    if args.params:
        base_name += "_" + "_".join(args.params)
    
    # Inspect files
    if args.inspect:
        for fp in filepaths:
            if fp.suffix == '.npz':
                inspect_npz_file(fp)
        return 0
    
    # Load samples (single or multiple chains)
    if len(filepaths) == 1:
        filepath = filepaths[0]
        if filepath.suffix == '.npz':
            inspect_npz_file(filepath)
        samples, params, metadata = load_emcee_samples(filepath)
    else:
        # Inspect first file
        if filepaths[0].suffix == '.npz':
            inspect_npz_file(filepaths[0])
        samples, params, metadata = load_multiple_chains(filepaths)
    
    n_steps, n_walkers, ndim = samples.shape
    
    # Auto-estimate burn-in if not specified
    burn_in = args.burn
    
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
                truths=truths, title=args.title,
                selected_params=args.params)
    
    # Scatter plot (for 2D case)
    if ndim >= 2:
        scatter_path = output_dir / f'{base_name}_scatter.png'
        plot_posterior_scatter(samples, params, scatter_path,
                               burn_in=burn_in, thin=args.thin)
    
    print(f"\nAll plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
