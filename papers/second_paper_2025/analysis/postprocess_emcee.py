"""
Postprocessing script for emcee MCMC chains from sampler.py

Produces paper-quality plots including:
- Corner plots with posterior distributions
- Trace plots showing walker convergence
- Autocorrelation analysis
- Summary statistics

Usage:
    # Single chain
    python postprocess_emcee.py chain.npz [--output <output_dir>] [--config <config_file>]
    
    # Multiple chains (explicit paths)
    python postprocess_emcee.py chain1.npz chain2.npz chain3.npz [--config <config_file>]
    
    # Multiple chains (auto-discovery from base path)
    python postprocess_emcee.py --base /path/to/chain [--output <output_dir>] [--config <config_file>]
    # This will automatically find chain1.npz, chain2.npz, etc. or chain_1.npz, chain_2.npz, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import re

from plot_utils_style import configure_paper_plot_style, get_single_chain_color
import postprocess_utils as putils

try:
    import getdist
    from getdist import plots as gplots
    from getdist.plots import GetDistPlotSettings
    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False
    print("Warning: 'getdist' package not found. Corner plots will not be generated.")

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
    # Omega parameters (Ω = w/h^2, calculated from fiducial values)
    # Ω_c = 0.1309 / (0.7^2) = 0.26714, Ω_b = 0.021 / (0.7^2) = 0.04286
    # Ω_m = Ω_c + Ω_b = 0.31
    "omega_c": 0.26714,
    "omega_b": 0.04286,
    "omega_m": 0.31,
}

DEFAULT_CONFIG = {
    'analysis': {
        'inspect': False,
        'burn': 0,
        'thin': 1,
        'truths': None,
        'title': None,
        'params': None,
        'param_labels': None,
        'omega_transform': False,  # Transform w params to omega = w/h^2
        'smooth_scale_1D': 1.5,  # Smoothing for 1D marginals (lower = sharper)
        'smooth_scale_2D': 1.5,  # Smoothing for 2D contours (lower = sharper)
        'corner_fig_size': None,  # GetDist corner plot figure size (e.g., 8 for 8 inches, or (8, 8) for (width, height))
    },
    'stuck_filter': {
        'enabled': False,
        'atol': 0.0,
        'rtol': 1e-12,
        'min_run': 5,
        'pre_padding': 1,
        'post_padding': 0,
        'remove_all_zero_steps': True,
        'zero_tol': 0.0,
        'remove_isolated_global_drops': True,
        'global_drop_rel': 0.02,
        'params': None,
    },
}

# Default LaTeX plotting labels (override via [analysis].param_labels in TOML)
DEFAULT_PARAM_LABELS = {
    's8': r'$S_8$',
    'w_c': r'$\omega_c$',
    'w_b': r'$\omega_b$',
    'h': r'$h$',
    'n_s': r'$n_s$',
    'A_IA': r'$A_{\mathrm{IA}}$',
    'omega_c': r'$\Omega_c$',
    'omega_b': r'$\Omega_b$',
    'omega_m': r'$\Omega_m$',
}

# Paper-quality plot settings (shared across analysis scripts)
configure_paper_plot_style()


def get_plot_labels(params, param_labels=None):
    """Return plotting labels (LaTeX if available) for a parameter list."""
    labels = dict(DEFAULT_PARAM_LABELS)
    if isinstance(param_labels, dict):
        labels.update(param_labels)
    return [labels.get(p, p) for p in params]


def load_postprocess_config(config_path):
    """Load TOML config file and merge with defaults."""
    return putils.load_postprocess_config(
        config_path,
        DEFAULT_CONFIG,
        relative_to=Path(__file__).resolve().parent,
    )


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

        # Safety: if this file contains checkpoint metadata, honor the number
        # of completed steps to avoid including any preallocated/invalid tail rows.
        if 'steps_completed' in metadata:
            completed = metadata['steps_completed']
            if completed < 0:
                raise ValueError(f"Invalid steps_completed={completed} in {filepath}")
            if completed < samples.shape[0]:
                print(
                    f"Trimming chain to steps_completed={completed} "
                    f"from raw shape {samples.shape}"
                )
                samples = samples[:completed]
                metadata['n_steps'] = completed
            
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


def plot_corner(samples, params, output_path, burn_in=0, thin=1,
                truths=None, title=None, quantiles=[0.16, 0.5, 0.84],
                selected_params=None, param_labels=None,
                smooth_scale_1D=0.6, smooth_scale_2D=0.6, corner_fig_size=None):
    """
    Create corner plot with posterior distributions using GetDist.
    
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
    smooth_scale_1D : float
        Smoothing scale for 1D marginal distributions (default 0.6; lower = sharper)
    smooth_scale_2D : float
        Smoothing scale for 2D contours (default 0.6; lower = sharper)
    corner_fig_size : float or tuple, optional
        Figure size in inches (e.g., 8 for square, or (8, 10) for (width, height))
    """
    if not HAS_GETDIST:
        print("GetDist package not available, skipping corner plot")
        return
    
    # Apply burn-in and thinning
    flat_samples = samples[burn_in::thin, :, :].reshape(-1, samples.shape[-1])
    
    # Select subset of parameters if specified
    plot_params = params
    plot_truths = truths
    if selected_params is not None:
        flat_samples, plot_params, plot_truths = putils.select_parameters(
            flat_samples, params, selected_params, truths
        )
        print(f"Plotting {len(plot_params)} parameters (marginalized over others): {plot_params}")
    
    print(f"Corner plot: using {flat_samples.shape[0]} samples "
          f"(burn-in={burn_in}, thin={thin})")

    # GetDist smoothing is controlled via MCSamples settings.
    # Values <= 0 trigger automatic bandwidth estimation.
    smoothing_settings = {
        'smooth_scale_1D': float(smooth_scale_1D),
        'smooth_scale_2D': float(smooth_scale_2D),
    }
    print(
        "GetDist smoothing settings: "
        f"smooth_scale_1D={smoothing_settings['smooth_scale_1D']}, "
        f"smooth_scale_2D={smoothing_settings['smooth_scale_2D']}"
    )
    
    # Create MCSamples object for GetDist
    plot_labels = get_plot_labels(plot_params, param_labels)
    samples_obj = getdist.MCSamples(
        samples=flat_samples,
        names=plot_params,
        labels=plot_labels,
        settings=smoothing_settings,
    )
    
    # Configure color scheme
    color = get_single_chain_color()
    
 
    # Create triangle plot (corner plot equivalent in GetDist)
    g = gplots.get_subplot_plotter(width_inch = corner_fig_size)
    g.triangle_plot(
        [samples_obj],
        filled=True,
        contour_colors=[color],
        line_args=[{'color': color, 'lw': 1.0}],
    )
    
    # Add truth values if provided
    if plot_truths is not None:
        valid_truths = [t for t in plot_truths if t is not None]
        if valid_truths:
            # Mark truths on diagonal (1D) and lower triangle (2D) panels.
            n_plot = len(plot_params)
            for row in range(n_plot):
                for col in range(n_plot):
                    ax = g.subplots[row, col]
                    if ax is None:
                        continue

                    # Diagonal: 1D marginal for parameter ``row``.
                    if row == col:
                        if row < len(plot_truths) and plot_truths[row] is not None:
                            ax.axvline(plot_truths[row], color='black', linestyle='--', alpha=0.7)

                    # Lower triangle: 2D panel for y=row, x=col.
                    elif row > col:
                        x_truth = plot_truths[col] if col < len(plot_truths) else None
                        y_truth = plot_truths[row] if row < len(plot_truths) else None
                        if x_truth is not None:
                            ax.axvline(x_truth, color='black', linestyle='--', alpha=0.7)
                        if y_truth is not None:
                            ax.axhline(y_truth, color='black', linestyle='--', alpha=0.7)
    
    # Add title if provided
    if title:
        g.fig.suptitle(title, fontsize=14, y=0.995)
    
    g.fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    
    # Also save as PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight',dpi=300)
    
    plt.close(g.fig)
    print(f"Corner plot saved: {output_path}")
    print(f"Corner plot (PDF) saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Postprocess emcee MCMC chains from sampler.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('filepaths', type=str, nargs='*',
                        help='Path(s) to the .npz file(s) with emcee samples')
    parser.add_argument('--base', type=str, default=None,
                        help='Base path for auto-discovering chains (e.g., /path/to/chain finds chain1.npz, chain2.npz, etc.)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory (default: same as input file)')
    parser.add_argument('--config', type=str, default='postprocess_emcee.toml',
                        help='TOML config file with analysis/plot/filter settings')
    
    args = parser.parse_args()

    config = load_postprocess_config(args.config)
    analysis_cfg = config['analysis']
    stuck_cfg = config['stuck_filter']

    burn_in = int(analysis_cfg['burn'])
    thin = int(analysis_cfg['thin'])
    if burn_in < 0:
        print(f"Error: burn must be >= 0, got {burn_in}")
        return 1
    if thin < 1:
        print(f"Error: thin must be >= 1, got {thin}")
        return 1

    params_to_plot = analysis_cfg['params']
    if params_to_plot is not None and not isinstance(params_to_plot, list):
        print("Error: [analysis].params in config must be a list of parameter names")
        return 1

    param_labels = analysis_cfg.get('param_labels')
    if param_labels is not None and not isinstance(param_labels, dict):
        print("Error: [analysis].param_labels in config must be a key/value table")
        return 1

    truths_from_config = analysis_cfg['truths']
    if truths_from_config is not None and not isinstance(truths_from_config, list):
        print("Error: [analysis].truths in config must be a list of numbers")
        return 1

    smooth_scale_1D = float(analysis_cfg.get('smooth_scale_1D', 0.6))
    smooth_scale_2D = float(analysis_cfg.get('smooth_scale_2D', 0.6))
    if smooth_scale_1D <= 0:
        print("Error: [analysis].smooth_scale_1D must be > 0 to disable auto bandwidth")
        return 1
    if smooth_scale_2D <= 0:
        print("Error: [analysis].smooth_scale_2D must be > 0 to disable auto bandwidth")
        return 1

    corner_fig_size = analysis_cfg.get('corner_fig_size')
    if corner_fig_size is not None:
        # Accept both scalar (for square) and tuple (width, height)
        if isinstance(corner_fig_size, (list, tuple)) and len(corner_fig_size) == 2:
            corner_fig_size = tuple(corner_fig_size)
        else:
            try:
                corner_fig_size = float(corner_fig_size)
            except (ValueError, TypeError):
                print(f"Warning: corner_fig_size '{corner_fig_size}' could not be parsed, using default")
                corner_fig_size = None

    if stuck_cfg['min_run'] < 1:
        print(f"Error: [stuck_filter].min_run must be >= 1, got {stuck_cfg['min_run']}")
        return 1
    if stuck_cfg['pre_padding'] < 0:
        print(f"Error: [stuck_filter].pre_padding must be >= 0, got {stuck_cfg['pre_padding']}")
        return 1
    if stuck_cfg['post_padding'] < 0:
        print(f"Error: [stuck_filter].post_padding must be >= 0, got {stuck_cfg['post_padding']}")
        return 1
    if stuck_cfg['zero_tol'] < 0:
        print(f"Error: [stuck_filter].zero_tol must be >= 0, got {stuck_cfg['zero_tol']}")
        return 1
   

    print(f"Using config: {args.config}")
    
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
    
    if params_to_plot:
        base_name += "_" + "_".join(params_to_plot)
    
    # Add omega suffix if omega transformation will be applied
    omega_requested = bool(params_to_plot and any(p.startswith('omega') for p in params_to_plot))
    if bool(analysis_cfg.get('omega_transform', False)) or omega_requested:
        base_name += "_omega"
    
    # Inspect files
    if analysis_cfg['inspect']:
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
    trace_step_indices = None

    # Optionally remove pathological stuck segments from the middle of the chain.
    if stuck_cfg['enabled']:
        samples, kept_mask, removed_runs, detection_params, trace_step_indices, n_zero_removed, n_global_drop_removed = putils.remove_stuck_steps(
            samples,
            params,
            atol=float(stuck_cfg['atol']),
            rtol=float(stuck_cfg['rtol']),
            min_run=int(stuck_cfg['min_run']),
            pre_padding=int(stuck_cfg['pre_padding']),
            post_padding=int(stuck_cfg['post_padding']),
            remove_all_zero_steps=bool(stuck_cfg['remove_all_zero_steps']),
            zero_tol=float(stuck_cfg['zero_tol']),
            remove_isolated_global_drops=bool(stuck_cfg['remove_isolated_global_drops']),
            global_drop_rel=float(stuck_cfg['global_drop_rel']),
            selected_params=stuck_cfg['params'],
        )

        n_removed = np.count_nonzero(~kept_mask)
        print(f"\nStuck-step filtering enabled:")
        print(f"  Detection params: {detection_params}")
        print(f"  Tolerance: atol={stuck_cfg['atol']}, rtol={stuck_cfg['rtol']}")
        print(f"  Minimum run length: {stuck_cfg['min_run']}")
        print(f"  Pre-padding: {stuck_cfg['pre_padding']}")
        print(f"  Post-padding: {stuck_cfg['post_padding']}")
        print(f"  Remove all-zero steps: {stuck_cfg['remove_all_zero_steps']} (tol={stuck_cfg['zero_tol']})")
        print(f"  Zero-steps removed: {n_zero_removed}")
        print(f"  Remove isolated global drops: {stuck_cfg['remove_isolated_global_drops']} (rel={stuck_cfg['global_drop_rel']})")
        print(f"  Isolated global-drop steps removed: {n_global_drop_removed}")
        print(f"  Removed steps: {n_removed} / {len(kept_mask)}")

        if removed_runs:
            print("  Removed step ranges (original indexing):")
            for start, end in removed_runs:
                print(f"    [{start}, {end}] (len={end - start + 1})")
        else:
            print("  No stuck runs meeting the minimum run length were found.")

        if samples.shape[0] == 0:
            print("Error: Stuck-step filtering removed all samples. Adjust tolerances or min run length.")
            return 1

        n_steps, n_walkers, ndim = samples.shape
    
    if burn_in >= n_steps:
        print(f"Error: burn-in ({burn_in}) must be smaller than available steps ({n_steps})")
        return 1
    
    # Determine if omega transformation is needed
    omega_transform = bool(analysis_cfg.get('omega_transform', False))
    # Also auto-enable if any omega parameters are requested to plot
    if params_to_plot:
        omega_params_requested = any(p.startswith('omega') for p in params_to_plot)
        omega_transform = omega_transform or omega_params_requested
        if omega_params_requested:
            print(f"Omega parameters requested in plot. Auto-enabling omega transformation.")
    
    # Apply omega transformation if needed
    if omega_transform:
        samples, params, truths_omega = putils.apply_omega_transform(
            samples,
            params,
            [FIDUCIALS.get(p, None) for p in params],
        )
        # Update truths if omega transform was applied
        if truths_from_config is None:
            truths_from_config = truths_omega
    
    
    
    # Build truths array from FIDUCIALS dict based on parameter names
    if truths_from_config:
        truths = truths_from_config
    else:
        truths = [FIDUCIALS.get(p, None) for p in params]
        if omega_transform:
            print(f"Updated fiducial values for transformed parameters: {dict(zip(params, truths))}")
        else:
            print(f"Fiducial values: {dict(zip(params, truths))}")
    
    # Create plots
    print("\nGenerating plots...")
    
    # Trace plot
    trace_path = output_dir / f'{base_name}_traces.png'
    putils.plot_traces(
        samples,
        params,
        trace_path,
        burn_in=burn_in,
        get_plot_labels_fn=get_plot_labels,
        param_labels=param_labels,
    )
    
    # Corner plot
    corner_path = output_dir / f'{base_name}_corner.png'
    plot_corner(samples, params, corner_path, 
                burn_in=burn_in, thin=thin,
                truths=truths, title=analysis_cfg['title'],
                selected_params=params_to_plot,
                param_labels=param_labels,
                smooth_scale_1D=smooth_scale_1D,
                smooth_scale_2D=smooth_scale_2D,
                corner_fig_size=corner_fig_size)
    
    # Scatter plot (for 2D case)
    if ndim >= 2:
        scatter_path = output_dir / f'{base_name}_scatter.png'
        putils.plot_posterior_scatter(
            samples,
            params,
            scatter_path,
            burn_in=burn_in,
            thin=thin,
            get_plot_labels_fn=get_plot_labels,
            param_labels=param_labels,
        )
    
    print(f"\nAll plots saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    exit(main())
