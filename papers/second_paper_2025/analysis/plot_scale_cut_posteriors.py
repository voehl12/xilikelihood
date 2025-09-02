"""
plot_scale_cut_posteriors.py

Plot and compare posteriors from scale cut sweep chains.
"""
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# Try to import ArviZ for nicer corner plots
try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False
    print("Warning: ArviZ not found. Falling back to corner for plotting.")

try:
    import corner
    CORNER_AVAILABLE = True
except ImportError:
    CORNER_AVAILABLE = False
    print("Warning: corner not found. Only 1D plots will be available.")

def load_chains(chain_dir, pattern="chain_scale_cut_*.npz"):
    files = sorted(glob.glob(os.path.join(chain_dir, pattern)))
    # Exclude files ending with _stuckwalker.npz
    files = [f for f in files if not f.endswith('_stuckwalker.npz')]
    chains = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        if 'samples' in data:
            samples = data['samples']
        elif 'points' in data:
            samples = data['points']
        else:
            continue
        param_names = data['param_names'] if 'param_names' in data else [f"param{i}" for i in range(samples.shape[1])]
        scale_cut = None
        for part in f.split('_'):
            try:
                scale_cut = float(part)
                break
            except ValueError:
                continue
        chains.append({'samples': samples, 'param_names': param_names, 'file': f, 'scale_cut': scale_cut*60.})
    return chains

def plot_posteriors(chains, outdir="plots_scale_cut_posteriors"):
    os.makedirs(outdir, exist_ok=True)
    for chain in chains:
        samples_raw = chain['samples']
        param_names = list(chain['param_names'])
        scale_cut = chain['scale_cut']
        label = f"scale_cut={scale_cut:.3f}"
        
        # Flatten samples if they're not already flattened
        if len(samples_raw.shape) == 3:
            # (n_steps, n_walkers, n_params) -> (n_samples, n_params)
            samples = samples_raw.reshape(-1, samples_raw.shape[-1])
        else:
            # Already flattened
            samples = samples_raw
            
        n_params = samples.shape[1]
        if n_params >= 2 and ARVIZ_AVAILABLE:
            # Use from_dict for robust variable naming
            posterior_dict = {name: samples[:, i] for i, name in enumerate(param_names)}
            az_data = az.from_dict(posterior=posterior_dict)
            az.plot_pair(az_data, var_names=param_names, marginals=True, kind='kde', divergences=False)
            plt.suptitle(f"ArviZ KDE Posterior: {label}")
            plt.savefig(os.path.join(outdir, f"arviz_corner_scale_cut_{scale_cut:.3f}.png"))
            plt.close()
        elif n_params >= 2 and CORNER_AVAILABLE:
            fig = corner.corner(samples, labels=param_names, show_titles=True, title_fmt=".3f", label_kwargs={"fontsize":12})
            plt.suptitle(f"Corner Posterior: {label}")
            plt.savefig(os.path.join(outdir, f"corner_scale_cut_{scale_cut:.3f}.png"))
            plt.close()
        else:
            # 1D histogram for each parameter
            for i, pname in enumerate(param_names):
                plt.hist(samples[:, i], bins=30, alpha=0.7)
                plt.title(f"{pname} ({label})")
                plt.xlabel(pname)
                plt.ylabel("Frequency")
                plt.savefig(os.path.join(outdir, f"hist_{pname}_scale_cut_{scale_cut:.3f}.png"))
                plt.close()
    print(f"Plots saved to {outdir}")

def plot_overlay_contours(chains, outdir="plots_scale_cut_posteriors", param_pair=None):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from config import FIDUCIAL_COSMO
    os.makedirs(outdir, exist_ok=True)
    # Sort chains by scale_cut for color order
    chains = sorted(chains, key=lambda c: c['scale_cut'])
    colors = cm.viridis(np.linspace(0, 1, len(chains)))
    if param_pair is None:
        # Use first two parameters by default
        param_pair = [chains[0]['param_names'][0], chains[0]['param_names'][1]]
    plt.figure(figsize=(6,6))
    for i, chain in enumerate(chains):
        samples_raw = chain['samples']
        param_names = list(chain['param_names'])
        scale_cut = chain['scale_cut']
        
        # Flatten samples if they're not already flattened
        if len(samples_raw.shape) == 3:
            # (n_steps, n_walkers, n_params) -> (n_samples, n_params)
            samples = samples_raw.reshape(-1, samples_raw.shape[-1])
        else:
            # Already flattened
            samples = samples_raw
            
        # Get indices for the chosen parameters
        idx1 = param_names.index(param_pair[0])
        idx2 = param_names.index(param_pair[1])
        x = samples[:, idx1]
        y = samples[:, idx2]
        # 2D KDE for contours
        try:
            import scipy.stats as st
            xx, yy = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            values = np.vstack([x, y])
            kernel = st.gaussian_kde(values)
            f = np.reshape(kernel(positions).T, xx.shape)
            # Find contour levels for 1, 2 sigma
            levels = np.percentile(f, [68, 95])
            plt.contour(xx, yy, f, levels=levels, colors=[colors[i]], linewidths=2, label=f"{scale_cut:.1f} arcmin")
        except Exception as e:
            print(f"KDE/contour failed for scale_cut={scale_cut}: {e}")
    # Plot fiducial cosmology
    fid_x = FIDUCIAL_COSMO[param_pair[0]]
    fid_y = FIDUCIAL_COSMO[param_pair[1]]
    plt.plot(fid_x, fid_y, 'r*', markersize=14, label='Fiducial')
    plt.xlabel(param_pair[0])
    plt.ylabel(param_pair[1])
    plt.title("Posterior contours for different scale cuts")
    # Custom legend
    from matplotlib.lines import Line2D
    legend_lines = [Line2D([0], [0], color=colors[i], lw=2) for i in range(len(chains))]
    legend_labels = [f"{c['scale_cut']:.1f} arcmin" for c in chains]
    legend_lines.append(Line2D([0], [0], marker='*', color='r', lw=0, markersize=14))
    legend_labels.append('Fiducial')
    plt.legend(legend_lines, legend_labels, title="Scale cut")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "overlay_contours.png"))
    plt.close()
    print(f"Overlay contour plot saved to {outdir}/overlay_contours.png")

def plot_traces_and_autocorr(chains, outdir="plots_scale_cut_posteriors_traces", n_walkers=6, n_steps=2000, burn_in=0):
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    for chain in chains:
        samples_raw = chain['samples']
        param_names = list(chain['param_names'])
        scale_cut = chain['scale_cut']
        
        # Handle different sample shapes: check if already in (n_steps, n_walkers, n_params) or flattened
        if len(samples_raw.shape) == 3:
            # Already in (n_steps, n_walkers, n_params) format
            samples = samples_raw[burn_in:]  # Remove burn-in
            actual_n_steps, actual_n_walkers, n_params = samples.shape
            print(f"Using unflattened samples: {samples.shape}")
        elif len(samples_raw.shape) == 2:
            # Flattened format (n_samples, n_params) - need to reshape
            n_params = samples_raw.shape[1]
            try:
                samples = samples_raw.reshape((n_steps, n_walkers, n_params))[burn_in:]
                actual_n_steps, actual_n_walkers, n_params = samples.shape
                print(f"Reshaped flattened samples: {samples.shape}")
            except Exception as e:
                print(f"Reshape failed for scale_cut={scale_cut}: {e}, shape was {samples_raw.shape}")
                continue
        else:
            print(f"Unexpected sample shape for scale_cut={scale_cut}: {samples_raw.shape}")
            continue
        # Trace plots: one plot per parameter, all walkers overplotted
        colors = plt.cm.tab10.colors
        for i, pname in enumerate(param_names):
            plt.figure(figsize=(8, 3))
            for w in range(actual_n_walkers):
                plt.plot(samples[:, w, i], color=colors[w % len(colors)], alpha=0.8, label=f"walker {w+1}")
            plt.title(f"Trace plot: {pname} (scale_cut={scale_cut:.1f} arcmin)")
            plt.xlabel("Step")
            plt.ylabel(pname)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f"trace_{pname}_scale_cut_{scale_cut:.1f}.png"))
            plt.close()
        # Autocorrelation times (flattened for all walkers)
        try:
            import emcee
            tau = emcee.autocorr.integrated_time(samples, quiet=True, has_walkers=True)
            for i, pname in enumerate(param_names):
                print(f"scale_cut={scale_cut:.1f} arcmin, {pname}: autocorr time = {tau[i]:.1f} steps (flattened)")
                with open(os.path.join(outdir, f"autocorr_scale_cut_{scale_cut:.1f}.txt"), "a") as f:
                    f.write(f"{pname}: autocorr time = {tau[i]:.1f} steps (flattened)\n")
        except Exception as e:
            print(f"Autocorr time calculation failed for scale_cut={scale_cut}: {e}")
            with open(os.path.join(outdir, f"autocorr_scale_cut_{scale_cut:.1f}.txt"), "a") as f:
                f.write(f"Autocorr time calculation failed: {e}\n")
    print(f"Trace plots and autocorr times saved to {outdir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot posteriors from scale cut sweep chains.")
    parser.add_argument('--chain_dir', type=str, default="/cluster/scratch/veoehl/scale_cut_chains", help="Directory with .npz chain files.")
    parser.add_argument('--outdir', type=str, default="plots_scale_cut_posteriors", help="Output directory for plots.")
    parser.add_argument('--overlay', action='store_true', help="Also plot overlayed contours for all scale cuts.")
    parser.add_argument('--traces', action='store_true', help="Also plot trace plots and compute autocorrelation times.")
    args = parser.parse_args()
    chains = load_chains(args.chain_dir)
    plot_posteriors(chains, outdir=args.outdir)
    if args.overlay:
        plot_overlay_contours(chains, outdir=args.outdir)
    if args.traces:
        plot_traces_and_autocorr(chains, outdir=args.outdir, n_walkers=6, n_steps=2000)