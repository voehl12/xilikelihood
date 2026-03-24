"""Shared postprocessing helpers for emcee chain analysis scripts."""

from typing import Callable, List, Optional, Sequence, Tuple
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_utils_style import get_exact_gaussian_colors

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:
    tomllib = None


def compute_autocorrelation_time(samples, quiet: bool = False):
    """Estimate integrated autocorrelation time for each parameter."""
    try:
        import emcee

        return emcee.autocorr.integrated_time(samples, quiet=quiet)
    except Exception as e:
        if not quiet:
            print(f"Warning: Could not compute autocorrelation time: {e}")
        return None


def load_postprocess_config(config_path, default_config, relative_to: Optional[Path] = None):
    """Load TOML config file and merge with defaults.

    Parameters
    ----------
    config_path : str or Path
        Config file path.
    default_config : dict
        Nested default configuration dict with at least 'analysis' and
        'stuck_filter' sections.
    relative_to : Path, optional
        Optional directory to resolve relative config path fallback.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        if relative_to is not None:
            alt_path = Path(relative_to) / config_path
            if alt_path.exists():
                config_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Config file not found: {config_path}. "
                    "Create one from the template or pass --config <file>."
                )
        else:
            raise FileNotFoundError(
                f"Config file not found: {config_path}. "
                "Create one from the template or pass --config <file>."
            )

    if tomllib is None:
        try:
            import tomli as toml_reader
        except ModuleNotFoundError as e:
            raise ImportError(
                "TOML parsing requires Python 3.11+ (tomllib) or tomli on older Python. "
                "Install with: pip install tomli"
            ) from e
    else:
        toml_reader = tomllib

    with open(config_path, "rb") as f:
        raw_config = toml_reader.load(f)

    config = {
        "analysis": dict(default_config["analysis"]),
        "stuck_filter": dict(default_config["stuck_filter"]),
    }

    if "comparison" in default_config:
        config["comparison"] = dict(default_config["comparison"])

    for section in ("analysis", "stuck_filter", "comparison"):
        if section in raw_config and section in config:
            config[section].update(raw_config[section])

    # Backward compatibility: map legacy symmetric padding key.
    if "stuck_filter" in raw_config:
        sf_raw = raw_config["stuck_filter"]
        if "padding" in sf_raw:
            if "pre_padding" not in sf_raw:
                config["stuck_filter"]["pre_padding"] = sf_raw["padding"]
            if "post_padding" not in sf_raw:
                config["stuck_filter"]["post_padding"] = sf_raw["padding"]

    for section, key in [
        ("analysis", "truths"),
        ("analysis", "params"),
        ("analysis", "param_labels"),
        ("stuck_filter", "params"),
    ]:
        if section in config:
            val = config[section].get(key)
            if isinstance(val, list) and len(val) == 0:
                config[section][key] = None

    return config


def _find_true_runs(mask):
    """Return list of (start, end) index pairs for consecutive True runs."""
    runs = []
    run_start = None
    for i, val in enumerate(mask):
        if val and run_start is None:
            run_start = i
        elif not val and run_start is not None:
            runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        runs.append((run_start, len(mask) - 1))
    return runs


def remove_stuck_steps(
    samples,
    params,
    atol=0.0,
    rtol=1e-12,
    min_run=5,
    pre_padding=1,
    post_padding=0,
    remove_all_zero_steps=True,
    zero_tol=0.0,
    remove_isolated_global_drops=True,
    global_drop_rel=0.02,
    selected_params=None,
):
    """Remove MCMC steps where walkers are effectively stuck.

    This is intentionally lightweight for plotting workflows:
    - detect low walker spread per step for selected parameters
    - remove runs longer than ``min_run`` with optional pre/post padding
    - optionally remove exact/near-zero steps

    Parameters ``remove_isolated_global_drops`` and ``global_drop_rel`` are
    accepted for backward compatibility but not used in this simplified version.
    """
    if samples.ndim != 3:
        raise ValueError(f"Expected samples with shape (n_steps, n_walkers, ndim), got {samples.shape}")

    n_steps, _, _ = samples.shape
    if n_steps == 0:
        return samples, np.array([], dtype=bool), [], [], np.array([], dtype=int), 0, 0

    min_run = max(1, int(min_run))
    pre_padding = max(0, int(pre_padding))
    post_padding = max(0, int(post_padding))
    zero_tol = max(0.0, float(zero_tol))

    if selected_params is None:
        param_indices = list(range(samples.shape[-1]))
        detection_params = list(params)
    else:
        param_indices = []
        detection_params = []
        for p in selected_params:
            if p in params:
                param_indices.append(params.index(p))
                detection_params.append(p)
            else:
                print(f"Warning: Stuck-detection parameter '{p}' not in chain. Ignoring it.")
        if not param_indices:
            raise ValueError(
                "No valid --stuck-params were found in the chain. "
                f"Available parameters: {params}"
            )

    selected = samples[:, :, param_indices]
    spread = np.ptp(selected, axis=1)  # shape: (n_steps, n_selected_params)
    scale = np.max(np.abs(selected), axis=1)
    tol = float(atol) + float(rtol) * np.maximum(scale, 1.0)

    is_step_stuck = np.all(spread <= tol, axis=1)
    stuck_runs = _find_true_runs(is_step_stuck)
    is_step_all_zero = np.all(np.abs(selected) <= zero_tol, axis=(1, 2))

    remove_mask = np.zeros(n_steps, dtype=bool)
    for start, end in stuck_runs:
        if (end - start + 1) >= min_run:
            remove_start = max(0, start - pre_padding)
            remove_end = min(n_steps - 1, end + post_padding)
            remove_mask[remove_start:remove_end + 1] = True

    if remove_all_zero_steps:
        remove_mask |= is_step_all_zero

    removed_runs = _find_true_runs(remove_mask)
    n_zero_removed = int(np.count_nonzero(remove_mask & is_step_all_zero))
    n_global_drop_removed = 0

    kept_mask = ~remove_mask
    filtered_samples = samples[kept_mask]
    kept_indices = np.where(kept_mask)[0]

    return (
        filtered_samples,
        kept_mask,
        removed_runs,
        detection_params,
        kept_indices,
        n_zero_removed,
        n_global_drop_removed,
    )


def select_parameters(samples, params, selected_params, truths=None):
    """Select a subset of parameters from samples (marginalizing over others)."""
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

    if samples.ndim == 3:
        selected_samples = samples[:, :, indices]
    else:
        selected_samples = samples[:, indices]

    selected_truths = None
    if truths is not None:
        selected_truths = [truths[i] for i in indices]

    return selected_samples, valid_params, selected_truths


def apply_omega_transform(samples, params, truths=None):
    """Transform density params from w-space to omega-space."""
    h_idx = params.index("h") if "h" in params else None
    w_c_idx = params.index("w_c") if "w_c" in params else None
    w_b_idx = params.index("w_b") if "w_b" in params else None

    if h_idx is None:
        raise ValueError("Parameter 'h' not found in chain. Cannot apply omega transformation.")

    print("\nApplying Omega transformation (Omega = w/h^2)...")

    if samples.ndim == 3:
        h_samples = samples[:, :, h_idx]
        omega_c_samples = samples[:, :, w_c_idx] / (h_samples ** 2) if w_c_idx is not None else None
        omega_b_samples = samples[:, :, w_b_idx] / (h_samples ** 2) if w_b_idx is not None else None
    else:
        h_samples = samples[:, h_idx : h_idx + 1]
        omega_c_samples = samples[:, w_c_idx : w_c_idx + 1] / (h_samples ** 2) if w_c_idx is not None else None
        omega_b_samples = samples[:, w_b_idx : w_b_idx + 1] / (h_samples ** 2) if w_b_idx is not None else None

    omega_m_samples = (
        omega_c_samples + omega_b_samples
        if omega_c_samples is not None and omega_b_samples is not None
        else None
    )

    new_params = []
    new_samples_list = []
    w_indices_to_remove = set()
    if w_c_idx is not None:
        w_indices_to_remove.add(w_c_idx)
    if w_b_idx is not None:
        w_indices_to_remove.add(w_b_idx)

    for i, p in enumerate(params):
        if i not in w_indices_to_remove:
            new_params.append(p)
            if samples.ndim == 3:
                new_samples_list.append(samples[:, :, i])
            else:
                new_samples_list.append(samples[:, i : i + 1])

    if omega_c_samples is not None:
        new_params.append("omega_c")
        new_samples_list.append(omega_c_samples)
    if omega_b_samples is not None:
        new_params.append("omega_b")
        new_samples_list.append(omega_b_samples)
    if omega_m_samples is not None:
        new_params.append("omega_m")
        new_samples_list.append(omega_m_samples)

    transformed_samples = (
        np.stack(new_samples_list, axis=-1)
        if samples.ndim == 3
        else np.concatenate(new_samples_list, axis=1)
    )

    transformed_truths = None
    if truths is not None:
        transformed_truths = []
        for i, _ in enumerate(params):
            if i not in w_indices_to_remove and i < len(truths):
                transformed_truths.append(truths[i])

        if w_c_idx is not None and w_c_idx < len(truths) and h_idx < len(truths):
            transformed_truths.append(truths[w_c_idx] / (truths[h_idx] ** 2))
        if w_b_idx is not None and w_b_idx < len(truths) and h_idx < len(truths):
            transformed_truths.append(truths[w_b_idx] / (truths[h_idx] ** 2))
        if omega_m_samples is not None and w_c_idx is not None and w_b_idx is not None:
            if w_c_idx < len(truths) and w_b_idx < len(truths) and h_idx < len(truths):
                transformed_truths.append((truths[w_c_idx] + truths[w_b_idx]) / (truths[h_idx] ** 2))

    print(f"  Transformed {len(new_params)} parameters:")
    print(f"  {', '.join(new_params)}")

    return transformed_samples, new_params, transformed_truths




def plot_traces(samples, params, output_path, burn_in=0, get_plot_labels_fn: Optional[Callable] = None, param_labels=None):
    """Create trace plots showing walker chains."""
    n_steps, n_walkers, ndim = samples.shape

    fig, axes = plt.subplots(ndim, 1, figsize=(6, 2.2 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]

    if get_plot_labels_fn is not None:
        plot_labels = get_plot_labels_fn(params, param_labels)
    else:
        plot_labels = params

    for i, ax in enumerate(axes):
        for j in range(n_walkers):
            ax.plot(samples[:, j, i], alpha=0.4, lw=0.5)

        ax.set_ylabel(plot_labels[i] if i < len(plot_labels) else f"param {i}")
        ax.set_xlim(0, n_steps)

        if burn_in > 0:
            ax.axvline(burn_in, color="red", linestyle="--", alpha=0.7, label="burn-in")

        if burn_in < n_steps:
            median_val = np.median(samples[burn_in:, :, i])
            ax.axhline(median_val, color="black", linestyle="-", alpha=0.5, lw=1)

    axes[-1].set_xlabel("Step")
    axes[0].set_title(f"MCMC Traces ({n_walkers} walkers, {n_steps} steps)")

    if burn_in > 0:
        axes[0].legend(loc="upper right")

    fig.tight_layout()
    plt.savefig(output_path)

    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, dpi=300)

    plt.close(fig)
    print(f"Trace plot saved: {output_path}")
    print(f"Trace plot (PDF) saved: {pdf_path}")


def plot_posterior_scatter(samples, params, output_path, burn_in=0, thin=1, get_plot_labels_fn: Optional[Callable] = None, param_labels=None):
    """Create a 2D scatter plot of posterior samples."""
    flat_samples = samples[burn_in::thin, :, :].reshape(-1, samples.shape[-1])

    if flat_samples.shape[1] < 2:
        print("Scatter plot requires at least 2 parameters, skipping")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    if get_plot_labels_fn is not None:
        plot_labels = get_plot_labels_fn(params, param_labels)
    else:
        plot_labels = params

    ax.scatter(flat_samples[:, 0], flat_samples[:, 1], s=2, alpha=0.3, c="C0", rasterized=True)

    ax.set_xlabel(plot_labels[0])
    ax.set_ylabel(plot_labels[1])
    ax.set_title(f"Posterior Samples (N={flat_samples.shape[0]})")

    try:
        from scipy import stats

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        kernel = stats.gaussian_kde(flat_samples[:, :2].T)
        f = np.reshape(kernel(positions).T, xx.shape)

        ax.contour(xx, yy, f, levels=4, colors="k", alpha=0.5, linewidths=0.5)
    except Exception as e:
        print(f"Could not add contours: {e}")

    fig.tight_layout()
    plt.savefig(output_path)

    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(pdf_path, dpi=300)

    plt.close(fig)
    print(f"Scatter plot saved: {output_path}")
    print(f"Scatter plot (PDF) saved: {pdf_path}")


def plot_corner_overlay(
    copula_samples,
    gaussian_samples,
    params,
    output_path,
    burn_in=0,
    thin=1,
    selected_params=None,
    truths=None,
    title=None,
    get_plot_labels_fn: Optional[Callable] = None,
    param_labels=None,
    smooth_scale_1D=1.5,
    smooth_scale_2D=1.5,
    corner_fig_size=None,
    filled_gaussian=True,
    filled_copula=True,
):
    """Create Gaussian vs Copula overlay corner plot using GetDist."""
    try:
        import getdist
        from getdist import plots as gplots
    except ImportError:
        print("GetDist package not available, skipping overlay corner plot")
        return

    flat_copula = copula_samples[burn_in::thin, :, :].reshape(-1, copula_samples.shape[-1])
    flat_gaussian = gaussian_samples[burn_in::thin, :, :].reshape(-1, gaussian_samples.shape[-1])

    plot_params = list(params)
    plot_truths = truths

    if selected_params is not None:
        flat_copula, plot_params, plot_truths = select_parameters(flat_copula, params, selected_params, truths)
        flat_gaussian, _, _ = select_parameters(flat_gaussian, params, selected_params, truths)

    if get_plot_labels_fn is not None:
        plot_labels = get_plot_labels_fn(plot_params, param_labels)
    else:
        plot_labels = plot_params

    settings = {
        "smooth_scale_1D": float(smooth_scale_1D),
        "smooth_scale_2D": float(smooth_scale_2D),
    }

    copula_obj = getdist.MCSamples(
        samples=flat_copula,
        names=plot_params,
        labels=plot_labels,
        settings=settings,
    )
    gaussian_obj = getdist.MCSamples(
        samples=flat_gaussian,
        names=plot_params,
        labels=plot_labels,
        settings=settings,
    )

    colors = get_exact_gaussian_colors()
    line_gauss = colors["linecolor_gauss"]
    line_exact = colors["linecolor_exact"]

    g = gplots.get_subplot_plotter(width_inch=corner_fig_size)
    g.triangle_plot(
        [gaussian_obj, copula_obj],
        filled=[bool(filled_gaussian), bool(filled_copula)],
        contour_colors=[line_gauss, line_exact],
        line_args=[{"color": line_gauss, "lw": 1.0}, {"color": line_exact, "lw": 1.0}],
        legend_labels=["Gaussian", "Copula"],
    )

    if plot_truths is not None:
        n_plot = len(plot_params)
        for row in range(n_plot):
            for col in range(n_plot):
                ax = g.subplots[row, col]
                if ax is None:
                    continue
                if row == col:
                    if row < len(plot_truths) and plot_truths[row] is not None:
                        ax.axvline(plot_truths[row], color="black", linestyle="--", alpha=0.7)
                elif row > col:
                    x_truth = plot_truths[col] if col < len(plot_truths) else None
                    y_truth = plot_truths[row] if row < len(plot_truths) else None
                    if x_truth is not None:
                        ax.axvline(x_truth, color="black", linestyle="--", alpha=0.7)
                    if y_truth is not None:
                        ax.axhline(y_truth, color="black", linestyle="--", alpha=0.7)

    if title:
        g.fig.suptitle(title, fontsize=14, y=0.995)

    g.fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    pdf_path = Path(output_path).with_suffix(".pdf")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(g.fig)

    print(f"Overlay corner plot saved: {output_path}")
    print(f"Overlay corner plot (PDF) saved: {pdf_path}")
