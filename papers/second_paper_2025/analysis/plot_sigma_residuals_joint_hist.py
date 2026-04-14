"""Plot joint histogram(s) of saved sigma residual arrays.

Usage:
    python plot_sigma_residuals_joint_hist.py file1_sigma_residuals.npz file2_sigma_residuals.npz
    python plot_sigma_residuals_joint_hist.py file1.npz file2.npz --labels Copula Gaussian --output sigma_joint
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from plot_utils_style import (
    configure_paper_plot_style,
    get_default_figsize,
    get_exact_gaussian_colors,
)


def _load_sigma_residuals(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path)
    if "sigma_residuals" not in data:
        raise KeyError(f"Missing 'sigma_residuals' key in {npz_path}")
    arr = np.asarray(data["sigma_residuals"]).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"No finite sigma residuals found in {npz_path}")
    return arr


def _default_colors(n: int):
    if n == 2:
        c = get_exact_gaussian_colors()
        return [c["linecolor_exact"], c["linecolor_gauss"]]
    return [plt.get_cmap("tab10")(i % 10) for i in range(n)]


def main() -> int:
    configure_paper_plot_style()

    parser = argparse.ArgumentParser(description="Plot joint histogram of sigma residual NPZ files")
    parser.add_argument("files", nargs="+", help="Input *.npz files containing 'sigma_residuals'")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels (same length as files)")
    parser.add_argument("--bins", type=int, default=60, help="Number of histogram bins")
    parser.add_argument("--alpha", type=float, default=0.35, help="Constant histogram alpha")
    parser.add_argument("--output", type=str, default="sigma_residuals_joint_hist", help="Output path stem")
    args = parser.parse_args()

    files = [Path(f).expanduser().resolve() for f in args.files]
    for f in files:
        if not f.exists():
            raise FileNotFoundError(f"Input file not found: {f}")

    if args.labels is not None and len(args.labels) not in {0, len(files)}:
        raise ValueError("--labels must be omitted or have the same length as files")

    labels = args.labels if args.labels else [f.stem.replace("_sigma_residuals", "") for f in files]
    sigma_sets = [_load_sigma_residuals(f) for f in files]
    # Use fixed bin edges from the first dataset and reuse for all datasets.
    bin_edges = np.histogram_bin_edges(sigma_sets[0], bins=args.bins)

    colors = _default_colors(len(sigma_sets))

    width, _ = get_default_figsize("single")
    fig, ax = plt.subplots(figsize=(width, 4.0))

    for values, label, color in zip(sigma_sets, labels, colors):
        frac2 = 100.0 * np.mean(np.abs(values) > 2.0)
        frac3 = 100.0 * np.mean(np.abs(values) > 3.0)
        frac4 = 100.0 * np.mean(np.abs(values) > 4.0)
        mean_val = float(np.mean(values))
        print(
            f"{label}: N={values.size}, "
            f"|delta/sigma|>2: {frac2:.2f}%, "
            f">3: {frac3:.2f}%, "
            f">4: {frac4:.2f}%"
        )
        ax.hist(
            values,
            bins=bin_edges,
            density=True,
            histtype="stepfilled",
            alpha=float(args.alpha),
            color=color,
            label=label,
        )
        ax.axvline(mean_val, color=color, linestyle="-", linewidth=1.5, alpha=0.9)
    ax.axvline(-3.0, color="black", linestyle="--", linewidth=1.0)
    ax.axvline(3.0, color="black", linestyle="--", linewidth=1.0)

    ax.set_xlabel(r"$\Delta / \sigma_{\mathrm{bootstrap}}$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)

    out_stem = Path(args.output).expanduser()
    out_png = out_stem.with_suffix(".png")
    out_pdf = out_stem.with_suffix(".pdf")
    out_bins = out_stem.with_name(out_stem.name + "_bins.npy")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    np.save(out_bins, bin_edges)
    plt.close(fig)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")
    print(f"Saved bin edges: {out_bins}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
