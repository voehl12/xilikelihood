"""Shared plotting style and color helpers for analysis scripts."""

import matplotlib.pyplot as plt

try:
    import cmasher as cmr
    HAS_CMASHER = True
except ImportError:
    HAS_CMASHER = False


def configure_paper_plot_style():
    """Apply shared paper-style matplotlib defaults."""
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "text.latex.preamble": r"\usepackage{amsmath}",
    })


def get_default_figsize(kind="single"):
    """Return default figure sizes used across plotting scripts."""
    sizes = {
        "single": (5, 5),
        "double": (10, 5),
    }
    return sizes.get(kind, sizes["single"])


def get_exact_gaussian_colors(n_levels=2):
    """Return consistent exact/gaussian contour colors.

    Exact/Copula colors follow the lower torch range and Gaussian colors follow
    the higher torch range, matching existing comparison plots.
    """
    
    if HAS_CMASHER:
        pdf_cm = cmr.torch
        colors_exact = cmr.take_cmap_colors(
            pdf_cm,
            n_levels,
            cmap_range=(0.3, 0.4),
            return_fmt="hex",
        )
        colors_gauss = cmr.take_cmap_colors(
            pdf_cm,
            n_levels,
            cmap_range=(0.65, 0.9),
            return_fmt="hex",
        )
    else:
        # Fallback colors when cmasher is unavailable.
        colors_exact = ["#cc6f5a", "#b35340"][:n_levels]
        colors_gauss = ["#6ca6d8", "#2f79b7"][:n_levels]

    return {
        "colors_exact": colors_exact,
        "colors_gauss": colors_gauss,
        "linecolor_exact": colors_exact[0],
        "linecolor_gauss": colors_gauss[0],
    }


def get_single_chain_color():
    """Return the standard single-chain fill/line color for GetDist plots."""
    if HAS_CMASHER:
        return cmr.take_cmap_colors("cmr.gem", 1, cmap_range=(0.2, 0.2), return_fmt="hex")[0]
    return "#2471a3"
