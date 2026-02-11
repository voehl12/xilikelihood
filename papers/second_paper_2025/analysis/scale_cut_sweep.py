"""
Sweep the large-angle threshold to assess Gaussian vs exact likelihood tradeoffs.

Runs a grid of `large_angle_threshold` values, evaluates the log-likelihood at
the fiducial cosmology, and records runtime for each threshold. Produces a
summary CSV and plot comparing exact and Gaussian values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
import tempfile
import logging
import argparse
import os
import cmasher as cmr
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')

import xilikelihood as xlh
from config import (
    FIDUCIAL_COSMO,
    EXACT_LMAX,
    MASK_CONFIG_MEDRES_STAGE3 as MASK_CONFIG,
    DATA_FILES,
    PACKAGE_DIR
)

# TODO: use larger mask, play with cf parameters, compare to gaussian likelihood in any case, plot times as well.


def make_plot(df, png_path, pdf_path=None):
    """Create paper-ready plot from DataFrame."""
    # Get colors from cmasher torch colormap (consistent with other paper plots)
    pdf_cm = cmr.torch
    colors = cmr.take_cmap_colors(pdf_cm, 2, cmap_range=(0.3, 0.5), return_fmt='hex')
    likelihood_color = colors[0]
    time_color = colors[1]
    
    fig, ax1 = plt.subplots(figsize=(5, 4))

    # Plot likelihood
    ax1.plot(df["scale_cut_arcmin"], df["loglike"], marker='o', linestyle='-', 
             color=likelihood_color, label=r'Copula Likelihood', markersize=5)
    
    # Add Gaussian likelihood as black dashed horizontal line
    # Use the first gausslike value (should be constant across all scale cuts)
    gauss_value = df["gausslike"].iloc[0]
    ax1.axhline(gauss_value, color='black', linestyle='--', linewidth=1.5, 
                label=r'Gaussian Likelihood')
    
    ax1.set_xlabel(r"Threshold for Gaussian marginals (arcmin)")
    ax1.set_ylabel(r"Log-Likelihood")
    ax1.set_xscale('log')
    
    # Fix y-axis labels to avoid scientific notation offset
    ax1.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset=False))
    ax1.ticklabel_format(style='plain', axis='y')
    
    # Create second axis for time (colored line, black axis labels)
    ax2 = ax1.twinx()
    ax2.plot(df["scale_cut_arcmin"], df["time_sec"], marker='s', linestyle='-', 
             color=time_color, label=r'Computation time', markersize=4, alpha=0.8)
    ax2.set_ylabel(r'Time (seconds)')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=False)
    
    # Adjust y-axis limits to make space above for legend
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymin, ymax + 0.25 * (ymax - ymin))

    plt.tight_layout()
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    if pdf_path:
        plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    
    return png_path, pdf_path


def run_sweep(csv_path, log_path):
    """Run the scale cut sweep and save results to CSV."""
    # Set up logging
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.propagate = False

    # Enable xilikelihood logging
    xili_logger = logging.getLogger('xilikelihood')
    xili_logger.setLevel(logging.DEBUG)
    xili_logger.addHandler(file_handler)

    logger.info("Starting scale cut sweep...")

    # Setup mask and dataspace
    mask = xlh.SphereMask(
        spins=MASK_CONFIG['spins'],
        circmaskattr=MASK_CONFIG['circmaskattr'],
        exact_lmax=EXACT_LMAX,
        l_smooth=MASK_CONFIG['l_smooth'],
        working_dir=PACKAGE_DIR
    )

    redshift_bins, angular_bins_in_deg = xlh.fiducial_dataspace(min_ang_cutoff_in_arcmin=5.0)
    logger.info(f"Redshift bins: {len(redshift_bins)}, Angular bins: {len(angular_bins_in_deg)}")

    # Create a base likelihood for mock data creation (use threshold=0.0)
    base_likelihood = xlh.XiLikelihood(
        mask=mask,
        redshift_bins=redshift_bins,
        ang_bins_in_deg=angular_bins_in_deg,
        include_ximinus=False,
        exact_lmax=EXACT_LMAX,
        large_angle_threshold=5.0/60,
    )
    base_likelihood.setup_likelihood()

    # Create mock data and covariance on the fly
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_data_path = f"{tmpdir}/mock_data.npz"
        cov_path = f"{tmpdir}/mock_cov.npz"
        mock_data, gaussian_covariance = xlh.mock_data.create_mock_data(
            base_likelihood,
            mock_data_path=mock_data_path,
            gaussian_covariance_path=cov_path,
            fiducial_cosmo=FIDUCIAL_COSMO,
            random=None
        )
    logger.info(f"Mock data shape: {mock_data.shape}")
    logger.info(f"Covariance shape: {gaussian_covariance.shape}")

    # Large angle thresholds need to be adjusted so bins are left out one by one
    #scale_cuts = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1200.]
    scale_cuts = [5.0, 10.0, 20.0, 50.0, 100.0, 300.0, 1200.]

    results = []

    for cut in scale_cuts:
        logger.info(f"Processing large_angle_threshold: {cut} arcmin")
        cut_in_deg = cut / 60.0
        # Setup likelihood with varying large_angle_threshold
        likelihood = xlh.XiLikelihood(
            mask=mask,
            redshift_bins=redshift_bins,
            ang_bins_in_deg=angular_bins_in_deg,
            include_ximinus=False,
            exact_lmax=EXACT_LMAX,
            large_angle_threshold=cut_in_deg,  # This is what we vary
        )
        likelihood.setup_likelihood()
        likelihood.gaussian_covariance = gaussian_covariance

        # Evaluate log-likelihood
        start_time = time.time()
        loglike, gausslike = likelihood.loglikelihood(mock_data, FIDUCIAL_COSMO, gausscompare=True)
        time_sec = time.time() - start_time

        logger.info(f"Log-likelihood: {loglike}, Time: {time_sec:.2f}s")
        logger.info(f"Gaussian comparison likelihood: {gausslike}")
        results.append({
            "scale_cut_arcmin": cut,
            "loglike": loglike,
            "gausslike": gausslike,
            "time_sec": time_sec
        })

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(csv_path, index=False)
    logger.info(f"Results saved to {csv_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Scale cut sweep analysis')
    parser.add_argument('--plot-only', action='store_true', 
                        help='Only create plot from existing CSV file')
    parser.add_argument('--csv', type=str, default='scale_cut_sweep_n1024_minang5.csv',
                        help='Path to CSV file (input for plot-only, output otherwise)')
    parser.add_argument('--output', type=str, default='scale_cut_sweep_n1024_minang5',
                        help='Base name for output files (without extension)')
    args = parser.parse_args()
    
    csv_path = args.csv
    png_path = f"{args.output}.png"
    pdf_path = f"{args.output}.pdf"
    log_path = f"{args.output}.log"
    
    if args.plot_only:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        print(f"Reading data from {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        df = run_sweep(csv_path, log_path)
    
    # Create plot
    make_plot(df, png_path, pdf_path)
    print(f"Plot saved to {png_path} and {pdf_path}")
    print("Scale cut sweep completed!")


if __name__ == "__main__":
    main()
