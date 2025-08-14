"""
Diagnostic tools for eigenvalue and likelihood analysis in XiLikelihood.

This module provides functions to analyze and visualize eigenvalues and their influence on likelihood products.
"""
import numpy as np
import os
import matplotlib.pyplot as plt

def analyze_significant_eigenvalues(eigvals, tol=1e-12, return_mask=False):
    """
    For each marginal, determine which eigenvalues are far enough from 1 to significantly alter the product.
    An eigenvalue is considered significant if |eigval - 1| > tol.
    Prints a summary and optionally returns a boolean mask.

    Parameters
    ----------
    eigvals : ndarray
        Array of eigenvalues, shape (..., n_eig)
    tol : float
        Tolerance for |eigval - 1| to consider an eigenvalue significant.
    return_mask : bool
        If True, return the boolean mask array.
    Returns
    -------
    significant_mask : ndarray (optional)
        Boolean array of shape (..., n_eig) where True means |eigval-1| > tol.
    significant_counts : ndarray
        Array with the number of significant eigenvalues per marginal.
    """
    abs_diff = np.abs(eigvals - 1)
    significant_mask = abs_diff > tol
    significant_counts = np.sum(significant_mask, axis=-1)
    print(f"Significant eigenvalues (|eigval-1| > {tol}):")
    print(f"Shape: {significant_mask.shape}")
    print(f"Mean count per marginal: {np.mean(significant_counts):.2f}")
    # Print a summary for each marginal
    for idx, count in np.ndenumerate(significant_counts):
        if count > 0:
            eigs = eigvals[idx][:]
            sigs = significant_mask[idx][:]
            print(f"Marginal {idx}: {count} significant eigenvalues.")
            print(f"  Eigenvalues far from 1: {[float(e) for e,s in zip(eigs,sigs) if s]}")
    if return_mask:
        return significant_mask, significant_counts
    else:
        return significant_counts

def plot_eigenvalue_distributions(eigvals, max_marginals=10, save_dir=None, show=True):
    """
    For each marginal, plot the distribution of eigenvalues in the complex plane (Re vs Im).
    Optionally save each figure to a directory. By default, only the first `max_marginals` are plotted.

    Parameters
    ----------
    eigvals : ndarray
        Array of eigenvalues, shape (n_corr, n_redshift, n_eig)
    max_marginals : int
        Maximum number of marginals to plot (for large problems).
    save_dir : str or None
        If provided, save each figure as a PNG in this directory.
    show : bool
        If True, display the figures interactively.
    """
    n_corr, n_redshift = eigvals.shape[:2]
    n_plots = min(max_marginals, n_corr * n_redshift)
    count = 0
    for i in range(n_corr):
        for j in range(n_redshift):
            if count >= n_plots:
                break
            vals = eigvals[i, j].flatten()
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(np.fabs(np.real(vals)), np.fabs(np.imag(vals)), alpha=0.7, s=30, c='tab:blue', edgecolor='k')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.axhline(0, color='gray', lw=1, ls='--')
            ax.axvline(1, color='red', lw=1, ls=':')
            ax.set_xlabel('Re(eigenvalue)')
            ax.set_ylabel('Im(eigenvalue)')
            ax.set_title(f'Eigenvalues for marginal (corr={i}, z={j})')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            if save_dir is not None:
                os.makedirs(save_dir, exist_ok=True)
                fname = os.path.join(save_dir, f'eigvals_corr{i}_z{j}.png')
                plt.savefig(fname)
            if show:
                plt.show()
            else:
                plt.close()
            count += 1
        if count >= n_plots:
            break
    if n_corr * n_redshift > n_plots:
        print(f"Plotted only the first {n_plots} marginals out of {n_corr * n_redshift}.")



def plot_pdfs_and_cdfs(pdfs, xs, cdfs=None, max_plots=16, savepath=None):
    """
    Plot PDFs and (optionally) CDFs for each (redshift, angular bin) combination.

    Parameters
    ----------
    pdfs : ndarray
        Array of PDFs, shape (n_rs, n_ang, n_points).
    xs : ndarray
        Array of x-values, shape (n_rs, n_ang, n_points).
    cdfs : ndarray, optional
        Array of CDFs, shape (n_rs, n_ang, n_points).
    max_plots : int
        Maximum number of subplots (default: 16).
    show : bool
        Whether to show the plot (default: True).
    savepath : str or None
        If provided, save the figure to this path.
    """
    import matplotlib.pyplot as plt
    n_rs, n_ang, n_points = pdfs.shape
    n_total = n_rs * n_ang
    n_plots = min(n_total, max_plots)
    ncols = min(4, n_plots)
    nrows = int(np.ceil(n_plots / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows))
    axes = np.array(axes).reshape(-1)
    for idx in range(n_plots):
        i = idx // n_ang
        j = idx % n_ang
        ax = axes[idx]
        ax.plot(xs[i, j], pdfs[i, j], label='PDF', color='tab:blue')
        ax.set_title(f'rs={i}, ang={j}')
        ax.set_ylabel('PDF')
        ax.set_xlabel('x')
        if cdfs is not None:
            ax2 = ax.twinx()
            ax2.plot(xs[i, j], cdfs[i, j], label='CDF', color='tab:orange', linestyle='--')
            ax2.set_ylabel('CDF', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax.legend(loc='upper left')
    for ax in axes[n_plots:]:
        ax.axis('off')
    #fig.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.close(fig)


def _testing_function():
    """
    INTERNAL: Development testing function.
    
    ⚠️ Warning: This is for development only and will be removed.
    Use proper test files for production testing.
    """
    import warnings
    warnings.warn(
        "This is an internal development function and will be removed. "
        "Use dedicated test files instead.",
        DeprecationWarning,
        stacklevel=2
    )
    copula = copula_funcs.joint_pdf(
        self._cdfs[1:],
        self._pdfs[1:],
        self._cov[1:, 1:],
    )

    fig, ((ax00, ax01, ax02), (ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(
        3, 3, gridspec_kw=dict(width_ratios=[1, 1, 1]), figsize=(11, 11)
    )
    # bincenters, mean, errors, mu_estimate, cov_estimate
    configpath = "config_adjusted.ini"
    simspath = "/cluster/work/refregier/veoehl/xi_sims/croco_3x2pt_kids_33_circ10000smoothl30_noisedefault_llim_None_newwpm/"
    config = postprocess_nd_likelihood.load_config(configpath)

    diag_fig, diag_ax = plt.subplots()
    sims_lmax = self.lmax if highell else self._exact_lmax
    bincenters, mean, errors, mu_estimate, cov_estimate = (
        postprocess_nd_likelihood.load_and_bootstrap_sims_nd(
            config,
            simspath,
            sims_lmax,
            axes=(ax00, ax1, ax3),
            vmax=None,
            n_bootstrap=1000,
            diagnostic_ax=diag_ax,
        )
    )
    x_vals = self._xs[1, 0]
    y_vals = self._xs[2, 0]
    diag_ax.plot(x_vals, self._pdfs[1, 0], label="xi55_analytic")
    diag_ax.plot(y_vals, self._pdfs[2, 0], label="xi53_analytic")
    diag_ax.legend()
    diag_fig.savefig("marginal_diagnostics_10000sqd_fullell_newwpm.png")

    x_grid, y_grid = np.meshgrid(x_vals, y_vals)
    test_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

    # x_exact, pdf_exact = postprocess_nd_likelihood.convert_nd_cf_to_pdf(config,highell_moms=highell_moms)
    vmax = np.max(copula)
    copula_grid = copula.reshape(x_grid.shape).T
    interp = RegularGridInterpolator(
        (x_vals[1:-1], y_vals[1:-1]), copula_grid[1:-1, 1:-1], method="cubic"
    )
    # interp_exact = RegularGridInterpolator((x_exact[:,0,0],x_exact[0,:,1]),pdf_exact,method='cubic')
    # marginals_exact = postprocess_nd_likelihood.get_marginal_likelihoods([x_exact[:,0,0],x_exact[0,:,1]],pdf_exact)
    # marginals_copula = postprocess_nd_likelihood.get_marginal_likelihoods([x_vals,y_vals],copula_grid)

    # grid_z_copula = griddata(test_points, copula, (x_grid, y_grid), method="cubic")
    gauss = self.gauss_compare().pdf(test_points)
    gauss_est = multivariate_normal(mean=mu_estimate, cov=cov_estimate)
    gauss_est = gauss_est.pdf(test_points)
    gauss_grid = gauss_est.reshape(x_grid.shape).T
    interp_gauss = RegularGridInterpolator((x_vals, y_vals), gauss_grid, method="cubic")
    (ax1, ax2, ax5), res_plot = postprocess_nd_likelihood.compare_to_sims_2d(
        [ax1, ax2, ax5], bincenters, mean, errors, interp, vmax
    )
    (ax3, ax4, ax6), gauss_res = postprocess_nd_likelihood.compare_to_sims_2d(
        [ax3, ax4, ax6], bincenters, mean, errors, interp_gauss, vmax
    )
    # (ax00,ax01,ax02), exact_res = postprocess_nd_likelihood.compare_to_sims_2d([ax00,ax01,ax02],bincenters,mean,errors,interp_exact,vmax)

    # fig, ax4 = plt.subplots()
    # c2 = ax4.contourf(x_grid, y_grid, grid_z_copula, levels=100, vmax=np.max(grid_z_copula))
    # ax4.set_title("Copula")

    fig.colorbar(res_plot, ax=ax5)
    fig.colorbar(gauss_res, ax=ax6)
    # fig.colorbar(exact_res, ax=ax02)
    fig.savefig("comparison_copula_sims_10000deg2_fullell_newwpm.png")
    pass