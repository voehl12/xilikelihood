import numpy as np
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import PchipInterpolator
from scipy.integrate import cumulative_trapezoid as cumtrapz
import matplotlib.pyplot as plt


def interpolate_along_last_axis(xs, pdfs, num_points=512):

    def interpolate_1d(xpdf):
        xpdf = xpdf.reshape(2, -1)
        x, pdf = xpdf[0], xpdf[1]
        interp = PchipInterpolator(x, pdf)
        x_vals = np.linspace(x[0], x[-1], num_points)
        pdf_vals = interp(x_vals)
        concat = np.concatenate((x_vals, pdf_vals), axis=-1)
        return concat

    concatenated_xpdf = np.concatenate((xs, pdfs), axis=-1)

    # Apply interpolation along the last axis
    xs_pdfs = np.apply_along_axis(interpolate_1d, -1, concatenated_xpdf)
    xs_pdfs = xs_pdfs.reshape(*xs_pdfs.shape[:-1], 2, -1)
    xs_interp = xs_pdfs[..., 0, :]
    pdfs_interp = xs_pdfs[..., 1, :]

    return xs_interp, pdfs_interp


def pdf_to_cdf(xs, pdfs):
    xs_interp, pdfs_interp = interpolate_along_last_axis(xs, pdfs)
    cdfs = cumtrapz(pdfs_interp, xs_interp, initial=0)

    assert np.all(np.fabs(cdfs[:, :, -1] - 1) < 1e-2), "CDF not normalized to 1"
    max_values = np.max(cdfs, axis=-1, keepdims=True)
    cdfs /= max_values

    return cdfs, pdfs_interp, xs_interp


def get_values_from_indices(array_3d, indices_2d):
    """
    Accesses the values in a 3D array corresponding to the indices in a 2D array along the last axis.

    Parameters:
    - array_3d: A 3D numpy array (shape: (c, b, n))
    - indices_2d: A 2D numpy array of indices (shape: (c, b))

    Returns:
    - values: A 2D numpy array of corresponding values (shape: (c, b))
    """
    values = np.take_along_axis(array_3d, indices_2d[..., np.newaxis], axis=-1).squeeze()
    return values


def pdf_and_cdf_point_eval(x_data, xs, pdfs, cdfs):
    # returns nd cdf value of datapoint and pdf value in each dimension, index wrt x?
    broadcasted_data = x_data[:, :, None]
    data_inds = np.argmin(np.abs(xs - broadcasted_data), axis=-1)
    pdf_point = get_values_from_indices(pdfs, data_inds)
    cdf_point = get_values_from_indices(cdfs, data_inds)
    return pdf_point, cdf_point


def covariance_to_correlation(cov_matrix):
    """
    Converts a covariance matrix to a correlation matrix.

    Parameters:
    - cov_matrix: A 2D numpy array representing the covariance matrix

    Returns:
    - corr_matrix: A 2D numpy array representing the correlation matrix
    """
    # Compute the standard deviations
    std_devs = np.sqrt(np.diag(cov_matrix))

    # Outer product of standard deviations
    std_devs_outer = np.outer(std_devs, std_devs)

    # Compute the correlation matrix
    corr_matrix = cov_matrix / std_devs_outer

    # Set the diagonal elements to 1
    np.fill_diagonal(corr_matrix, 1)

    return corr_matrix


def gaussian_copula_density(cdfs, covariance_matrix):
    # Convert u and v to normal space
    cdfs_flat = cdfs.reshape(-1, cdfs.shape[-1])
    z = norm.ppf(cdfs_flat)  # same shape as cdfs

    corr_matrix = covariance_to_correlation(covariance_matrix)
    mean = np.zeros(len(corr_matrix))
    mvn = multivariate_normal(
        mean=mean, cov=corr_matrix
    )  # multivariate normal with right correlation structure
    ppf_grid = np.meshgrid(*z)
    stacked_ppfgrid = np.stack(ppf_grid, axis=-1)
    ppf_points = stacked_ppfgrid.reshape(-1, stacked_ppfgrid.shape[-1])
    mvariate_pdf = mvn.pdf(ppf_points)  # evaluate mv normal at ppf points

    pdf = norm.pdf(z)  # evaluate normal at the inverse cdf points
    pdf_grid = np.meshgrid(*pdf)
    pdf_points = np.stack(pdf_grid, axis=-1)  # stack the pdfs along new axis
    pdf_points = pdf_points.reshape(-1, pdf_points.shape[-1])  # reshape to 2D array

    # Copula density
    copula_density = mvariate_pdf / (np.prod(pdf_points, axis=1))
    return copula_density


def gaussian_copula_point_density(cdf_point, covariance_matrix):
    z = norm.ppf(cdf_point)
    corr_matrix = covariance_to_correlation(covariance_matrix)
    mean = np.zeros(len(corr_matrix))
    mvn = multivariate_normal(
        mean=mean, cov=corr_matrix
    )  # multivariate normal with right correlation structure
    z = z.flatten()
    mvariate_pdf = mvn.pdf(z)
    pdf = norm.pdf(z)
    return mvariate_pdf / np.prod(pdf)


def gaussian_copula_density_2d(u, v, covariance_matrix):
    # Convert u and v to normal space
    z1 = norm.ppf(u)
    z2 = norm.ppf(v)

    # Bivariate normal PDF with correlation rho
    corr_matrix = covariance_to_correlation(covariance_matrix)
    mvn = multivariate_normal(mean=[0, 0], cov=corr_matrix)
    x_grid, y_grid = np.meshgrid(z1, z2)
    test_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    bivariate_pdf = mvn.pdf(test_points)

    # Standard normal PDFs
    pdf_z1 = norm.pdf(z1)
    pdf_z2 = norm.pdf(z2)

    pdf_z1_grid, pdf_z2_grid = np.meshgrid(pdf_z1, pdf_z2)
    pdf_points = np.vstack([pdf_z1_grid.ravel(), pdf_z2_grid.ravel()]).T

    # Copula density
    copula_density = bivariate_pdf / (np.prod(pdf_points, axis=1))
    return copula_density


def joint_pdf(cdfs, pdfs, cov):

    pdfs_flat = pdfs.reshape(-1, pdfs.shape[-1])  # all pdfs stacked, two-dim array
    pdf_meshgrid = np.meshgrid(
        *pdfs_flat
    )  # meshgrid of all pdfs -> evaluate one point by pdf_meshgrid[k, i, j, ...] where k is the coordinate of the kth dimension, and i,j,... are the indices of the meshgrid
    stacked_meshgrid = np.stack(pdf_meshgrid, axis=-1)  # stack the meshgrid along new axis
    pdf_points = stacked_meshgrid.reshape(
        -1, stacked_meshgrid.shape[-1]
    )  # reshape to 2D array, list of points in pdf space

    # Compute copula density
    copula_density = gaussian_copula_density(cdfs, cov)
    # this returns the full pdf, still needs to be adjusted to just evaluate at single points
    # i.e. cdf is only needed for one point in each dimension, only integrate up to data point x, might be able to avoid loop and interpolation
    # Joint PDF
    return copula_density * np.prod(pdf_points, axis=1)


def joint_pdf_2d(cdf_X, cdf_Y, pdf_X, pdf_Y, cov):
    # Compute marginals
    u = cdf_X[1:-1]
    v = cdf_Y[1:-1]
    pdf_x_grid, pdf_y_grid = np.meshgrid(pdf_X[1:-1], pdf_Y[1:-1])
    pdf_points = np.vstack([pdf_x_grid.ravel(), pdf_y_grid.ravel()]).T

    # Compute copula density
    copula_density = gaussian_copula_density(u, v, cov)

    # Joint PDF
    return copula_density * np.prod(pdf_points, axis=1)


def evaluate(x_data, xs, pdfs, cdfs, cov):
    pdf_point, cdf_point = pdf_and_cdf_point_eval(x_data=x_data, xs=xs, pdfs=pdfs, cdfs=cdfs)
    copula_density = gaussian_copula_point_density(cdf_point, cov)
    return copula_density * np.prod(pdf_point)


def testing():
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
