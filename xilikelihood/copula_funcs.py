"""
Copula-based statistical modeling for multivariate correlation functions.

This module provides:
- Gaussian and Student-t copula implementations
- PDF/CDF interpolation and evaluation utilities
- Joint probability density computation
- Covariance matrix utilities and conditioning
- Data validation and quality checks

Used for modeling dependencies between cosmological observables while
preserving marginal distributions from theory predictions.
"""

import numpy as np
from scipy.stats import norm, multivariate_normal, t
from scipy.interpolate import PchipInterpolator, UnivariateSpline, RegularGridInterpolator
from scipy.integrate import cumulative_trapezoid as cumtrapz
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.special import gamma
import logging

logger = logging.getLogger(__name__)

DEFAULT_MIN_EIGENVALUE = 5e-4
DEFAULT_INTERPOLATION_POINTS = 512
DEFAULT_PDF_THRESHOLD = 0.01
DEFAULT_NORMALIZATION_TOLERANCE = 1e-3
DEFAULT_SMALL_NUMBER = 1e-300

__all__ = [
    # Core copula functions
    'gaussian_copula_density',
    'student_t_copula_density', 
    'gaussian_copula_point_density',
    'joint_pdf',
    'joint_pdf_2d',
    
    # PDF/CDF utilities
    'pdf_to_cdf',
    'interpolate_and_evaluate',
    'interpolate_and_evaluate_with_log',
    'pdf_and_cdf_point_eval',
    
    # Matrix utilities
    'covariance_to_correlation',
    'get_well_conditioned_matrix',
    'test_correlation_matrix',
    
    # Data utilities
    'validate_pdfs',
    'cov_subset',
    'data_subset',
    'evaluate',
    
    # Statistical functions
    'multivariate_student_t_logpdf',
]

# ============================================================================
# Probability Density Functions
# ============================================================================


def multivariate_student_t_logpdf(z, df, scale):
    """
    Compute the log-PDF of the multivariate Student-t distribution.

    Parameters:
    - z: ndarray
        Points at which to evaluate the PDF (shape: (n_points, n_dims)).
    - df: float
        Degrees of freedom for the Student-t distribution.
    - scale: ndarray
        Scale matrix (correlation matrix scaled by df / (df - 2)) (shape: (n_dims, n_dims)).

    Returns:
    - log_pdf: ndarray
        Log-PDF values (shape: (n_points,)).
    """
    d = z.shape[1]  # Dimensionality
    inv_scale = np.linalg.inv(scale)
    det_scale = np.linalg.det(scale)

    # Mahalanobis distance
    quad_form = np.sum(z @ inv_scale * z, axis=1)

    # Log-PDF computation
    log_norm_const = (
        np.log(gamma((df + d) / 2))
        - np.log(gamma(df / 2))
        - 0.5 * d * np.log(df * np.pi)
        - 0.5 * np.log(det_scale)
    )
    log_kernel = -0.5 * (df + d) * np.log(1 + quad_form / df)

    return log_norm_const + log_kernel

# ============================================================================
# Validation and Quality Control
# ============================================================================


def validate_pdfs(pdfs, xs, cdfs=None,atol=DEFAULT_NORMALIZATION_TOLERANCE, plot=False, max_plots=16, savepath=None):
    """
    Perform checks on the PDFs to ensure they are valid.

    Parameters
    ----------
    pdfs : ndarray
        Array of PDFs.
    xs : ndarray
        Array of x-values corresponding to the PDFs.
    cdfs : ndarray, optional
        Array of CDFs corresponding to the PDFs. If provided, checks monotonicity.
    plot : bool, optional
        If True, plot the PDFs and CDFs for visual inspection.
    max_plots : int, optional
        Maximum number of subplots to show (default: 16).
    show : bool, optional
        Whether to show the plot (default: True).
    savepath : str or None, optional
        If provided, save the figure to this path.

    Logs warnings if any issues are found.
    """
    for i in range(pdfs.shape[0]):  # Loop over redshift combinations
        for j in range(pdfs.shape[1]):  # Loop over angular bins
            pdf = pdfs[i, j, :]
            x = xs[i, j, :]
 
            # Check normalization
            integral = np.trapz(pdf, x)
            if not np.isclose(integral, 1.0, atol=atol):
                logger.warning(f"PDF {i}-{j} is not normalized: integral={integral}")

            # Check non-negativity
            if np.any(pdf < 0):
                logger.warning(f"PDF {i}-{j} has negative values.")

            # Check domain coverage
            if x[0] > np.min(x) or x[-1] < np.max(x):
                logger.warning(f"PDF {i}-{j} does not cover the expected domain.")

            # Check CDF monotonicity (if CDFs are provided)
            if cdfs is not None:
                cdf = cdfs[i, j, :]
                if not np.all(np.diff(cdf) >= 0):
                    logger.warning(f"CDF {i}-{j} is not monotonic.")

            # Check if PDF drops to zero at domain boundaries (relative to max value)
            max_pdf = np.max(pdf)
            if not (np.isclose(pdf[0], 0, atol=atol * max_pdf) and np.isclose(pdf[-1], 0, atol=atol * max_pdf)):
                logger.warning(f"PDF {i}-{j} does not drop to zero at domain boundaries (relative to max value).")

    logger.info("All PDFs have been checked.")
    if plot or logger.isEnabledFor(logging.DEBUG):
        plot_pdfs_and_cdfs(pdfs, xs, cdfs=cdfs, max_plots=max_plots, savepath=savepath)


# ============================================================================
# PDF/CDF Interpolation and Evaluation
# ============================================================================


def interpolate_along_last_axis(xs, pdfs, num_points=DEFAULT_INTERPOLATION_POINTS, thres=DEFAULT_PDF_THRESHOLD):
    # only interpolate relevant part?

    def interpolate_1d(xpdf):
        xpdf = xpdf.reshape(2, -1)
        x, pdf = xpdf[0], xpdf[1]

        # Only interpolate the relevant part of the PDF
        len_relevant = 0
        threshold = thres * np.max(pdf)
        while len_relevant < len(x) // 5:
            relevant_indices = pdf > threshold
            len_relevant = np.sum(relevant_indices)
            threshold *= 0.1
        x, pdf = x[relevant_indices], pdf[relevant_indices]

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
    xs_interp, pdfs_interp = interpolate_along_last_axis(xs, pdfs, num_points=1024)
    cdfs = cumtrapz(pdfs_interp, xs_interp, initial=0)

    assert np.all(np.fabs(cdfs[..., -1] - 1) < 1e-2), "CDF not normalized to 1"
    max_values = np.max(cdfs, axis=-1, keepdims=True)
    cdfs /= max_values

    return cdfs, pdfs_interp, xs_interp


def interpolate_and_evaluate(x_data, xs, pdfs):
    """
    Interpolates the given xs and pdfs and evaluates the PDF at the given points.

    Parameters:
    x_data (ndarray): 2D array of points where the PDF should be evaluated.
    xs (ndarray): 3D array of x values corresponding to the pdfs (last axis along x).
    pdfs (ndarray): 3D array of pdf values.


    Returns:
    ndarray: 2D array of evaluated PDF values at the given points.
    """
    # Flatten the first two dimensions
    flat_xs = xs.reshape(-1, xs.shape[-1])
    flat_pdfs = pdfs.reshape(-1, pdfs.shape[-1])
    flat_points = x_data.reshape(-1)
    # Create interpolators for each flattened pair of xs and pdfs
    interpolators = [PchipInterpolator(flat_xs[i], flat_pdfs[i]) for i in range(flat_xs.shape[0])]

    # Evaluate the interpolators at the flattened points
    interpolated_point_evs = np.array(
        [interpolators[i](flat_points[i]) for i in range(flat_points.shape[0])]
    )

    # Reshape the result back to the original 2D shape
    interpolated_point_evs = interpolated_point_evs.reshape(x_data.shape)

    return interpolated_point_evs

def interpolate_and_evaluate_with_log(x_data, xs, pdfs):
    flat_xs = xs.reshape(-1, xs.shape[-1])
    flat_pdfs = pdfs.reshape(-1, pdfs.shape[-1])
    flat_points = x_data.reshape(-1)

    # Replace zeros in the PDF with a small value to avoid log(0)
    flat_pdfs[flat_pdfs <= 0] = DEFAULT_SMALL_NUMBER
    log_pdfs = np.log(flat_pdfs)

    # Create interpolators for the log-PDF
    interpolators = [PchipInterpolator(flat_xs[i], log_pdfs[i]) for i in range(flat_xs.shape[0])]

    # Evaluate the interpolators and exponentiate
    log_interpolated_point_evs = np.array(
        [(interpolators[i](flat_points[i])) for i in range(flat_points.shape[0])]
    )

    # Reshape back to the original shape
    log_interpolated_point_evs = log_interpolated_point_evs.reshape(x_data.shape)
    return log_interpolated_point_evs


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
    # broadcasted_data = x_data[:, :, None]
    # data_inds = np.argmin(np.abs(xs - broadcasted_data), axis=-1)
    cdf_point = interpolate_and_evaluate(x_data, xs, cdfs)
    log_pdf_point = interpolate_and_evaluate_with_log(x_data, xs, pdfs)
    
    return log_pdf_point, cdf_point

# ============================================================================
# Copula Density Functions
# ============================================================================

def gaussian_copula_density(cdfs, covariance_matrix):
    # Convert u and v to normal space
    cdfs_flat = cdfs.reshape(-1, cdfs.shape[-1])
    z = norm.ppf(cdfs_flat)  # same shape as cdfs

    corr_matrix = covariance_to_correlation(covariance_matrix)
    mean = np.zeros(len(corr_matrix))
    mvn = multivariate_normal(
        mean=mean, cov=corr_matrix
    )  # multivariate normal with right correlation structure
    
    ppf_points = meshgrid_and_recast(z)
    mvariate_pdf = mvn.logpdf(ppf_points)  # evaluate mv normal at ppf points

    pdf = norm.logpdf(z)  # evaluate normal at the inverse cdf points
    
    pdf_points = meshgrid_and_recast(pdf)  # reshape to 2D array

    # log Copula density
    copula_density = mvariate_pdf - np.sum(pdf_points, axis=1)
    
   
    return copula_density

def gaussian_copula_point_density(cdf_point, covariance_matrix):
    z = norm.ppf(cdf_point)
    corr_matrix = covariance_to_correlation(covariance_matrix)
    
    
    mean = np.zeros(len(corr_matrix))
    mvn = multivariate_normal(
        mean=mean, cov=corr_matrix
    )  # multivariate normal with right correlation structure
    z = z.flatten()
    mvariate_pdf = mvn.logpdf(z)
    pdf = norm.logpdf(z)
    return mvariate_pdf - np.sum(pdf)


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

def student_t_copula_density(cdfs, covariance_matrix, df):
    """
    Compute the log-density of the Student-t copula.

    Parameters:
    - cdfs: ndarray
        CDF values for each dimension (shape: (n_corr,n_ang,n_points_per_dim)).
    - covariance_matrix: ndarray
        Covariance matrix for the copula (shape: (n_dims, n_dims)).
    - df: float
        Degrees of freedom for the Student-t distribution.

    Returns:
    - copula_density: ndarray
        Log-density of the Student-t copula (shape: (n_points,)).
    """
    # Convert CDFs to Student-t space
    cdfs_flat = cdfs.reshape(-1, cdfs.shape[-1])
    z = t.ppf(cdfs_flat, df)  # same shape as cdfs: (n_dims, n_points_per_dim)
    ppf_points = meshgrid_and_recast(z) # (n_points, n_dims)
    # Convert covariance matrix to correlation matrix
    corr_matrix = covariance_to_correlation(covariance_matrix)

    # Compute the multivariate Student-t log PDF
    
    # Scale matrix for the multivariate Student-t distribution
    scale = corr_matrix * (df - 2)/ df

    # Compute the multivariate Student-t log PDF
    mv_t_logpdf = multivariate_student_t_logpdf(ppf_points, df, scale)

    # Compute the marginal Student-t log PDFs
    marginal_logpdf = t.logpdf(z, df)  # Evaluate Student-t at the inverse CDF points
    marginal_logpdf_grid = meshgrid_and_recast(marginal_logpdf)  # Stack the PDFs along new axis
    marginal_logpdf_sum = np.sum(marginal_logpdf_grid, axis=1)

    # Log Copula density
    copula_density = mv_t_logpdf - marginal_logpdf_sum

    return copula_density

# ============================================================================
# Matrix Utilities and Conditioning
# ============================================================================


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
    corr_matrix = test_correlation_matrix(corr_matrix)
    return corr_matrix


def meshgrid_and_recast(funcs):
    """
    Create a meshgrid from the given points and recast it to a 2D array.

    Parameters:
    - funcs: A list of arrays representing pdfs/cdfs to recast, shape (n_dims, n_points_per_dim).

    Returns:
    - meshgrid: A 2D numpy array representing the meshgrid, shape (n_points, n_dims).
    """
    meshgrid = np.meshgrid(*funcs)
    stacked_meshgrid = np.stack(meshgrid, axis=-1)
    return stacked_meshgrid.reshape(-1, stacked_meshgrid.shape[-1])  # Reshape to 2D array



def get_well_conditioned_matrix(corr_matrix, min_eigenvalue=DEFAULT_MIN_EIGENVALUE):
    eigvals, eigvecs = eigh(corr_matrix)
    eigvals = np.maximum(eigvals, min_eigenvalue)  # Clip small eigenvalues
    return eigvecs @ np.diag(eigvals) @ eigvecs.T  # Reconstruct matrix


def test_correlation_matrix(matrix, iscov=False):
    if iscov:
        corr_matrix = covariance_to_correlation(matrix)
    else:
        corr_matrix = matrix
    cond_number = np.linalg.cond(corr_matrix)
    logger.info('Condition number of correlation matrix: {:.2f}'.format(cond_number))
    if cond_number > 1e4:
        logger.warning('Regularizing...')
        corr_matrix = get_well_conditioned_matrix(corr_matrix)
        new_cond = np.linalg.cond(corr_matrix)
        logger.info('New condition number of correlation matrix: {:.2f}'.format(new_cond))
    return corr_matrix


# ============================================================================
# High-level Joint PDF Computation
# ============================================================================


def  joint_pdf(cdfs, pdfs, cov, copula_type="gaussian", df=None):
    """
    Compute the joint PDF for the given CDFs, PDFs, and covariance matrix.

    Parameters
    ----------
    cdfs : ndarray
        CDF values for each dimension (shape: (n_correlations, n_angular_bins, n_points_per_dim)).
    pdfs : ndarray
        PDF values for each dimension (shape: (n_correlations, n_angular_bins, n_points_per_dim)).
    cov : ndarray
        Covariance matrix for the Gaussian copula (shape: (n_dims, n_dims)).
    - copula_type: str
        Type of copula to use ("gaussian" or "student-t").
    - df: float, optional
        Degrees of freedom for the Student-t copula (required if copula_type="student-t").

    Returns
    -------
    ndarray
        Joint PDF values reshaped to be ready to plot, shape = (n_points_per_dim,) * n_dim.
    """
    if copula_type not in ["gaussian", "student-t"]:
        raise ValueError(f"Unsupported copula type: {copula_type}. Use 'gaussian' or 'student-t'.")
    
    if copula_type == "student-t" and df is None:
        raise ValueError("Degrees of freedom (df) must be provided for Student-t copula.")
    
    if copula_type == "student-t" and df <= 2:
        logger.warning(f"Low degrees of freedom (df={df}) may cause numerical issues.")
    

    # Flatten the PDFs along the first two axes
    
    n_points_per_dim = cdfs.shape[-1]
    pdfs_flat = pdfs.reshape(-1, pdfs.shape[-1])  # Shape: (n_dims, n_points_per_dim)
    n_dim = pdfs_flat.shape[0]

    if copula_type == "gaussian":
        copula_density = gaussian_copula_density(cdfs, cov)  # Shape: (n_total_points,)
    elif copula_type == "student-t":
        copula_density = student_t_copula_density(cdfs, cov, df)
    else:
        raise ValueError(f"Unsupported copula type: {copula_type}")
    
    pdf_points = meshgrid_and_recast(pdfs_flat)
    log_pdf_points = np.log(pdf_points)  # Shape: (n_total_points, n_dims)
  

    # Compute joint PDF
    joint_pdf_values = copula_density + np.sum(log_pdf_points, axis=1)  # Shape: (n_total_points,)

    # Reshape the joint PDF to match the original data space
    shape = (n_points_per_dim,) * n_dim
    joint_pdf_reshaped = joint_pdf_values.reshape(shape)  # Shape: (n_points_per_dim,) * ndim

    return joint_pdf_reshaped


def joint_pdf_2d(cdf_X, cdf_Y, pdf_X, pdf_Y, cov):
    # Compute marginals
    u = cdf_X
    v = cdf_Y
    pdf_x_grid, pdf_y_grid = np.meshgrid(pdf_X, pdf_Y)
    pdf_points = np.vstack([pdf_x_grid.ravel(), pdf_y_grid.ravel()]).T
    # check shapes here and in joint_pdf - copula_density has been reshaped to match data
    # Compute copula density
    copula_density = gaussian_copula_density(u, v, cov)

    # Joint PDF
    return copula_density * np.prod(pdf_points, axis=1)


# ============================================================================
# Data Subsetting and Evaluation
# ============================================================================


def cov_subset(cov, subset, num_angs):
    """
    Extracts a subset of the covariance matrix based on the provided indices.

    Parameters:
    - cov: The covariance matrix (shape: (rs_combs * ang, rs_combs * ang)).
    - subset: A list of (rs_combs, ang) tuples specifying the dimensions to extract.
    - num_angs: Total number of angular bins (including xi+ and xi-), i.e twice the actual number if xi- is included.

    Returns:
    - A reduced covariance matrix corresponding to the specified subset.
    """
    # Convert the subset of indices to flat indices
    flat_indices = [rs_comb * num_angs + ang for rs_comb, ang in subset]
    logger.info(f"Extracting covariance subset with indices: {flat_indices}")
    logger.info(f"New covariance has shape: {len(flat_indices)} x {len(flat_indices)}")
    return cov[np.ix_(flat_indices, flat_indices)]

def data_subset(data, subset,full_grid=False):
    """
    Extracts a subset of the data based on the provided indices.

    Parameters
    ----------
    data : ndarray
        Input data array. Can be 2D (e.g., shape: (n_correlations, n_angular_bins)) or 3D
        (e.g., shape: (n_correlations, n_angular_bins, n_points_per_dim)).
    subset : list of tuples
        List of (rs_combs, ang) tuples specifying the indices to extract.

    Returns
    -------
    ndarray
        Subset of the data. If the input is 3D, the third axis is preserved.
        For 2D data, always returns shape (n_rs_combs_subset, n_ang_bins_subset),
        where the axes correspond to all unique redshift and angular bin indices in the subset.
    """
    rs_combs_indices, ang_indices = zip(*subset)
    rs_combs_indices = np.array(rs_combs_indices)
    ang_indices = np.array(ang_indices)
    if full_grid:
        # Check if subset forms a full grid
        unique_rs = np.unique(rs_combs_indices)
        unique_ang = np.unique(ang_indices)
        expected_pairs = set((i, j) for i in unique_rs for j in unique_ang)
        actual_pairs = set(zip(rs_combs_indices, ang_indices))
        if expected_pairs == actual_pairs:
            # Return full 2D array
            if data.ndim == 2:
                logger.info(f"Returning full grid data subset with shape {data[np.ix_(unique_rs, unique_ang)].shape}.")
                return data[np.ix_(unique_rs, unique_ang)]
            elif data.ndim == 3:
                logger.info(f"Returning full grid data subset with shape {data[np.ix_(unique_rs, unique_ang, np.arange(data.shape[2]))].shape}.")
                # Preserve the third axis
                return data[np.ix_(unique_rs, unique_ang, np.arange(data.shape[2]))]
        # If not a full grid, fall through to default behavior

    if data.ndim == 2:
        logger.info(f"Returning data subset with shape {data[rs_combs_indices, ang_indices].shape}.")
        return data[rs_combs_indices, ang_indices] # returns 1d array of the datapoints desired
    elif data.ndim == 3:
        logger.info(f"Returning data subset with shape {data[rs_combs_indices, ang_indices].shape}.")
        return data[rs_combs_indices, ang_indices]  # Preserve the third axis, i.e. 2d array is returned
    else:
        raise ValueError("data must be a 2D or 3D array.")

def expand_subset_for_ximinus(subset, n_angbins):
    """
    Expand a subset of (rs, ang) pairs to include both xi+ and xi- bins, and sort so that all xi+ bins come first, then all xi- bins.
    Returns a list of (rs, ang) pairs with the desired ordering.
    """
    # Collect unique rs and ang indices from the input subset
    rs_set = sorted(set(rs for rs, _ in subset))
    ang_set = sorted(set(ang for _, ang in subset if ang < n_angbins))
    expanded = []
    # First all xi+ (ang in [0, n_angbins-1])
    for rs in rs_set:
        for ang in ang_set:
            if (rs, ang) in subset:
                expanded.append((rs, ang))
    # Then all xi- (ang+n_angbins)
    for rs in rs_set:
        for ang in ang_set:
            if (rs, ang) in subset:
                expanded.append((rs, ang + n_angbins))
    return expanded

def evaluate(x_data, xs, pdfs, cdfs, cov,subset=None):
    """
    Evaluates the joint PDF at a given data point using the Gaussian copula.

    Parameters:
    - x_data: The data point to evaluate (shape: (rs_combs, ang)).
    - xs: The x-values of the PDFs (shape: (rs_combs, ang, n)).
    - pdfs: The PDF values (shape: (rs_combs, ang, n)).
    - cdfs: The CDF values (shape: (rs_combs, ang, n)).
    - cov: The covariance matrix (shape: (rs_combs * ang, rs_combs * ang)).
    - subset: A list of (rs_combs, ang) tuples specifying the dimensions to evaluate. If None, all dimensions are used.

    Returns:
    - The log joint PDF value.
    """
    
    if subset is not None:
        # Select the subset of dimensions
        
        num_angs = pdfs.shape[1]
        x_data = data_subset(x_data, subset)
        xs = data_subset(xs, subset)
        pdfs = data_subset(pdfs, subset)
        cdfs = data_subset(cdfs, subset)
        # Extract the covariance matrix for the subset
        cov = cov_subset(cov, subset, num_angs)
        # Reduce the covariance matrix to the subset
    log_pdf_point, cdf_point = pdf_and_cdf_point_eval(x_data=x_data, xs=xs, pdfs=pdfs, cdfs=cdfs)
    

    copula_density = gaussian_copula_point_density(cdf_point, cov)

    return np.sum(log_pdf_point) + copula_density

# ============================================================================
# Internal/Development Functions
# ============================================================================


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
        ax.plot(xs[i, j], pdfs[i, j], label='PDF')
        if cdfs is not None:
            ax.plot(xs[i, j], cdfs[i, j], label='CDF')
        ax.set_title(f'rs={i}, ang={j}')
        ax.legend()
    for ax in axes[n_plots:]:
        ax.axis('off')
    #fig.tight_layout()
    if savepath:
        plt.savefig(savepath)
    plt.close(fig)
