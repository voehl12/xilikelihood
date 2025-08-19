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
from .diagnostic_tools import plot_pdfs_and_cdfs
from scipy.linalg import eigh
from scipy.special import gamma
import logging

logger = logging.getLogger(__name__)


DEFAULT_MIN_EIGENVALUE = 5e-4
DEFAULT_INTERPOLATION_POINTS = 512
DEFAULT_PDF_THRESHOLD = 1e-3
DEFAULT_NORMALIZATION_TOLERANCE = 1e-3
DEFAULT_SMALL_NUMBER = 1e-300

__all__ = [
    # Core copula functions
    'gaussian_copula_density',
    'student_t_copula_density', 
    'gaussian_copula_point_density',
    'joint_logpdf',
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


def validate_pdfs(pdfs, xs, cdfs=None,atol=DEFAULT_NORMALIZATION_TOLERANCE, plot=False, max_plots=32, savepath=None):
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


def pdf_to_cdf(xs, pdfs, num_points):
    xs_interp, pdfs_interp = interpolate_along_last_axis(xs, pdfs, num_points=num_points)
    cdfs = cumtrapz(pdfs_interp, xs_interp, initial=0)

    # Vectorized normalization check
    final_vals = cdfs[..., -1]
    normalization_tolerance = 1e-2
    mask = np.abs(final_vals - 1) >= normalization_tolerance
    problematic_indices = list(zip(*np.where(mask)))
    if problematic_indices:
        logger.warning(f"CDFs not normalized at indices: {problematic_indices}. Final values: {final_vals[mask]}")

    # Normalize all CDFs regardless
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
    # Convert CDFs to Student-t space with boundary protection
    cdfs_flat = cdfs.reshape(-1, cdfs.shape[-1])
    
    # Check for problematic CDF values that cause infinite quantiles
    n_zeros = np.sum(cdfs_flat == 0.0)
    n_ones = np.sum(cdfs_flat == 1.0)
    if n_zeros > 0 or n_ones > 0:
        logger.warning(f"Student-t copula: Found {n_zeros} exact zeros and {n_ones} exact ones in CDFs. "
                      f"This will cause infinite quantiles and copula failure. "
                      f"Consider using Gaussian copula for realistic cosmological data.")
    
    # Clip CDFs to avoid infinite quantiles (essential for realistic marginals)
    epsilon = 1e-12  # Small but not too small to avoid numerical issues
    cdfs_clipped = np.clip(cdfs_flat, epsilon, 1 - epsilon)
    
    # Warn if clipping was necessary
    n_clipped = np.sum((cdfs_flat != cdfs_clipped))
    if n_clipped > 0:
        logger.warning(f"Student-t copula: Clipped {n_clipped} CDF values to avoid infinite quantiles. "
                      f"This may affect marginal recovery accuracy.")
    
    z = t.ppf(cdfs_clipped, df)  # same shape as cdfs: (n_dims, n_points_per_dim)
    ppf_points = meshgrid_and_recast(z) # (n_points, n_dims)
    
    # Convert covariance matrix to correlation matrix
    corr_matrix = covariance_to_correlation(covariance_matrix)

    # Scale matrix for the multivariate Student-t distribution
    # Use the standard parameterization: scale = corr_matrix * (df - 2) / df
    # This ensures the covariance matrix equals the correlation matrix
    scale = corr_matrix * (df - 2) / df

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


def joint_logpdf(cdfs, pdfs, cov, copula_type="gaussian", df=None):
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
    
    if copula_type == "student-t":
        # Enhanced warnings for numerical stability
        if df <= 2:
            logger.warning(f"Very low degrees of freedom (df={df} ≤ 2) will cause numerical instability. "
                          f"Student-t copulas require df > 2. Consider df ≥ 5 for stable results.")
        elif df < 5:
            logger.warning(f"Low degrees of freedom (df={df} < 5) may cause numerical issues with high correlations. "
                          f"Consider df ≥ 5 for more stable results.")
        
        # Check for problematic correlation combinations
        if hasattr(cov, 'shape') and len(cov.shape) == 2:
            # Convert to correlation matrix to check correlations
            corr_matrix = covariance_to_correlation(cov)
            max_corr = np.max(np.abs(corr_matrix[~np.eye(corr_matrix.shape[0], dtype=bool)]))
            
            if df < 5 and max_corr > 0.8:
                logger.warning(f"High correlation (max |ρ|={max_corr:.3f}) with low df={df} "
                              f"may cause marginal recovery errors >5%. Consider df ≥ 5 or |ρ| ≤ 0.7.")
            elif df < 10 and max_corr > 0.9:
                logger.warning(f"Very high correlation (max |ρ|={max_corr:.3f}) with df={df} "
                              f"may cause numerical issues. Consider df ≥ 10 for |ρ| > 0.9.")
    

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
  

    # Compute joint PDF (in log space, then exponentiate)
    log_joint_pdf_values = copula_density + np.sum(log_pdf_points, axis=1)  # Shape: (n_total_points,)

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
        Input data array. Can be 2D (e.g., shape: (n_correlations, n_angular_bins)), 3D
        (e.g., shape: (n_correlations, n_angular_bins, n_points_per_dim)), or 4D
        (e.g., shape: (n_correlations, n_angular_bins, n_points_per_dim, n_extra)).
    subset : list of tuples
        List of (rs_combs, ang) tuples specifying the indices to extract.

    Returns
    -------
    ndarray
        Subset of the data. For 2D, shape (n_rs_combs_subset, n_ang_bins_subset).
        For 3D, shape (n_rs_combs_subset, n_ang_bins_subset, n_points_per_dim).
        For 4D, shape (n_rs_combs_subset, n_ang_bins_subset, n_points_per_dim, n_extra).
    """
    rs_combs_indices, ang_indices = zip(*subset)
    rs_combs_indices = np.array(rs_combs_indices)
    ang_indices = np.array(ang_indices)
    if full_grid:
        unique_rs = np.unique(rs_combs_indices)
        unique_ang = np.unique(ang_indices)
        expected_pairs = set((i, j) for i in unique_rs for j in unique_ang)
        actual_pairs = set(zip(rs_combs_indices, ang_indices))
        if expected_pairs == actual_pairs:
            if data.ndim == 2:
                logger.info(f"Returning full grid data subset with shape {data[np.ix_(unique_rs, unique_ang)].shape}.")
                return data[np.ix_(unique_rs, unique_ang)]
            elif data.ndim == 3:
                logger.info(f"Returning full grid data subset with shape {data[np.ix_(unique_rs, unique_ang, np.arange(data.shape[2]))].shape}.")
                return data[np.ix_(unique_rs, unique_ang, np.arange(data.shape[2]))]
            elif data.ndim == 4:
                logger.info(f"Returning full grid data subset with shape {data[np.ix_(unique_rs, unique_ang, np.arange(data.shape[2]), np.arange(data.shape[3]))].shape}.")
                return data[np.ix_(unique_rs, unique_ang, np.arange(data.shape[2]), np.arange(data.shape[3]))]
        # If not a full grid, fall through to default behavior

    if data.ndim == 2:
        logger.info(f"Returning data subset with shape {data[rs_combs_indices, ang_indices].shape}.")
        return data[rs_combs_indices, ang_indices]
    elif data.ndim == 3:
        logger.info(f"Returning data subset with shape {data[rs_combs_indices, ang_indices].shape}.")
        return data[rs_combs_indices, ang_indices]
    elif data.ndim == 4:
        logger.info(f"Returning data subset with shape {data[rs_combs_indices, ang_indices].shape}.")
        return data[rs_combs_indices, ang_indices]
    else:
        raise ValueError("data must be a 2D, 3D, or 4D array.")

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





