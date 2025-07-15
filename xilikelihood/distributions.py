"""
Characteristic function calculation utilities for 2-point correlation functions.

This module provides functions for:
- Converting characteristic functions to PDFs
- Computing Gaussian covariances for characteristic functions of correlation functions  
- Batch processing of characteristic functions with JAX
- High-ell Gaussian extensions

For legacy functions from first paper, see deprecation warnings.
New analyses should primarily use the likelihood module.
"""

import os

import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import UnivariateSpline
from scipy.stats import multivariate_normal
import warnings

from .noise_utils import get_noisy_cl
from .cl2xi_transforms import get_integrated_wigners, pcls2xis, precompute_wigners_cache
from .core_utils import check_property_equal
from .theoretical_moments import get_moments_from_combination_matrix_1d

# Constants
FULL_SKY_AREA_DEG2 = 4 * np.pi * (180 / np.pi)**2  # Full sky area in square degrees

__all__ = [
    # Constants
    'FULL_SKY_AREA_DEG2',
    
    # Grid setup utilities
    'setup_t',

    # Gaussian characteristic functions (moved from helper_funcs)
    'gaussian_cf',
    'gaussian_cf_nD',
    
    # Core CF/PDF conversion functions
    'batched_cf_1d_jitted',
    'high_ell_gaussian_cf_1d', 
    'cf_to_pdf_1d',
    'get_exact',

    # Normalization/Marginalization
    'exp_norm_mean',
       
    # Gaussian covariance functions
    'gaussian_2d',
    'cov_xi_gaussian_nD',
    'mean_xi_gaussian_nD',
    'cov_xi_nD',
    
    # C_ℓ covariance functions
    'cov_cl_nD',
    'cov_cl_gaussian',
    'cov_cl_gaussian_mixed',
]

# ============================================================================
# Grid setup utilities
# ============================================================================

def setup_t(xi_max, steps):
    dts, t0s, ts = [], [], []

    for xi in xi_max:

        dt = 0.45 * 2 * np.pi / xi
        t0 = -0.5 * dt * (steps - 1)
        t = np.linspace(t0, -t0, steps - 1)
        dts.append(dt)
        t0s.append(t0)
        ts.append(t)

    t_inds = np.arange(len(t))
    t_sets = np.stack(np.meshgrid(ts[0], ts[1]), -1).reshape(-1, 2)

    return t_inds, t_sets, t0s, dts

# ============================================================================
# Gaussian characteristic functions (moved from helper_funcs)
# ============================================================================

def gaussian_cf(t, mu, sigma):

    return np.exp(1j * t * mu - 0.5 * sigma**2 * t**2)


def gaussian_cf_nD(t_sets, mu, cov):
    tmu = np.dot(t_sets, mu)
    cov_t = np.einsum("ij,kj->ki", cov, t_sets)
    tct = np.einsum("ki,ki->k", t_sets, cov_t)
    return np.exp(1j * tmu - 0.5 * tct)

# ============================================================================
# Core CF/PDF conversion functions
# ============================================================================

def batched_cf_1d(eigvals, max_vals, steps=1024):

    all_dt = 0.45 * 2 * jnp.pi / max_vals
    all_t0 = -0.5 * all_dt * (steps - 1)
    all_t = jnp.linspace(all_t0, -all_t0, steps - 1, axis=-1)
    t_evals = all_t[:, :, :, None] * eigvals[:, :, None, :]
    cfs = jnp.prod(jnp.sqrt(1 / (1 - 2 * 1j * t_evals)), axis=-1)

    return all_t, cfs


batched_cf_1d_jitted = jax.jit(batched_cf_1d, static_argnums=(2,))

def high_ell_gaussian_cf_1d(t_lowell, means, vars):
    """
    calculates the characteristic function for a set of 1d Gaussians used as high mulitpole moment extension.

    Parameters
    ----------
    t_lowell : 3d array
        Fourier space t grid used for the low multipole moment part
        shape: (number of redshift bin combinations, number of angular bins, number of t steps)
    means : 2d array
        mean values of the Gaussian extensions
        shape: (number of redshift bin combinations, number of angular bins)
    vars : 2d array
        variance values of the Gaussian extensions
        shape: (number of redshift bin combinations, number of angular bins)

    Returns
    -------
    3d complex array
        characteristic function of the Gaussian extensions on the same t grid as the low ell part
        shape: (number of redshift bin combinations, number of angular bins, number of t steps)
    """

    xip_max = np.fabs(means) + 500 * np.sqrt(vars)
    dt_xip = 0.45 * 2 * np.pi / xip_max
    steps = 4096
    all_t0 = -0.5 * dt_xip * (steps - 1)
    all_t = np.linspace(all_t0, -all_t0, steps - 1, axis=-1)

    gauss_cf = np.exp(1j * means[:, :, None] * all_t - 0.5 * vars[:, :, None] * all_t**2)

    interp_to_lowell_real = np.zeros_like(t_lowell, dtype=np.float64)
    interp_to_lowell_imag = np.zeros_like(t_lowell, dtype=np.float64)

    for i in range(gauss_cf.shape[0]):
        for j in range(gauss_cf.shape[1]):
            interp_to_lowell_real[i, j] = UnivariateSpline(
                all_t[i, j], gauss_cf[i, j].real, k=5, s=0
            )(t_lowell[i, j])
            interp_to_lowell_imag[i, j] = UnivariateSpline(
                all_t[i, j], gauss_cf[i, j].imag, k=5, s=0
            )(t_lowell[i, j])

    interp_to_lowell = 1j * interp_to_lowell_imag + interp_to_lowell_real

    return interp_to_lowell

def cf_to_pdf_1d(t, cf):
    """
    Converts a characteristic function phi(t) to a probability density function f(x).

    Parameters:
    - cf: The characteristic function as a function of t
    - t0: The first value of t at which the characteristic function has been evaluated
    - dt: The increment in t used

    Returns tuple of (x, pdf).
    """
    if t.ndim > 1:
        t = t.reshape(-1, t.shape[-1])
        cf_array = cf.reshape(-1, cf.shape[-1])
        pdf_array = np.fft.fft(cf_array, axis=-1)
        dt = t[:, 1] - t[:, 0]  # assuming uniform spacing
        t0 = t[:, 0]
        x_array = np.fft.fftfreq(cf_array.shape[1]) * 2 * np.pi / dt[:, None]
        pdf_array *= dt[:, None] * np.exp(1j * x_array * t0[:, None]) / (2 * np.pi)
        xs, pdfs = [], []

        for x, pdf in zip(x_array, pdf_array):
            x_sorted, pdf_sorted = list(zip(*sorted(zip(x, np.abs(pdf)))))
            xs.append(x_sorted)
            pdfs.append(pdf_sorted)
        # TODO: build in a check that the pdfs are close to zero at the boundaries
        pdfs = np.array(pdfs)
        xs = np.array(xs)
        pdfs = pdfs.reshape(cf.shape)
        xs = xs.reshape(cf.shape)

        return xs, pdfs

    else:
        # Main part of the pdf is given by a FFT
        pdf = np.fft.fft(cf)
        dt = t[1] - t[0]
        t0 = t[0]
        # x is given by the FFT frequencies multiplied by some normalisation factor 2pi/dt
        x = np.fft.fftfreq(cf.size) * 2 * np.pi / dt

        # Multiply pdf by factor to account for the differences between numpy's FFT and a true continuous FT
        pdf *= dt * np.exp(1j * x * t0) / (2 * np.pi)

        # Take the real part of the pdf, and sort both x and pdf by x
        # x, pdf = list(zip(*sorted(zip(x, pdf.real))))
        x, pdf = list(
            zip(*sorted(zip(x, np.abs(pdf))))
        )  # I have found this tends to be more numerically stable

        return (np.array(x), np.array(pdf))

def get_exact(m, cov, steps=4096):
    mean_trace, second, _ = get_moments_from_combination_matrix_1d(m, cov)
    var_trace = second - mean_trace**2
    ximax = mean_trace + 20 * np.sqrt(var_trace)
    # m = np.diag(m)
    t, cf = calc_quadcf_1D(ximax, steps, cov, m, is_diag=False)
    x_low, pdf_low = cf_to_pdf_1d(t, cf)
    return x_low, pdf_low, t, cf

# ============================================================================
# Normalization/Marginalization
# ============================================================================

def exp_norm_mean(x,posterior,reg=350):
    
    posterior = np.array(posterior)
    posterior = posterior - reg
    posterior = np.exp(posterior)
    diffs = np.diff(posterior)
    nonan_diffs = diffs[~np.isnan(diffs)]
    extended_diffs = np.append(diffs, 0)
    
    median_diffs = np.median(np.fabs(nonan_diffs))
    threshold = 500 * median_diffs
    
    
    cond = ~np.isnan(posterior) & (np.fabs(extended_diffs) < threshold)
    if np.sum(cond) == 0:
        raise ValueError("No values in posterior satisfy the condition for normalization.")
    integral = np.trapz(posterior[cond], x=x[cond])
    if not np.isfinite(integral) or integral <= 0:
        # Import matplotlib only when needed for debugging
        try:
            import matplotlib.pyplot as plt
            debugfig, ax = plt.subplots()
            ax.plot(x, posterior)
            ax.set_yscale("log")
            ax.set_title("Posterior not finite or integral <= 0")
            debugfig.savefig("posterior_not_finite.png")
            plt.close(debugfig)
        except ImportError:
            pass  # Skip plotting if matplotlib not available
        raise ValueError("Integral of posterior is not finite or less than or equal to zero.")
    # check convergence
    conditioned_posterior = posterior[cond]
    conditioned_x = x[cond]
    endvalues = conditioned_posterior[0]/np.max(conditioned_posterior), conditioned_posterior[-1]/np.max(conditioned_posterior)
    #print(endvalues)
    if endvalues[0] > 0.1 or endvalues[1] > 0.1:
        print("Warning: Posterior does not converge to zero at the edges.")
        print("End values:", endvalues)
    normalized_post = conditioned_posterior / integral
    mean = np.trapz(conditioned_x * normalized_post, x=conditioned_x)
    std = np.sqrt(np.trapz((conditioned_x-mean)**2 * normalized_post, x=conditioned_x))
    return conditioned_x, normalized_post, mean, std


# ============================================================================
# Gaussian covariance Functions 
# ============================================================================

def gaussian_2d(xs,mean,cov):
    mean_flat = mean.flatten()
    n_dim = mean_flat.shape[0]
    n_points_per_dim = xs.shape[-1]
    xs_flat = xs.reshape(-1, n_points_per_dim)

    # Create a meshgrid for the xs_flat
    x_grid = np.meshgrid(*xs_flat)
    x_points = np.stack(x_grid, axis=-1)
    x_points = x_points.reshape(-1, x_points.shape[-1])

    # Compute Gaussian PDF on the meshgrid
    gaussian_pdf = multivariate_normal.logpdf(x_points, mean=mean_flat, cov=cov)
    shape = (n_points_per_dim,) * n_dim
    # Reshape Gaussian PDF to match the 2D subset
    gaussian_pdf_reshaped = gaussian_pdf.reshape(shape)
    return gaussian_pdf_reshaped

def cov_xi_gaussian_nD(cl_objects, redshift_bin_combs, angbins_in_deg, eff_area, lmin=0, lmax=None, include_ximinus=True):
    # cov_xi_gaussian(lmin=0, noise_apo=False)
    # Calculates Gaussian  covariance (shot noise and sample variance) of a xi_plus correlation function
    # cl_objects order like in GLASS. assume mask and lmax is the same as for cl_object[0] for all

    # e.g. https://www.aanda.org/articles/aa/full_html/2018/07/aa32343-17/aa32343-17.html

    
    if lmax is None:
        if not check_property_equal(cl_objects, "lmax"):
            raise ValueError("lmax not equal for all cl objects.")
        else:
            lmax = cl_objects[0].lmax
    cov_cl2 = cov_cl_nD(cl_objects, lmax, redshift_bin_combs=redshift_bin_combs)

    fsky = eff_area / FULL_SKY_AREA_DEG2  # assume that all fields have at least similar enough fsky
    c_tot = cov_cl2[:, :, lmin : lmax + 1] / fsky
    ell = 2 * np.arange(lmin, lmax + 1) + 1
    
    # Double angular bins if ximinus is included
    n_angbins_orig = len(angbins_in_deg)
    if include_ximinus:
        n_angbins_effective = 2 * n_angbins_orig  # [xi+, xi+, ..., xi-, xi-, ...]
    else:
        n_angbins_effective = n_angbins_orig
    
    # Shape: (n_redshift_combs * n_angbins_effective, n_redshift_combs * n_angbins_effective)
    n_redshift_combs = len(cov_cl2)
    data_vector_size = n_redshift_combs * n_angbins_effective
    xi_cov = np.zeros((data_vector_size, data_vector_size))
    
    # Pre-compute Wigner functions once
    wigners_cache = precompute_wigners_cache(lmin, lmax, angbins_in_deg, include_ximinus)
    
    # Use np.ndindex to efficiently iterate over upper triangle only
    for flat_idx1, flat_idx2 in np.ndindex(data_vector_size, data_vector_size):
        # Only compute upper triangle (including diagonal)
        if flat_idx1 <= flat_idx2:
            # Convert flat indices back to (redshift_comb, angbin) pairs
            i = flat_idx1 // n_angbins_effective  # redshift comb for first index
            k = flat_idx1 % n_angbins_effective   # angular bin for first index
            j = flat_idx2 // n_angbins_effective  # redshift comb for second index
            l = flat_idx2 % n_angbins_effective   # angular bin for second index
            
            # Determine correlation type and original angular bin index
            k_orig = k % n_angbins_orig
            k_is_minus = k >= n_angbins_orig if include_ximinus else False
            l_orig = l % n_angbins_orig  
            l_is_minus = l >= n_angbins_orig if include_ximinus else False
            
            # Get appropriate Wigner functions from cache
            if include_ximinus:
                wigners1_plus, wigners1_minus = wigners_cache[angbins_in_deg[k_orig]]
                wigners2_plus, wigners2_minus = wigners_cache[angbins_in_deg[l_orig]]
                
                wigners1 = wigners1_minus if k_is_minus else wigners1_plus
                wigners2 = wigners2_minus if l_is_minus else wigners2_plus
            else:
                wigners1 = wigners_cache[angbins_in_deg[k_orig]][0]  # Only xi+ 
                wigners2 = wigners_cache[angbins_in_deg[l_orig]][0]
            
            # Compute covariance element
            xi_cov[flat_idx1, flat_idx2] = np.sum(wigners1 * wigners2 * c_tot[i, j] * ell)
    
    # Fill lower triangle by symmetry
    xi_cov = xi_cov + xi_cov.T - np.diag(np.diag(xi_cov))
    
    assert np.all(np.linalg.eigvals(xi_cov) >= 0), "Covariance matrix not positive-semidefinite"
    assert np.allclose(xi_cov, xi_cov.T), "Covariance matrix not symmetric"

    return xi_cov


def mean_xi_gaussian_nD(prefactors, pseudo_cl, lmin=0, lmax=None, kind="p"):
    pseudo_cl = jnp.array(pseudo_cl)

    pcl_means_p, pcl_means_m = pcls2xis(
        pseudo_cl,
        prefactors,
        lmax,
        lmin=lmin,
    )
    """ if hasattr(auto_cov_object, "_noise_sigma"):
        cl_e = auto_cov_object.ee.copy() + auto_cov_object.noise_cl
        cl_b = auto_cov_object.bb.copy() + auto_cov_object.noise_cl
    else:
        cl_e, cl_b = auto_cov_object.ee.copy(), auto_cov_object.bb.copy()
    cl_mean_p, cl_mean_m = helper_funcs.cl2xi((cl_e, cl_b), angbins_in_deg[i], lmax, lmin=lmin)
    # assert np.allclose(pcl_mean_p[0], cl_mean_p, rtol=1e-2), (pcl_mean_p[0], cl_mean_p)
    print(
        "lmin: {:d}, lmax: {:d}, pCl mean: {:.5e}, Cl mean: {:.5e}".format(
            lmin, lmax, pcl_mean_p[i], cl_mean_p
        )
    )
    assert np.allclose(pcl_mean_p[i], cl_mean_p, rtol=1e-1) """
    if kind == "p":
        return np.array(pcl_means_p)
    elif kind == "m":
        return np.array(pcl_means_m)
    elif kind == "both":
        return np.array(pcl_means_p), np.array(pcl_means_m)
    else:
        # Default: return both for backward compatibility
        return np.array(pcl_means_p), np.array(pcl_means_m)


def cov_xi_nD(cov_objects):
    """
    Patches together several pseudo alm covariances to a full covariance matrix.
    Only needed for multi-dimensional characteristic functions and moments, computationally challenging due to matrix size.
    """
    n = len(cov_objects)
    sidelen_almcov = int(0.5 * (-1 + np.sqrt(1 + 8 * n)))
    
    k = 0
    for i in range(sidelen_almcov):
        for j in reversed(list(range(sidelen_almcov)[: i + 1])):
            cov_object = cov_objects[k]

            sub_cov = cov_object.cov_alm_xi()
            if k == 0:
                len_sub = len(sub_cov)
                cov = np.zeros((len(sub_cov) * sidelen_almcov, len(sub_cov) * sidelen_almcov))
            cov[i * len_sub : (i + 1) * len_sub, j * len_sub : (j + 1) * len_sub] = sub_cov
            if i != j:
                cov[j * len_sub : (j + 1) * len_sub, i * len_sub : (i + 1) * len_sub] = sub_cov
            k += 1

    assert np.allclose(cov, cov.T), "Covariance matrix not symmetric"
    return cov

# ============================================================================
# C_ℓ covariance functions
# ============================================================================

def cov_cl_nD(cl_objects, lmax, redshift_bin_combs=None, n_redshift_bins=None):
    """
    Compute covariance matrix for C_ell for multiple redshift bins.
    
    Parameters:
    -----------
    cl_objects : list
        List of CL objects in triangular (GLASS) order
    lmax : int
        Maximum multipole
    redshift_bin_combs : array_like, optional
        Array of redshift bin combinations
    n_redshift_bins : int, optional
        Number of redshift bins (alternative to redshift_bin_combs)
        
    Returns:
    --------
    ndarray
        Covariance matrix of shape (n_combinations, n_combinations, lmax+1)
    """
    from .theory_cl import BinCombinationMapper
    
    n_cl_objects = len(cl_objects)
    # Determine number of redshift bins and combinations
    if redshift_bin_combs is None:
        if n_redshift_bins is None:
            raise ValueError("Provide either redshift_bin_combs or n_redshift_bins.")
        
        # Verify consistency
        expected_combinations = n_redshift_bins * (n_redshift_bins + 1) // 2
        if n_cl_objects != expected_combinations:
            raise ValueError(
                f"Number of cl_objects ({n_cl_objects}) not compatible with "
                f"n_redshift_bins ({n_redshift_bins}). Expected {expected_combinations}."
            )
        # Create mapper and get combinations
        mapper = BinCombinationMapper(n_redshift_bins)
        redshift_bin_combs = mapper.combinations
        
    else:
        redshift_bin_combs = np.array(redshift_bin_combs)

    n_combinations = len(redshift_bin_combs)
    # Create mapper for efficient indexing
    max_bin = np.max(redshift_bin_combs) + 1
    mapper = BinCombinationMapper(max_bin)
    
    # Initialize covariance matrix
    cov = np.zeros((n_combinations, n_combinations, lmax + 1))

    
    for i in range(n_combinations):
        for j in range(i, n_combinations):  # Only compute upper triangle
            
            (k, l), (m, n) = redshift_bin_combs[i], redshift_bin_combs[j]

            mix_combinations = [(k, m), (l, n), (k, n), (l, m)]
            mix_cl_objects = []
            for comb in mix_combinations:
                # Sort combination to match triangular storage
                sorted_comb = tuple(sorted(comb, reverse=True))
                cl_index = mapper.get_index(sorted_comb)
                
                if cl_index < len(cl_objects):
                    mix_cl_objects.append(cl_objects[cl_index])
                else:
                    # Handle case where combination doesn't exist (shouldn't happen with proper input)
                    mix_cl_objects.append(None)
          

            sub_cov = cov_cl_gaussian_mixed(tuple(mix_cl_objects), lmax)
            cov[i, j] = sub_cov
            cov[j, i] = sub_cov

    assert np.allclose(cov[:, :, 0], cov[:, :, 0].T), "Covariance matrix not symmetric"
    return cov


def cov_cl_gaussian(cl_object):
    cl_e = cl_object.ee.copy()
    cl_b = cl_object.bb.copy()
    noise2 = np.zeros_like(cl_e)

    if hasattr(cl_object, "_noise_sigma"):
        noise_B = noise_E = cl_object.noise_cl

        cl_e += noise_E
        cl_b += noise_B
        noise2 += np.square(noise_E) + np.square(noise_B)

    cl2 = np.square(cl_e) + np.square(cl_b)

    diag = 2 * cl2
    noise_diag = 2 * noise2
    return diag, noise_diag


def cov_cl_gaussian_mixed(mixed_cl_objects, lmax):
    cl_es, cl_bs = get_noisy_cl(
        mixed_cl_objects, lmax
    )  # should return all cle and clb needed with noise added
    one_ee, two_ee, three_ee, four_ee = cl_es
    one_bb, two_bb, three_bb, four_bb = cl_bs
    cl2 = one_ee * two_ee + three_ee * four_ee + one_bb * two_bb + three_bb * four_bb
    return cl2



# ============================================================================
# Deprecated functions
# ============================================================================



def cf_to_pdf_nd(cf_grid, t0, dt, verbose=True):
    """DEPRECATED: Use likelihood module instead."""
    import warnings
    warnings.warn(
        "cf_to_pdf_nd is deprecated. Use likelihood module for new analyses.",
        DeprecationWarning,
        stacklevel=2
    )
    



def pdf_xi_1D(*args, **kwargs):
    """DEPRECATED: Use likelihood module instead."""
    warnings.warn(
        "pdf_xi_1D is deprecated. Use likelihood module for new analyses. "
        "For reproducibility of first paper, see legacy/calc_pdf_v1.py",
        DeprecationWarning,
        stacklevel=2
    )

def high_ell_gaussian_cf(t_lowell, cov_object, angbin):
    """DEPRECATED: Use high_ell_gaussian_cf_1d instead."""
    warnings.warn(
        "high_ell_gaussian_cf is deprecated. Use high_ell_gaussian_cf_1d for new analyses. "
        "For reproducibility of first paper, see legacy/calc_pdf_v1.py",
        DeprecationWarning,
        stacklevel=2
    )

def get_cf_nD(tset, mset, cov):
    """DEPRECATED: Use likelihood module instead."""
    warnings.warn(
        "get_cf_nD is deprecated. Use likelihood module for new analyses. "
        "For reproducibility of first paper, see legacy/calc_pdf_v1.py",
        DeprecationWarning,
        stacklevel=2
    )

def high_ell_gaussian_cf_nD(t_sets, mu, cov):
    """DEPRECATED: Use high_ell_gaussian_cf_1d instead."""
    warnings.warn(
        "high_ell_gaussian_cf_nD is deprecated. Use high_ell_gaussian_cf_1d for new analyses. "
        "For reproducibility of first paper, see legacy/calc_pdf_v1.py",
        DeprecationWarning,
        stacklevel=2
    )
    # gauss_cf = helper_funcs.gaussian_cf_nD(t_sets, mu, cov)
    # return gauss_cf

def generate_combinations(n):
    warnings.warn(
        "generate_combinations is deprecated, use comb_mapper in theory_cl. "
        "This will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    combinations = []
    for i in range(n):
        for j in range(i, -1, -1):
            combinations.append([i, j])
    return np.array(combinations)

def get_cov_n(comb):
    warnings.warn(
        "get_cov_n is deprecated, use comb_mapper in theory_cl. "
        "This will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    row, column = get_cov_pos(comb)
    if row == 0:
        return 0
    else:
        rowlengths = np.arange(1, row + 1)
        n = np.sum(rowlengths) + column
        return n

def get_combs(cov_n):
    warnings.warn(
        "get_combs is deprecated, use comb_mapper in theory_cl. "
        "This will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2
    )
    row = -1
    sum_rows = 0
    while sum_rows < cov_n:
        row += 1
        rowlength = row + 1
        sum_rows += rowlength

    return (row, sum_rows - cov_n - 1)

def get_cov_pos(comb):
    """DEPRECATED: Use BinCombinationMapper instead."""
    warnings.warn(
        "Use BinCombinationMapper instead of get_cov_pos.",
        DeprecationWarning,
        stacklevel=2
    )
    if comb[0] < comb[1]:
        raise RuntimeError("Cross-correlation: provide combination with larger number first.")
    column = int(np.fabs(comb[1] - comb[0]))
    row = comb[0]
    return (row, column)

def get_cov_triang(cov_objects):
    # order of cov_objects: as in GLASS gaussian fields creation
    """DEPRECATED: Use BinCombinationMapper instead."""
    warnings.warn(
        "Use BinCombinationMapper instead of get_cov_triang.",
        DeprecationWarning,
        stacklevel=2
    )
    n = len(cov_objects)
    sidelen_xicov = int(0.5 * (-1 + np.sqrt(1 + 8 * n)))
    rowlengths = np.arange(1, sidelen_xicov + 1)
    cov_triang = [
        cov_objects[np.sum(rowlengths[:i]) : np.sum(rowlengths[: i + 1])]
        for i in range(rowlengths[-1])
    ]
   
    return cov_triang

def calc_quadcf_1D(val_max, steps, cov, m, is_diag=False):
    """
    DEPRECATED: Use batched_cf_1d_jitted instead where eigenvalues are directly provided. Use likelihood module for new analyses.
    """
    warnings.warn(
        "calc_quadcf_1D is deprecated. Use batched_cf_1d_jitted instead where eigenvalues are directly provided. "
        "Use likelihood module for new analyses."
        "For reproducibility of first paper, see legacy/calc_pdf_v1.py",
        DeprecationWarning,
        stacklevel=2
    )
    all_dt = 0.45 * 2 * np.pi / val_max
    all_t0 = -0.5 * all_dt * (steps - 1)
    all_t = np.linspace(all_t0, -all_t0, steps - 1, axis=-1)

    if is_diag:
        eigvals = jnp.diag(cov)
    else:
        eigvals = cov[m, m]

    t_evals = all_t[:, :, None] * eigvals[:, None]
    cfs = jnp.prod(jnp.sqrt(1 / (1 - 2 * 1j * t_evals)), axis=-1)

    return all_t, cfs