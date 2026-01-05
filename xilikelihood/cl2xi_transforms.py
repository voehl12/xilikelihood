"""
Power spectrum to correlation function transforms.

This module provides functions for transforming between angular power spectra (C_l)
and two-point correlation functions (xi_+, xi_-) using Wigner d-functions and
Legendre polynomials. Supports both individual angle calculations and binned averages.

Main Functions
--------------
pcl2xi : Transform pseudo-C_l to correlation functions
pcls2xis : Batch transform multiple pseudo-C_l realizations  
cl2xi : Transform true C_l to correlation functions
prep_prefactors : Prepare angular-dependent prefactors
cl2pseudocl : Apply mask coupling to convert true to pseudo C_l

Examples
--------
>>> import numpy as np
>>> # Prepare angular bins and prefactors
>>> angular_bins = [(1.0, 2.0), (2.0, 4.0)]  # degrees
>>> wl = np.ones(100)  # mask power spectrum
>>> prefactors = prep_prefactors(angular_bins, wl, 99, 50)
>>> 
>>> # Transform pseudo-C_l to correlation functions
>>> pcl = (cl_ee, cl_bb, cl_eb)  # E, B, EB power spectra
>>> xi_plus, xi_minus = pcl2xi(pcl, prefactors, lmax=50)
"""


import numpy as np
from scipy.integrate import quad_vec
from scipy.special import eval_legendre
import logging

logger = logging.getLogger(__name__)

try:
    import jax
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np  # Fallback to numpy
    logger.warning("JAX not available - some functions will be slower")

try:
    import wigner
    HAS_WIGNER = True
except ImportError:
    HAS_WIGNER = False
    logger.error("wigner module required for correlation function calculations")

__all__ = [
    # Core transformation functions
    'pcl2xi',
    'pcls2xis', 
    'cl2xi',
    'cl2pseudocl',
    
    # Prefactor calculation
    'prep_prefactors',
    'ang_prefactors',
    'bin_prefactors',
    
    # Utility functions
    'get_int_lims',
    'get_integrated_wigners',
    'compute_kernel',
    
    # JAX optimized versions
    'pcls2xis_jit',
    'cl2pseudocl_einsum',
    
    # Optional dependency flags
    'HAS_JAX',
    'HAS_WIGNER',
]

# ============================================================================
# Core transformation functions
# ============================================================================

def pcl2xi(pcl, prefactors, out_lmax=None, lmin=0):
    """
    Transform pseudo-C_l to correlation functions xi_+ and xi_-.
    
    Parameters
    ----------
    pcl : tuple of array_like
        Three arrays (cl_ee, cl_bb, cl_eb) containing E, B, and EB power spectra
    prefactors : array_like
        Prefactor array from prep_prefactors with shape (n_angles, 2, lmax+1)
    out_lmax : int, optional
        Maximum multipole for summation. If None, uses length of pcl arrays
    lmin : int, optional
        Minimum multipole for summation (default: 0)
        
    Returns
    -------
    xi_plus, xi_minus : array_like
        Correlation functions at the angular separations
        
    Raises
    ------
    ValueError
        If input arrays have incompatible shapes
    ImportError
        If wigner module is not available
    """
    if not HAS_WIGNER:
        raise ImportError("wigner module required for correlation function calculations")
    
    # Input validation
    if len(pcl) != 3:
        raise ValueError("pcl must contain exactly 3 arrays (EE, BB, EB)")
    
    pcl_e, pcl_b, pcl_eb = pcl
    
    # Check array shapes
    if not (len(pcl_e) == len(pcl_b) == len(pcl_eb)):
        raise ValueError("All power spectrum arrays must have the same length")
    
    if out_lmax is None:
        out_lmax = len(pcl_e) - 1
    
    if lmin < 0:
        raise ValueError("lmin must be non-negative")
    
    if out_lmax >= len(pcl_e):
        raise ValueError(f"out_lmax ({out_lmax}) must be less than pcl length ({len(pcl_e)})")
    
    # Check prefactors shape
    expected_shape = (prefactors.shape[0], 2, out_lmax + 1)
    if prefactors.shape[2] < out_lmax + 1:
        raise ValueError(f"Prefactors array too small. Expected at least {expected_shape}, got {prefactors.shape}")
    

    kernel_xip, kernel_xim = compute_kernel(pcl, prefactors, out_lmax, lmin)
    xip = np.sum(kernel_xip, axis=-1)
    xim = np.sum(kernel_xim, axis=-1)
    return xip, xim


def pcls2xis(pcls, prefactors, out_lmax=None, lmin=0):
    """
    Generate xi+ and xi- from pseudo-Cl data.

    Parameters
    ----------
    pcls : numpy array
        (3, batchsize, lmax+1) ; batchsize can also be number of correlations
        if array is 4-dim (3, batchsize, n_corr, lmax+1) it is assumed that the batchsize is the first dimension
        and the second dimension is the number of correlations.
    prefactors : np.array
        Array with bin or angle prefactors from prep_prefactors (i.e., an (len(angles), 2, out_lmax) array).
    out_lmax : int
        lmax to which sum over pcl is taken.
    lmin : int, optional
        Minimum l value, by default 0.

    Returns
    -------
    xips, xims: (batchsize, n_angbins) arrays
    """
    pcls_e, pcls_b, pcls_eb = pcls[0], pcls[1], pcls[2]
    if out_lmax is None:
        out_lmax = len(pcls_e[0]) - 1
    l = 2 * jnp.arange(lmin, out_lmax + 1) + 1
    p_cl_prefactors_p, p_cl_prefactors_m = jnp.array(prefactors[:, 0]), jnp.array(prefactors[:, 1])

    if pcls_e.ndim == 2:
        xips = jnp.sum(
            p_cl_prefactors_p[None, :, lmin : out_lmax + 1]
            * l
            * (pcls_e[:, None, lmin : out_lmax + 1] + pcls_b[:, None, lmin : out_lmax + 1]),
            axis=-1,
        )
        xims = jnp.sum(
            p_cl_prefactors_m[None, :, lmin : out_lmax + 1]
            * l
            * (
                pcls_e[:, None, lmin : out_lmax + 1]
                - pcls_b[:, None, lmin : out_lmax + 1]
                - 2j * pcls_eb[:, None, lmin : out_lmax + 1]
            ),
            axis=-1,
        )
    elif pcls_e.ndim == 3:
        xips = jnp.sum(
            p_cl_prefactors_p[None,None, :, lmin : out_lmax + 1]
            * l
            * (pcls_e[:, :, None, lmin : out_lmax + 1] + pcls_b[:, :, None, lmin : out_lmax + 1]),
            axis=-1,
        )
        xims = jnp.sum(
            p_cl_prefactors_m[None, None,:, lmin : out_lmax + 1]
            * l
            * (
                pcls_e[:, :, None, lmin : out_lmax + 1]
                - pcls_b[:, :, None, lmin : out_lmax + 1]
                - 2j * pcls_eb[:, :, None, lmin : out_lmax + 1]
            ),
            axis=-1,
        )
    else:
        print("pcls2xis: pcls has to be 2 or 3 dimensional, returning 1d pcl2xi")
        return pcl2xi(pcls, prefactors, out_lmax, lmin)
    

    return xips, xims


# Only create jitted version if JAX is available
if 'jax' in locals():
    pcls2xis_jit = jax.jit(pcls2xis, static_argnums=(2, 3))
else:
    pcls2xis_jit = pcls2xis  # Fallback to non-jitted version


def cl2xi(cl, ang_bin_in_deg, out_lmax, lmin=0):
    cl_e, cl_b = cl
    l = 2 * np.arange(lmin, out_lmax + 1) + 1
    wigner_int_p = lambda theta_in_rad: theta_in_rad * wigner.wigner_dl(
        lmin, out_lmax, 2, 2, theta_in_rad
    )
    wigner_int_m = lambda theta_in_rad: theta_in_rad * wigner.wigner_dl(
        lmin, out_lmax, 2, -2, theta_in_rad
    )
    norm = 1 / (4 * np.pi)
    lower, upper = ang_bin_in_deg
    lower, upper = np.radians(lower), np.radians(upper)
    t_norm = 2 / (upper**2 - lower**2)
    integrated_wigners_p = quad_vec(wigner_int_p, lower, upper)[0]
    integrated_wigners_m = quad_vec(wigner_int_m, lower, upper)[0]
    xip = (
        t_norm
        * norm
        * np.sum(integrated_wigners_p * l * (cl_e[lmin : out_lmax + 1] + cl_b[lmin : out_lmax + 1]))
    )
    xim = (
        t_norm
        * norm
        * np.sum(integrated_wigners_m * l * (cl_e[lmin : out_lmax + 1] - cl_b[lmin : out_lmax + 1]))
    )

    return xip, xim

def cl2pseudocl(mllp, theorycls):
    """
    Apply mask coupling matrix to convert true C_l to pseudo-C_l.
    
    This function applies the mask-induced mode coupling to transform
    true angular power spectra into the pseudo power spectra that would
    be measured from a masked sky survey.
    
    Parameters
    ----------
    mllp : array_like
        Mode coupling matrix with shape (3, lmax+1, lmax+1) containing
        [M^EE, M^BB, M^EB] coupling matrices
    theorycls : list of TheoryCl objects
        Theory power spectra objects containing ee, bb attributes
        
    Returns
    -------
    array_like
        Pseudo power spectra with shape (3, n_spectra, lmax+1) containing
        [pseudo_EE, pseudo_BB, pseudo_EB]
        
    Notes
    -----
    The transformation follows the relation from NaMaster:
    C^pseudo_l = sum_l' M_ll' C^true_l'
    
    References
    ----------
    Alonso et al. 2019, MNRAS, 484, 4127 (NaMaster paper) + Scientific Documentation
    https://namaster.readthedocs.io/en/latest/index.html
    """

    cl_es, cl_bs = [], []
    for theorycl in theorycls:
        if hasattr(theorycl, "_noise_sigma"):
            cl_e = theorycl.ee.copy() + theorycl.noise_cl
            cl_b = theorycl.bb.copy() + theorycl.noise_cl

        else:
            cl_e = theorycl.ee.copy()
            cl_b = theorycl.bb.copy()
        cl_es.append(cl_e)
        cl_bs.append(cl_b)

    mllp = jnp.array(mllp)
    cl_es, cl_bs = jnp.array(cl_es), jnp.array(cl_bs)
    p_ee, p_bb, p_eb = cl2pseudocl_einsum(mllp, cl_es, cl_bs)
    return jnp.array([p_ee, p_bb, p_eb])


# Conditionally apply JIT if available
def cl2pseudocl_einsum(mllp, cl_e, cl_b):
    cl_eb = cl_be = cl_b
    p_ee = jnp.einsum("lm,nm->nl", mllp[0], cl_e) + jnp.einsum("lm,nm->nl", mllp[1], cl_b)
    p_bb = jnp.einsum("lm,nm->nl", mllp[1], cl_e) + jnp.einsum("lm,nm->nl", mllp[0], cl_b)
    p_eb = jnp.einsum("lm,nm->nl", mllp[0], cl_eb) - jnp.einsum("lm,nm->nl", mllp[1], cl_be)
    return p_ee, p_bb, p_eb

# Apply JIT decoration if JAX is available
if 'jax' in locals():
    cl2pseudocl_einsum = jax.jit(cl2pseudocl_einsum)

# ============================================================================
# Prefactor calculation
# ============================================================================

def ang_prefactors(t_in_deg, wl, norm_lmax, out_lmax, kind="p"):
    # normalization factor automatically has same l_max as the pseudo-Cl summation

    t = np.radians(t_in_deg)
    norm_l = np.arange(norm_lmax + 1)
    legendres = eval_legendre(norm_l, np.cos(t))
    norm = 1 / np.sum((2 * norm_l + 1) * legendres * wl) / (2 * np.pi)

    if kind == "p":
        wigners = wigner.wigner_dl(0, out_lmax, 2, 2, t)
    elif kind == "m":
        wigners = wigner.wigner_dl(0, out_lmax, 2, -2, t)
    else:
        raise RuntimeError("correlation function kind needs to be p or m")

    return 2 * np.pi * norm * wigners


def bin_prefactors(ang_bin_in_deg, wl, norm_lmax, out_lmax, kind="p"):
    lower, upper = ang_bin_in_deg
    lower, upper = np.radians(lower), np.radians(upper)
    buffer = 0
    norm_l = np.arange(norm_lmax + buffer + 1)
    legendres = lambda t_in_rad: eval_legendre(norm_l, np.cos(t_in_rad))
    # TODO: check whether this sum needs to be done after the integration as well (even possible?)
    # -> sum is the same, the 1/ x is a problem when lots of the wl are zero, so it's actually better to do it this way, the order does not seem to matter in this case.
    norm = (
        lambda t_in_rad: 1
        / np.sum((2 * norm_l + 1) * legendres(t_in_rad) * wl[: norm_lmax + buffer + 1])
        / (2 * np.pi)
    )
    if kind == "p":
        wigners = lambda t_in_rad: wigner.wigner_dl(0, out_lmax, 2, 2, t_in_rad)
    elif kind == "m":
        wigners = lambda t_in_rad: wigner.wigner_dl(0, out_lmax, 2, -2, t_in_rad)
    else:
        raise RuntimeError("correlation function kind needs to be p or m")

    integrand = lambda t_in_rad: norm(t_in_rad) * wigners(t_in_rad) * t_in_rad
    # norm * d_l * weights
    W = 0.5 * (upper**2 - lower**2)  # weights already integrated
    A_ell = quad_vec(integrand, lower, upper)

    return 2 * np.pi * A_ell[0] / W


def prep_prefactors(angs_in_deg, wl, norm_lmax, out_lmax):
    """
    Calculate angular-dependent prefactors for correlation function transforms.

    Parameters
    ----------
    angs_in_deg : list
        Angular separations in degrees. Can be floats for point angles
        or tuples for angular bins
    wl : array_like
        Mask power spectrum up to at least norm_lmax
    norm_lmax : int
        Maximum multipole for normalization calculation
    out_lmax : int
        Maximum multipole for output prefactors

    Returns
    -------
    array_like
        Prefactor array with shape (len(angs_in_deg), 2, out_lmax+1)
        
    Raises
    ------
    ValueError
        If input parameters are invalid
    ImportError
        If wigner module is not available
    """
    if not HAS_WIGNER:
        raise ImportError("wigner module required for prefactor calculations")
    
    # Input validation
    if norm_lmax < 0 or out_lmax < 0:
        raise ValueError("norm_lmax and out_lmax must be non-negative")
    
    if len(wl) <= norm_lmax:
        raise ValueError(f"wl array too short. Need at least {norm_lmax + 1} elements, got {len(wl)}")
    
    if len(angs_in_deg) == 0:
        raise ValueError("angs_in_deg cannot be empty")
    
    # Check for valid angular ranges
    for i, ang in enumerate(angs_in_deg):
        if isinstance(ang, tuple):
            if len(ang) != 2:
                raise ValueError(f"Angular bin {i} must be a 2-tuple, got {ang}")
            if ang[0] >= ang[1]:
                raise ValueError(f"Invalid angular bin {i}: min >= max ({ang})")
            if ang[0] <= 0:
                raise ValueError(f"Angular separations must be positive, got {ang}")
            

    prefactors_arr = np.zeros((len(angs_in_deg), 2, out_lmax + 1))
    if type(angs_in_deg[0]) is tuple:
        prefactors = bin_prefactors
    else:
        prefactors = ang_prefactors
    for i, ang_in_deg in enumerate(angs_in_deg):
        prefactors_arr[i, 0] = prefactors(ang_in_deg, wl, norm_lmax, out_lmax)
        prefactors_arr[i, 1] = prefactors(ang_in_deg, wl, norm_lmax, out_lmax, kind="m")

    return prefactors_arr


# ============================================================================
# Utility functions
# ============================================================================


def get_int_lims(bin_in_deg):
    """Convert angular bin limits from degrees to radians."""
    binmin_in_deg, binmax_in_deg = bin_in_deg
    return np.radians(binmin_in_deg), np.radians(binmax_in_deg)

def precompute_wigners_cache(lmin, lmax, angbins_in_deg, include_ximinus=False):
    """
    Pre-compute all needed Wigner integrals once.
    
    Returns
    -------
    dict
        Cache with angular bins as keys, Wigner arrays as values
    """
    cache = {}
    for angbin in angbins_in_deg:
        if include_ximinus:
            cache[angbin] = get_integrated_wigners_both(lmin, lmax, angbin)
        else:
            cache[angbin] = (get_integrated_wigners(lmin, lmax, angbin), None)
    return cache

def get_integrated_wigners(lmin, lmax, bin_in_deg):
    wigner_int = lambda theta_in_rad: theta_in_rad * wigner.wigner_dl(
        lmin, lmax, 2, 2, theta_in_rad
    )
    lower, upper = get_int_lims(bin_in_deg)
    t_norm = 2 / (upper**2 - lower**2)
    integrated_wigners = quad_vec(wigner_int, lower, upper)[0]
    norm = 1 / (4 * np.pi)
    return norm * integrated_wigners * t_norm


def get_integrated_wigners_both(lmin, lmax, bin_in_deg):
    """
    Get integrated Wigner d-functions for both xi_plus and xi_minus.
    
    Returns
    -------
    tuple
        (wigners_plus, wigners_minus) for (2,2) and (2,-2) spin combinations
    """
    # Define separate integration functions
    def wigner_int_plus(theta_in_rad):
        return theta_in_rad * wigner.wigner_dl(lmin, lmax, 2, 2, theta_in_rad)
    
    def wigner_int_minus(theta_in_rad):
        return theta_in_rad * wigner.wigner_dl(lmin, lmax, 2, -2, theta_in_rad)
    
    lower, upper = get_int_lims(bin_in_deg)
    t_norm = 2 / (upper**2 - lower**2)
    norm = 1 / (4 * np.pi)
    
    # Separate integrations for each function
    integrated_plus = quad_vec(wigner_int_plus, lower, upper)[0]
    integrated_minus = quad_vec(wigner_int_minus, lower, upper)[0]
    
    return (norm * integrated_plus * t_norm, 
            norm * integrated_minus * t_norm)



def compute_kernel(pcl, prefactors, out_lmax=None, lmin=0):
    
    pcl_e, pcl_b, pcl_eb = pcl
    if out_lmax is None:
        out_lmax = len(pcl_e) - 1
    l = 2 * np.arange(lmin, out_lmax + 1) + 1
    p_cl_prefactors_p, p_cl_prefactors_m = prefactors[:, 0], prefactors[:, 1]
    kernel_xip = p_cl_prefactors_p[:, lmin : out_lmax + 1] * l * (pcl_e[lmin : out_lmax + 1] + pcl_b[lmin : out_lmax + 1])
    kernel_xim = p_cl_prefactors_m[:, lmin : out_lmax + 1] * l * (pcl_e[lmin : out_lmax + 1] - pcl_b[lmin : out_lmax + 1] - 2j * pcl_eb[lmin : out_lmax + 1])
    return kernel_xip, kernel_xim

