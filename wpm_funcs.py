"""
Wigner 3j-symbol computation and mask coupling functions.

This module provides functions for computing Wigner 3j-symbols, mask coupling
matrices, and related spherical harmonic operations for cosmic shear analysis.

Main Functions
--------------
prepare_wigners : Prepare Wigner symbol arrays for given spins
calc_w_element_from_mask : Calculate W matrix elements from mask
compute_w_arrays : Compute full W arrays for mask coupling
m_llp : Compute coupling matrix elements
smooth_cl : Apply smoothing to power spectra

Examples
--------
>>> import numpy as np
>>> l_arr = np.arange(100)
>>> wigners_arr = prepare_wigners(2, 10, 20, 1, -1, 99)
>>> smoothed = smooth_cl(l_arr, 30)
"""

import wigner
import healpy as hp
import numpy as np
import logging
from scipy.special import factorial

logger = logging.getLogger(__name__)

__all__ = [
    # Core Wigner functions
    'prepare_wigners',
    'wigners_on_array', 
    'wigners',
    'wigners_2',
    'wigners_0',
    
    # W matrix computation
    'calc_w_element_from_mask',
    'compute_w_arrays',
    'assemble_w_array',
    
    # Coupling matrix functions
    'm_llp',
    'w_factor',
    
    # Utility functions
    'get_wlm_l',
    'shorten_wigners',
    
    # Smoothing functions
    'smooth_gauss',
    'smooth_cl',
    'smooth_alm',
]



def w_factor(l, l1, l2):
    return np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l + 1) / (4 * np.pi))


def prepare_wigners(spin, L1, L2, M1, M2, lmax):
    # prepare an array of wigners (or products thereof) of allowed l to be summed for given m1,m2,l1,l2

    m = M1 - M2
    l_arr = np.arange(lmax + 1)
    len_wigners = len(l_arr)
    wigners1 = wigners_on_array(L1, L2, -M1, M2, lmax)

    if spin == 0:
        wigners0 = wigners_on_array(L1, L2, 0, 0, lmax)
        return l_arr, wigners1 * wigners0

    elif spin == 2:
        wigners2p, wigners2m = wigners_on_array(L1, L2, 2, -2, lmax), wigners_on_array(
            L1, L2, -2, 2, lmax
        )

        w2sum = wigners2p + wigners2m
        w2diff = wigners2p - wigners2m
        wp_l = w2sum * wigners1
        wm_l = w2diff * wigners1
        return l_arr, wp_l, wm_l

    else:
        raise RuntimeError("Wigner 3j-symbols can only be calculated for spin 0 or 2 fields.")


def get_wlm_l(wlm, m, allowed_l):
    lmax = hp.sphtfunc.Alm.getlmax(len(wlm))
    if m < 0:
        wlm_l = (-1) ** -m * np.conj(wlm[hp.sphtfunc.Alm.getidx(lmax, allowed_l, -m)])
    else:
        wlm_l = wlm[hp.sphtfunc.Alm.getidx(lmax, allowed_l, m)]

    return wlm_l


def shorten_wigners(wigner_lmax, lmax, wigners_l):
    if wigner_lmax > lmax:
        diff = int(wigner_lmax) - lmax
        wigner_lmax = lmax
        return wigner_lmax, wigners_l[:-diff]
    else:
        return wigner_lmax, wigners_l


def wigners_on_array(l1, l2, m1, m2, lmax):
    l_array = np.arange(lmax + 1)
    wigners = np.zeros(len(l_array))
    wlmin, wlmax, wcof = wigner.wigner_3jj(l1, l2, m1, m2)
    wlmax, wcof = shorten_wigners(wlmax, lmax, wcof)
    if len(wcof) == 0:
        return wigners
    else:
        min_ind, max_ind = np.argmin(np.fabs(wlmin - l_array)), np.argmin(np.fabs(wlmax - l_array))
        wigners[min_ind : max_ind + 1] = wcof
        return wigners


def wigners(l_arr, lp, lpp, purified=False):
    lmax = l_arr[-1]

    wigners2 = np.zeros(len(l_arr))

    wigners0 = wigners_on_array(lp, lpp, -2, 0, lmax)
    wigners00 = wigners_on_array(lp, lpp, 0, 0, lmax)
    if purified == False:
        return wigners0, wigners00
    else:
        if lpp >= 1:
            prefac1 = 2 * np.sqrt(
                (factorial(l_arr + 1) * factorial(l_arr - 1) * factorial(lpp + 1))
                / (factorial(l_arr - 1) * factorial(l_arr + 2) * factorial(lpp - 1))
            )

            wigners1 = wigners_on_array(lp, lpp, -2, 1, lmax)
            wigners1 *= prefac1
        if lpp >= 2:
            prefac2 = np.sqrt(
                (factorial(l_arr + 2) * factorial(lpp + 2))
                / (factorial(l_arr + 2) * factorial(lpp - 2))
            )

            wigners2 = wigners_on_array(lp, lpp, -2, 2)
            wigners2 *= prefac2

        return wigners0 + wigners1 + wigners2
    
def wigners_2(l_arr, lp, lpp):
    lmax = l_arr[-1]
    wigners0 = wigners_on_array(lp, lpp, -2, 0, lmax)
    return wigners0

def wigners_0(l_arr, lp, lpp):
    lmax = l_arr[-1]
    wigners00 = wigners_on_array(lp, lpp, 0, 0, lmax)
    return wigners00



def m_llp(wl, lmax, spin0=False):
    # take wl of any length, return mllp arrays to max given lmax
    wl_full = np.zeros(lmax + 1)
    wl_full[: len(wl)] = wl if len(wl) < lmax + 1 else wl[: lmax+1]

    m_3d_pp = np.zeros((lmax+1, lmax+1, lmax + 1))
    m_3d_mm = np.zeros_like(m_3d_pp)
    m_3d_zero = np.zeros_like(m_3d_pp)
    l = lp = np.arange(lmax + 1)
    lpp = np.arange(lmax + 1)

    for cp, i in enumerate(lp):
        for cpp, j in enumerate(lpp):
            if i < 2:
                # skip the first two rows for spin 2
                continue
            else:
                prefac = (2 * i + 1) * (2 * j + 1) * wl_full[cpp] / (4 * np.pi)
                wigners_l = wigners_2(l, i, j)
                m_3d = prefac * np.square(wigners_l)
                m_3d_pp[:, cp, cpp] = m_3d * 0.5 * (1 + (-1) ** (l + i + j))
                m_3d_mm[:, cp, cpp] = m_3d * 0.5 * (1 - (-1) ** (l + i + j))
                if spin0:
                    wigners0_l = wigners_0(l, i, j)
                    m_3d_zero[:, cp, cpp] = prefac * np.square(wigners0_l)
                
    if not spin0:
        return np.sum(m_3d_pp, axis=-1), np.sum(m_3d_mm, axis=-1) 
    else:
        return np.sum(m_3d_pp, axis=-1), np.sum(m_3d_mm, axis=-1), np.sum(m_3d_zero, axis=-1) 


def smooth_gauss(l, l_smooth):
    sigma2 = l_smooth**2 / 12 * np.log10(np.e)
    return np.exp(-(l**2) / (2 * sigma2))


def smooth_cl(l, l_smooth):
    sigma2 = l_smooth**2 / 6 * np.log10(np.e)
    return np.exp(-(l**2) / (2 * sigma2))


def smooth_alm(l_smooth, lmax):
    l_array = [np.arange(i, lmax + 1) for i in range(lmax + 1)]
    l_array = np.concatenate(l_array, axis=0)
    smoothing_arr = smooth_gauss(l_array, l_smooth)
    return smoothing_arr


def calc_w_element_from_mask(wlm_lmax, exact_lmax, L1, L2, M1, M2, spin0=True, spin2=True):
    """
    Calculate W matrix element for given l,m indices.
    
    Parameters
    ----------
    wlm_lmax : array
        Spherical harmonic coefficients of mask
    exact_lmax : int
        Maximum multipole for exact calculations
    L1, L2, M1, M2 : int
        Multipole and azimuthal indices
    spin0, spin2 : bool
        Whether to compute spin-0 and spin-2 components
        
    Returns
    -------
    tuple
        (w0, wp, wm) - W matrix elements for each spin
    """
    m = M1 - M2
    buffer_lmax = exact_lmax
    
    # Spin-0 calculation
    if not spin0:
        w0 = 0
    else:
        allowed_l, wigners0 = prepare_wigners(0, L1, L2, M1, M2, buffer_lmax)
        wlm_l = get_wlm_l(wlm_lmax, m, allowed_l)
        prefac = w_factor(allowed_l, L1, L2)
        w0 = (-1) ** np.abs(M1) * np.sum(wigners0 * prefac * wlm_l)
    
    # Spin-2 calculation
    if not spin2 or np.logical_or(L1 < 2, L2 < 2):
        wp, wm = 0, 0
    else:
        allowed_l, wp_l, wm_l = prepare_wigners(2, L1, L2, M1, M2, buffer_lmax)
        prefac = w_factor(allowed_l, L1, L2)
        wlm_l = get_wlm_l(wlm_lmax, m, allowed_l)
        wp = 0.5 * (-1) ** np.abs(M1) * np.sum(prefac * wlm_l * wp_l)
        wm = 0.5 * 1j * (-1) ** np.abs(M1) * np.sum(prefac * wlm_l * wm_l)
    
    return w0, wp, wm

def compute_w_arrays(wlm_lmax, exact_lmax, cov_ell_buffer, spin0=True, spin2=True, verbose=True):
    """
    Compute full W arrays for mask coupling matrix.
    
    Parameters
    ----------
    wlm_lmax : array
        Spherical harmonic coefficients of mask
    exact_lmax : int
        Maximum multipole for exact calculations
    cov_ell_buffer : int
        Buffer in multipole space
    spin0, spin2 : bool
        Whether to compute spin components
    verbose : bool
        Whether to print progress
        
    Returns
    -------
    tuple
        (w0_arr, wpm_arr) - computed W arrays
    """
    import logging
    logger = logging.getLogger(__name__)
    
    L = np.arange(exact_lmax + cov_ell_buffer + 1)
    M = np.arange(-exact_lmax - cov_ell_buffer, exact_lmax + cov_ell_buffer + 1)
    Nl, Nm = len(L), len(M)
    
    # Initialize arrays
    w0_arr = np.zeros((Nl, Nm, Nl, Nm), dtype=complex) if spin0 else None
    wpm_arr = np.zeros((2, Nl, Nm, Nl, Nm), dtype=complex) if spin2 else None
    
    # Generate argument list - matching your original indexing
    arglist = []
    for l1, L1 in enumerate(L):
        M1_arr = np.arange(-L1, L1 + 1)
        for l2, L2 in enumerate(L):
            M2_arr = np.arange(-L2, L2 + 1)
            for m1, M1 in enumerate(M1_arr):
                for m2, M2 in enumerate(M2_arr):
                    # Calculate the actual indices into the M array
                    m1_ind = np.where(M == M1)[0][0]  # Find index of M1 in M array
                    m2_ind = np.where(M == M2)[0][0]  # Find index of M2 in M array
                    arglist.append((l1, m1_ind, l2, m2_ind, L1, L2, M1, M2))
    
    logger.info(f"Computing W arrays for {len(arglist)} elements...")
    
    # Compute elements
    for i, (l1_ind, m1_ind, l2_ind, m2_ind, L1, L2, M1, M2) in enumerate(arglist):
        if verbose and i % 1000 == 0:
            logger.info(f"Progress: {i/len(arglist)*100:.1f}%")
        
        w0, wp, wm = calc_w_element_from_mask(
            wlm_lmax, exact_lmax, L1, L2, M1, M2, spin0, spin2
        )
        
        # Store results using the correct indices
        inds = (l1_ind, m1_ind, l2_ind, m2_ind)
        if w0 and spin0:
            w0_arr[inds] = w0
        if (wp or wm) and spin2:
            wpm_arr[0, l1_ind, m1_ind, l2_ind, m2_ind] = wp
            wpm_arr[1, l1_ind, m1_ind, l2_ind, m2_ind] = wm
    
    logger.info("W array computation completed")
    return w0_arr, wpm_arr

def assemble_w_array(w0_arr, wpm_arr, spin0=True, spin2=True):
    """Assemble final W array from spin components."""
    if spin0 and spin2:
        return np.append(wpm_arr, w0_arr[None, :, :, :, :], axis=0)
    elif spin0:
        helper = np.empty_like(w0_arr)[None, :, :, :, :]
        return np.append(np.append(helper, helper, axis=0), w0_arr[None, :, :, :, :], axis=0)
    elif spin2:
        return wpm_arr
    else:
        raise ValueError("Must have at least one of spin0 or spin2")