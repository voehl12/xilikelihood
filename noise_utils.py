"""
Noise utilities for cosmological weak lensing surveys.

This module provides functions for computing various types of observational noise:
- Shape noise in correlation functions (xi)
- Shape noise in power spectra (Cl)
- Pixel-level noise estimates
- Noisy power spectra construction

Typical usage for KiDS-like surveys with default parameters.
"""

import numpy as np
import healpy as hp

__all__ = [
    'get_noise_xi_cov',
    'get_noise_cl', 
    'get_noise_pixelsigma',
    'noise_cl_cube',
    'get_noisy_cl'
]

# Full sky area in square degrees
FULL_SKY_AREA_SQDEQ = 41253.0

def get_noise_xi_cov(
    survey_area_in_sqd, binmin_in_deg, binmax_in_deg, sigma_e=(0.282842712474619, 1.207829761642)
):
    """
    Compute shape noise covariance for correlation function measurements.
    
    Parameters
    ----------
    survey_area_in_sqd : float
        Survey area in square degrees
    binmin_in_deg, binmax_in_deg : float
        Angular bin edges in degrees
    sigma_e : tuple, optional
        (sigma_single_component, n_gal_per_arcmin2). Default is KiDS-like.
        
    Returns
    -------
    float
        Variance of xi measurement in this angular bin
    """
    if not isinstance(sigma_e, tuple) or len(sigma_e) != 2:
        raise ValueError("sigma_e must be tuple (sigma_single_component, n_gal_per_arcmin2)")
    
    sigma, n_gal = sigma_e
    pp_sigma = sigma * np.sqrt(2)
    A_annulus = np.pi * ((binmax_in_deg * 60) ** 2 - (binmin_in_deg * 60) ** 2)
    N_pair = (
        survey_area_in_sqd * 3600 * A_annulus * (n_gal) ** 2
    )  # this is 2 * N_pair but it comes from defining the shotnoise this way
    return (pp_sigma**2 / np.sqrt(N_pair)) ** 2


def get_noise_cl(sigma_e=(0.282842712474619, 1.207829761642)):
    # TODO: should be extended to mixed redshift bin C_ell with different shape noise parameters.
    """
    Compute shape noise level for power spectrum measurements.
    
    Parameters
    ----------
    sigma_e : tuple, optional
        (sigma_single_component, n_gal_per_arcmin2). Default is KiDS-like.
        
    Returns
    -------
    float
        Shape noise power spectrum level (constant across all l)
    """
    if not isinstance(sigma_e, tuple) or len(sigma_e) != 2:
        raise ValueError("sigma_e must be tuple (sigma_single_component, n_gal_per_arcmin2)")
    
    sigma, n_gal_per_arcmin2 = sigma_e
    return (sigma) ** 2 / (n_gal_per_arcmin2 * 3600 * 41253 / (4 * np.pi))


def get_noise_pixelsigma(nside=256, sigma_e=(0.282842712474619, 1.207829761642)):
    """
    Compute per-pixel shape noise standard deviation for HEALPix maps.
    
    Parameters
    ----------
    nside : int, optional
        HEALPix resolution parameter
    sigma_e : tuple, optional
        (sigma_single_component, n_gal_per_arcmin2). Default is KiDS-like.
        
    Returns
    -------
    float
        Standard deviation of shape noise per HEALPix pixel
    """
    if not isinstance(sigma_e, tuple) or len(sigma_e) != 2:
        raise ValueError("sigma_e must be tuple (sigma_single_component, n_gal_per_arcmin2)")
    
    sigma, n_gal_per_arcmin2 = sigma_e
    pixel_area_arcmin2 = hp.nside2pixarea(nside, degrees=True) * 3600
    return sigma / np.sqrt(n_gal_per_arcmin2 * pixel_area_arcmin2)


def noise_cl_cube(noise_cl):
    # TODO: separately for e,b and n?
    """
    Create a 3x3 noise covariance cube for E, B, and T modes.
    
    Returns shape (3, 3, lmax+1) with identical noise on diagonal.
    """
    c_all = np.zeros((3, 3, len(noise_cl)))
    for i in range(3):
        c_all[i, i] = noise_cl
    return c_all


def get_noisy_cl(cl_objects, lmax):
    """
    Extract E and B mode power spectra with noise added.
    
    Parameters
    ----------
    cl_objects : list
        List of Cl objects with .ee, .bb attributes. None entries create zeros.
    lmax : int
        Maximum multipole for output spectra
        
    Returns
    -------
    tuple
        (cl_es, cl_bs) tuples of noisy power spectra
    """
    cl_es, cl_bs = [], []
    for cl_object in cl_objects:
        if cl_object is None:
            # assert that this is in place of a cross cl?
            cl_e = cl_b = np.zeros((lmax + 1))
        else:
            cl_e = cl_object.ee.copy()
            cl_b = cl_object.bb.copy()
            if hasattr(cl_object, "_noise_sigma"):
                noise_B = noise_E = cl_object.noise_cl

                cl_e += noise_E
                cl_b += noise_B
        cl_es.append(cl_e)
        cl_bs.append(cl_b)
    return tuple(cl_es), tuple(cl_bs)