"""
Correlation function simulation utilities.

This module provides functions for simulating two-point correlation functions
from theoretical power spectra using either pseudo-C_l estimators or TreeCorr.

Main Functions
--------------
simulate_correlation_functions : Unified simulation interface
create_maps : Create Gaussian random maps using GLASS
compute_pseudo_cl : Compute pseudo-C_l from masked maps
compute_correlation_functions : Convert pseudo-C_l to correlation functions or use TreeCorr directly

Examples
--------
>>> # 1D simulation
>>> results = simulate_correlation_functions(
...     theory_cl_list=[theory_cl],
...     masks=[mask],
...     angular_bins=angular_bins,
...     method="pcl_estimator"
... )

>>> # nD simulation  
>>> results = simulate_correlation_functions(
...     theory_cl_list=[cl1, cl2, cl3],
...     masks=[mask1, mask2, mask3],
...     angular_bins=angular_bins,
...     method="pcl_estimator"
... )
"""

import os
import logging
import warnings
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
from numpy.random import default_rng
from pathlib import Path

# Optional imports with graceful degradation
try:
    import treecorr
    HAS_TREECORR = True
except ImportError:
    HAS_TREECORR = False
    

import glass.fields
import time

from .cl2xi_transforms import pcl2xi, prep_prefactors
from .noise_utils import get_noise_pixelsigma
from .pseudo_alm_cov import Cov
from .core_utils import check_property_equal

__all__ = [
    # Main API
    'simulate_correlation_functions',
    
    # Core utilities that users might need
    'create_maps',
    'add_noise_to_maps',
    'compute_pseudo_cl',
    'compute_correlation_functions',
    'get_noise_sigma',
    'limit_noise',
    
    # TreeCorr utilities
    'prep_cat_treecorr',
    'prep_angles_treecorr',
    'get_xi_treecorr',
    
    # Backward compatibility
    'TwoPointSimulation',

]

# Initialize logging
logger = logging.getLogger(__name__)


def create_maps(power_spectra, nside, lmax=None):
    """
    Create Gaussian random maps using GLASS for 1D or nD cases.
    
    Parameters
    ----------
    power_spectra : list of arrays
        List of C_l arrays for cross-correlations
        - For 1D: [cl_ee] (single power spectrum)
        - For nD: [cl_11, cl_22, cl_12, cl_33, cl_32, cl_31, ...]
        Only E-mode C_ells (T and B set to zero in GLASS)
    nside : int
        HEALPix resolution parameter
    lmax : int, optional
        Maximum multipole (not used by GLASS but kept for compatibility)
        
    Returns
    -------
    maps : ndarray
        - For 1D: shape (3, n_pix) - single T,Q,U map
        - For nD: shape (n_fields, 3, n_pix) - multiple T,Q,U maps
    """
    if len(power_spectra) == 0:
        npix = hp.nside2npix(nside)
        return np.zeros((3, npix))
    
    # Use GLASS generator correctly (your implementation is right!)
    fields = glass.fields.generate_gaussian(power_spectra, nside=nside)
    field_list = []
    while True:
        try:
            maps_TQU = next(fields)
            field_list.append(maps_TQU)
        except StopIteration:
            break
    
    maps_array = np.array(field_list)
    
    # For single field case, remove extra dimension for consistency
    if len(power_spectra) == 1:
        return maps_array[0]  # Shape: (3, n_pix)
    else:
        return maps_array     # Shape: (n_fields, 3, n_pix)


def get_noise_sigma(theory_cl, nside):
    """Extract noise sigma from theory C_l object."""
    if theory_cl.sigma_e is not None:
        if isinstance(theory_cl.sigma_e, str):
            return get_noise_pixelsigma(nside)
        elif isinstance(theory_cl.sigma_e, tuple):
            return get_noise_pixelsigma(nside, theory_cl.sigma_e)
        else:
            raise ValueError("sigma_e must be string for default or tuple (sigma_e, n_gal)")
    return None


def limit_noise(noisemap, nside, lmax=None):
    almq = hp.map2alm(noisemap)
    clq = hp.sphtfunc.alm2cl(almq)

    np.random.seed()
    return hp.sphtfunc.synfast(clq, nside, lmax=lmax)


def add_noise_to_maps(maps, nside, noise_sigmas=None, lmax=None):
    """
    Add shape noise to Q and U maps for 1D or nD cases.
    
    Parameters
    ----------
    maps : ndarray
        Maps from create_maps_glass
        - 1D case: shape (3, n_pix) 
        - nD case: shape (n_fields, 3, n_pix)
    nside : int
        HEALPix resolution parameter
    noise_sigmas : list or float, optional
        Noise sigma for each field
        - For 1D: single float or None
        - For nD: list of floats matching number of fields
    lmax : int, optional
        Maximum multipole for noise maps
        
    Returns
    -------
    noisy_maps : ndarray
        Maps with added noise (same shape as input)
    """
    if noise_sigmas is None:
        return maps
    
    # Handle 1D case - add extra dimension temporarily
    if maps.ndim == 2:  # Shape (3, n_pix)
        maps_nd = maps[None, ...]  # Shape (1, 3, n_pix)
        if not isinstance(noise_sigmas, list):
            noise_sigmas = [noise_sigmas]
        is_1d = True
    else:
        maps_nd = maps
        is_1d = False
    
    maps_noisy = maps_nd.copy()
    n_pix = maps_nd.shape[-1]
    
    for i, sigma in enumerate(noise_sigmas):
        if sigma is None:
            continue
            
        rng = default_rng()
        noise_q = limit_noise(rng.normal(size=n_pix, scale=sigma), nside, lmax)
        noise_u = limit_noise(rng.normal(size=n_pix, scale=sigma), nside, lmax)
        
        maps_noisy[i, 1] += noise_q  # Q map
        maps_noisy[i, 2] += noise_u  # U map
    
    # Return original shape
    if is_1d:
        return maps_noisy[0]  # Shape (3, n_pix)
    else:
        return maps_noisy     # Shape (n_fields, 3, n_pix)


def compute_pseudo_cl(maps_list, masks, fullsky=False, healpy_datapath=None):
    """
    Compute pseudo-C_l from masked maps.
    
    Parameters
    ----------
    maps_list : ndarray
        Maps of shape (n_fields, 3, n_pix)
    masks : ndarray
        Mask for each field, shape (n_fields, n_pix)
    fullsky : bool, optional
        If True, compute full-sky C_l, otherwise pseudo-C_l
        
    Returns
    -------
    pcl_array : ndarray
        Pseudo-C_l array of shape (n_cross, 3, n_ell)
        Order: 00, 11, 10, 22, 21, 20, ... (auto then cross)
        Contains [cl_e, cl_b, cl_eb] for each cross-correlation
    """
    anafast_kwargs = {
        'iter': 5,
        'use_pixel_weights': True
    }
    
    if healpy_datapath is not None:
        anafast_kwargs['datapath'] = healpy_datapath

    if fullsky:
        cl_list = []
        for i, field_i in enumerate(maps_list):
            for j, field_j in reversed(list(enumerate(maps_list[:i+1]))):
                # cl_t, cl_e, cl_b, cl_te, cl_eb, cl_tb order
                cl_results = hp.anafast(field_i, field_j)
                cl_list.append([cl_results[1], cl_results[2], cl_results[4]])  # E, B, EB
        return np.array(cl_list)
    else:
        pcl_list = []
        masked_fields = masks[:, None, :] * maps_list
        
        for i, field_i in enumerate(masked_fields):
            for j, field_j in reversed(list(enumerate(masked_fields[:i+1]))):
                try:
                    pcl_t, pcl_e, pcl_b, pcl_te, pcl_eb, pcl_tb = hp.anafast(
                        field_i,
                        field_j,
                        **anafast_kwargs
                    )

                except Exception as e:
                    logger.warning(f"Failed with pixel weights: {e}")
                    logger.warning("Retrying without pixel weights")
                    pcl_t, pcl_e, pcl_b, pcl_te, pcl_eb, pcl_tb = hp.anafast(
                        field_i,
                        field_j,
                        iter=5,
                        use_pixel_weights=False
                    )

                pcl_list.append([pcl_e, pcl_b, pcl_eb])  # (n_croco, 3, n_ell)

        return np.array(pcl_list)


def compute_correlation_functions(pcl_array, prefactors, lmax, lmin=0):
    """
    Convert pseudo-C_l to correlation functions.
    
    Parameters
    ----------
    pcl_array : ndarray
        Pseudo-C_l array from compute_pseudo_cl
    prefactors : ndarray
        Prefactors for C_l to xi transformation
    lmax : int
        Maximum multipole
    lmin : int
        Minimum multipole
        
    Returns
    -------
    xi_array : ndarray
        Correlation functions of shape (n_cross, 2, n_bins)
        [xi_plus, xi_minus] for each cross-correlation
    """
    n_cross = len(pcl_array)
    n_ang_bins = len(prefactors)
    xi_array = np.zeros((n_cross, 2, n_ang_bins))
    
    for i, pcl in enumerate(pcl_array):
        xi_array[i] = np.array(pcl2xi(pcl, prefactors, lmax, lmin=lmin))
    
    return xi_array


def simulate_correlation_functions(
    theory_cl_list,
    masks,
    angular_bins,
    job_id=0,
    n_batch=1,
    method="pcl_estimator",
    lmax=None,
    lmin=0,
    add_noise=True,
    save_path=None,
    save_pcl=False,
    plot_diagnostics=False,
    run_name="simulation"
):
    """
    Simulate correlation functions for 1D or nD cases.
    
    Parameters
    ----------
    theory_cl_list : list of TheoryC_l
        Theory power spectra objects
    masks : list of SphereMask
        Survey masks (same length as theory_cl_list)
    angular_bins : array or list
        Angular separation bins
    job_id : int
        Job identifier for batch processing
    n_batch : int
        Number of simulations in this batch
    method : str
        'pcl_estimator' or 'treecorr'
    lmax : int, optional
        Maximum multipole
    lmin : int
        Minimum multipole
    add_noise : bool
        Whether to add shape noise
    save_path : str, optional
        Path to save results
    save_pcl : bool
        Whether to save pseudo-C_l
    plot_diagnostics : bool
        Whether to create diagnostic plots
    run_name : str
        Name for this simulation run
        
    Returns
    -------
    results : dict
        Dictionary with simulation results
    """
    # Validate method availability
    if method == "treecorr" and not HAS_TREECORR:
        raise ImportError("TreeCorr not available. Install with: pip install treecorr")
    

    
    if not check_property_equal(
        masks, "nside"
    ) or not check_property_equal(masks, "eff_area"):
        raise RuntimeError("Different masks not implemented yet.")

    # Setup simulation parameters
    mask = masks[0]
    
    # Set lmax
    if lmax is None:
        lmax = mask.lmax
    
  
    # Auto-generate save path if not provided
    if save_path is None:
        sigma_name = theory_cl_list[0].sigmaname
        folder_name = f"croco_{run_name}_{mask.name}_{sigma_name}_{method}_llim_{lmax}"
        save_path = os.path.join("simulations", folder_name)
    
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Simulation folder: {save_path}")
    
    

    # Branch based on method
    if method == "pcl_estimator":
        return _simulate_pcl_estimator(
            theory_cl_list, mask, angular_bins, job_id, n_batch,
            lmax, lmin, add_noise, save_path, save_pcl, plot_diagnostics
        )
    elif method == "treecorr":
        return _simulate_treecorr(
            theory_cl_list, mask, angular_bins, job_id, n_batch,
            add_noise, save_path, plot_diagnostics
        )
    else:
        raise ValueError(f"Unknown method: {method}")



def _simulate_pcl_estimator(
    theory_cl_list, mask, angular_bins, job_id, n_batch,
    lmax, lmin, add_noise, save_path, save_pcl, plot_diagnostics
):
    """Pseudo_C_ell estimator simulation implementation."""
  
    nside = mask.nside
    
    logger.info(f"Simulating using pcl_estimator with nside={nside}, lmax={lmax}, lmin={lmin}")
    # Use smooth mask if available
    if hasattr(mask, "smooth_mask"):
        sim_mask = mask.smooth_mask
    else:
        logger.warning("Using unsmoothed mask for simulations")
        sim_mask = mask.mask
    # Prepare prefactors for C_l to xi transformation
    prefactors = prep_prefactors(angular_bins, mask.wl, mask.lmax, mask.lmax)
    
    # Extract power spectra for GLASS
    power_spectra = [cl.ee.copy() for cl in theory_cl_list]
    
    # Setup noise sigmas
    noise_sigmas = None
    if add_noise:
        noise_sigmas = [get_noise_sigma(cl, nside) for cl in theory_cl_list]
        if all(s is None for s in noise_sigmas):
            noise_sigmas = None
    
    # Run simulations
    xi_batch = []
    pcl_batch = []
    times = []
    
    for i in range(n_batch):
        tic = time.perf_counter()
        logger.info(f"Simulating batch {i+1}/{n_batch} ({(i+1)/n_batch*100:.1f}%)")
        
        # Create maps using GLASS
        maps = create_maps(power_spectra, nside, lmax)
        
        # Add noise if requested
        if add_noise and noise_sigmas is not None:
            maps = add_noise_to_maps(maps, nside, noise_sigmas, lmax)
        
        # Ensure maps are in nD format for consistent processing
        if maps.ndim == 2:  # 1D case: (3, n_pix) -> (1, 3, n_pix)
            maps = maps[None, ...]
            masks_array = np.array([sim_mask])
        else:  # nD case: already (n_fields, 3, n_pix)
            masks_array = np.array([sim_mask for _ in range(len(maps))])
        
        # Compute pseudo-C_l and correlation functions
        pcl_array = compute_pseudo_cl(maps, masks_array, fullsky=False)
        xi_array = compute_correlation_functions(pcl_array, prefactors, lmax, lmin)
        # (n_cross,2,n_ang_bins)
        xi_batch.append(xi_array)
        pcl_batch.append(pcl_array)
        
        toc = time.perf_counter()
        times.append(toc - tic)
    
    # Convert to arrays
    xi_batch = np.array(xi_batch)
    pcl_batch = np.array(pcl_batch)
    
    logger.info(f"Simulation times: {times}, Average: {np.mean(times):.2f}s")
    
    # Save and return results
    results = _save_and_return_results(
        xi_batch, pcl_batch, angular_bins, 'pcl_estimator', lmax, lmin, 
        n_batch, job_id, save_path, save_pcl, plot_diagnostics, prefactors
    )
    
    return results


def _save_and_return_results(
    xi_batch, pcl_batch, angular_bins, method, lmax, lmin, 
    n_batch, job_id, save_path, save_pcl, plot_diagnostics, prefactors
):
    """Save simulation results using existing file format for compatibility."""
    
    # Save in the EXISTING format that file_handling expects
    save_file = os.path.join(save_path, f"job{job_id:d}.npz")
    np.savez(
        save_file,
        mode=method,           
        theta=angular_bins,    
        lmin=lmin,
        lmax=lmax,
        xip=xi_batch[:, :, 0, :],  
        xim=xi_batch[:, :, 1, :], 
    )
    logger.info(f"Saved results to {save_file}")
    
    # Save pseudo-C_l if requested (keep existing format)
    if save_pcl and pcl_batch is not None:
        pcl_file = os.path.join(save_path, f"pcljob{job_id:d}.npz")
        np.savez(
            pcl_file,
            lmin=lmin,
            lmax=lmax,
            pcl_e=pcl_batch[:, :, 0],   # Keep existing key names
            pcl_b=pcl_batch[:, :, 1], 
            pcl_eb=pcl_batch[:, :, 2],
            prefactors=prefactors,
        )
        logger.info(f"Saved pseudo-C_l to {pcl_file}")
    
    # Create diagnostic plots
    if plot_diagnostics:
        _create_diagnostic_plots(xi_batch, pcl_batch, save_path, job_id)
    
    # Return results in a convenient dictionary format for immediate use
    # (but file format remains compatible)
    results = {
        'xi_plus': xi_batch[:, :, 0, :],
        'xi_minus': xi_batch[:, :, 1, :], 
        'theta': angular_bins,
        'method': method,
        'n_batch': n_batch,
        'job_id': job_id,
        'lmax': lmax,
        'lmin': lmin
    }
    
    return results

def _create_diagnostic_plots(xi_batch, pcl_batch, save_path, job_id):
    """Create diagnostic plots without hardcoded data loading."""
    # Correlation function histogram
    plt.figure(figsize=(8, 6))
    plt.hist(xi_batch[:, 0, 0, 0], bins=30, alpha=0.7)
    plt.xlabel('xi_plus value')
    plt.ylabel('Count')
    plt.title(f'Distribution of xi_plus (job {job_id})')
    plt.savefig(os.path.join(save_path, f"xi_hist_{job_id}.png"))
    plt.close()
    
    # Power spectrum plots (if available)
    if pcl_batch is not None:
        plt.figure(figsize=(10, 6))
        ell_range = np.arange(pcl_batch.shape[-1])
        mean_pcl = np.mean(pcl_batch, axis=0)
        plt.plot(ell_range, mean_pcl[0, 0, :], 'k-', label='E-mode', linewidth=2)
        plt.xlabel('Multipole l')
        plt.ylabel('C_l^EE')
        plt.title(f'Power spectrum (job {job_id})')
        plt.legend()
        plt.savefig(os.path.join(save_path, f"pcl_comparison_{job_id}.png"))
        plt.close()




def _simulate_treecorr(
    theory_cl_list, mask, angular_bins, job_id, n_batch,
    add_noise, save_path, plot_diagnostics
):
    """TreeCorr simulation implementation."""
    if not HAS_TREECORR:
        raise ImportError("TreeCorr not available")
    
    
    nside = mask.nside

    logger.info(f"Simulating using TreeCorr with nside={nside}, angular_bins={angular_bins}")

    # Use smooth mask if available
    if hasattr(mask, "smooth_mask"):
        sim_mask = mask.smooth_mask
    else:
        logger.warning("Using unsmoothed mask for simulations")
        sim_mask = mask.mask
    
    # Prepare catalog properties for TreeCorr
    cat_props = prep_cat_treecorr(nside, sim_mask)
    
    # Extract power spectra for GLASS (TreeCorr mainly for 1D)
    power_spectra = [theory_cl_list[0].ee.copy()]
    
    # Setup noise
    noise_sigma = get_noise_sigma(theory_cl_list[0], nside) if add_noise else None
    
    # Run simulations
    xi_batch = []
    
    for i in range(n_batch):
        logger.info(f"Simulating batch {i+1}/{n_batch} ({(i+1)/n_batch*100:.1f}%)")
        
        # Create maps using GLASS
        maps = create_maps(power_spectra, nside)
        
        # Add noise if requested
        if add_noise and noise_sigma is not None:
            maps = add_noise_to_maps(maps, nside, noise_sigma)
        
        # Compute correlation functions using TreeCorr
        xi_p, xi_m, theta = get_xi_treecorr(maps, angular_bins, cat_props)
        
        # Store results (reshape to match pcl format)
        xi_array = np.array([xi_p, xi_m]).T[None, ...]  # (1, 2, n_bins)
        xi_batch.append(xi_array)
    
    # Convert to arrays
    xi_batch = np.array(xi_batch)
    
    # Save and return results
    results = _save_and_return_results(
        xi_batch, None, theta, "treecorr", None, None,
        n_batch, job_id, save_path, False, plot_diagnostics, None
    )
    
    return results





def prep_cat_treecorr(nside, mask=None):
    """takes healpy mask, returns mask needed for treecorr simulation"""
    if mask is None:
        all_pix = np.arange(hp.nside2npix(nside))
        phi, thet = hp.pixelfunc.pix2ang(nside, all_pix, lonlat=True)
        treecorr_mask_cat = (None, phi, thet)
    else:
        all_pix = np.arange(len(mask))
        phi, thet = hp.pixelfunc.pix2ang(nside, all_pix, lonlat=True)
        treecorr_mask_cat = (mask, phi, thet)
    return treecorr_mask_cat


def prep_angles_treecorr(seps_in_deg):
    if type(seps_in_deg[0]) is tuple:
        bin_size = seps_in_deg[0][1] - seps_in_deg[0][0]
        thetamin = seps_in_deg[0][0]
        thetamax = seps_in_deg[-1][1]
    else:
        raise RuntimeError("Angular separations need to be bins for treecorr!")
    return thetamin, thetamax, bin_size


def get_xi_treecorr(maps_TQU, ang_bins_in_deg, cat_props):
    """takes maps and mask, returns treecorr correlation function"""
    """need to run cat and angles prep once before simulating"""
    mask, phi, thet = cat_props

    cat = treecorr.Catalog(
        ra=phi, dec=thet, g1=-maps_TQU[1], g2=maps_TQU[2], ra_units="deg", dec_units="deg", w=mask
    )

    xip, xim, angs = [], [], []
    for ang_bin in ang_bins_in_deg:
        thetamin, thetamax = ang_bin

        gg = treecorr.GGCorrelation(
            min_sep=thetamin,
            max_sep=thetamax,
            nbins=1,
            bin_type="Linear",
            sep_units="deg",
            bin_slop=0,
        )
        gg.process(cat)
        xip.append(gg.xip)
        xim.append(gg.xim)
        angs.append(gg.rnom)
    cat.clear_cache()
    return np.array(xip)[:, 0], np.array(xim)[:, 0], np.array(angs)


class TwoPointSimulation:
    """
    Two-point correlation function simulation class.
    
    .. deprecated:: 
        TwoPointSimulation is deprecated and will be removed in a future version.
        This class has been moved to papers/first_paper_method/analysis/simulate_1d.py
        and is maintained there for reproducibility of the first paper results.
        
        For new simulations, use the functional simulation API:
        - simulate_correlation_functions() for single redshift bins
        - simulate_correlation_functions_nd() for multiple redshift bins
        
    This class is limited to single redshift bin combinations and has been
    superseded by more flexible functional approaches.
    """
    
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "TwoPointSimulation is deprecated. Use the functional simulation API instead. "
            "This class will be removed in a future version. "
            "See papers/first_paper_method/analysis/ for the maintained version.",
            DeprecationWarning,
            stacklevel=2
        )

