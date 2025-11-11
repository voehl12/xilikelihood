"""
Correlation function simulation utilities.

This module provides functions for simulating two-point correlation functions
from theoretical power spectra using either pseudo-C_l estimators or TreeCorr.
It requires the GLASS package for map generation, version >2025.1, requiring numpy>2.1

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
    

try:
    import glass
    import glass.fields
    import glass.lensing
    HAS_GLASS = True
except ImportError:
    HAS_GLASS = False
    _GLASS_ERROR_MSG = (
        "GLASS package is required for map simulations. "
        "Install with: pip install xilikelihood[simulate]\n"
        "Or: pip install glass"
    )
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
    'apply_mask_to_maps',
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


def _plot_field_distribution_sanity_check(kappa_list, gamma1_list, gamma2_list, field_type, n_fields):
    """
    Plot histograms of kappa and shear distributions for sanity checking.
    
    Parameters
    ----------
    kappa_list : list of arrays
        List of convergence (kappa) maps
    gamma1_list : list of arrays
        List of gamma1 (real shear component) maps
    gamma2_list : list of arrays
        List of gamma2 (imaginary shear component) maps
    field_type : str
        Type of field ("gaussian" or "lognormal")
    n_fields : int
        Number of fields generated
    """
    import scipy.stats as stats
    
    # Flatten all maps into single arrays
    kappa_all = kappa_list[0].ravel()
    gamma1_all = gamma1_list[0].ravel()
    gamma2_all = gamma2_list[0].ravel()
    
    # Create figure with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Field Distribution Sanity Check: {field_type.upper()} ({n_fields} fields)', 
                 fontsize=14, fontweight='bold')
    
    # Row 1: Histograms
    # Kappa histogram
    ax = axes[0, 0]
    counts, bins, _ = ax.hist(kappa_all, bins=100, density=True, alpha=0.7, 
                               color='blue', label='Data')
    ax.set_xlabel('κ (convergence)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Kappa Distribution')
    
    # Overlay theoretical distribution
    if field_type.lower() == "gaussian":
        # Gaussian: should be centered at 0
        mu, sigma = kappa_all.mean(), kappa_all.std()
        x = np.linspace(kappa_all.min(), kappa_all.max(), 200)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
                label=f'Gaussian(μ={mu:.2e}, σ={sigma:.2e})')
    else:
        # Lognormal: plot lognormal fit
        # Note: GLASS applies shift, so distribution might not be exactly lognormal
        mu, sigma = kappa_all.mean(), kappa_all.std()
        ax.axvline(mu, color='r', linestyle='--', linewidth=2, label=f'Mean={mu:.2e}')
        ax.text(0.05, 0.95, f'Skewness: {stats.skew(kappa_all):.3f}\nKurtosis: {stats.kurtosis(kappa_all):.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gamma1 histogram
    ax = axes[0, 1]
    ax.hist(gamma1_all, bins=100, density=True, alpha=0.7, color='green', label='Data')
    ax.set_xlabel('γ₁ (shear component 1)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Gamma1 Distribution')
    mu_g1, sigma_g1 = gamma1_all.mean(), gamma1_all.std()
    x = np.linspace(gamma1_all.min(), gamma1_all.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu_g1, sigma_g1), 'r-', linewidth=2,
            label=f'Gaussian(μ={mu_g1:.2e}, σ={sigma_g1:.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Gamma2 histogram
    ax = axes[0, 2]
    ax.hist(gamma2_all, bins=100, density=True, alpha=0.7, color='orange', label='Data')
    ax.set_xlabel('γ₂ (shear component 2)')
    ax.set_ylabel('Probability Density')
    ax.set_title('Gamma2 Distribution')
    mu_g2, sigma_g2 = gamma2_all.mean(), gamma2_all.std()
    x = np.linspace(gamma2_all.min(), gamma2_all.max(), 200)
    ax.plot(x, stats.norm.pdf(x, mu_g2, sigma_g2), 'r-', linewidth=2,
            label=f'Gaussian(μ={mu_g2:.2e}, σ={sigma_g2:.2e})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Row 2: Q-Q plots
    # Kappa Q-Q plot
    ax = axes[1, 0]
    if field_type.lower() == "gaussian":
        stats.probplot(kappa_all, dist="norm", plot=ax)
        ax.set_title('Kappa Q-Q Plot (Gaussian)')
    else:
        # For lognormal, show Q-Q against normal (should deviate)
        stats.probplot(kappa_all, dist="norm", plot=ax)
        ax.set_title('Kappa Q-Q Plot (vs Gaussian)')
    ax.grid(True, alpha=0.3)
    
    # Gamma1 Q-Q plot
    ax = axes[1, 1]
    stats.probplot(gamma1_all, dist="norm", plot=ax)
    ax.set_title('Gamma1 Q-Q Plot (Gaussian)')
    ax.grid(True, alpha=0.3)
    
    # Gamma2 Q-Q plot
    ax = axes[1, 2]
    stats.probplot(gamma2_all, dist="norm", plot=ax)
    ax.set_title('Gamma2 Q-Q Plot (Gaussian)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    filename = f'field_sanity_check_{field_type.lower()}_{n_fields}fields.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    logger.info(f"Saved sanity check plot: {filename}")
    
    # Print statistics
    logger.info(f"\n{'='*70}")
    logger.info(f"FIELD DISTRIBUTION STATISTICS: {field_type.upper()}")
    logger.info(f"{'='*70}")
    logger.info(f"Kappa (κ):")
    logger.info(f"  Mean:     {kappa_all.mean():.6e}")
    logger.info(f"  Std:      {kappa_all.std():.6e}")
    logger.info(f"  Min/Max:  {kappa_all.min():.6e} / {kappa_all.max():.6e}")
    logger.info(f"  Skewness: {stats.skew(kappa_all):.6f}")
    logger.info(f"  Kurtosis: {stats.kurtosis(kappa_all):.6f}")
    
    logger.info(f"\nGamma1 (γ₁):")
    logger.info(f"  Mean:     {gamma1_all.mean():.6e}")
    logger.info(f"  Std:      {gamma1_all.std():.6e}")
    logger.info(f"  Min/Max:  {gamma1_all.min():.6e} / {gamma1_all.max():.6e}")
    logger.info(f"  Skewness: {stats.skew(gamma1_all):.6f}")
    logger.info(f"  Kurtosis: {stats.kurtosis(gamma1_all):.6f}")
    
    logger.info(f"\nGamma2 (γ₂):")
    logger.info(f"  Mean:     {gamma2_all.mean():.6e}")
    logger.info(f"  Std:      {gamma2_all.std():.6e}")
    logger.info(f"  Min/Max:  {gamma2_all.min():.6e} / {gamma2_all.max():.6e}")
    logger.info(f"  Skewness: {stats.skew(gamma2_all):.6f}")
    logger.info(f"  Kurtosis: {stats.kurtosis(gamma2_all):.6f}")
    logger.info(f"{'='*70}\n")
    
    if field_type.lower() == "lognormal":
        logger.info("Note: For lognormal fields, kappa should show positive skewness.")
        logger.info("      Shear components (γ₁, γ₂) may still appear approximately Gaussian")
        logger.info("      due to the derivative operation in the shear conversion.")
    
    plt.show()


def create_maps(theory_cl_list, nside, field_type="gaussian", sanity_check=False):
    """
    Create random maps using GLASS for 1D or nD cases.
    
    Parameters
    ----------
    theory_cl_list : list of TheoryCl
        Theory power spectra objects containing power spectra and z_bins
    nside : int
        HEALPix resolution parameter
    lmax : int, optional
        Maximum multipole for kappa → shear conversion
    field_type : str, optional
        Type of random field to generate. Options:
        - "gaussian": Gaussian random fields (default)
        - "lognormal": Log-normal random fields
    sanity_check : bool, optional
        If True, plot histograms of kappa and shear maps to verify distributions.
        Default is False.
        
    Returns
    -------
    maps : ndarray
        - For 1D: shape (3, n_pix) - single T,Q,U map (T=0, Q=γ1, U=-γ2)
        - For nD: shape (n_fields, 3, n_pix) - multiple T,Q,U maps
        
    Notes
    -----
    This function generates convergence (kappa) maps using GLASS, then converts
    them to shear (γ1, γ2) using glass.fields.from_convergence(), and finally
    converts to polarization maps (T=0, Q=γ1, U=-γ2) for cosmic shear analysis.
    
    For lognormal fields, effective redshifts are computed from the z_bins in
    theory_cl_list to properly account for the nonlinear transformation.

    Raises
    ------
    ImportError
        If GLASS package is not installed.

    """
    if not HAS_GLASS:
        raise ImportError(_GLASS_ERROR_MSG)
    # Extract power spectra from theory_cl_list
    power_spectra = [cl.ee.copy() for cl in theory_cl_list]
    n_spectra = len(power_spectra)
    n_fields = glass.nfields_from_nspectra(n_spectra) 
        
    
    if len(power_spectra) == 0:
        npix = hp.nside2npix(nside)
        return np.zeros((3, npix))
    
    # Generate fields based on field type
    if field_type.lower() == "gaussian":
        fields = [glass.grf.Normal() for _ in range(n_fields)]
        gls = power_spectra
        
    elif field_type.lower() == "lognormal":
        # Power spectra are ordered by BinCombinationMapper: for n bins we have n(n+1)/2 spectra
        

        # Extract effective redshifts from the first n_bins theory_cl objects (the auto-correlations)
        # Auto-correlations are at indices 0, 1, 3, 6, ... (triangular numbers)
        z_effs = []
        auto_indices = [i*(i+1)//2 for i in range(n_fields)]  # 0, 1, 3, 6, 10, ...
        
        for idx in auto_indices:
            theory_cl = theory_cl_list[idx]
            if theory_cl.z_bins is None:
                raise ValueError(f"TheoryCl object must have z_bins for lognormal generation")
            # For auto-correlation, both bins are the same
            bin1, bin2 = theory_cl.z_bins
            z_effs.append(bin1.z_eff)
        
        logger.info(f"Using {n_fields} unique redshift bins for lognormal")
        logger.info(f"Effective redshifts: {z_effs}")
        
        # Apply lognormal transformation following GLASS methodology
        shift = glass.lognormal_shift_hilbert2011
        fields = [glass.grf.Lognormal(shift(z)) for z in z_effs]
        gls = glass.compute_gaussian_spectra(fields=fields, spectra=power_spectra)
        gls = np.array(gls)
        # Set first two elements of each row to zero (ℓ=0,1 monopole/dipole)
        gls[:, :2] = 0
    else:
        raise ValueError(f"Unknown field_type: {field_type}. Must be 'gaussian' or 'lognormal'.")

    samples = glass.generate(fields, gls, nside)
    
    field_list = []
    kappa_list = []  # Store kappa maps for sanity checks
    gamma1_list = []  # Store gamma1 maps for sanity checks
    gamma2_list = []  # Store gamma2 maps for sanity checks
    
    while True:
        try:
            kappa_map = next(samples)  # This is the convergence map
            
            if sanity_check:
                kappa_list.append(kappa_map.copy() if hasattr(kappa_map, 'copy') else np.array(kappa_map))
            
            # Convert kappa to shear using GLASS
            # Use lmax if provided, otherwise let GLASS use its default
            # from_convergence returns a tuple when shear=True: (shear, ...)
            result = glass.lensing.from_convergence(kappa_map, shear=True, lmax=None)
            
            # Extract shear from tuple (first element)
            gamma = result[0] if isinstance(result, tuple) else result
            
            # Flatten to 1D if needed (in case of shape (1, n_pix))
            if gamma.ndim > 1:
                gamma = gamma.ravel()
            
            gamma1 = np.real(gamma)
            gamma2 = np.imag(gamma)
            
            if sanity_check:
                gamma1_list.append(gamma1.copy())
                gamma2_list.append(gamma2.copy())
            
            # Convert shear components (γ1, γ2) to polarization (Q, U)
            
            # This matches healpy.synfast output and ensures proper E/B mode separation
            T_map = np.zeros_like(gamma1)  # No temperature for pure shear
            Q_map = gamma1
            U_map = gamma2
            
            # Stack into T,Q,U format expected by the rest of the pipeline
            maps_TQU = np.array([T_map, Q_map, U_map])
            field_list.append(maps_TQU)
        except StopIteration:
            break
    
    # Sanity check: plot histograms of kappa and shear maps
    if sanity_check and len(kappa_list) > 0:
        _plot_field_distribution_sanity_check(
            kappa_list, gamma1_list, gamma2_list, field_type, n_fields
        )
    
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
        Maps from create_maps
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


def apply_mask_to_maps(maps, mask):
    """
    Apply mask(s) to maps in pixel space with proper broadcasting.
    
    Parameters
    ----------
    maps : ndarray
        Maps to mask
        - 1D case: shape (3, n_pix) - single T,Q,U map
        - nD case: shape (n_fields, 3, n_pix) - multiple T,Q,U maps
    mask : ndarray
        Mask array. Values should be 0 (masked) or 1 (unmasked),
        or any weight between 0 and 1. Can be:
        - shape (n_pix,): Single mask applied to all fields
        - shape (n_fields, n_pix): One mask per field (for nD maps only)
        
    Returns
    -------
    masked_maps : ndarray
        Maps with mask applied (same shape as input)
        
    Examples
    --------
    >>> # Single field with single mask
    >>> maps = create_maps(theory_cl_list, nside)  # shape (3, n_pix)
    >>> masked = apply_mask_to_maps(maps, mask.mask)  # mask shape (n_pix,)
    
    >>> # Multiple fields with per-field masks
    >>> maps = create_maps(theory_cl_list, nside)  # shape (n_fields, 3, n_pix)
    >>> masks = np.array([mask1, mask2, ...])  # shape (n_fields, n_pix)
    >>> masked = apply_mask_to_maps(maps, masks)
    """
    if maps.ndim == 2:  # 1D case: (3, n_pix)
        # Broadcast mask across T, Q, U
        return maps * mask[None, :]
    else:  # nD case: (n_fields, 3, n_pix)
        # Handle both single mask and per-field masks via broadcasting
        # mask (n_pix,) broadcasts to (n_fields, 3, n_pix)
        # mask (n_fields, n_pix) broadcasts to (n_fields, 3, n_pix)
        return maps * mask[:, None, :] if mask.ndim == 2 else maps * mask[None, None, :]


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
        # Apply masks using the reusable function
        masked_fields = apply_mask_to_maps(maps_list, masks)
        
        pcl_list = []
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
    run_name="simulation",
    field_type="gaussian"
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
    field_type : str, optional
        Type of random field to generate ("gaussian" or "lognormal")
        
    Returns
    -------
    results : dict
        Dictionary with simulation results

    Raises
    ------
    ImportError
        If GLASS package is not installed.

    """
    if not HAS_GLASS:
        raise ImportError(_GLASS_ERROR_MSG)
    
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
        folder_name = f"croco_{run_name}_{mask.name}_{sigma_name}_{method}_{field_type}_llim_{lmax}"
        save_path = os.path.join("simulations", folder_name)
    
    os.makedirs(save_path, exist_ok=True)
    logger.info(f"Simulation folder: {save_path}")
    
    

    # Branch based on method
    if method == "pcl_estimator":
        return _simulate_pcl_estimator(
            theory_cl_list, mask, angular_bins, job_id, n_batch,
            lmax, lmin, add_noise, save_path, save_pcl, plot_diagnostics,field_type
        )
    elif method == "treecorr":
        return _simulate_treecorr(
            theory_cl_list, mask, angular_bins, job_id, n_batch,
            add_noise, save_path, plot_diagnostics, field_type
        )
    else:
        raise ValueError(f"Unknown method: {method}")



def _simulate_pcl_estimator(
    theory_cl_list, mask, angular_bins, job_id, n_batch,
    lmax, lmin, add_noise, save_path, save_pcl, plot_diagnostics, field_type
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
        maps = create_maps(theory_cl_list, nside, field_type=field_type)
        
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
        n_batch, job_id, save_path, save_pcl, plot_diagnostics, prefactors, field_type
    )
    
    return results


def _save_and_return_results(
    xi_batch, pcl_batch, angular_bins, method, lmax, lmin, 
    n_batch, job_id, save_path, save_pcl, plot_diagnostics, prefactors, field_type
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
        field_type=field_type  # Document field type in saved data
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
        'lmin': lmin,
        'field_type': field_type
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
    add_noise, save_path, plot_diagnostics, field_type
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
    
    # Setup noise
    noise_sigma = get_noise_sigma(theory_cl_list[0], nside) if add_noise else None
    
    # Run simulations
    xi_batch = []
    
    for i in range(n_batch):
        logger.info(f"Simulating batch {i+1}/{n_batch} ({(i+1)/n_batch*100:.1f}%)")
        
        # Create maps using GLASS (TreeCorr mainly for 1D)
        maps = create_maps(theory_cl_list, nside, field_type=field_type)
        
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
        n_batch, job_id, save_path, False, plot_diagnostics, None, field_type
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

