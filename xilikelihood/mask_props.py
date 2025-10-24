"""
Spherical mask properties and calculations for cosmological surveys.

This module provides the SphereMask class for handling survey masks on the sphere,
including coupling matrix calculations and spherical harmonic decompositions.
"""

import os
import time
import pickle
import logging
from pathlib import Path
from typing import List, Optional, Union, Tuple

import numpy as np
import healpy as hp

# Local imports
from . import wpm_funcs
from . import cov_funcs
from .file_handling import save_arrays, load_arrays, generate_filename, check_for_file, ensure_directory_exists
from .core_utils import computation_phase

logger = logging.getLogger(__name__)

__all__ = ['SphereMask', 'save_maskobject']



class SphereMask:
    """
    Survey mask properties and calculations for spherical cosmological surveys.
    
    Handles mask loading, smoothing, spherical harmonic decomposition, and 
    coupling matrix calculations for correlation function covariance estimation.
    
    Parameters
    ----------
    spins : list of int, default=[0, 2]
        Spin fields to consider (0=scalar, 2=tensor/shear)
    maskpath : str, optional
        Path to FITS file containing the mask
    circmaskattr : tuple, optional
        (area_in_sqd, nside) for circular mask, or ('fullsky', nside)
    lmin : int, default=0
        Minimum multipole for calculations
    exact_lmax : int, optional
        Maximum multipole for exact calculations. Defaults to 3*nside-1
    maskname : str, default="mask"
        Name identifier for saving/loading cached arrays
    l_smooth : int or 'auto', optional
        Smoothing scale parameter. If 'auto', uses exact_lmax
    working_dir : str, optional
        Directory for caching arrays. Defaults to current directory
        
    Attributes
    ----------
    mask : ndarray
        HEALPix mask map
    nside : int
        HEALPix resolution parameter
    npix : int
        Number of pixels in the map
    area : float
        Survey area in square degrees
    eff_area : float
        Effective area after smoothing
    lmax : int
        Bandlimit of the mask (3*nside-1)
    exact_lmax : int
        Maximum multipole for exact calculations
        
    Examples
    --------
    >>> # Create circular mask
    >>> mask = SphereMask(circmaskattr=(1000, 256))  # 1000 sq deg at nside=256
    >>> 
    >>> # Load mask from file
    >>> mask = SphereMask(maskpath="survey_mask.fits", exact_lmax=30)
    >>> 
    >>> # Precompute arrays for covariance calculations
    >>> mask.precompute_for_cov_masked(cov_ell_buffer=10)
    """

    def __init__(
        self,
        spins: List[int] = [0, 2],
        maskpath: Optional[str] = None,
        circmaskattr: Optional[Tuple] = None,
        lmin: int = 0,
        exact_lmax: Optional[int] = None,
        maskname: str = "mask",
        l_smooth: Optional[Union[int, str]] = None,
        working_dir: Optional[str] = None,
    ) -> None:
        
        if maskpath is None and circmaskattr is None:
            raise ValueError("Must specify either maskpath or circmaskattr")
        
        if not any(spin in [0, 2] for spin in spins):
            raise ValueError("spins must contain 0 and/or 2")

        if working_dir is None:
            working_dir = os.getcwd()
        self.working_dir = Path(working_dir)
        
        # Initialize mask
        if maskpath is not None:
            self.name = maskname
            self.maskpath = Path(maskpath)
            self._read_maskfile()

        elif circmaskattr is not None:
            self._create_mask_from_attributes(circmaskattr)
        
        # Set up mask properties
        self._initialize_properties(spins, lmin, exact_lmax)
        
        # Smoothing if requested
        if l_smooth is not None:
            self._setup_smoothing(l_smooth)

    def _initialize_properties(self, spins, lmin, exact_lmax):
        """Initialize mask properties and validate parameters."""
        self._precomputed = False
        self.npix = hp.nside2npix(self.nside)
        self.spins = spins
        self.lmin = lmin
        self.lmax = 3 * self.nside - 1
        
        # Set exact_lmax with validation
        if exact_lmax is not None:
            if exact_lmax > self.lmax:
                raise ValueError(f"exact_lmax ({exact_lmax}) cannot exceed mask bandlimit ({self.lmax})")
            self._exact_lmax = exact_lmax
        else:
            self._exact_lmax = self.lmax
            logger.warning(f"exact_lmax not specified, defaulting to mask bandlimit: {self._exact_lmax}")
        # Initialize spin flags
        self.spin0 = 0 in spins
        self.spin2 = 2 in spins
        self.n_field = sum([1 if self.spin0 else 0, 2 if self.spin2 else 0])
        
        # Initialize arrays to None
        self._reset_arrays()

    def _reset_arrays(self):
        """Reset all computed arrays to None."""
        self.L = None
        self.M = None
        self.w0_arr = None
        self.wpm_arr = None
        self._w_arr = None

        # Reset cached spherical harmonic properties
        if hasattr(self, '_wlm'):
            delattr(self, '_wlm')
        if hasattr(self, '_wlm_lmax'):
            delattr(self, '_wlm_lmax')
        if hasattr(self, '_wl'):
            delattr(self, '_wl')
        if hasattr(self, '_eff_area'):
            delattr(self, '_eff_area')
        # Add this line:
        if hasattr(self, '_m_llp'):
            delattr(self, '_m_llp')

        
        
    def _setup_smoothing(self, l_smooth: Union[int, str]) -> None:
        """Set up mask smoothing parameters."""
        if l_smooth == "auto":
            self.l_smooth_auto = True
            self.l_smooth = self._exact_lmax
        else:
            if not isinstance(l_smooth, int) or l_smooth <= 0:
                raise ValueError("l_smooth must be positive integer or 'auto'")
            self.l_smooth_auto = False
            self.l_smooth = l_smooth
        
        self._apply_smoothing()
        self.name += f"smoothl{self.l_smooth}"
        # Reset all cached arrays since mask has changed
        self._reset_arrays()

    def _apply_smoothing(self) -> None:
        """Apply smoothing to the mask."""
        logger.info(f"Applying smoothing with l_smooth={self.l_smooth}")
        try:
            sigma = np.deg2rad(1 / self.l_smooth * 300)
            self.smooth_mask = hp.smoothing(
                self.mask,
                sigma=np.abs(sigma),
                iter=50,
                use_pixel_weights=True,
            )
            logger.info("Mask smoothing completed successfully")
        except Exception as e:
            logger.error(f"Mask smoothing failed: {e}")
            raise RuntimeError(f"Mask smoothing failed: {e}")
        

          

    def _read_maskfile(self) -> None:
        """Read mask from FITS file and set properties."""
        if not self.maskpath.exists():
            raise FileNotFoundError(f"Mask file not found: {self.maskpath}")
        
        try:
            self.mask = hp.read_map(str(self.maskpath))
            self.nside = hp.get_nside(self.mask)
            self.area = hp.nside2pixarea(self.nside, degrees=True) * np.sum(self.mask)
        except Exception as e:
            raise RuntimeError(f"Failed to read mask file {self.maskpath}: {e}")

    def _create_mask_from_attributes(self, circmaskattr: Tuple) -> None:
        """Create mask from circular or fullsky attributes."""
        if len(circmaskattr) != 2:
            raise ValueError("circmaskattr must be (area_in_sqd, nside) or ('fullsky', nside)")
        
        area_or_type, nside = circmaskattr
        
        if area_or_type == "fullsky":
            self._create_fullsky_mask(nside)
        else:
            self._create_circular_mask(area_or_type, nside)
    

    def _create_circular_mask(self, area: float, nside: int) -> None:
        """Create circular mask with specified area."""
        if area <= 0:
            raise ValueError("Area must be positive")
        if nside <= 0 or not isinstance(nside, int):
            raise ValueError("nside must be positive integer")
            
        self.area = area
        self.nside = nside
        self.name = f"circ{area:.0f}"
        
        # Create masks directory and set path
        masks_dir = Path("masks")
        ensure_directory_exists(str(masks_dir))
        self.maskpath = masks_dir / f"circular_{area:.0f}sqd_nside{nside}.fits"
        
        if self.maskpath.exists():
            self.mask = hp.read_map(str(self.maskpath))
            # Validate area matches
            actual_area = hp.nside2pixarea(nside, degrees=True) * np.sum(self.mask)
            if not np.isclose(area, actual_area, rtol=0.1):
                raise ValueError(f"Existing mask area {actual_area:.1f} doesn't match requested {area:.1f}")
        else:
            self._generate_circular_mask()

    
    def _generate_circular_mask(self) -> None:
        """Generate and save circular mask."""
        npix = hp.nside2npix(self.nside)
        mask = np.zeros(npix)
        vec = hp.ang2vec(np.pi / 2, 0)  # North pole
        radius = np.sqrt(self.area / np.pi)
        disc = hp.query_disc(nside=self.nside, vec=vec, radius=np.radians(radius))
        mask[disc] = 1
        self.mask = mask
        hp.write_map(str(self.maskpath), mask, overwrite=True)


    def _create_fullsky_mask(self, nside: int) -> None:
        """Create full-sky mask."""
        if nside <= 0 or not isinstance(nside, int):
            raise ValueError("nside must be positive integer")
            
        self.nside = nside
        npix = hp.nside2npix(nside)
        self.mask = np.ones(npix)
        
        # Create masks directory and set path
        masks_dir = Path("masks")
        ensure_directory_exists(str(masks_dir))
        self.maskpath = masks_dir / f"fullsky_nside{nside}.fits"
        
        self.name = "fullsky"
        self.area = hp.nside2pixarea(nside, degrees=True) * npix
        hp.write_map(str(self.maskpath), self.mask, overwrite=True)

    
    @property
    def is_precomputed(self) -> bool:
        """Check if mask arrays are precomputed for covariance calculations."""
        return self._precomputed

    @property
    def exact_lmax(self):
        return self._exact_lmax

    
    @property
    def wlm(self):
        """Spherical harmonic coefficients of the mask to exact_lmax."""
        if not hasattr(self, '_wlm'):
            if hasattr(self, "l_smooth"):
                self._wlm = hp.sphtfunc.map2alm(self.smooth_mask, lmax=self._exact_lmax)
            else:
                self._wlm = hp.sphtfunc.map2alm(self.mask, lmax=self._exact_lmax)
        return self._wlm

    @property
    def wlm_lmax(self):
        """Spherical harmonic coefficients of the mask to mask bandlimit."""
        if not hasattr(self, '_wlm_lmax'):
            if hasattr(self, "l_smooth"):
                bandlimit = self.nside // 2
                if self._exact_lmax > bandlimit:
                    raise ValueError(
                        f"exact_lmax ({self._exact_lmax}) exceeds smoothed mask bandlimit ({bandlimit}). "
                        "Consider reducing exact_lmax or filling mixing matrices with zeros."
                    )
                self._wlm_lmax = hp.sphtfunc.map2alm(self.smooth_mask, lmax=bandlimit)
            else:
                self._wlm_lmax = hp.sphtfunc.map2alm(self.mask, lmax=self.lmax)
        return self._wlm_lmax

    

    @property
    def wl(self):
        """Power spectrum of the mask."""
        if not hasattr(self, '_wl'):
            wl = np.zeros(self.lmax + 1)
            wlm = self.wlm_lmax
            wl_bandlimited = hp.sphtfunc.alm2cl(wlm)
            wl[:len(wl_bandlimited)] = wl_bandlimited
            self._wl = wl
        return self._wl

    @property
    def eff_area(self):
        """Effective area of the mask."""
        if not hasattr(self, '_eff_area'):
            if hasattr(self, "smooth_mask"):
                self._eff_area = hp.nside2pixarea(self.nside, degrees=True) * np.sum(self.smooth_mask)
            else:
                logger.warning("Computing effective area of unsmoothed mask, returning actual area")
                self._eff_area = self.area
        return self._eff_area
      

    def w_arr(self, cov_ell_buffer=0, path=None):
        """Compute or load W coupling arrays."""
        logger.info(f"Computing W arrays with buffer={cov_ell_buffer}")
        
        if path is None:
            self.set_wpmpath(cov_ell_buffer)
        else:
            self.wpm_path = path
        
        if check_for_file(self.wpm_path, kind="wpm"):
            logger.info("Loading existing W arrays from disk")
            self.load_w_arr()
            return self._w_arr
        


        # Compute arrays
        with computation_phase("W array computation", log_memory=True):
            logger.info("Starting W array computation...")
            
                        
            # Compute using wpm_funcs
            w0_arr, wpm_arr = wpm_funcs.compute_w_arrays(
                self.wlm_lmax, 
                self._exact_lmax, 
                cov_ell_buffer,
                spin0=self.spin0,
                spin2=self.spin2,
            )

            # Assemble final array
            self._w_arr = wpm_funcs.assemble_w_array(w0_arr, wpm_arr, self.spin0, self.spin2)
            
            # Clean up intermediate arrays
            del w0_arr, wpm_arr
            
            # Save to disk
            logger.info("Saving W arrays to disk")
            self.save_w_arr()
            
        logger.info("W array computation completed successfully")
        return self._w_arr

    @property
    def m_llp(self):
        """M_ll' coupling arrays."""
        if hasattr(self, '_m_llp'):
            return self._m_llp
            
        self.set_mllppath()
        if check_for_file(self.mllp_path, kind="mllp"):
            self.load_mllp_arr()
        else:
            self._m_llp = wpm_funcs.m_llp(self.wl, self.lmax, spin0=self.spin0)
            self.save_mllp_arr()
        return self._m_llp

    def precompute_for_cov_masked(self, cov_ell_buffer: int = 0) -> None:
        """
        Precompute all arrays needed for masked covariance calculations.
        
        Parameters
        ----------
        cov_ell_buffer : int, default=0
            Buffer in multipole space for covariance calculations
        """
        logger.info("Starting mask precomputation")
        with computation_phase("mask precomputation", log_memory=True):
            w_arr = self.w_arr(cov_ell_buffer=cov_ell_buffer)
            # Precompute xi arrays
            logger.info("Precomputing xi coupling matrices")
            self._wpm_delta, self._wpm_stack = cov_funcs.precompute_xipm(w_arr)
            # Compute M_ll' arrays
            logger.info("Computing M_ll' arrays")
            _ = self.m_llp  # Trigger computation
            self._precomputed = True

    def __repr__(self) -> str:
        """String representation of the mask."""
        status = "precomputed" if self._precomputed else "ready"
        smoothed = " (smoothed)" if hasattr(self, 'smooth_mask') else ""
        return (f"SphereMask(name='{self.name}', area={self.area:.1f} sq deg, "
                f"nside={self.nside}, exact_lmax={self._exact_lmax}, {status}){smoothed}")

    @property
    def status(self) -> str:
        """Current computation status of the mask."""
        if not hasattr(self, '_w_arr'):
            return "initialized"
        elif not hasattr(self, '_wpm_delta'):
            return "w_arrays_computed"
        elif self._precomputed:
            return "fully_precomputed"
        else:
            return "partially_computed"

    def save_w_arr(self):
        save_arrays(data={"wpm0": self._w_arr}, filepath=self.wpm_path)
        

    def save_mllp_arr(self):
        """Save M_ll' arrays to disk."""
        save_arrays(data={"m_llp_p": self._m_llp[0], "m_llp_m": self._m_llp[1]}, filepath=self.mllp_path)

    def load_w_arr(self):
        """Load WPM arrays from disk."""
        wpmfile = load_arrays(self.wpm_path, keys=["wpm0"])
        self._w_arr = wpmfile["wpm0"]

    def load_mllp_arr(self):
        """Load M_ll' arrays from disk."""
        mllpfile = load_arrays(self.mllp_path, keys=["m_llp_p", "m_llp_m"])
        self._m_llp = mllpfile["m_llp_p"], mllpfile["m_llp_m"]



    def set_wpmpath(self, cov_ell_buffer):
        """Set path for WPM arrays."""
        self.wpm_path = generate_filename('wpm', {
            "lmax": self._exact_lmax+cov_ell_buffer,
            "nside": self.nside,
            "mask": self.name,
        }, base_dir=self.working_dir)
        

    def set_mllppath(self):
        """Set path for MLLP arrays."""
        self.mllp_path = generate_filename(
            "mllp", {
                "lmax": self._exact_lmax,
                "nside": self.nside,
                "mask": self.name
            }, base_dir=self.working_dir)
        


def save_maskobject(maskobject: SphereMask, directory: str = "") -> str:
    """
    Save SphereMask object to pickle file.
    
    Parameters
    ----------
    maskobject : SphereMask
        Mask object to save
    directory : str, optional
        Directory to save in
        
    Returns
    -------
    str
        Path to saved file
    """
    filename = f"{maskobject.name}_l{maskobject.lmax}_n{maskobject.nside}.pkl"
    filepath = Path(directory) / filename
    
    with open(filepath, "wb") as f:
        pickle.dump(maskobject, f)
    
    return str(filepath)