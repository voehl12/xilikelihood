"""
Spherical mask properties and calculations for cosmological surveys.

This module provides the SphereMask class for handling survey masks on the sphere,
including coupling matrix calculations and spherical harmonic decompositions.
"""

import os
import time
import pickle
from pathlib import Path
from typing import List, Optional, Union, Tuple

import numpy as np
import healpy as hp

# Local imports
import wpm_funcs
import cov_funcs
import file_handling
from core_utils import computation_phase

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
        self._initialize_properties(spins, lmin, exact_lmax, l_smooth)
        
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
            print("Warning: exact lmax has been set to {:d}.".format(self._exact_lmax))
            
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

        if maskpath is not None:
            self.name = maskname
            self.maskpath = maskpath
            self.read_maskfile()

        elif circmaskattr is not None:
            if circmaskattr[0] == "fullsky":
                self.nside = circmaskattr[1]
                self.fullsky_mask()
                self.area = 41253

            else:
                self.area, self.nside = circmaskattr
                self.get_circmask()
       
        
        
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
        self.name += f"_smoothl{self.l_smooth}"
        self.wl
        self.wlm

    def _apply_smoothing(self) -> None:
        """Apply smoothing to the mask."""
        try:
            sigma = np.deg2rad(1 / self.l_smooth * 300)
            self.smooth_mask = hp.smoothing(
                self.mask,
                sigma=np.abs(sigma),
                iter=50,
                use_pixel_weights=True,
                datapath="/cluster/home/veoehl/2ptlikelihood/masterenv/lib/python3.8/site-packages/healpy/data/",
            )
        except Exception as e:
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
        self.maskpath = Path(f"circular_{area:.0f}sqd_nside{nside}.fits")
        
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
        self.maskpath = Path(f"fullsky_nside{nside}.fits")
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

    """     @property
    def smooth_alm(self):
        self._smooth_alm = wpm_funcs.smooth_alm(self.l_smooth, self._exact_lmax)
        return self._smooth_alm

    @property
    def smooth_alm_lmax(self):
        self._smooth_alm_lmax = wpm_funcs.smooth_alm(self.l_smooth, self.lmax)
        return self._smooth_alm_lmax
    """
    @property
    def wlm(self):
        """Calculates spherical harmonic coefficients of the mask

        Returns
        -------
        array
            spin 0 alm of mask in healpix ordering
        """

        if hasattr(self, "l_smooth"):
            self._wlm = hp.sphtfunc.map2alm(self.smooth_mask, lmax=self._exact_lmax)
        else:
            self._wlm = hp.sphtfunc.map2alm(self.mask, lmax=self._exact_lmax)
        return self._wlm

    @property
    def wlm_lmax(self):
        """Calculates spherical harmonic coefficients of the mask to bandlimit of mask

        Returns
        -------
        array
            spin 0 alm of mask in healpix ordering
        """

        if hasattr(self, "l_smooth"):
            self._wlm_lmax = hp.sphtfunc.map2alm(self.smooth_mask, lmax=self.nside // 2)
            if self._exact_lmax > self.nside // 2:
                ValueError(
                    "Exact lmax exceeds mask bandlimit after smoothing, requires careful reconsideration of mask calculations (mixing matrices need to be filled up with zeros)."
                )

            return self._wlm_lmax
        else:
            self._wlm_lmax = hp.sphtfunc.map2alm(self.mask, lmax=self.lmax)
            return self._wlm_lmax

    

    @property
    def wl(self):
        wl = np.zeros(self.lmax + 1)
        wlm = self.wlm_lmax
        wl_bandlimited = hp.sphtfunc.alm2cl(wlm)
        wl[: len(wl_bandlimited)] = wl_bandlimited
        self._wl = wl

        return self._wl

    @property
    def eff_area(self):
        if hasattr(self, "smooth_mask"):
            self._eff_area = hp.nside2pixarea(self.nside, degrees=True) * np.sum(self.smooth_mask)
            return self._eff_area
        else:
            print("Warning: attempting effective area of unsmoothed mask, returning actual area...")
            return self.area

    def initiate_w_arrs(self, cov_ell_buffer):

        buffer = cov_ell_buffer
        self.L = np.arange(self._exact_lmax + buffer + 1)
        print(
            "4D W_llpmmp will be calculated for lmax = {:d} + {:d}".format(self._exact_lmax, buffer)
        )
        self.M = np.arange(-self._exact_lmax - buffer, self._exact_lmax + buffer + 1)
        Nl = len(self.L)
        Nm = len(self.M)
        if self.spin0:
            self.w0_arr = np.zeros((Nl, Nm, Nl, Nm), dtype=complex)
        if self.spin2:
            self.wpm_arr = np.zeros((2, Nl, Nm, Nl, Nm), dtype=complex)

    def calc_w_element(
        self, L1, L2, M1, M2
    ):  # move this to wpm_funcs, since it does not attach anything to the object? but it heavily relies on wlm
        m = M1 - M2
        m1_ind = np.argmin(np.fabs(M1 - self.M))
        m2_ind = np.argmin(np.fabs(M2 - self.M))
        l1_ind = np.argmin(np.fabs(L1 - self.L))
        l2_ind = np.argmin(np.fabs(L2 - self.L))
        inds = (l1_ind, m1_ind, l2_ind, m2_ind)
        self.buffer_lmax = self._exact_lmax  # buffer in these sums does not change anything

        if self.spin0 is None:
            w0 = 0
        else:
            allowed_l, wigners0 = wpm_funcs.prepare_wigners(0, L1, L2, M1, M2, self.buffer_lmax)

            wlm_l = wpm_funcs.get_wlm_l(
                self._wlm_lmax, m, allowed_l
            )  # needs the lmax the wlm are calculated for.
            prefac = wpm_funcs.w_factor(allowed_l, L1, L2)
            w0 = (-1) ** np.abs(M1) * np.sum(wigners0 * prefac * wlm_l)

        if self.spin2 is None or np.logical_or(L1 < 2, L2 < 2):
            wp, wm = 0, 0
        else:
            allowed_l, wp_l, wm_l = wpm_funcs.prepare_wigners(2, L1, L2, M1, M2, self.buffer_lmax)

            prefac = wpm_funcs.w_factor(allowed_l, L1, L2)
            wlm_l = wpm_funcs.get_wlm_l(self._wlm_lmax, m, allowed_l)
            wlm_l_large = np.where(np.abs((wlm_l)) > 1e-17, wlm_l, 0)
            wp = 0.5 * (-1) ** np.abs(M1) * np.sum(prefac * wlm_l * wp_l)
            """ assert np.allclose(
                np.sum(prefac * wlm_l * wp_l).real - np.cumsum(prefac * wlm_l * wp_l).real[-10:],
                np.zeros(10),
            )
            assert np.allclose(
                np.sum(prefac * wlm_l * wp_l).imag - np.cumsum(prefac * wlm_l * wp_l).imag[-10:],
                np.zeros(10),
            ) """
            wm = 0.5 * 1j * (-1) ** np.abs(M1) * np.sum(prefac * wlm_l * wm_l)

        return (inds, w0, wp, wm)

    def save_w_element(self, result):
        inds, w0, wp, wm = result
        if w0:
            self.w0_arr[inds] = w0
        if wp or wm:
            inds = (slice(0, 2), *inds)
            self.wpm_arr[inds] = [wp, wm]

    def w_arr(self, cov_ell_buffer=0, verbose=True, path=None):
        if path is None:
            self.set_wpmpath(cov_ell_buffer)
        else:
            self.wpm_path = path
        if file_handling.check_for_file(self.wpm_path, kind="wpm"):
            self.load_w_arr()
            return self._w_arr
        if self.w0_arr is None and self.wpm_arr is None:
            self.initiate_w_arrs(cov_ell_buffer)

        arglist = []
        if verbose:
            print("Preparing list of l, m arguments...")
        for l1, L1 in enumerate(self.L):
            # TODO: implement pos_m stuff here, so m1 is only calculated for positive m. Check, whether positive m also suffice for m2.
            M1_arr = np.arange(-L1, L1 + 1)
            for l2, L2 in enumerate(self.L):
                M2_arr = np.arange(-L2, L2 + 1)

                for m1, M1 in enumerate(M1_arr):
                    for m2, M2 in enumerate(M2_arr):
                        arglist.append((L1, L2, M1, M2))

        self.wlm
        self.wlm_lmax
        print("Starting computation of 4D W_llpmmp arrays... ")
        for i, arg in enumerate(arglist):
            print(
                "Computing 4D W_llpmmp arrays......{:4.1f}%".format(i / len(arglist) * 100),
                end="\r",
            )
            result = self.calc_w_element(*arg)
            self.save_w_element(result)
            # pool.apply_async(self.calc_w_element, args=arg,callback=self.save_w_element)
            # pool.close()
            # pool.join()
        print()
        print("Finished.")
        if self.spin0 and self.spin2:
            self._w_arr = np.append(self.wpm_arr, self.w0_arr, axis=0)
            self.w0_arr = (
                None  # could also delete these attributes from the instance itself to make space
            )
            self.wpm_arr = None
            self.save_w_arr()
            return self._w_arr

        elif self.spin0:
            helper = np.empty_like(self.w0_arr)[None, :, :, :, :]
            self._w_arr = np.append(np.append(helper, helper, axis=0), self.w0_arr, axis=0)
            self.w0_arr = (
                None  # could also delete these attributes from the instance itself to make space
            )
            self.wpm_arr = None
            self.save_w_arr()
            return self._w_arr

        elif self.spin2:
            self._w_arr = self.wpm_arr
            self.w0_arr = None
            self.wpm_arr = None
            self.save_w_arr()
            return self._w_arr

    @property
    def m_llp(self):
        self.set_mllppath()
        if file_handling.check_for_file(self.mllp_path, kind="mllp"):
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
        with computation_phase("mask precomputation", log_memory=True):
            w_arr = self.w_arr(cov_ell_buffer=cov_ell_buffer)
            self._wpm_delta, self._wpm_stack = cov_funcs.precompute_xipm(w_arr)
            _ = self.m_llp  # Trigger computation
            self._precomputed = True

    def __repr__(self) -> str:
        """String representation of the mask."""
        status = "precomputed" if self._precomputed else "ready"
        smoothed = " (smoothed)" if hasattr(self, 'smooth_mask') else ""
        return (f"SphereMask(name='{self.name}', area={self.area:.1f} sq deg, "
                f"nside={self.nside}, exact_lmax={self._exact_lmax}, {status}){smoothed}")

    

    def save_w_arr(self):
        print("Saving Wpm0 arrays.")
        np.savez(self.wpm_path, wpm0=self._w_arr)

    def save_mllp_arr(self):
        np.savez(self.mllp_path, m_llp_p=self._m_llp[0], m_llp_m=self._m_llp[1])

    def load_w_arr(self):
        print("Loading Wpm0 arrays.")
        wpmfile = np.load(self.wpm_path)
        print("Loaded with size {} mb.".format(getsizeof(wpmfile["wpm0"]) / 1024**2))
        self._w_arr = wpmfile["wpm0"]

    def load_mllp_arr(self):
        print("Loading Mllp arrays.")
        mllpfile = np.load(self.mllp_path)
        m_llp_p, m_llp_m = mllpfile["m_llp_p"], mllpfile["m_llp_m"]
        self._m_llp = m_llp_p, m_llp_m

    def set_wpm_string(self, cov_ell_buffer):

        charstring = "_l{:d}_n{:d}_{}.npz".format(
            self._exact_lmax + cov_ell_buffer,
            self.nside,
            self.name,
        )
        return charstring

    def set_wpmpath(self, cov_ell_buffer):
        charac = self.set_wpm_string(cov_ell_buffer)
        # covname = "covariances/cov_xi" + charac
        if not os.path.isdir(self.working_dir + "/wpm_arrays"):
            command = "mkdir " + self.working_dir + "/wpm_arrays"
            os.system(command)
        wpm_name = self.working_dir + "/wpm_arrays/wpm" + charac
        self.wpm_path = wpm_name

    def set_mllppath(self):
        charstring = "_l{:d}_n{:d}_{}.npz".format(
            self._exact_lmax,
            self.nside,
            self.name,
        )

        if not os.path.isdir(self.working_dir + "/mllp_arrays"):
            command = "mkdir " + self.working_dir + "/mllp_arrays"
            os.system(command)
        mllp_name = self.working_dir + "/mllp_arrays/mllp" + charstring
        self.mllp_path = mllp_name


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