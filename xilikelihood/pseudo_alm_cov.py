"""
Pseudo-alm covariance computation for masked spherical analyses.

This module provides the Cov class for computing covariances of pseudo
spherical harmonic coefficients (pseudo-alm) needed for likelihood analyses
of cosmological correlation functions on incomplete sky coverage.

The covariance calculation accounts for mode coupling induced by survey masks
and includes both signal and noise contributions.
"""

import numpy as np
import os.path
from sys import getsizeof
import time
import logging
from . import cov_funcs
from . import noise_utils
from .file_handling import generate_filename, save_arrays, check_for_file, load_arrays
from .core_utils import check_property_equal

# Set up module logger
logger = logging.getLogger(__name__)

__all__ = ['Cov']

DEFAULT_COV_ELL_BUFFER = 10

class Cov:
    """
    Calculate and store pseudo-alm covariances for masked spherical Gaussian random fields.
    
    This class computes the covariance matrix of pseudo spherical harmonic coefficients
    needed for likelihood analyses of correlation functions (xi+/-) or power spectra
    when working with incomplete sky coverage.
    
    Parameters:
    -----------
    mask : SphereMask
        Mask object containing survey geometry and precomputed coupling matrices
    theorycl : TheoryCl  
        Theory power spectra object including signal and noise
    exact_lmax : int
        Maximum multipole for exact calculations
    lmax : int, optional
        Maximum multipole for mask (defaults to 3*nside-1)
    cov_ell_buffer : int, default=10
        Buffer in multipole space to ensure convergence
    working_dir : str, optional
        Working directory for output files (defaults to current directory)
        
    Attributes:
    -----------
    cov_alm : ndarray
        2D covariance matrix of pseudo-alm coefficients
    p_ee, p_bb, p_eb : ndarray
        Pseudo power spectra (E-mode, B-mode, cross-correlation)
    p_tt : ndarray
        Pseudo temperature power spectrum (if spin-0 modes included)
        
    Methods:
    --------
    cov_alm_xi(ischain=False)
        Compute covariance matrix for xi+/- correlation functions
    cov_alm_general(alm_kinds, pos_m=True, lmin=0, ischain=False)
        General covariance computation for specified alm modes
    cl2pseudocl(ischain=False) 
        Convert theory power spectra to pseudo power spectra
    
    Notes:
    ------
    - Supports both full-sky and masked analyses
    - Automatically handles caching and loading of precomputed covariances
    - Memory usage is reported during computation for large matrices
    - Uses JAX-optimized functions from cov_funcs module when available
    """

    def __init__(
        self,
        mask,
        theorycl,
        exact_lmax,
        lmax=None,
        cov_ell_buffer=DEFAULT_COV_ELL_BUFFER,
        working_dir=None,
        ischain=True,
    ):

        # Validate inputs
        if exact_lmax <= 0:
            raise ValueError("exact_lmax must be positive")
        if cov_ell_buffer < 0:
            raise ValueError("cov_ell_buffer must be non-negative")
            
        self.working_dir = working_dir or os.getcwd()
        self.mask = mask
        self.theorycl = theorycl
        self._exact_lmax = exact_lmax
        self.ischain = ischain
        
        # Validate consistency between mask and covariance lmax
        if not check_property_equal([self, self.mask], "_exact_lmax"):
            raise RuntimeError(
                "exact_lmax mismatch between Cov class and mask. "
                f"Cov: {exact_lmax}, Mask: {getattr(mask, '_exact_lmax', 'undefined')}"
            )

        self._lmax = lmax if lmax is not None else 3 * self.mask.nside - 1
        self._cov_ell_buffer = cov_ell_buffer
        # Set up file paths
        if not self.ischain:
            self.covalm_path = generate_filename("cov_xi", {
                "lmax":self._exact_lmax,
                "nside":self.mask.nside,
                "mask":self.mask.name,
                "theory":self.theorycl.name,
                "sigma":self.theorycl.sigmaname,},
                base_dir=self.working_dir
            )

    @property
    def exact_lmax(self):
        """Maximum multipole for exact calculations."""
        return self._exact_lmax

    def cell_cube(self, lmax):
        c_all = np.zeros((3, 3, lmax + 1))
        c_all[0, 0] = self.theorycl.ee.copy()[: lmax + 1]
        c_all[0, 2] = self.theorycl.ne.copy()[: lmax + 1]
        c_all[2, 0] = self.theorycl.ne.copy()[: lmax + 1]
        c_all[2, 2] = self.theorycl.nn.copy()[: lmax + 1]
        return c_all

    def cov_alm_xi(self, ischain=True):
        """
        Calculates covariance of pseudo-alm needed for the spin-2 correlation function xi+/-.

        Parameters
        ----------
        exact_lmax : int, optional
            maximum multipole moment ell to be used, None defaults to class instance exact_lmax set earlier, by default None
        pos_m : bool, optional
            only calculate covariances of pseudo-alm with positive m, by default True

        Returns
        -------
        2D array
            Covariance matrix of pseudo-alm: E, and B-modes, real and imaginary parts, sorted by ell, then m (see (cov_alm_general))
        """

        alm_kinds = [
            "ReE",
            "ImE",
            "ReB",
            "ImB",
        ]
        self.cov_alm = self.cov_alm_general(alm_kinds, pos_m=True, ischain=ischain)
        return self.cov_alm

    def cov_alm_general(self, alm_kinds, pos_m=True, lmin=0, ischain=True, check_conditioning=False):
        """
        Calculate covariance matrix for specified pseudo-alm modes.

        This method computes covariances of pseudo spherical harmonic coefficients
        for given mode combinations. The computation accounts for mode coupling
        from survey masks and supports both full-sky and masked analyses.
        

        Parameters:
        -----------
        alm_kinds : list of str
            Pseudo-alm modes to include (e.g., ['ReE', 'ImE', 'ReB', 'ImB'])
        pos_m : bool, default=True
            If True, only compute positive m contributions (sufficient for xi+)
        lmin : int, default=0
            Minimum multipole to include
        ischain : bool, default=False
            If True, skip file I/O operations
        check_conditioning : bool, default=False
            If True, check matrix conditioning (eigenvalues, condition number)

        Returns:
        --------
        ndarray
            Covariance matrix for specified alm modes
            
        Raises:
        -------
        ValueError
            If alm_kinds contains invalid mode strings
        NotImplementedError
            If full-sky calculation with pos_m=False is requested

        Notes:
        ------
        - Matrix ordering: first by E/B and Re/Im, then by l, then by m
        - For xi+ correlation functions, only positive m contributions needed
        - Supports both full-sky and masked methods
        - Always a covariance of pseudo alms - order and number depends on two point statistics considered
        - Could implement limitation of m to current l, would make n_cov smaller overall
        
        """

        if not alm_kinds:
            raise ValueError("alm_kinds cannot be empty")
        if lmin < 0:
            raise ValueError("lmin must be non-negative")
        if lmin > self._exact_lmax:
            raise ValueError(f"lmin ({lmin}) cannot exceed exact_lmax ({self._exact_lmax})")
    
        logger.info(f"Computing covariance for alm modes: {alm_kinds}")
        logger.info(f"Parameters: pos_m={pos_m}, lmin={lmin}, exact_lmax={self._exact_lmax}")


        # Set up theory power spectra with buffer
        buffer = self._cov_ell_buffer
        theory_cell = self.cell_cube(self._exact_lmax + buffer)
        
        # Try to load from cache if not in chain mode
        if not ischain and self.check_cov():
            logger.info("Loading covariance from cache")
            self.load_cov()
            return self.cov_alm

        
        alm_inds = cov_funcs.match_alm_inds(alm_kinds)
        n_alm = len(alm_inds)
        logger.debug(f"Processing {n_alm} alm modes: indices {alm_inds}")

        # Add noise contributions if available
        if hasattr(self.theorycl, "_noise_sigma") and self.theorycl._noise_sigma is not None:
            logger.debug("Adding noise contributions to theory power spectra")
            theory_cell += noise_utils.noise_cl_cube(
                self.theorycl.noise_cl[:self._exact_lmax + 1 + buffer]
            )
            

        # Calculate matrix dimensions
        if pos_m:
            n_cov = n_alm * (self._exact_lmax - lmin + 1) * (self._exact_lmax + 1)
        else:
            n_cov = n_alm * (self._exact_lmax - lmin + 1) * (2 * self._exact_lmax + 1)

        logger.info(f"Covariance matrix dimensions: {n_cov} x {n_cov}")
        matrix_size_mb = (n_cov * n_cov * 8) / 1024**2  # 8 bytes per float64

        logger.info(f"Estimated memory usage: {matrix_size_mb:.1f} MB")
    
        if matrix_size_mb > 1000:  # Warn if > 1GB
            logger.warning(f"Large matrix detected: {matrix_size_mb:.1f} MB")
    
        # Choose computation method based on mask type
        if self.mask.name == "fullsky":
            if not pos_m:
                raise NotImplementedError(
                    "Full-sky covariance matrix only implemented for positive m"
                )
            logger.info("Computing full-sky covariance (analytical method)")
            cov_matrix = self.cov_fullsky(alm_inds, n_cov, theory_cell)
        else:
            logger.info("Computing masked-sky covariance (numerical method)")
            cov_matrix = self.cov_masked(alm_inds, n_cov, theory_cell, lmin=lmin, pos_m=pos_m)
            

        cov_matrix = np.where(np.isnan(cov_matrix), cov_matrix.T, cov_matrix)
        if not np.allclose(cov_matrix, cov_matrix.T, rtol=1e-10, atol=1e-12):
            logger.warning("Covariance matrix is not symmetric - checking tolerance")
            max_asymmetry = np.max(np.abs(cov_matrix - cov_matrix.T))
            logger.warning(f"Maximum asymmetry: {max_asymmetry:.2e}")
            if max_asymmetry > 1e-8:
                raise RuntimeError("Covariance matrix severely non-symmetric")
            else:
                logger.info("Asymmetry within acceptable tolerance - symmetrizing")
                cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)

        # Check matrix conditioning (debug mode only)
        if check_conditioning:
            try:
                eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                min_eigenval = np.min(eigenvals)
                condition_number = np.max(eigenvals) / max(min_eigenval, 1e-16)
                neg_eig_mask = eigenvals < 0
                num_neg = np.sum(neg_eig_mask)
                if num_neg > 0:
                    logger.warning(f"Setting {num_neg} eigenvalues < 0 to zero out of {len(eigenvals)} total.")
                    eigenvals[neg_eig_mask] = 0.0
                    cov_matrix = (eigenvecs @ np.diag(eigenvals) @ eigenvecs.T)
                    cov_matrix = 0.5 * (cov_matrix + cov_matrix.T)
                if min_eigenval <= 0:
                    if abs(min_eigenval) < 1e-14:
                        logger.debug(f"Small negative eigenvalue (numerical precision): min eigenvalue = {min_eigenval:.2e}")
                    else:
                        logger.warning(f"Non-positive definite matrix: min eigenvalue = {min_eigenval:.2e}")
                if condition_number > 1e12:
                    logger.warning(f"Ill-conditioned matrix: condition number = {condition_number:.2e}")
                logger.debug(f"Matrix condition number: {condition_number:.2e}")
            except Exception as e:
                logger.warning(f"Could not compute matrix condition: {e}") 

        self.cov_alm = cov_matrix
        # Save to cache if not in chain mode
        if not ischain:
            logger.info("Saving covariance to cache")
            self.save_cov()
            
        logger.info("Covariance computation completed successfully")
        return self.cov_alm

    def cov_fullsky(self, alm_inds, n_cov, theory_cell,lmin=0):
        """
        Compute covariance matrix for full-sky spherical harmonic coefficients.
        
        For full-sky analysis (no mask), the covariance matrix is diagonal
        because different (ℓ,m) modes are uncorrelated. This provides a much
        faster analytical computation compared to masked-sky methods.
        
        Parameters:
        -----------
        alm_inds : list of int
            Numerical indices for alm modes from match_alm_inds()
        n_cov : int
            Side length of covariance matrix
        theory_cell : ndarray
            Theory C_ℓ cube [mode_i, mode_j, ℓ]
        lmin : int, default=0
            Minimum multipole to include
            
        Returns:
        --------
        ndarray
            Diagonal covariance matrix for pseudo-alm coefficients
            
        Notes:
        -----
        For full-sky analysis:
        - Covariance is diagonal: Cov(a_ℓm, a_ℓ'm') = C_ℓ δ_ℓℓ' δ_mm'
        - Real parts have factor 2 for m≥0 (since a_ℓ,-m = (-1)^m a*_ℓm)
        - Imaginary parts of m=0 modes vanish (a_ℓ0 is real)
        - Factor 0.5 base normalization with mode-dependent corrections
        
        This implementation corrects the factor calculation bug in the original 
        new implementation where real parts of m>0 modes had incorrect factors.
        """
        logger.info("Computing full-sky covariance matrix")
        logger.debug(f"Matrix size: {n_cov}×{n_cov}, modes: {len(alm_inds)}")

        if lmin != 0:
            logger.warning("lmin≠0 not fully tested for full-sky case")
        cov_matrix = np.zeros((n_cov, n_cov))
        diagonal_elements = np.zeros(n_cov)
        for mode_idx, alm_mode in enumerate(alm_inds):
            logger.debug(f"Processing alm mode {alm_mode} ({mode_idx+1}/{len(alm_inds)})")
            
            # Map alm mode to theory C_ℓ index (E/B modes share same C_ℓ)
            theory_idx = alm_mode // 2  # 0→0, 1→0, 2→1, 3→1 (ReE,ImE→EE, ReB,ImB→BB)
            is_real_part = (alm_mode % 2 == 0)  # Even indices are real parts
            # Build covariance for this mode across all (ℓ,m)
            mode_covariances = self._build_fullsky_mode_covariance(
                theory_idx, theory_cell, is_real_part, lmin
            )

            # Insert into full diagonal
            len_mode = len(mode_covariances)
            start_idx = len_mode * mode_idx
            end_idx = start_idx + len_mode
            
            diagonal_elements[start_idx:end_idx] = mode_covariances
        if len(diagonal_elements) != n_cov:
            raise RuntimeError(f"Diagonal length mismatch: {len(diagonal_elements)} ≠ {n_cov}")
        
        logger.debug("Converting to diagonal matrix")
        cov_matrix = np.diag(diagonal_elements)
        
        logger.info("Full-sky covariance computation completed")
        return cov_matrix
        
    
    def _build_fullsky_mode_covariance(self, theory_idx, theory_cell, is_real_part, lmin):
        """
        Build covariance diagonal for a single alm mode (E or B, real or imaginary).
        
        This is an elegant vectorized implementation inspired by the original
        cov_fullsky_old approach, avoiding explicit loops over ℓ and m.
        
        Parameters:
        -----------
        theory_idx : int
            Index into theory_cell (0 for E-modes, 1 for B-modes)
        theory_cell : ndarray
            Theory C_ℓ cube
        is_real_part : bool
            True for real parts, False for imaginary parts
        lmin : int
            Minimum multipole
            
        Returns:
        --------
        ndarray
            Covariance diagonal elements for this mode
            
        Notes:
        ------
        Implements the standard full-sky spherical harmonic covariance:
        - Cov(a_ℓm, a_ℓ'm') = C_ℓ δ_ℓℓ' δ_mm' (diagonal covariance)
        - Base factor 0.5 for all modes
        - Real parts of m=0: additional factor 2 → total factor 1.0
        - Imaginary parts of m=0: set to 0 (a_ℓ0 is real)
        - Imaginary parts of m>0: no additional factor → total factor 0.5
        
        This vectorized approach matches the original cov_fullsky_old logic:
        1. Create ranges of C_ℓ values repeated for each m (0 to ℓ)
        2. Pad to uniform length and flatten into a single array
        3. Apply mode-dependent factors using array indexing
        """
        max_ell = self._exact_lmax
        len_sub = max_ell + 1
        
        # Extract C_ℓ values for this theory mode (only from lmin onwards)
        c_ell_values = theory_cell[theory_idx, theory_idx, lmin:max_ell + 1]
        
        # For each ℓ, repeat C_ℓ for each m (from 0 to ℓ)
        # This creates the triangular structure: [C_0, C_1 C_1, C_2 C_2 C_2, ...]
        cell_ranges = [np.repeat(c_ell_values[ell_idx], lmin + ell_idx + 1) 
                      for ell_idx in range(len(c_ell_values))]
        
        # Pad each range to uniform length (len_sub) and stack into array
        full_ranges = [
            np.append(cell_ranges[i], np.zeros(len_sub - len(cell_ranges[i])))
            for i in range(len(cell_ranges))
        ]
        
        # Flatten into 1D array and apply base factor 0.5
        cov_part = 0.5 * np.ndarray.flatten(np.array(full_ranges))
        
        # Apply mode-specific factors using elegant array indexing
        if is_real_part:
            # Real parts: multiply m=0 elements (every len_sub-th element) by 2
            # This gives total factor 1.0 for both m=0 and m>0 real parts
            cov_part[::len_sub] *= 2
        else:
            # Imaginary parts: set m=0 elements (every len_sub-th element) to 0
            # m>0 elements keep the base factor 0.5
            cov_part[::len_sub] *= 0
            
        return cov_part

    def cov_masked(self, alm_inds, n_cov, theory_cell, lmin=0, pos_m=True):
        """
        Sets up covariance matrix for pseudo-alms. So far only works for spin-0 if spin-2 are calculated as well.

        Parameters
        ----------
        alm_inds : list of int
            integers according to alm kinds from cov_funcs.match_alm_inds()
        n_cov : int
            side length of covariance matrix
        theory_cell : 3D array
            cube of theory c_ell. first & second dimension: pairings of modes, third dimension: ell (see self.cell_cube())
        lmin : int
            minimum ell considered , by default 0
        pos_m : bool, optional
            only calculate contributions for positive m (seems to be sufficient for plus correlation function, couldn't show analytically why yet), by default True


        Returns
        -------
        2D array
            Covariance matrix of pseudo-alm
        """
        tic = time.perf_counter()
        buffer = self._cov_ell_buffer

        # Get W-array (coupling matrices)
        if self.mask._precomputed:
            logger.debug("Using precomputed W-arrays")
            w_arr = self.mask._w_arr
            precomputed = self.mask._wpm_delta, self.mask._wpm_stack
        else:
            logger.debug("Computing W-arrays on-the-fly")
            w_arr = self.mask.w_arr(cov_ell_buffer=buffer)
            precomputed = None

        cov_matrix = np.full((n_cov, n_cov), np.nan)
        memory_mb = getsizeof(cov_matrix) / 1024**2
        logger.info(f"Initialized covariance matrix: {memory_mb:.1f} MB")
        # Calculate number of unique matrix blocks
        num_alm = len(alm_inds)
        total_blocks = (num_alm * (num_alm + 1)) // 2  # Upper triangular
        logger.info(f"Computing {total_blocks} matrix blocks")

        # Main computation loop
        block_counter = 0
        for i, alm_i in enumerate(alm_inds):
            for j, alm_j in enumerate(alm_inds[i:], start=i):  # Only upper triangular
                block_counter += 1
            
                # Progress logging every 10% or for small numbers every block
                if total_blocks <= 10 or block_counter % max(1, total_blocks // 10) == 0:
                    progress = (block_counter / total_blocks) * 100
                    logger.info(f"Progress: {progress:.1f}% (block {block_counter}/{total_blocks})")

                 # Compute covariance block
                if self.mask._precomputed:
                    cov_part = cov_funcs.optimized_cov_4D_jit(
                        alm_i, alm_j, precomputed, self._exact_lmax + buffer, lmin, theory_cell
                    )
                else:
                    cov_part = cov_funcs.cov_4D_jit(
                        alm_i, alm_j, w_arr, self._exact_lmax + buffer, lmin, theory_cell
                    )    

                # Reshape to 2D and insert into matrix
                if pos_m:
                    len_2D = (cov_part.shape[0] - buffer) * (cov_part.shape[1] - buffer)
                    cov_2D = np.reshape(
                        cov_part[:-buffer, :-buffer, :-buffer, :-buffer], 
                        (len_2D, len_2D)
                    )
                else:
                    len_2D = (cov_part.shape[0] - buffer) * (cov_part.shape[1] - 2 * buffer)
                    cov_2D = np.reshape(
                        cov_part[:-buffer, buffer:-buffer, :-buffer, buffer:-buffer],
                        (len_2D, len_2D)
                    )
            
                # Insert block into full matrix
                pos_i = (len_2D * i, len_2D * (i + 1))
                pos_j = (len_2D * j, len_2D * (j + 1))
                cov_matrix[pos_i[0]:pos_i[1], pos_j[0]:pos_j[1]] = cov_2D
                
                # Insert symmetric block if off-diagonal
                if i != j:
                    cov_matrix[pos_j[0]:pos_j[1], pos_i[0]:pos_i[1]] = cov_2D.T
                
                # Clean up memory
                del cov_2D, cov_part   
                    
                
        toc = time.perf_counter()
        logger.info(f"Masked covariance computation completed in {toc-tic:.2f} seconds")
        return cov_matrix

    def save_cov(self):
        """Save covariance matrix using centralized file handling."""
        save_arrays(data={"cov": self.cov_alm}, filepath=self.covalm_path)

    def check_cov(self):
        """Check if covariance file exists."""
        return check_for_file(self.covalm_path)

    def load_cov(self):
        """Load covariance matrix using centralized file handling."""
        cov_dict = load_arrays(self.covalm_path, "cov")
        self.cov_alm = cov_dict['cov']

    def _get_pseudo_cl_path(self):
        """Generate file path for pseudo-Cl cache."""
        return generate_filename(
            "pcl",
            {
                "nside": self.mask.nside,
                "mask": self.mask.name,
                "theory": self.theorycl.name,
                "sigma": self.theorycl.sigmaname,
            },
            base_dir=self.working_dir
        )
    
    def cl2pseudocl(self, ischain=False):
        """
        Convert theory power spectra to pseudo power spectra.
        
        Computes pseudo-Cl from theory Cl using mask coupling matrices.
        Includes noise contributions if available in theory object.
        
        Parameters:
        -----------
        ischain : bool, default=False
            If True, skip file I/O operations (for MCMC chains)
            
        Notes:
        ------
        Pseudo-Cl account for mode coupling: P_ell = M_ell,ell' * C_ell'
        where M is the mask-induced coupling matrix.

        For weak lensing cosmology:
        - B-modes are expected to be zero from theory (cl_bb ≈ 0)
        - Cross-correlations EB and BE are also negligible
        - This is why cl_eb = cl_be = cl_b is appropriate

        References:
        -----------
        Based on NaMaster scientific documentation paper.
        """
        # Handle file I/O for caching (skip if running in chain)
        if not ischain:
            pcl_path = self._get_pseudo_cl_path()
            
            # Try to load from cache first
            try:
                cached_pcl = load_arrays(pcl_path)
                if cached_pcl is not None:
                    self.p_ee = cached_pcl["pcl_ee"]
                    self.p_bb = cached_pcl["pcl_bb"] 
                    self.p_eb = cached_pcl["pcl_eb"]
                    if "pcl_tt" in cached_pcl:
                        self.p_tt = cached_pcl["pcl_tt"]
                    logger.info("Loaded pseudo-Cl from cache")
                    return
            except Exception as e:
                logger.debug(f"Could not load pseudo-Cl from cache: {e}")
    
        logger.info("Computing pseudo-Cl from theory power spectra")
        # Prepare E and B mode power spectra with noise if available
        cl_ee, cl_bb = self._prepare_power_spectra()
        # For weak lensing: B-modes and EB cross-correlations are negligible
        # B-modes arise only from systematics, noise, or non-cosmological sources
        cl_eb = cl_be = cl_bb  # Appropriate for weak lensing where C_EB ≈ C_BB ≈ 0

    
        # Compute pseudo power spectra using mask coupling matrices
        self._compute_pseudo_spectra(cl_ee, cl_bb, cl_eb, cl_be)
        
        # Save results if not in chain mode
        if not ischain:
            save_dict = {
                "pcl_ee": self.p_ee,
                "pcl_bb": self.p_bb, 
                "pcl_eb": self.p_eb
            }
            if hasattr(self, 'p_tt'):
                save_dict["pcl_tt"] = self.p_tt
            
            
            save_arrays(save_dict, pcl_path)
        logger.info("Pseudo-Cl computation completed")
        
  
    def _prepare_power_spectra(self):
        """Prepare E and B mode power spectra with noise if available."""
        if hasattr(self.theorycl, "_noise_sigma") and self.theorycl._noise_sigma is not None:
            logger.debug("Adding noise contributions to power spectra")
            cl_ee = self.theorycl.ee.copy() + self.theorycl.noise_cl
            cl_bb = self.theorycl.bb.copy() + self.theorycl.noise_cl
        else:
            logger.debug("Using theory power spectra without noise")
            cl_ee = self.theorycl.ee.copy()
            cl_bb = self.theorycl.bb.copy()
        
        return cl_ee, cl_bb
    
    def _compute_pseudo_spectra(self, cl_ee, cl_bb, cl_eb, cl_be):
        """Compute pseudo power spectra using mask coupling matrices."""
        # Validate that coupling matrices are available
        if not hasattr(self.mask, 'm_llp'):
            raise RuntimeError("Mask coupling matrices (m_llp) not available")
        
        if self.mask.spin0:
            # Include temperature (spin-0) modes
            m_llp_plus, m_llp_minus, m_llp_zero = self.mask.m_llp
            
            self.p_ee = (np.einsum("lm,m->l", m_llp_plus, cl_ee) + 
                        np.einsum("lm,m->l", m_llp_minus, cl_bb))
            self.p_bb = (np.einsum("lm,m->l", m_llp_minus, cl_ee) + 
                        np.einsum("lm,m->l", m_llp_plus, cl_bb))
            self.p_eb = (np.einsum("lm,m->l", m_llp_plus, cl_eb) - 
                        np.einsum("lm,m->l", m_llp_minus, cl_be))
            self.p_tt = np.einsum("lm,m->l", m_llp_zero, cl_ee)
            
            logger.debug(f"Computed pseudo-Cl with temperature: shapes EE={self.p_ee.shape}, "
                        f"BB={self.p_bb.shape}, EB={self.p_eb.shape}, TT={self.p_tt.shape}")
        else:
            # Only polarization (spin-2) modes
            m_llp_plus, m_llp_minus = self.mask.m_llp
            
            self.p_ee = (np.einsum("lm,m->l", m_llp_plus, cl_ee) + 
                        np.einsum("lm,m->l", m_llp_minus, cl_bb))
            self.p_bb = (np.einsum("lm,m->l", m_llp_minus, cl_ee) + 
                        np.einsum("lm,m->l", m_llp_plus, cl_bb))
            self.p_eb = (np.einsum("lm,m->l", m_llp_plus, cl_eb) - 
                        np.einsum("lm,m->l", m_llp_minus, cl_be))
            
            logger.debug(f"Computed pseudo-Cl (polarization only): shapes EE={self.p_ee.shape}, "
                        f"BB={self.p_bb.shape}, EB={self.p_eb.shape}")
            

    def set_noise_pixelsigma(self):
        """DEPRECATED: Noise handling moved to simulation modules."""
        import warnings
        warnings.warn(
            "set_noise_pixelsigma is deprecated. Noise handling has been "
            "moved to simulation modules and should not be needed here.",
            DeprecationWarning,
            stacklevel=2
        )
        if self.theorycl._sigma_e is not None:
            if isinstance(self.theorycl._sigma_e, str):
                self.pixelsigma = noise_utils.get_noise_pixelsigma(self.mask.nside)

            elif isinstance(self.theorycl._sigma_e, tuple):
                self.pixelsigma = noise_utils.get_noise_pixelsigma(
                    self.mask.nside, self.theorycl._sigma_e
                )
            else:
                raise RuntimeError(
                    "sigma_e needs to be string for default or tuple (sigma_e,n_gal)"
                )

        else:
            try:

                del self.pixelsigma
            except:
                pass