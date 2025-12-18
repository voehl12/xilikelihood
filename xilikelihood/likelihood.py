"""
Likelihood computation for 2-point correlation function analyses.

This module provides the XiLikelihood class for computing likelihoods of
cosmological parameters from shear correlation function measurements using
characteristic function methods.

Key features:
- Exact low-ℓ likelihood via characteristic functions    def _compute_variances(self, auto_prods, cross_prods, cross_combs):
        logger.info("Computing variances...")
        auto_transposes = np.transpose(auto_prods, (0, 1, 3, 2))
        
        # Handle doubled data vector when xi_minus is included
        n_correlation_types = 2 if self.include_ximinus else 1
        variances = np.zeros((self._n_redshift_bin_combs, n_correlation_types * len(self.ang_bins_in_deg)))

        # Auto terms
        variances = variances.at[~self._is_cov_cross].set(
            2 * np.sum(auto_prods * auto_transposes, axis=(-2, -1))
        )

        # Cross terms
        cross_transposes = np.transpose(cross_prods, (0, 1, 3, 2))
        auto_normal, auto_transposed = cross_combs[:, 0], cross_combs[:, 1]
        variances[self._is_cov_cross] = np.sum(
            cross_prods * cross_transposes, axis=(-2, -1)
        ) + np.sum(
            auto_prods[auto_normal] * auto_transposes[auto_transposed], axis=(-2, -1)
        )

        return variancespproximation for computational efficiency  
- Copula-based joint PDF construction
- Support for tomographic redshift bins
- JAX-optimized computations for performance

Classes:
--------
XiLikelihood : Main likelihood computation class

Functions:
----------
fiducial_dataspace : Generate standard KiDS-like analysis setup
"""

import os
import re
import logging
import gc
import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from scipy.stats import multivariate_normal

# Graceful JAX import
try:
    import jax
    import jax.numpy as jnp
    # Configure JAX to use double precision by default
    jax.config.update("jax_enable_x64", False)
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    


# Internal imports
from .theory_cl import prepare_theory_cl_inputs, generate_theory_cl, RedshiftBin, load_kids_redshift_bins
from .pseudo_alm_cov import Cov
from .distributions import (
    mean_xi_gaussian_nD,
    batched_cf_1d_jitted, 
    high_ell_gaussian_cf_1d,
    gaussian_pdf_1d,
    cf_to_pdf_1d,
    cov_xi_gaussian_nD,
    gaussian_2d,
)
from .cl2xi_transforms import prep_prefactors, cl2pseudocl
from .diagnostic_tools import plot_eigenvalue_distributions, analyze_significant_eigenvalues
from . import copula_funcs as cop
from .core_utils import LikelihoodConfig, temporary_arrays, computation_phase, check_property_equal

logger = logging.getLogger(__name__)

__all__ = [
    'XiLikelihood',
    'fiducial_dataspace',
]


class XiLikelihood:
    """
    Likelihood computation for cosmological shear correlation functions.
    
    Computes exact likelihoods using characteristic function methods for low-ℓ
    modes and Gaussian approximations for high-ℓ modes. Supports tomographic
    analyses with multiple redshift bins.
    
    Parameters:
    -----------
    mask : Mask object
        Survey mask defining sky coverage and properties
    redshift_bins : list of RedshiftBin
        Tomographic redshift bin definitions
    ang_bins_in_deg : list of tuples
        Angular bin edges in degrees [(min1, max1), (min2, max2), ...]
    config : LikelihoodConfig, optional
        Configuration object with computation settings
    exact_lmax : int, optional
        Maximum ℓ for exact CF computation (default: mask.exact_lmax)
    lmax : int, optional  
        Maximum ℓ for analysis (default: 3*nside-1)
    noise : str or dict, optional
        Noise model specification (default: 'default')
    **kwargs
        Additional parameters to override config settings
        
    Examples:
    ---------
    Standard likelihood evaluation:
    >>> redshift_bins, ang_bins = fiducial_dataspace()
    >>> config = LikelihoodConfig(cf_steps=8192, ximax_sigma_factor=50.0)
    >>> likelihood = XiLikelihood(mask, redshift_bins, ang_bins, config=config)
    >>> likelihood.setup_likelihood()
    >>> logL = likelihood.loglikelihood(observed_data, test_cosmology)  # ✅ Recommended
    
    2D visualization:
    
    >>> data_subset = [0, 5]  # First redshift combo, bin 0 and bin 5
    >>> xs_2d, pdf_2d = likelihood.likelihood_function_2d(cosmology, data_subset)  # ✅ Good
    
    Full joint PDF (only for low dimensions):
    
    >>> # Only do this for 2-3 dimensional data!
    >>> if likelihood_data_dim <= 4:
    ...     xs, pdf = likelihood.likelihood_function(cosmology)  # ⚠️ Use with caution
    """
   

    def __init__(
        self,
        mask,
        redshift_bins: List[RedshiftBin],
        ang_bins_in_deg: List[Tuple[float, float]],
        config: Optional[LikelihoodConfig] = None,
        exact_lmax: Optional[int] = None,
        lmax: Optional[int] = None,
        noise: str = 'default',
        include_ximinus: bool = False,
        **kwargs
    ):
        # Input validation
        if not hasattr(mask, 'nside'):
            raise TypeError("mask must have 'nside' attribute")
        
        if not redshift_bins:
            raise ValueError("redshift_bins cannot be empty")
            
            
        # Validate angular bins are in ascending order
        for i, (min_ang, max_ang) in enumerate(ang_bins_in_deg):
            if min_ang >= max_ang:
                raise ValueError(f"Angular bin {i}: min_ang ({min_ang}) >= max_ang ({max_ang})")
        
        # Configuration setup
        self.config = config or LikelihoodConfig()
        
        # Override config with any kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Override config.{key} = {value}")
            else:
                # Handle legacy parameters
                setattr(self, key, value)
                logger.debug(f"Set legacy parameter {key} = {value}")
        
        # Set working directory
        self.working_dir = self.config.working_dir or os.getcwd()

        # Core attributes
        self.mask = mask
        self.lmax = lmax if lmax is not None else 3 * self.mask.nside - 1
        self.noise = noise
        self.include_ximinus = include_ximinus
        # Set n_correlation_types as an attribute
        self.n_correlation_types = 2 if self.include_ximinus else 1
        # Angular bins
        self.ang_bins_in_deg = ang_bins_in_deg
        self._n_ang_bins = len(ang_bins_in_deg)
        self._is_large_angle = np.array([ang[0] >= self.config.large_angle_threshold for ang in ang_bins_in_deg])
        self.n_large_angles = int(np.sum(self._is_large_angle))
        logger.info("Initialized XiLikelihood with %d large angles out of %d total angular bins, large angles at indices %s.",
                    self.n_large_angles, self._n_ang_bins, np.where(self._is_large_angle)[0])
        self.n_small_angles = self._n_ang_bins - self.n_large_angles
        # Repeat _is_large_angle for both correlation types if needed
        if self.n_correlation_types == 2:
            self._is_large_angle = np.tile(self._is_large_angle, 2)
        
        
        # Redshift bins
        self.redshift_bins = redshift_bins
        self.n_redshift_bins = len(self.redshift_bins)
        
        # Prepare theory Cl inputs
        (
            self._numerical_redshift_bin_combinations,
            self.redshift_bin_combinations,
            self._is_cov_cross,
            self.shot_noise,
            self._n_to_bin_comb_mapper,
        ) = prepare_theory_cl_inputs(redshift_bins, noise)
        
        self._n_redshift_bin_combs = len(self._numerical_redshift_bin_combinations)
        # Data shape attributes for full, large, and small angles
        self.data_shape_full = (self._n_redshift_bin_combs, self.n_correlation_types * self._n_ang_bins)
        self.n_data_points_full = self.data_shape_full[0] * self.data_shape_full[1]
        self.n_data_points_per_rs_comb = self.data_shape_full[1]
        
        self.data_shape_large_angles = (self._n_redshift_bin_combs, self.n_correlation_types * self.n_large_angles)
        self.data_shape_small_angles = (self._n_redshift_bin_combs, self.n_correlation_types * self.n_small_angles)
        
        # Exact lmax setup
        if exact_lmax is None:
            self._exact_lmax = mask.exact_lmax
        else:
            self._exact_lmax = exact_lmax
            if not check_property_equal([self, self.mask], "_exact_lmax"):
                raise RuntimeError("Exact lmax does not align for mask and likelihood.")
        
        # State variables
        self._highell = False
        self.gaussian_covariance = None
        
        logger.info(f"Initialized XiLikelihood: {self.n_redshift_bins} redshift bins, "
                   f"{self._n_ang_bins} angular bins, lmax={self.lmax}, exact_lmax={self._exact_lmax}")

 

    def initiate_mask_specific(self):
        # initiate the mask specific quantities

        self.mask.precompute_for_cov_masked(cov_ell_buffer=self.config.cov_ell_buffer)
        self._eff_area = self.mask.eff_area

    def prep_data_array(self):
        # Data array size depends on whether xi_minus is included

        return np.zeros(self.data_shape_full)

    def check_pdfs(self):
        """
        Wrapper for PDF validation. Delegates the actual checks to a utility function.
        """
        cop.validate_pdfs(self._pdfs, self._xs, self._cdfs,plot=False,savepath="pdfs_validation.png")

    def precompute_combination_matrices(self):
        # prefactors is a 3D array with shape (len(angles), 2, out_lmax)
        # only necessary for exact data dimensions: idea: only pass large angles, or actually...
        prefactors = prep_prefactors(
            self.ang_bins_in_deg, self.mask.wl, self.lmax, self.lmax
        )
        # ...compute prefactors for all angles, because needed for means and covariances also for Gaussian marginals
        self._prefactors = prefactors
        len_sub_m = self._exact_lmax + 1
        # need all combination matrices because we need products of all scales for full covariance matrix
        if self.include_ximinus:
            # Use both xi_plus and xi_minus prefactors
            xiplus_prefactors = prefactors[:, 0, :self._exact_lmax + 1]
            ximinus_prefactors = prefactors[:, 1, :self._exact_lmax + 1]
            
            # Create xi_plus matrices
            m_xiplus = 2 * np.repeat(xiplus_prefactors, len_sub_m, axis=1)
            m_xiplus[:, ::len_sub_m] *= 0.5
            self._m_xiplus = np.tile(m_xiplus, (1, 4))
            
            # Create xi_minus matrices  
            # these are wrong!! need to be adjusted to the way xi- is set up (including minus and EB modes!)
           
            m_ximinus = 2 * np.repeat(ximinus_prefactors, len_sub_m, axis=1)
            m_ximinus[:, ::len_sub_m] *= 0.5
            tiled = np.tile(m_ximinus, (1, 4))
            tiled[:, 2 * m_ximinus.shape[1]:] *= -1 # this already makes the real means match, but the -2j * C_EB part should be implemented for the covariance.
            self._m_ximinus = tiled
            # ximinus m won't be diagonal anymore, so multiplying will need some more effort
            #self._m_ximinus = np.tile(m_ximinus, (1, 4))
            
            # Concatenate xi_plus and xi_minus matrices
            # This preserves the is_cov_cross indexing for both correlation functions
            self._m_combined = np.concatenate([self._m_xiplus, self._m_ximinus], axis=0)
        else:
            # Only xi_plus (legacy behavior)
            xiplus_prefactors = prefactors[:, 0, :self._exact_lmax + 1]
            m_xiplus = 2 * np.repeat(xiplus_prefactors, len_sub_m, axis=1)
            m_xiplus[:, ::len_sub_m] *= 0.5
            self._m_xiplus = np.tile(m_xiplus, (1, 4))
            self._m_combined = self._m_xiplus
        
   
       
        

    def initiate_theory_cl(self, cosmo):
        # get the theory Cl for the given cosmology
        self._theory_cl = generate_theory_cl(self.lmax, self.redshift_bin_combinations, self.shot_noise, cosmo,self.mask.nside)
    
        return self._theory_cl

    def _get_pseudo_alm_covariances(self):
        """Get pseudo alm covariances - potentially memory intensive."""
        # all redshift bin combinations, organized as in GLASS
        # this loop could be mpi parallelized
        with computation_phase("pseudo alm covariances", 
                          log_memory=self.config.log_memory_usage):
            pseudo_alm_covs = [
                Cov(self.mask, theory_cl, self._exact_lmax, ischain=True).cov_alm_xi(ischain=True)
                for theory_cl in self._theory_cl
            ]
            return pseudo_alm_covs

    def _prepare_matrix_products(self):
        """Prepare the matrix products for moments and marginals."""
        # put together every combination of two different redshift bins n_redshift over 2 -> 10 combinations
        with computation_phase("matrix products preparation", log_memory=self.config.log_memory_usage):
    
            covs = self._get_pseudo_alm_covariances()
            # Use combined matrices for both xi_plus and xi_minus (or just xi_plus if flag is False)
            
            self._products = np.array(
                [self._m_combined[:, :, None] * covs[i] for i in range(len(covs))]
            )  # all angular bins each, all covariances, includes xi+ and optionally xi-
            # products shape: (n_redshift_bin_combs, n_correlation_types*len(angular_bins), len(cov), len(cov))
            logger.info(f"products shape: {self._products.shape}, dtype: {self._products.dtype}, size: {self._products.nbytes/1e9:.2f} GB")
            logger.info("Calculating 1D means...")
            self.pseudo_cl = cl2pseudocl(self.mask.m_llp, self._theory_cl)
            
            if self.include_ximinus:
                # Calculate both xi_plus and xi_minus means at once
                means_both = mean_xi_gaussian_nD(
                    self._prefactors, self.pseudo_cl, lmin=0, lmax=self._exact_lmax, kind="both"
                )
                means_xiplus, means_ximinus = means_both
                # Concatenate along the angular bin axis (axis=1), not redshift bin axis (axis=0)
                means = np.concatenate([means_xiplus, means_ximinus], axis=1)
                self._means_lowell = means.real if np.iscomplexobj(means) and np.allclose(means.imag, 0) else means
                if np.iscomplexobj(self._means_lowell):
                    logger.warning("Means for low-ell contain complex values, using real part only.")
                    logger.info("Imaginary parts are fraction of real parts: %s", self._means_lowell.imag / self._means_lowell.real)
                    self._means_lowell = means.real
                
                # should contain means for all scales
            else:
                # Only xi_plus
                self._means_lowell = mean_xi_gaussian_nD(
                    self._prefactors, self.pseudo_cl, lmin=0, lmax=self._exact_lmax, kind="p"
                )
            
            # Optional validation check (can be expensive for large matrices)
            if self.config.validate_means:
                # careful, this can only be run if all scales are included in the _products....
                logger.info("Validating means with einsum computation...")
                einsum_means = np.einsum("cbll->cb", self._products.copy())
                diff = einsum_means - self._means_lowell
                
                if np.any(np.abs(diff) > 1e-10):
                    logger.error("Means do not match: max diff = %.2e", np.max(np.abs(diff)))
                    logger.error("Einsum means: %s", einsum_means)
                    logger.error("Analytical means: %s", self._means_lowell)
                    # raise RuntimeError("Means do not match")
                else:
                    logger.debug("Means validation passed: max diff = %.2e", np.max(np.abs(diff)))
                
                if self.config.enable_memory_cleanup:
                    del einsum_means, diff
            
            if self.config.enable_memory_cleanup:
                del covs
            
        
      
    def _compute_auto_eigenvalues(self, auto_prods):
        logger.info("Computing auto eigenvalues...")
        # shape (n_redshift_bins, len(angular_bins),len(cov))
        #auto_prods_cpu = jax.device_put(jnp.array(auto_prods), device=jax.devices("cpu")[0])
        auto_prods_numpy = np.asarray(auto_prods)
        eigvals_auto = np.linalg.eigvals(auto_prods_numpy)
        eigvals_auto_padded = np.pad(
            eigvals_auto, ((0, 0), (0, 0), (0, eigvals_auto.shape[-1])), "constant"
        )
        return jnp.array(eigvals_auto_padded)

    def _compute_cross_eigenvalues(self, cross_prods, auto_prods, cross_combs):
        logger.info("Computing cross eigenvalues...")
        # shape (n_redshift_bin_cross_combs, len(angular_bins), 2*len(cov), 2*len(cov))
        cross_eigvals = []
        # Preallocate output array for eigenvalues
        n_cross = len(cross_combs)
        # Assume eigvals shape is (len(angular_bins)*2, ) for each mat (adjust if needed)
        # We'll get the shape from the first computation
        if n_cross == 0:
            return np.array([])
        cross_prods, auto_prods = np.asarray(cross_prods), np.asarray(auto_prods)
        # Compute shape from first matrix
        diag_elem = cross_prods[0]
        off_diag_elem_1 = auto_prods[cross_combs[0][0]]
        off_diag_elem_2 = auto_prods[cross_combs[0][1]]
        mat = 0.5 * np.block(
            [[diag_elem, off_diag_elem_2], [off_diag_elem_1, diag_elem]]
        )
        #mat_cpu = jax.device_put(mat, device=jax.devices("cpu")[0])
        mat = np.array(mat)
        eigvals0 = np.linalg.eigvals(mat)
        eigval_shape = eigvals0.shape
        cross_eigvals = np.zeros((n_cross,) + eigval_shape, dtype=eigvals0.dtype)
        cross_eigvals[0] = eigvals0
        for c, comb in enumerate(cross_combs[1:], start=1):
            diag_elem = cross_prods[c]
            off_diag_elem_1 = auto_prods[comb[0]]
            off_diag_elem_2 = auto_prods[comb[1]]
            mat = 0.5 * np.block(
                [[diag_elem, off_diag_elem_2], [off_diag_elem_1, diag_elem]]
            )
            mat = np.array(mat)
            eigvals = np.linalg.eigvals(mat)
            cross_eigvals[c] = eigvals
        return jnp.array(cross_eigvals)
    
    def _compute_cfs(self):
        """Compute characteristic functions using configured parameters."""
        # now returns combination of exact and gaussian cfs, to be combined with high ell end and converted to pdf all together.
        logger.info("Computing characteristic functions...")
        ts = np.zeros(self.data_shape_large_angles + (self.config.cf_steps-1,))
        cfs = np.zeros(self.data_shape_large_angles + (self.config.cf_steps-1,),dtype=complex)
        t_lowell, cfs_lowell = batched_cf_1d_jitted(
            self._eigvals,  self._ximin[:,self._is_large_angle],self._ximax[:,self._is_large_angle], steps=self.config.cf_steps
        ) # need to only pass ximax 
        ts = np.asarray(t_lowell)
        cfs = np.asarray(cfs_lowell)
        
        return ts, cfs

    def _get_cfs_1d_lowell(self):
        """Compute characteristic functions for low-ell part."""
        with computation_phase("Characteristic function computation (low-ell)", log_memory=self.config.log_memory_usage):
            
            products = self._products.view() 
            
            large_angle_products = products[:, self._is_large_angle, ...]
            large_angle_products.flags.writeable = False
            
            eigvals_shape = self.data_shape_large_angles + (2 * large_angle_products.shape[-1],)
            self._eigvals = jnp.zeros(eigvals_shape,dtype=complex)
            
            cross_prods = np.array(large_angle_products[self._is_cov_cross])
            auto_prods = np.array(large_angle_products[~self._is_cov_cross])
            if self.config.enable_memory_cleanup:
                del large_angle_products, products

            self._ximax = jnp.array(self._means_lowell + self.config.ximax_sigma_factor * jnp.sqrt(self._variances_lowell))
            self._ximin = jnp.array(self._means_lowell - self.config.ximin_sigma_factor * jnp.sqrt(self._variances_lowell))

            # Compute eigenvalues
            eigvals_auto_padded = self._compute_auto_eigenvalues(auto_prods)
            self._eigvals = self._eigvals.at[~self._is_cov_cross].set(eigvals_auto_padded)
            cross_combs = self._numerical_redshift_bin_combinations[self._is_cov_cross]
            if len(cross_combs) > 0:
                cross_eigvals = self._compute_cross_eigenvalues(cross_prods, auto_prods, cross_combs)
                self._eigvals = self._eigvals.at[self._is_cov_cross].set(cross_eigvals)

            if self.config.analyze_eigenvalues:
                analyze_significant_eigenvalues(self._eigvals)
                plot_eigenvalue_distributions(self._eigvals,save_dir=self.working_dir)
            # Compute CFs
            self._t_lowell, self._cfs_lowell = self._compute_cfs() 

            # Cleanup
            # Free memory after CF computation
            if self.config.enable_memory_cleanup:
                del auto_prods, cross_prods, cross_combs
    

    def precompute_covariance(self):
        # Precompute all (i, j, k, l, cov_pos) for i <= j
        index_tuples = []
        cov_pos_list = []
        i = 0
        for rcomb1 in self._numerical_redshift_bin_combinations:
                # Iterate over all data points (xi+ and optionally xi- for each angular bin)
                for k in range(self.n_data_points_per_rs_comb):
                    j = 0
                    for rcomb2 in self._numerical_redshift_bin_combinations:
                        for l in range(self.n_data_points_per_rs_comb):
                            if i <= j:
                                all_combs = [
                                    (rcomb1[0], rcomb2[0]),
                                    (rcomb1[1], rcomb2[1]),
                                    (rcomb1[1], rcomb2[0]),
                                    (rcomb1[0], rcomb2[1]),
                                ]
                                
                                cov_pos = [self._n_to_bin_comb_mapper.get_index(comb) for comb in all_combs]
                                index_tuples.append([i, j, k, l])
                                cov_pos_list.append(cov_pos)
                            j += 1
                    i += 1

        self._index_tuples = np.array(index_tuples)
        self._cov_pos_list = np.array(cov_pos_list)

    @staticmethod
    @jax.jit
    def compute_cov_block(products, cov_pos, k, l):
        # products: shape (...), cov_pos: shape (4,)
        return jnp.sum(products[cov_pos[0],k] * products[cov_pos[1],l].T +
                    products[cov_pos[2],k] * products[cov_pos[3],l].T)

    def get_covariance_matrix_lowell(self):
        # get the covariance matrix for the full data vector exact part
        # use products of combination matrix and pseudo alm covariance
        # here I need products of all scales - maybe keep products after all
        
       
        self._cov_lowell = jnp.full((self.n_data_points_full,self.n_data_points_full), jnp.nan)
        with computation_phase("Covariance matrix computation (low-ell)", log_memory=self.config.log_memory_usage):
            
            products = self._products.view()
            products.flags.writeable = False
            products = jnp.array(products)
            # quadruple for loop has been moved to prepare_covariance, now need to write a jax vmap here to speed things up.
            vectorized_compute_cov_block = jax.vmap(self.compute_cov_block, in_axes=(None,0,0,0))
            cov_mat_entries = vectorized_compute_cov_block(
                products, 
                self._cov_pos_list, 
                self._index_tuples[:, 2],  # k
                self._index_tuples[:, 3]   # l
            )
            i_list = self._index_tuples[:, 0]  # i
            j_list = self._index_tuples[:, 1]  # j
            self._cov_lowell = self._cov_lowell.at[(i_list, j_list)].set(cov_mat_entries)
            if self.config.enable_memory_cleanup:
                del products
        # Fill lower triangle by symmetry
        self._cov_lowell = jnp.where(
            jnp.isnan(self._cov_lowell), self._cov_lowell.T, self._cov_lowell
        )
        # Convert to numpy array for compatibility
        self._cov_lowell = np.asarray(self._cov_lowell)
        self._variances_lowell = np.diag(self._cov_lowell).reshape(self.data_shape_full)
        return self._cov_lowell

    def get_covariance_matrix_highell(self):
        # get the covariance matrix for the full data vector Gaussian part
        # use C_ell approximation
        self._cov_highell = cov_xi_gaussian_nD(
            self._theory_cl,
            self._numerical_redshift_bin_combinations,
            self.ang_bins_in_deg,
            self._eff_area,
            lmin=self._exact_lmax + 1,
            lmax=self.lmax,
            include_ximinus=self.include_ximinus,
        )

    def _get_means_highell(self):
        # get the mean for the full data vector Gaussian part
        if self.include_ximinus:
            # Calculate both xi_plus and xi_minus means for high-ell
            means_both = mean_xi_gaussian_nD(
                self._prefactors, self.pseudo_cl, lmin=self._exact_lmax + 1, lmax=self.lmax, kind="both"
            )
            means_xiplus, means_ximinus = means_both
            # Concatenate along the angular bin axis (axis=1)
            means = np.concatenate([means_xiplus, means_ximinus], axis=1)
            self._means_highell = means.real if np.iscomplexobj(means) and np.allclose(means.imag, 0) else means
            if np.iscomplexobj(self._means_highell):
                logger.warning("Means for high-ell contain complex values, using real part only.")
                self._means_highell = means.real  
            
        else:
            # Only xi_plus (legacy behavior)
            self._means_highell = mean_xi_gaussian_nD(
                self._prefactors, self.pseudo_cl, lmin=self._exact_lmax + 1, lmax=self.lmax
            )

    def _get_cfs_1d_highell(self):
        # get the Gaussian cf for the high ell part
        vars_highell = np.diag(self._cov_highell)
        vars_highell = vars_highell.reshape(self.data_shape_full)
        vars_highell = vars_highell[:,self._is_large_angle]  # only large angles
        return high_ell_gaussian_cf_1d(self._t_lowell, self._means_highell[:,self._is_large_angle], vars_highell)

    def _get_gaussian_marginals(self):
        pass

    def marginals_exact(self):
        # get the marginal pdfs
        self._get_cfs_1d_lowell()

        if self._highell:
            logger.info("Adding high-ell contribution to CFs...")
            self._cfs = self._cfs_lowell * self._get_cfs_1d_highell()
        else:
            self._cfs = self._cfs_lowell
        
        self._xs[:,self._is_large_angle], self._pdfs[:,self._is_large_angle] = cf_to_pdf_1d(self._t_lowell, self._cfs) # is tuple (x,pdf) with shape (croco,large_angs,n_points) each
        if self.config.enable_memory_cleanup:
            del self._t_lowell, self._cfs_lowell, self._cfs
        return self._xs, self._pdfs

    def marginals_gaussian(self):
        gauss_means = self._mean[:,~self._is_large_angle]
        gauss_vars = np.diag(self._cov).reshape(self.data_shape_full)[:,~self._is_large_angle]
        logger.info("Means for the Gaussian marginals: %s", gauss_means)
        logger.info("Variances for the Gaussian marginals: %s", gauss_vars)
        xs_gauss, pdfs_gauss = gaussian_pdf_1d(gauss_means, gauss_vars, n_grid=self.config.cf_steps)
        self._xs[:,~self._is_large_angle] = xs_gauss
        self._pdfs[:,~self._is_large_angle] = pdfs_gauss
        if self.config.enable_memory_cleanup:
            del gauss_means, gauss_vars, xs_gauss, pdfs_gauss

        return self._xs, self._pdfs

    def gauss_compare(self, data, data_subset=None):
        # should always use a fixed covariance, produce on initialization?
        mean = self._mean
        if self.gaussian_covariance is None:
            logger.warning("Using cosmology-dependent covariance for Gaussian likelihood!")
            cov = self._cov
        else:
            cov = self.gaussian_covariance
        if data_subset is not None:
            data = cop.data_subset(data, data_subset)
            cov = cop.cov_subset(cov, data_subset, self._n_ang_bins)  # Use _n_ang_bins
            mean = cop.data_subset(mean, data_subset)

        mean = mean.flatten()
        mvn = multivariate_normal(mean=mean, cov=cov)
        data_flat = data.flatten()
        return mvn.logpdf(data_flat)
    
    def _prepare_likelihood_components(self, cosmology, highell: bool = True):
        """Prepare all components needed for likelihood computation."""
        if cosmology is None:
            raise ValueError("cosmology cannot be None")
    
        if not hasattr(self, '_prefactors'):
            raise RuntimeError("Combination matrices not precomputed. Call precompute_combination_matrices() first")
    

        if highell:
            self._highell = True
        else:  # Highell is False
            self._highell = False

        self.initiate_theory_cl(cosmology)
        self._prepare_matrix_products()

        self._cov = self.get_covariance_matrix_lowell()
        self._mean = self._means_lowell

        if self._highell:
            self.get_covariance_matrix_highell()
            self._get_means_highell()
            self._cov = self._cov_lowell + self._cov_highell
            self._mean = self._means_lowell + self._means_highell
        logger.info("Covariance matrix and means prepared.")
        logger.info("Mean samples: %s", self._mean[:5, :5])
        
        self._xs = np.zeros(self.data_shape_full + (self.config.cf_steps-1,))
        self._pdfs = np.zeros(self.data_shape_full + (self.config.cf_steps-1,))
        self.marginals_exact()
        self.marginals_gaussian()
        self._cdfs, self._pdfs, self._xs = cop.pdf_to_cdf(self._xs, self._pdfs,num_points=self.config.pdf_steps)  # Interpolated xs and pdfs
        self.check_pdfs()

        del self._products


    def loglikelihood(self, data, cosmology, highell=True, gausscompare=False, data_subset=None):
        """
        Compute the log-likelihood for a given cosmology and data point.

        Parameters
        ----------
        data : ndarray
            Data point to evaluate the likelihood at.
        cosmology : dict
            Cosmological parameters.
        highell : bool, optional
            Whether to include high-ell contributions, by default True.
        gausscompare : bool, optional
            Whether to compare with Gaussian likelihood, by default False.
        data_subset : list, optional
            Subset of data to evaluate, by default None.

        Returns
        -------
        float or tuple
            Log-likelihood value, optionally with Gaussian comparison.
        """
        
        with computation_phase("Log-likelihood evaluation", log_memory=self.config.log_memory_usage):
            self._prepare_likelihood_components(cosmology, highell)

            # Quick check to ensure data and self._xs have matching first two dimensions
            if data.shape[:2] != self._xs.shape[:2]:
                raise ValueError("Mismatch in dimensions: data and self._xs must have the same first two dimensions. But got: {} and {}".format(data.shape, self._xs.shape))

            # Ensure all arrays passed to copula-based likelihood evaluation are NumPy arrays
            likelihood = cop.evaluate(np.asarray(data), np.asarray(self._xs), np.asarray(self._pdfs), np.asarray(self._cdfs), np.asarray(self._cov), subset=data_subset)

        if gausscompare:
            return likelihood, self.gauss_compare(data, data_subset=data_subset)
        else:
            return likelihood
    
    def likelihood_function(self, cosmology, highell=True):
        """
        Return the likelihood as a joint PDF for the entire data space.
        
        .. deprecated:: 
            This method is deprecated for high-dimensional data (>3D) due to 
            computational and memory constraints. Use alternatives:
            
            - `loglikelihood(data, cosmology)` for point evaluation
            - `likelihood_function_2d(cosmology, data_subset)` for 2D visualization
            - Custom sampling approaches for MCMC/optimization
        
        Parameters
        ----------
        cosmology : dict
            Cosmological parameters.
        highell : bool, optional
            Whether to include high-ell contributions, by default True.

        Returns
        -------
        tuple
            (xs, joint_pdf_values) - coordinate grids and PDF values
            
        See Also
        --------
        loglikelihood : Evaluate likelihood at specific data points
        likelihood_function_2d : 2D joint PDF for visualization
        """
        import warnings
        
        data_dim = self._n_redshift_bin_combs * len(self.ang_bins_in_deg)
        
        # Always warn about usage
        warnings.warn(
            f"likelihood_function() is deprecated for {data_dim}D data. "
            f"Use loglikelihood() for point evaluation or likelihood_function_2d() "
            f"for visualization. This method may be removed in future versions.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if data_dim > 4:
            raise ValueError(
                f"likelihood_function() is not supported for {data_dim}D data. "
                f"Use loglikelihood() or likelihood_function_2d() instead."
            )
        # Prepare the likelihood components
        self._prepare_likelihood_components(cosmology, highell)

        # Compute the joint PDF using cop.joint_logpdf
        joint_logpdf_values = cop.joint_logpdf(self._cdfs, self._pdfs, self._cov)

        return self._xs, joint_logpdf_values




    def likelihood_function_2d(self, data_subset=None,gausscompare=False):
        """
        Return a 2D subset of the likelihood as a joint PDF at a given cosmology.

        Parameters
        ----------
        cosmology : dict
            Cosmological parameters.
        
        data_subset : list
            Subset of data to evaluate (e.g., 2D indices).

        Returns
        -------
        tuple
            xs and joint PDF values for the specified 2D subset.
        """
        if data_subset is None:
            raise ValueError("data_subset must be specified for likelihood_function_2d.")
        
        # Ensure data_subset selects exactly 2 data points
        if len(data_subset) != 2:
            raise ValueError("data_subset must select exactly 2 data dimensions.")

        if not hasattr(self, '_cov') or self._cov is None:
            raise RuntimeError("Likelihood not prepared. Call prepare_likelihood_components() first")
    
        
        cov_2d = cop.cov_subset(self._cov, data_subset, self._n_ang_bins)  # Use _n_ang_bins
        xs_subset = cop.data_subset(self._xs, data_subset)
        pdfs_subset = cop.data_subset(self._pdfs, data_subset)
        cdfs_subset = cop.data_subset(self._cdfs, data_subset)
        log_pdf_2d = cop.joint_logpdf(cdfs_subset, np.asarray(pdfs_subset), np.asarray(cov_2d))
        
        
        if gausscompare:
            means_subset = cop.data_subset(self._mean, data_subset)
            gaussian_log_pdf = gaussian_2d(xs_subset,np.asarray(means_subset),np.asarray(cov_2d))
            return xs_subset, log_pdf_2d, gaussian_log_pdf
        else:
            return xs_subset, log_pdf_2d


    def setup_likelihood(self) -> None:
        """
        Complete likelihood setup workflow for one-time initialization.
        
        Convenience method that runs the initialization sequence that only 
        needs to be done once (independent of cosmology):
        1. Mask-specific setup
        2. Combination matrices precomputation
        3. Optional memory validation
        
        Note: Theory Cl computation and likelihood preparation are handled
        automatically in loglikelihood() for each cosmology.
        
        Parameters:
        -----------
        validate_memory : bool, default=True
            Whether to validate memory requirements
            
        Examples:
        ---------
        >>> likelihood = XiLikelihood(mask, redshift_bins, ang_bins)
        >>> likelihood.setup_likelihood()  # One-time setup
        >>> 
        >>> # Now ready for multiple cosmologies
        >>> logL1 = likelihood.loglikelihood(data, cosmology1)
        >>> logL2 = likelihood.loglikelihood(data, cosmology2)
        """

        if hasattr(self, '_prefactors'):
            logger.warning("Likelihood already set up - skipping redundant setup")
            return
    
        logger.info("Setting up likelihood computation (one-time initialization)...")
        
        with computation_phase("likelihood setup", log_memory=self.config.log_memory_usage):
            
                       
            self.initiate_mask_specific()
            self.precompute_combination_matrices()
            self.precompute_covariance()
            
            logger.info("Likelihood setup complete! Ready for cosmology-dependent computations.")

    
    

def fiducial_dataspace(
    redshift_directory: Optional[str] = None,
    ang_min: float = 0.5, 
    ang_max: float = 300,
    n_bins: int = 9, 
    min_ang_cutoff_in_arcmin: float = 15
) -> Tuple[List[RedshiftBin], List[Tuple[float, float]]]:
    """
    Generate standard KiDS-like analysis setup.
    
    Parameters:
    -----------
    redshift_directory : str, optional
        Directory containing redshift bin files. If None, uses package default.
    ang_min : float  
        Minimum angular scale in arcminutes
    ang_max : float
        Maximum angular scale in arcminutes
    n_bins : int
        Number of angular bins
    min_ang_cutoff_in_arcmin : float
        Minimum angular scale cutoff in arcminutes
        
    Returns:
    --------
    tuple
        (redshift_bins, angular_bins) for likelihood initialization
    """
    
    # Load KiDS redshift bins using the standalone utility function
    redshift_bins_sorted = load_kids_redshift_bins(directory=redshift_directory)

    # Create angular bins
    initial_ang_bins_in_arcmin = np.logspace(
        np.log10(ang_min), np.log10(ang_max), num=n_bins, endpoint=True
    )
    # Filter out bins smaller than cutoff
    filtered_ang_bins_in_arcmin = initial_ang_bins_in_arcmin[
        initial_ang_bins_in_arcmin >= min_ang_cutoff_in_arcmin
    ]
    # Add one more bin on the larger side according to the same pattern
    last_bin_ratio = filtered_ang_bins_in_arcmin[-1] / filtered_ang_bins_in_arcmin[-2]
    new_bin = filtered_ang_bins_in_arcmin[-1] * last_bin_ratio
    extended_ang_bins_in_arcmin = np.append(filtered_ang_bins_in_arcmin, new_bin)
    # Convert to degrees
    ang_bins_in_deg = extended_ang_bins_in_arcmin / 60

    # Create tuples representing the bin edges
    ang_bins_in_deg = [
        (ang_bins_in_deg[i], ang_bins_in_deg[i + 1]) for i in range(len(ang_bins_in_deg) - 1)
    ]
    logger.info(f"Created {len(redshift_bins_sorted)} redshift bins and {len(ang_bins_in_deg)} angular bins")
    return redshift_bins_sorted, ang_bins_in_deg




""" [[4.26316685e-07 1.12676419e-06]
 [4.06130857e-07 1.73571351e-06]
 [3.35446133e-07 1.41249739e-06]]


[[5.68059428e-07 1.19825625e-06]
 [5.62009356e-07 1.72977291e-06]
 [5.58122970e-07 1.43061433e-06]] 
 
 [[4.53476258e-07 1.07737428e-06]
 [4.54624196e-07 1.73445360e-06]
 [4.83807631e-07 1.32179069e-06]]
 
 
 low ell [[1.53723621e-06 2.19769249e-08]
 [1.83496196e-06 3.17630054e-08]
 [1.21273912e-06 2.58645384e-08]]
 
 
 [[8.5676675e-07 1.9434999e-06]
 [1.0004612e-06 2.8053128e-06]
 [9.1155823e-07 2.2863794e-06]]

 [[4.7412379e-07 1.1633111e-06]
 [5.3981921e-07 1.6721029e-06]
 [4.9302201e-07 1.3668407e-06]]

 """