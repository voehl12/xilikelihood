"""
Exact 2D characteristic function computation for copula validation.

This module provides exact characteristic function calculation for 2D subsets of correlation
functions, enabling validation of copula-based approximations and estimation
of Student-t copula parameters.
"""

import numpy as np
import logging
import os
from .core_utils import computation_phase
from .distributions import setup_t

logger = logging.getLogger(__name__)


class Exact2DCF:
    """
    Computes exact 2D characteristic functions for correlation function pairs.
    
    This class is initialized for specific redshift bin pairs and angular bins,
    eliminating the need for repeated setup calls. It works with essential
    XiLikelihood components:
    - _m_combined: M matrices for correlation function extraction
    - pseudo_alm_covs: covariance matrices for each redshift bin combination
    - _ximax: maximum xi values for t-grid setup (computed automatically)
    
    Example:
    --------
    # Create instance for specific correlation function pair
    exact_cf = Exact2DCF(
        xi_likelihood_instance=likelihood,
        redshift_bin_pairs=((0, 1), (2, 3)),  # Correlate bins 0x1 with 2x3
        angular_bins=(5, 10),                 # Angular bins 5 and 10
        steps=129                             # T-grid resolution
    )
    
    # Compute characteristic function (no further setup needed)
    cf_results = exact_cf.compute_2d_exact_cf()
    """
    
    def __init__(self, xi_likelihood_instance, redshift_bin_pairs, angular_bins, steps=129, cosmo=None):
        """
        Initialize with an existing XiLikelihood instance and set up for specific correlation functions.
        
        Parameters:
        -----------
        xi_likelihood_instance : XiLikelihood
            XiLikelihood instance (cosmology will be initialized if needed)
        redshift_bin_pairs : tuple of tuples
            Two redshift bin combinations as ((bin_i1, bin_i2), (bin_j1, bin_j2))
            Example: ((0, 1), (2, 3)) for correlating bins 0x1 with bins 2x3
        angular_bins : tuple of int
            Angular bin indices for each correlation function (ang_bin_i, ang_bin_j)
        steps : int, default 129
            Number of t-grid steps per dimension
        cosmo : optional
            Cosmology object. If None and XiLikelihood not initialized, will raise error.
        """
        self.xi_likelihood = xi_likelihood_instance
        
        # Check if cosmology is initialized
        if not hasattr(xi_likelihood_instance, '_theory_cl') or xi_likelihood_instance._theory_cl is None:
            if cosmo is None:
                raise RuntimeError(
                    "XiLikelihood instance not initialized with cosmology. "
                    "Either pass cosmo parameter or call likelihood.initiate_theory_cl(cosmo) first."
                )
            # Initialize cosmology
            logger.info("Initializing XiLikelihood with provided cosmology")
            xi_likelihood_instance.initiate_theory_cl(cosmo)
        
        # Ensure M matrices are set up
        if not hasattr(xi_likelihood_instance, '_m_combined'):
            logger.info("M matrices not found, computing combination matrices...")
            xi_likelihood_instance.precompute_combination_matrices()
        
        # Get the minimal required components directly
        self._m_combined = xi_likelihood_instance._m_combined
        self.pseudo_alm_covs = xi_likelihood_instance._get_pseudo_alm_covariances()
        
        # Access precomputed products for efficient CF computation
        if hasattr(xi_likelihood_instance, '_products'):
            self._products = xi_likelihood_instance._products
        else:
            self._products = None
            logger.warning("XiLikelihood instance has no precomputed products. Call prepare_matrix_products() first.")
        
        # Get pseudo_alm_cov_size from XiLikelihood (set when covariances are computed)
        self.pseudo_alm_cov_size = xi_likelihood_instance.pseudo_alm_cov_size
        
        # Access xi_max values from XiLikelihood for t-grid setup
        if hasattr(xi_likelihood_instance, '_ximax') and xi_likelihood_instance._ximax is not None:
            self._ximax = xi_likelihood_instance._ximax
        else:
            # Automatically trigger covariance computation to get _ximax
            logger.info("Computing covariance matrix to access ximax values for t-grid setup")
            xi_likelihood_instance.get_covariance_matrix_lowell()
            self._ximax = xi_likelihood_instance._ximax
        
        # Set up 2D components immediately during initialization
        logger.info(f"Setting up 2D components for redshift bins {redshift_bin_pairs}, angular bins {angular_bins}")
        self._setup_2d_components(redshift_bin_pairs, angular_bins, steps)
        
        logger.info(f"Initialized Exact2DCF: "
                   f"M {self._m_combined.shape}, "
                   f"covs {len(self.pseudo_alm_covs)}, size {self.pseudo_alm_cov_size}")
        if self._products is not None:
            logger.debug(f"Products available: shape {self._products.shape}")
        else:
            logger.debug("Products not available (call prepare_matrix_products() first)")
        logger.debug(f"Xi_max available: shape {self._ximax.shape}")
        logger.info(f"2D setup complete: t-grid {len(self._t_grid_info['t_sets'])} points, "
                   f"products {self._products_2d.shape}")
    
    def _setup_2d_components(self, redshift_bin_pairs, angular_bins, steps):
        """
        Set up 2D components using actual redshift bin combinations (internal method).
        
        This method is called during initialization to set up the 2D likelihood
        for the specified redshift bin pairs and angular bins.
        
        Parameters:
        -----------
        redshift_bin_pairs : tuple of tuples
            Two redshift bin combinations as ((bin_i1, bin_i2), (bin_j1, bin_j2))
        angular_bins : tuple of int
            Angular bin indices for each correlation function (ang_bin_i, ang_bin_j)
        steps : int
            Number of t-grid steps per dimension
        """
        mapper = self.xi_likelihood._n_to_bin_comb_mapper
        
        # Convert redshift bin pairs to combination indices
        rs_bins_i, rs_bins_j = redshift_bin_pairs
        ang_bin_i, ang_bin_j = angular_bins
        
        # Find the combination indices for these redshift bin pairs
        rs_comb_i = mapper.get_index(rs_bins_i)
        rs_comb_j = mapper.get_index(rs_bins_j)
        
        if rs_comb_i is None:
            raise ValueError(f"Redshift bin combination {rs_bins_i} not found in mapper. "
                           f"Available combinations: {[mapper.get_combination(i) for i in range(mapper.n)]}")
        
        if rs_comb_j is None:
            raise ValueError(f"Redshift bin combination {rs_bins_j} not found in mapper. "
                           f"Available combinations: {[mapper.get_combination(i) for i in range(mapper.n)]}")
        
        logger.debug(f"Redshift bin pairs {redshift_bin_pairs} mapped to combination indices ({rs_comb_i}, {rs_comb_j})")
        
        # Store the components and original input for reference
        self._components = ((rs_comb_i, ang_bin_i), (rs_comb_j, ang_bin_j))
        self._redshift_bin_pairs = redshift_bin_pairs  # Store original input
        self._angular_bins = angular_bins              # Store original input
        
        # Convert to flat indices for compatibility
        flat_i = self._components_to_flat_index(rs_comb_i, ang_bin_i)
        flat_j = self._components_to_flat_index(rs_comb_j, ang_bin_j)
        self._indices = (flat_i, flat_j)
        
        # Set up t-grid once
        self._t_grid_info = self._setup_t_grid_internal(steps)
        
        # Extract 2D products for all involved redshift bin combinations
        self._products_2d = self._extract_2d_products(rs_bins_i, ang_bin_i, rs_bins_j, ang_bin_j)
    
    def compute_2d_exact_cf(self, steps=None, save_path=None):
        """
        Compute exact 2D characteristic function across complete t-grid.
        
        Convenience wrapper for single-job CF computation using the redshift bins
        and angular bins specified during initialization.
        
        Parameters:
        -----------
        steps : int, optional
            Number of t-grid steps per dimension. If None, uses the steps from initialization.
            If different from initialization, will regenerate the t-grid.
        save_path : str, optional
            Path to save CF results. If None, results are not saved.
            
        Returns:
        --------
        dict with keys:
            'cf_values': complex array of exact CF values at t-grid points
            't_grid_info': t-grid setup information
            'matrix_info': covariance and M matrix metadata
            'computation_info': timing information
        """
        # Use steps from initialization if not specified
        if steps is None:
            steps = self._t_grid_info['steps']
        
        logger.info(f"Computing exact 2D CF with t-grid {steps}×{steps}")
        
        # Check if we need to regenerate t-grid for different steps
        if steps != self._t_grid_info['steps']:
            logger.debug(f"Regenerating t-grid for {steps} steps (was {self._t_grid_info['steps']})")
            self._t_grid_info = self._setup_t_grid_internal(steps)
        
        if save_path:
            # Use the job function for consistent saving
            result_path = self.compute_2d_exact_cf_job(
                steps=steps, job_id=0, total_jobs=1, 
                save_dir=os.path.dirname(save_path) if os.path.dirname(save_path) else "."
            )
            
            # Load and return the results in the expected format
            saved_data = np.load(result_path, allow_pickle=True)
            
            return {
                'cf_values': saved_data['cf_values'],
                't_grid_info': saved_data['t_grid_info'].item(),
                'matrix_info': saved_data['matrix_info'].item(),
                'computation_info': {
                    'total_time': float(saved_data['computation_time']),
                    'cf_rate': len(saved_data['cf_values']) / float(saved_data['computation_time']),
                    'failed_points': saved_data['job_info'].item()['failed_points'],
                    'steps': steps
                }
            }
        else:
            # Direct computation without saving - use job function in memory mode
            import tempfile
            import os
            
            with tempfile.TemporaryDirectory() as temp_dir:
                result_path = self.compute_2d_exact_cf_job(
                    steps=steps, job_id=0, total_jobs=1, 
                    save_dir=temp_dir
                )
                
                # Load results and return in expected format
                saved_data = np.load(result_path, allow_pickle=True)
                
                return {
                    'cf_values': saved_data['cf_values'],
                    't_grid_info': saved_data['t_grid_info'].item(),
                    'matrix_info': saved_data['matrix_info'].item(),
                    'computation_info': {
                        'total_time': float(saved_data['computation_time']),
                        'cf_rate': len(saved_data['cf_values']) / float(saved_data['computation_time']),
                        'failed_points': saved_data['job_info'].item()['failed_points'],
                        'steps': steps
                    }
                }
    
    def compute_2d_exact_cf_job(self, steps=None, job_id=0, total_jobs=1, save_dir="cf_results"):
        """
        Compute exact 2D characteristic functions for a subset of t-grid points (job-based).
        
        This method is designed for parallel execution across multiple jobs, where each job
        computes characteristic functions for a subset of the t-grid.
        
        Parameters
        ----------
        steps : int, optional
            Number of steps for t-grid. If None, uses the steps from initialization.
        job_id : int, optional
            Job identifier (0-indexed). Default is 0.
        total_jobs : int, optional
            Total number of jobs. Default is 1.
        save_dir : str, optional
            Directory to save results. Default is "cf_results".
            
        Returns
        -------
        str
            Path to saved results file.
        """
        
        
        
        # Use steps from initialization if not specified
        if steps is None:
            steps = self._t_grid_info['steps']
        
        is_single_job = (total_jobs == 1)
        
        # Setup for the computation
        phase_name = f"Complete 2D CF computation" if is_single_job else f"2D CF job {job_id}/{total_jobs}"
        
        # Check if we need to regenerate t-grid for different steps
        if steps != self._t_grid_info['steps']:
            logger.debug(f"Regenerating t-grid for {steps} steps (was {self._t_grid_info['steps']})")
            self._t_grid_info = self._setup_t_grid_internal(steps)
        
        with computation_phase(phase_name):
            # Use cached components
            t_grid_info = self._t_grid_info
            indices = self._indices
            
            # Compute CF (single job or job slice)
            t_sets = t_grid_info['t_sets']
            
            if not is_single_job:
                # Split t-grid across jobs
                n_points = len(t_sets)
                points_per_job = n_points // total_jobs
                remainder = n_points % total_jobs
                
                start_idx = job_id * points_per_job + min(job_id, remainder)
                end_idx = start_idx + points_per_job + (1 if job_id < remainder else 0)
                
                t_subset = t_sets[start_idx:end_idx]
                logger.info(f"Job {job_id}/{total_jobs}: Processing t-points {start_idx}-{end_idx-1} ({len(t_subset)} points)")
            else:
                t_subset = t_sets
                start_idx = 0
                end_idx = len(t_sets)
                logger.info(f"Single job: Processing all {len(t_subset)} t-points")
            
            # Compute CF at each t-point  
            cf_values = []
            failed_points = 0
            
            logger.info(f"Computing eigenvalues for {len(t_subset)} matrix products...")
            
            for i, t_point in enumerate(t_subset):
                try:
                    # Compute characteristic function using precomputed products
                    _, cf_value = self._get_cf_2d(t_point)
                    cf_values.append(cf_value)
                    
                    if (i + 1) % max(1, len(t_subset) // 10) == 0:
                        logger.debug(f"Progress: {i+1}/{len(t_subset)} ({100*(i+1)/len(t_subset):.1f}%)")
                        
                except Exception as e:
                    logger.warning(f"CF computation failed at t-point {i}: {e}")
                    cf_values.append(np.nan + 1j*np.nan)
                    failed_points += 1
            
            cf_values = np.array(cf_values)
            
            logger.info(f"CF computation complete: {len(cf_values) - failed_points} successful, "
                       f"{failed_points} failed, range [{np.abs(cf_values).min():.2e}, {np.abs(cf_values).max():.2e}]")
            
            # Package results
            results = {
                'job_id': job_id,
                'total_jobs': total_jobs,
                'indices': indices,
                'cf_values': cf_values,
                't_points': t_subset if not is_single_job else t_grid_info['t_sets'],
                'job_info': {
                    'job_id': job_id,
                    'total_jobs': total_jobs,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'failed_points': failed_points
                },
                't_grid_info': t_grid_info,
                'matrix_info': {
                    'redshift_bin_pairs': self._redshift_bin_pairs,
                    'angular_bins': self._angular_bins,
                    'products_shape': self._products_2d.shape
                }
            }
        
        # Save results
        os.makedirs(save_dir, exist_ok=True)
        
        if is_single_job:
            save_path = os.path.join(save_dir, f"cf_complete_indices_{indices[0]}_{indices[1]}_steps_{steps}.npz")
        else:
            save_path = os.path.join(save_dir, f"cf_job_{job_id:03d}_indices_{indices[0]}_{indices[1]}.npz")
        
        np.savez_compressed(save_path, **results)
        logger.info(f"Results saved to {save_path}")
        
        return save_path
    
    @staticmethod
    def combine_cf_job_results(save_dir, indices, total_jobs, output_path=None):
        """
        Combine CF results from multiple jobs into complete t-grid.
        
        After all SLURM jobs complete, use this to reconstruct the full CF grid
        and convert to PDF.
        
        Parameters:
        -----------
        save_dir : str
            Directory containing individual job result files
        indices : tuple of int
            Correlation function indices (for file naming)
        total_jobs : int
            Total number of jobs to combine
        output_path : str, optional
            Path to save combined results
            
        Returns:
        --------
        dict : Combined CF results ready for PDF conversion
        """
        
        
        logger.info(f"Combining {total_jobs} CF job results for indices {indices}")
        
        # Load all job results
        all_cf_values = []
        all_t_points = []
        job_info = {}
        
        for job_id in range(total_jobs):
            job_file = os.path.join(save_dir, f"cf_job_{job_id:03d}_indices_{indices[0]}_{indices[1]}.npz")
            
            if not os.path.exists(job_file):
                logger.warning(f"Missing job file: {job_file}")
                continue
                
            job_data = np.load(job_file, allow_pickle=True)
            all_cf_values.append(job_data['cf_values'])
            all_t_points.append(job_data['t_points'])
            
            if job_id == 0:
                # Get metadata from first job
                t_grid_info = job_data['t_grid_info'].item()
                job_info['indices'] = job_data['indices']
                job_info['total_jobs'] = job_data['total_jobs']
        
        # Reconstruct full CF grid
        combined_cf = np.concatenate(all_cf_values)
        combined_t_points = np.concatenate(all_t_points, axis=0)
        
        # Verify we have the complete grid
        expected_points = len(t_grid_info['t_sets'])
        if len(combined_cf) != expected_points:
            logger.warning(f"CF grid incomplete: {len(combined_cf)}/{expected_points} points")
        
        combined_results = {
            'cf_values': combined_cf,
            't_points': combined_t_points,
            't_grid_info': t_grid_info,
            'job_info': job_info,
            'n_jobs_combined': len(all_cf_values)
        }
        
        # Save combined results
        if output_path:
            np.savez_compressed(output_path, **combined_results)
            logger.info(f"Combined CF results saved to {output_path}")
        
        logger.info(f"CF combination complete: {len(combined_cf)} total points")
        
        return combined_results
            
            
            
       
    
    def _flat_index_to_components(self, flat_index):
        """
        Convert flat data vector index to (redshift_comb, angular_bin) components.
        
        The data vector is organized as:
        [rs_comb_0_ang_0, rs_comb_0_ang_1, ..., rs_comb_1_ang_0, rs_comb_1_ang_1, ...]
        
        Parameters:
        -----------
        flat_index : int
            Flat index into the full data vector
            
        Returns:
        --------
        rs_comb : int
            Redshift combination index
        ang_bin : int
            Angular bin index within that redshift combination
        """
        n_data_per_rs_comb = self.xi_likelihood.n_data_points_per_rs_comb
        
        rs_comb = flat_index // n_data_per_rs_comb
        ang_bin = flat_index % n_data_per_rs_comb
        
        return rs_comb, ang_bin
    
    def _components_to_flat_index(self, rs_comb, ang_bin):
        """
        Convert (redshift_comb, angular_bin) components to flat data vector index.
        
        Inverse of _flat_index_to_components().
        
        Parameters:
        -----------
        rs_comb : int
            Redshift combination index
        ang_bin : int
            Angular bin index within that redshift combination
            
        Returns:
        --------
        flat_index : int
            Flat index into the full data vector
        """
        n_data_per_rs_comb = self.xi_likelihood.n_data_points_per_rs_comb
        return rs_comb * n_data_per_rs_comb + ang_bin
    
    def _setup_t_grid_internal(self, steps):
        """
        Set up t-grid for characteristic function computation.
        
        Uses the same approach as distributions.setup_t() for consistency.
        
        Parameters:
        -----------
        steps : int
            Number of grid points per dimension
            
        Returns:
        --------
        dict
            t-grid information including t_sets and metadata
        """
        # ximax is structured as (redshift_combs, angular_bins)
        # Extract ximax values for the specific correlation functions
        (rs_comb_i, ang_bin_i), (rs_comb_j, ang_bin_j) = self._components
        
        xi_max_i = self._ximax[rs_comb_i, ang_bin_i]
        xi_max_j = self._ximax[rs_comb_j, ang_bin_j]
        xi_max_2d = [xi_max_i, xi_max_j]
        
        logger.debug(f"Using ximax values for 2D grid: {xi_max_2d}")
        
        # Use setup_t to get the grid
        t_inds, t_sets, t0s, dts = setup_t(xi_max_2d, steps)
        
        logger.debug(f"T-grid with distributions.setup_t: {len(t_sets)} points, "
                    f"t0s={t0s}, dts={dts}")
        
        return {
            'steps': steps,
            't_sets': t_sets,
            't0s': t0s,
            'dts': dts,
        }
    
    def _extract_2d_products(self, rs_bins_i, ang_bin_i, rs_bins_j, ang_bin_j):
        """
        Extract precomputed M @ cov products for all redshift bin combinations involved.
        
        Similar to the covariance matrix building, we need products for all combinations 
        of the unique redshift bins involved, multiplied by both M matrices (both angular bins).
        
        Parameters:
        -----------
        rs_bins_i : tuple
            Redshift bin combination for first correlation function (bin_i1, bin_i2)
        ang_bin_i : int
            Angular bin for first correlation function
        rs_bins_j : tuple
            Redshift bin combination for second correlation function (bin_j1, bin_j2)
        ang_bin_j : int
            Angular bin for second correlation function
            
        Returns:
        --------
        products_2d : array_like, shape (n_combinations, 2, size, size)
            Precomputed products for all involved redshift combinations with both M matrices
        """
        if self._products is None:
            raise RuntimeError("No precomputed products available. Call likelihood.prepare_matrix_products() first.")
        
        # Find all unique redshift bins involved
        all_bins = set()
        all_bins.update(rs_bins_i)
        all_bins.update(rs_bins_j)
        unique_bins = sorted(list(all_bins))
        
        mapper = self.xi_likelihood._n_to_bin_comb_mapper
        
        # Get all combinations of these unique bins
        involved_combinations = []
        for i, bin_i in enumerate(unique_bins):
            for j in range(i, len(unique_bins)):  # Only upper triangle: j >= i
                bin_j = unique_bins[j]
                
                # Get the combination index for this redshift bin pair
                comb = (bin_i, bin_j)
                comb_idx = mapper.get_index(comb)
                involved_combinations.append(comb_idx)
        
        logger.debug(f"Building products for unique bins {unique_bins}: "
                    f"{len(involved_combinations)} combinations")
        
        # Extract products for all combinations with both angular bins using slicing
        involved_combinations = np.array(involved_combinations)  # Convert to numpy array for slicing
        angular_bins = np.array([ang_bin_i, ang_bin_j])  # Angular bins as array
        
        # Store involved combinations and unique bins info as attributes for reference
        self._involved_combinations = involved_combinations
        self._unique_bins = unique_bins
        self._n_unique_bins = len(unique_bins)
        
        # Extract all products at once using 2D advanced indexing
        # Shape: (n_combinations, 2, size, size) - consistent with _products structure
        products_2d = self._products[np.ix_(involved_combinations, angular_bins)]
        
        logger.debug(f"Extracted 2D products: {len(involved_combinations)} combinations "
                    f"× 2 angular bins -> {products_2d.shape}")
        
        return products_2d
    
    def _get_cf_2d(self, tset):
        """
        Compute exact 2D characteristic function using precomputed products.
        
        Parameters:
        -----------
        tset : array_like, shape (2,)
            2D point in characteristic function space
            
        Returns:
        --------
        tset : array_like
            t-grid point 
        cf_value : complex
            Characteristic function value
        """
        # Classify correlation types using is_croco attribute
        correlation_type = self._classify_correlation_type()
        
        # Dispatch to appropriate matrix constructor
        if correlation_type == "auto-auto":
            combined_matrix = self._build_auto_auto_matrix(tset)
        elif correlation_type == "cross-cross":
            combined_matrix = self._build_cross_cross_matrix(tset)
        else:  # cross-auto
            combined_matrix = self._build_cross_auto_matrix(tset)
        
        evals = np.linalg.eigvals(combined_matrix)
        return tset, np.prod((1 - 2j * evals) ** -0.5)
    
    def _classify_correlation_type(self):
        """
        Classify the type of correlation pair for matrix construction.
        
        Uses the _is_cov_cross attribute from XiLikelihood to determine if redshift
        bin combinations correspond to auto-correlations or cross-correlations.
        
        Returns:
        --------
        str
            One of "auto-auto", "cross-cross", or "cross-auto"
        """
        (rs_comb_i, ang_bin_i), (rs_comb_j, ang_bin_j) = self._components
        
        # Check if each correlation function is auto or cross using _is_cov_cross
        # _is_cov_cross[i] = True means rs_comb i is a cross-correlation (different redshift bins)
        # _is_cov_cross[i] = False means rs_comb i is an auto-correlation (same redshift bins)
        is_auto_i = not self.xi_likelihood._is_cov_cross[rs_comb_i]  
        is_auto_j = not self.xi_likelihood._is_cov_cross[rs_comb_j]  
        
        if is_auto_i and is_auto_j:
            correlation_type = "auto-auto"
        elif not is_auto_i and not is_auto_j:
            correlation_type = "cross-cross"
        else:
            correlation_type = "cross-auto"
        
        logger.debug(f"Correlation type: {correlation_type} (rs_comb {rs_comb_i}: auto={is_auto_i}, "
                    f"rs_comb {rs_comb_j}: auto={is_auto_j})")
        logger.debug(f"Involved combinations: {self._involved_combinations}")
        
        return correlation_type
    
    def _build_auto_auto_matrix(self, tset):
        """
        Build combined matrix for auto-auto correlation case.
        
        For two auto-correlations ξ⁺(z_i,z_i) and ξ⁺(z_j,z_j), the covariance 
        structure involves 2 unique redshift bins with 3 combinations:
        - (i,i): auto-covariance for bin i
        - (j,j): auto-covariance for bin j  
        - (i,j): cross-covariance between bins i and j (if i != j)
        
        Parameters:
        -----------
        tset : array_like, shape (2,)
            2D point in characteristic function space (t_i, t_j)
            
        Returns:
        --------
        combined_matrix : array_like, shape (2, 2)
            Combined covariance matrix for the two correlation functions
        """
        products_2d = self._products_2d  # Shape: (n_combinations, 2, size, size)
        involved_combinations = self._involved_combinations
        
        (rs_comb_i, ang_bin_i), (rs_comb_j, ang_bin_j) = self._components
        t_i, t_j = tset
        
        # For auto-auto case, we expect either 1 or 3 combinations:
        # - If 1 combination: same redshift bins (i,i) 
        # - If 3 combinations: different redshift bins (i,i), (j,j), (i,j)
        
        logger.debug(f"Auto-auto case with {len(involved_combinations)} combinations")
        
        if len(involved_combinations) == 1:
            # Same redshift bins - simplified case with only 1 covariance block
            
            # Single auto-correlation covariance block
            auto_product = products_2d[0]  # Shape: (2, size, size)
            
            # Build diagonal matrix using both angular bins
            combined_matrix = t_i * auto_product[0] + t_j * auto_product[1]
           
            
        else:
            # Different redshift bins - full case with 3 combinations
            
            # The involved_combinations are ordered: (min_bin, min_bin), (min_bin, max_bin), (max_bin, max_bin)
            # We need to map these to our specific correlation functions
            
            # Extract the M @ cov products for both angular bins
            auto_product_i = products_2d[0, 0]  # First auto-correlation, first angular bin
            auto_product_j = products_2d[2, 1]  # Second auto-correlation, second angular bin
            # need to check whether this selection is correct (ordering of products_2d)
            # Cross-correlation terms: off-diagonal elements
            cross_product_ij = products_2d[1, 0]  # Cross-covariance, first angular bin
            cross_product_ji = products_2d[1, 1]  # Cross-covariance, second angular bin
            
            # Build the 2x2 combined matrix
            combined_matrix = np.array([
                [t_i * auto_product_i,      t_i * cross_product_ij],
                [t_j * cross_product_ji,    t_j * auto_product_j]
            ])
        
        logger.debug(f"Built auto-auto matrix for t=({t_i:.3f}, {t_j:.3f}): "
                    f"shape {combined_matrix.shape}, det={np.linalg.det(combined_matrix):.3e}")
        
        return combined_matrix
    
    def _build_cross_cross_matrix(self, tset):
        """
        Build combined matrix for cross-cross correlation case.
        
        TODO: Implement cross-cross correlation matrix construction.
        This is more complex due to potentially 4 unique redshift bins.
        """
        logger.debug("Cross-cross matrix construction not yet implemented")
        # Placeholder - return identity for now
        return np.eye(2)
    
    def _build_cross_auto_matrix(self, tset):
        """
        Build combined matrix for cross-auto correlation case.
        
        TODO: Implement cross-auto correlation matrix construction.
        This has intermediate complexity with 2-3 unique redshift bins.
        """
        logger.debug("Cross-auto matrix construction not yet implemented")
        # Placeholder - return identity for now
        return np.eye(2)
    
    # Note: The following methods (_build_2d_matrices_from_components, _build_full_redshift_covariance, 
    # _build_correlation_m_matrix) are no longer needed since we use precomputed _products directly.
    # They are kept for reference in case manual matrix building is needed in the future.

    def __str__(self):
        """
        String representation of the 2D likelihood setup.
        
        Returns
        -------
        str
            Human-readable description of the setup.
        """
        return (f"Exact2DCF(\n"
                f"  redshift_bin_pairs={self._redshift_bin_pairs},\n"
                f"  angular_bins={self._angular_bins},\n"
                f"  t_grid_steps={self._t_grid_info['steps']},\n"
                f"  t_grid_points={len(self._t_grid_info['t_sets'])},\n"
                f"  products_shape={self._products_2d.shape if self._products_2d is not None else None},\n"
                f"  products_available={self._products is not None}\n"
                f")")
    
    def __repr__(self):
        """
        Developer representation of the 2D likelihood setup.
        
        Returns
        -------
        str
            Detailed representation for debugging.
        """
        return (f"Exact2DCF(redshift_bin_pairs={self._redshift_bin_pairs}, "
                f"angular_bins={self._angular_bins}, steps={self._t_grid_info['steps']})")


class Exact2DLikelihood:
    """
    Complete 2D likelihood computation pipeline.
    
    This class combines exact characteristic function computation with Gaussian CF 
    multiplication and PDF conversion to provide full likelihood evaluation for 
    2D correlation function pairs.
    
    TODO: Add CF loading functionality - CF computation is very heavy and should 
          be done once, then saved/loaded for multiple PDF computations.
          Need methods like:
          - load_cf_from_file(cf_path)
          - compute_pdf_from_loaded_cf(gaussian_cf_params=None)
          - Support for pre-computed CF results as input to avoid recomputation
    
    Example:
    --------
    # Create instance for specific correlation function pair
    exact_likelihood = Exact2DLikelihood(
        xi_likelihood_instance=likelihood,
        redshift_bin_pairs=((0, 1), (2, 3)),
        angular_bins=(5, 10),
        steps=129
    )
    
    # Compute full likelihood
    log_likelihood = exact_likelihood.compute_likelihood(data_vector)
    
    # Or get intermediate results
    pdf_results = exact_likelihood.compute_pdf()
    
    # TODO: Future usage with pre-computed CF
    # exact_likelihood.load_cf_from_file("heavy_cf_computation.npz")
    # pdf_results = exact_likelihood.compute_pdf_from_loaded_cf()
    """
    
    def __init__(self, xi_likelihood_instance, redshift_bin_pairs, angular_bins, steps=129, cosmo=None):
        """
        Initialize the 2D likelihood pipeline.
        
        Parameters:
        -----------
        xi_likelihood_instance : XiLikelihood
            XiLikelihood instance (cosmology will be initialized if needed)
        redshift_bin_pairs : tuple of tuples
            Two redshift bin combinations as ((bin_i1, bin_i2), (bin_j1, bin_j2))
        angular_bins : tuple of int
            Angular bin indices for each correlation function (ang_bin_i, ang_bin_j)
        steps : int, default 129
            Number of t-grid steps per dimension
        cosmo : optional
            Cosmology object. If None and XiLikelihood not initialized, will raise error.
        """
        # Create the CF computer as a component
        self.cf_computer = Exact2DCF(
            xi_likelihood_instance, redshift_bin_pairs, angular_bins, steps, cosmo
        )
        
        # Store parameters for likelihood computation
        self.redshift_bin_pairs = redshift_bin_pairs
        self.angular_bins = angular_bins
        self.steps = steps
        
        # TODO: Add CF caching/loading functionality
        self._loaded_cf_results = None  # For storing pre-computed CF results
        
        logger.info(f"Initialized Exact2DLikelihood pipeline for "
                   f"redshift bins {redshift_bin_pairs}, angular bins {angular_bins}")
    
    def compute_pdf(self, gaussian_cf_params=None, save_path=None):
        """
        Compute 2D PDF from characteristic function.
        
        Parameters:
        -----------
        gaussian_cf_params : dict, optional
            Parameters for Gaussian CF multiplication for high-ell modes.
            If None, no Gaussian multiplication is applied.
        save_path : str, optional
            Path to save PDF results. If None, results are not saved.
            
        Returns:
        --------
        dict
            PDF results with keys:
            - 'pdf_values': 2D PDF array
            - 'xi_grid': correlation function grid points
            - 'pdf_info': metadata about PDF computation
        """
        # Step 1: Get exact CF grid
        logger.info("Computing exact 2D characteristic function...")
        cf_results = self.cf_computer.compute_2d_exact_cf()
        
        # Step 2: Apply Gaussian CF for high-ell modes (if specified)
        if gaussian_cf_params:
            logger.info("Applying Gaussian CF multiplication for high-ell modes...")
            cf_results = self._apply_gaussian_cf(cf_results, gaussian_cf_params)
        
        # Step 3: Convert CF to PDF
        logger.info("Converting characteristic function to PDF...")
        pdf_results = self._cf_to_pdf(cf_results)
        
        # Step 4: Save if requested
        if save_path:
            self._save_pdf_results(pdf_results, save_path)
        
        return pdf_results
    
    def load_cf_from_file(self, cf_path):
        """
        Load pre-computed CF results from file to avoid heavy recomputation.
        
        TODO: Implement CF loading functionality.
        CF computation can be very expensive (large t-grids, many matrix eigenvalue 
        computations), so this allows computing CF once and reusing for multiple 
        PDF computations with different parameters.
        
        Parameters:
        -----------
        cf_path : str
            Path to saved CF results (.npz file from compute_2d_exact_cf)
            
        Notes:
        ------
        This is critical for practical usage since CF computation can take hours
        while PDF conversion and likelihood evaluation are much faster.
        """
        # TODO: Implement loading logic
        # self._loaded_cf_results = np.load(cf_path, allow_pickle=True)
        # Validate that loaded CF matches current setup (redshift bins, angular bins, etc.)
        logger.debug(f"CF loading not yet implemented (would load from {cf_path})")
        raise NotImplementedError("CF loading functionality needed - CF computation is very heavy!")
    
    def compute_pdf_from_loaded_cf(self, gaussian_cf_params=None, save_path=None):
        """
        Compute PDF from pre-loaded CF results instead of recomputing CF.
        
        TODO: Implement PDF computation from loaded CF.
        This enables the workflow: compute heavy CF once → load CF → fast PDF computation.
        
        Parameters:
        -----------
        gaussian_cf_params : dict, optional
            Parameters for Gaussian CF multiplication for high-ell modes
        save_path : str, optional
            Path to save PDF results
            
        Returns:
        --------
        dict
            PDF results
        """
        if self._loaded_cf_results is None:
            raise RuntimeError("No CF results loaded. Call load_cf_from_file() first or use compute_pdf().")
        
        # TODO: Implement PDF computation from loaded CF
        logger.debug("PDF computation from loaded CF not yet implemented")
        raise NotImplementedError("PDF from loaded CF functionality needed!")
    
    def compute_likelihood(self, data_vector, gaussian_cf_params=None):
        """
        Compute exact 2D likelihood value.
        
        Parameters:
        -----------
        data_vector : array_like, shape (2,)
            Observed correlation function values for the two angular bins
        gaussian_cf_params : dict, optional
            Parameters for Gaussian CF multiplication for high-ell modes
            
        Returns:
        --------
        float
            Log-likelihood value
        """
        # Get PDF
        pdf_results = self.compute_pdf(gaussian_cf_params)
        
        # Evaluate likelihood
        log_likelihood = self._evaluate_likelihood(pdf_results, data_vector)
        
        logger.info(f"Computed 2D likelihood: {log_likelihood:.6f}")
        return log_likelihood
    
    def _apply_gaussian_cf(self, cf_results, gaussian_params):
        """
        Multiply CF with Gaussian CF for high-ell modes.
        
        TODO: Implement Gaussian CF multiplication logic.
        This would use existing functions from the codebase.
        """
        # Placeholder - would implement Gaussian CF multiplication
        logger.debug("Gaussian CF multiplication not yet implemented")
        return cf_results
    
    def _cf_to_pdf(self, cf_results):
        """
        Convert characteristic function to probability density function.
        
        TODO: Implement CF to PDF conversion logic.
        This would use existing functions from the codebase.
        """
        # Placeholder - would implement CF to PDF conversion
        logger.debug("CF to PDF conversion not yet implemented")
        return {
            'pdf_values': None,  # Would be 2D PDF array
            'xi_grid': None,     # Would be correlation function grid
            'pdf_info': {'status': 'placeholder'}
        }
    
    
    
    def _save_pdf_results(self, pdf_results, save_path):
        """Save PDF results to file."""
        # Placeholder - would save PDF results
        logger.debug(f"PDF saving not yet implemented (would save to {save_path})")
    
    def __str__(self):
        """String representation of the likelihood pipeline."""
        return (f"Exact2DLikelihood(\n"
                f"  redshift_bin_pairs={self.redshift_bin_pairs},\n"
                f"  angular_bins={self.angular_bins},\n"
                f"  cf_computer={self.cf_computer}\n"
                f")")
    
    def __repr__(self):
        """Developer representation of the likelihood pipeline."""
        return (f"Exact2DLikelihood(redshift_bin_pairs={self.redshift_bin_pairs}, "
                f"angular_bins={self.angular_bins}, steps={self.steps})")
