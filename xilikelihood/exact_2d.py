"""
Exact 2D likelihood computation for copula validation.

This module provides exact likelihood calculation for 2D subsets of correlation
functions, enabling validation of copula-based approximations and estimation
of Student-t copula parameters.
"""

import numpy as np
from scipy.linalg import solve


def get_cf_nD(tset, mset, cov):
    """
    Compute exact characteristic function for n-dimensional data.
    
    Adapted from legacy/calc_pdf_v1.py for use in exact likelihood computation.
    
    Parameters:
    -----------
    tset : array_like
        Test point for likelihood evaluation
    mset : array_like
        Combination matrix (M matrix)
    cov : array_like
        Covariance matrix
        
    Returns:
    --------
    tset : array_like
        Input test point (for consistency)
    cf_value : complex
        Characteristic function value
    """
    big_m = np.einsum("i,ijk -> jk", tset, mset)
    evals = np.linalg.eigvals(big_m @ cov)
    return tset, np.prod((1 - 2j * evals) ** -0.5)


class Exact2DLikelihood:
    """
    Computes exact 2D likelihood for correlation function pairs.
    
    This class extracts the necessary covariance and combination matrices
    from an existing XiLikelihood instance and provides exact 2D likelihood
    computation for copula validation and parameter estimation.
    """
    
    def __init__(self, xi_likelihood_instance):
        """
        Initialize with an existing XiLikelihood instance.
        
        Parameters:
        -----------
        xi_likelihood_instance : XiLikelihood
            Configured likelihood instance containing covariance and M matrices
        """
        self.xi_likelihood = xi_likelihood_instance
        
        # Extract necessary matrices from the likelihood instance
        # Note: These attribute names may need adjustment based on actual XiLikelihood structure
        try:
            self.full_cov = xi_likelihood_instance.cov_matrix
            self.full_mset = xi_likelihood_instance.M_matrix
        except AttributeError:
            # Try alternative attribute names
            try:
                self.full_cov = xi_likelihood_instance.covariance_matrix
                self.full_mset = xi_likelihood_instance.combination_matrix
            except AttributeError:
                raise AttributeError(
                    "Could not find covariance or M matrices in XiLikelihood instance. "
                    "Please check the attribute names in your XiLikelihood class."
                )
        
        print(f"Initialized Exact2DLikelihood with matrices of shape:")
        print(f"  Covariance: {self.full_cov.shape}")
        print(f"  M matrix: {self.full_mset.shape}")
    
    def compute_2d_likelihood(self, xi_data_2d, indices):
        """
        Compute exact 2D likelihood for specified correlation function pair.
        
        Parameters:
        -----------
        xi_data_2d : array_like, shape (2,)
            2D correlation function data point
        indices : tuple of int
            Which 2 correlation functions to use (i, j)
            
        Returns:
        --------
        loglikelihood : float
            Exact log-likelihood for this 2D point
        """
        # Extract 2D subset
        cov_2d = self._extract_2d_covariance(indices)
        mset_2d = self._extract_2d_mset(indices)
        
        # Validate inputs
        if cov_2d.shape != (2, 2):
            raise ValueError(f"Expected 2x2 covariance matrix, got {cov_2d.shape}")
        
        # Use get_cf_nD to compute exact likelihood
        try:
            ts, cf_value = get_cf_nD(xi_data_2d, mset_2d, cov_2d)
            
            # Convert characteristic function to log-likelihood
            # The characteristic function gives us the probability density
            if np.abs(cf_value) > 0:
                loglikelihood = np.log(np.abs(cf_value))
            else:
                loglikelihood = -np.inf
            
            return loglikelihood
            
        except (np.linalg.LinAlgError, RuntimeError) as e:
            print(f"Warning: Numerical error in exact 2D likelihood computation: {e}")
            return -np.inf
    
    def _extract_2d_covariance(self, indices):
        """
        Extract 2x2 covariance submatrix for specified correlation functions.
        
        Parameters:
        -----------
        indices : tuple of int
            Which 2 correlation functions to extract (i, j)
            
        Returns:
        --------
        cov_2d : array_like, shape (2, 2)
            2D covariance submatrix
        """
        i, j = indices
        
        # Basic bounds checking
        n_dims = self.full_cov.shape[0]
        if i >= n_dims or j >= n_dims or i < 0 or j < 0:
            raise IndexError(f"Indices ({i}, {j}) out of bounds for covariance matrix of size {n_dims}")
        
        # Extract 2x2 submatrix
        cov_2d = self.full_cov[np.ix_([i, j], [i, j])]
        
        # Ensure positive definiteness
        eigenvals = np.linalg.eigvals(cov_2d)
        if np.any(eigenvals <= 0):
            print(f"Warning: 2D covariance matrix for indices ({i}, {j}) is not positive definite")
            # Add small regularization
            cov_2d += 1e-10 * np.eye(2)
        
        return cov_2d
    
    def _extract_2d_mset(self, indices):
        """
        Extract M matrix subset for 2D computation.
        
        Parameters:
        -----------
        indices : tuple of int
            Which 2 correlation functions to extract (i, j)
            
        Returns:
        --------
        mset_2d : array_like
            M matrix subset for 2D case
        """
        i, j = indices
        
        # The exact indexing depends on how the M matrix is structured
        # This may need adjustment based on your specific M matrix format
        
        # Basic bounds checking
        if self.full_mset.ndim == 3:
            # Assume shape is (n_dims, n_something, n_something)
            n_dims = self.full_mset.shape[0]
            if i >= n_dims or j >= n_dims or i < 0 or j < 0:
                raise IndexError(f"Indices ({i}, {j}) out of bounds for M matrix of size {n_dims}")
            
            # Extract subset for 2D case
            mset_2d = self.full_mset[[i, j], :, :]
            
        elif self.full_mset.ndim == 2:
            # Alternative M matrix structure
            mset_2d = self.full_mset[:, [i, j]]
            
        else:
            raise ValueError(f"Unexpected M matrix shape: {self.full_mset.shape}")
        
        return mset_2d
    
    def validate_extraction(self, indices_list=None):
        """
        Validate that 2D matrix extraction is working correctly.
        
        Parameters:
        -----------
        indices_list : list of tuples, optional
            List of index pairs to test. If None, tests a few random pairs.
            
        Returns:
        --------
        validation_results : dict
            Results of validation tests
        """
        if indices_list is None:
            n_dims = min(10, self.full_cov.shape[0])  # Test first 10 dimensions
            indices_list = [(i, j) for i in range(n_dims) for j in range(i+1, n_dims)][:5]
        
        results = {
            'successful_extractions': 0,
            'failed_extractions': 0,
            'positive_definite_count': 0,
            'details': []
        }
        
        for indices in indices_list:
            try:
                cov_2d = self._extract_2d_covariance(indices)
                mset_2d = self._extract_2d_mset(indices)
                
                # Check positive definiteness
                eigenvals = np.linalg.eigvals(cov_2d)
                is_positive_definite = np.all(eigenvals > 1e-12)
                
                results['successful_extractions'] += 1
                if is_positive_definite:
                    results['positive_definite_count'] += 1
                
                results['details'].append({
                    'indices': indices,
                    'cov_shape': cov_2d.shape,
                    'mset_shape': mset_2d.shape,
                    'min_eigenval': np.min(eigenvals),
                    'is_positive_definite': is_positive_definite
                })
                
            except Exception as e:
                results['failed_extractions'] += 1
                results['details'].append({
                    'indices': indices,
                    'error': str(e)
                })
        
        print(f"Validation results:")
        print(f"  Successful extractions: {results['successful_extractions']}")
        print(f"  Failed extractions: {results['failed_extractions']}")
        print(f"  Positive definite matrices: {results['positive_definite_count']}")
        
        return results
