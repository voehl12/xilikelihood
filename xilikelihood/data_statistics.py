"""
Statistical utilities for bootstrap sampling and moment calculations.

This module provides tools for statistical analysis of simulation data,
including bootstrap resampling, moment calculations, and comparison utilities
for validating theoretical predictions against simulation results.

Main Functions
--------------
bootstrap : Bootstrap resampling with custom statistics
bootstrap_2d : Bootstrap for 2D histogram statistics
compute_simulation_moments : Calculate moments from simulation data
compare_theory_vs_simulations : Compare theoretical and simulated moments

Examples
--------
>>> import numpy as np
>>> data = np.random.normal(0, 1, (1000, 10))
>>> boot_vars = bootstrap(data, n=100, func=np.var)
>>> print(f"Bootstrap variance estimates: {boot_vars.mean():.3f} ± {boot_vars.std():.3f}")
"""

import numpy as np
import itertools
import logging

logger = logging.getLogger(__name__)

__all__ = [
    # Bootstrap functions
    'bootstrap',
    'bootstrap_2d',
    
    # Moment calculations
    'compute_simulation_moments',
    'compute_moments_nd',
    
    # Comparison utilities
    'compare_theory_vs_simulations',
]

def bootstrap(data, n, axis=0, func=np.var, func_kwargs={"ddof": 1}):
    """Produce n bootstrap samples of data of the statistic given by func.

    Arguments
    ---------
    data : numpy.ndarray
        Data to resample.
    n : int
        Number of bootstrap trails.
    axis : int, optional
        Axis along which to resample. (Default ``0``).
    func : callable, optional
        Statistic to calculate. (Default ``numpy.var``).
    func_kwargs : dict, optional
        Dictionary with extra arguments for func. (Default ``{"ddof" : 1}``).

    Returns
    -------
    samples : numpy.ndarray
        Bootstrap samples of statistic func on the data.
    """

    fiducial_output = func(data, axis=axis, **func_kwargs)

    if isinstance(data, list):
        if axis != 0:
            raise NotImplementedError("Only axis == 0 supported for list inputs")
        
        # Validate list shapes
        shapes = [d.shape[1:] for d in data]
        if not all(shape == shapes[0] for shape in shapes):
            raise ValueError("All arrays in list must have matching shapes except along axis 0")


    samples = np.zeros((n, *fiducial_output.shape), dtype=fiducial_output.dtype)

    progress_interval = max(1, n // 10)

    for i in range(n):
        if i % progress_interval == 0:
            progress = i / n * 100
            logger.info(f"Bootstrap progress: {progress:.1f}% ({i}/{n})")
        
        if isinstance(data, list):
            idx = [np.random.choice(d.shape[0], size=d.shape[0], replace=True) for d in data]
            samples[i] = func([d[i] for d, i in zip(data, idx)], axis=axis, **func_kwargs)
        else:
            idx = np.random.choice(data.shape[axis], size=data.shape[axis], replace=True)
            idx_tuple = tuple(idx if ax == axis else slice(None) for ax in range(data.ndim))
            samples[i] = func(data[idx_tuple], axis=axis, **func_kwargs)
            
    logger.info("Bootstrap sampling completed")
    return samples


def bootstrap_statistic_2d(data, binedges=None,axis=0):
        x,y = data
        hist,_,_ = np.histogram2d(
            x,y,
            bins=binedges,
            density=True,
        )
        return hist.T





def compute_simulation_moments(sims, orders=[1, 2, 3], axis=1,center_moments=True):
    """
    Compute statistical moments from simulation data.
    
    Parameters
    ----------
    simulations : ndarray
        Simulation data with shape (..., n_sims, ...)
    orders : list of int
        Which moment orders to compute
    axis : int
        Axis along which simulations vary
    center_moments : bool
        Whether to compute central moments (subtract mean)
        
    Returns
    -------
    moments : ndarray or list of ndarrays
        Computed moments. If single order requested, returns array.
        Otherwise returns list of arrays.
    """
    dims = np.arange(sims.shape[axis - 1])
    n_sims = sims.shape[axis]
    logger.info(f"Computing moments from {n_sims} simulations")

    # np.cov needs to have the data in the form of (n, m) where n is the number of variables and m is the number of samples
    def _moments_nd(order, sims):
        """Compute specific moment order."""
        if order == 1:
            return np.mean(sims, axis=axis)
        elif order == 2:
            if center_moments:
                sims_centered = sims - np.mean(sims, axis=axis, keepdims=True)
            else:
                sims_centered = sims
            
            if axis == 0:
                sims_centered = sims_centered.T
            
            cov = cov = sims_centered @ sims_centered.T.conj() / (n_sims - 1)
            np_cov = np.cov(sims_centered, ddof=1)
            if not np.allclose(cov, np_cov):
                logger.warning("Covariance calculation mismatch with numpy.cov")
            
            return cov

        else:

            if center_moments:
                sims_centered = sims - np.mean(sims, axis=axis, keepdims=True)
            else:
                sims_centered = sims
                
            if axis == 0:
                sims_centered = sims_centered.T
            
            # Get all combinations with replacement
            combinations = list(itertools.combinations_with_replacement(dims, order))
            combs_array = np.array(combinations)
            
            # Compute moment for each combination
            moment_values = np.mean(np.prod(sims_centered[combs_array], axis=1), axis=1)
            return np.ravel(moment_values)

    stats = [_moments_nd(order, sims) for order in orders]
    if len(orders) == 1:
        return np.array(stats)
    else:
        return stats



def compare_theory_vs_simulations(t_grid, theory_cf, sim_data, n_moments=3):
    """Compare theoretical CF moments to simulation moments."""
    from theoretical_moments import nth_moment
    
    theory_moments = nth_moment(n_moments, t_grid, theory_cf)
    sim_moments = compute_simulation_moments(sim_data, orders=list(range(1, n_moments+1)))
    
    return {
        'theory': [m.real for m in theory_moments],
        'simulation': sim_moments,
        'difference': [t.real - s for t, s in zip(theory_moments, sim_moments)]
    }