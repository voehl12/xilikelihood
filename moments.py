"""
Statistical moments computation from characteristic functions.

This module provides utilities for computing statistical moments (mean, variance,
skewness, etc.) from characteristic functions using spline interpolation and
numerical differentiation.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline

__all__ = [
    'nth_moment',
    'skewness'
]


def nth_moment(n, t, cf):
    """
    Compute the first n moments from a characteristic function.
    
    Uses spline interpolation and numerical differentiation to compute moments
    from the characteristic function CF(t) via the relationship:
    E[X^k] = (-i)^k * d^k/dt^k CF(t) |_{t=0}
    
    Parameters
    ----------
    n : int
        Number of moments to compute (1 to n)
    t : array_like
        Sample points where characteristic function is evaluated
    cf : array_like (complex)
        Characteristic function values at points t
        
    Returns
    -------
    list of complex
        First n moments: [E[X], E[X^2], ..., E[X^n]]
        
    Examples
    --------
    >>> t = np.linspace(-5, 5, 1000)
    >>> cf = np.exp(-0.5 * t**2)  # Standard normal CF
    >>> moments = nth_moment(3, t, cf)
    >>> print(f"Mean: {moments[0].real:.6f}")  # Should be ~0
    >>> print(f"Variance: {moments[1].real:.6f}")  # Should be ~1
    
    Notes
    -----
    Uses 5th degree spline interpolation for numerical stability.
    Higher-order moments may become numerically unstable.
    """

    # Input validation
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    
    t = np.asarray(t)
    cf = np.asarray(cf)
    
    if len(t) != len(cf):
        raise ValueError("t and cf must have the same length")
    
    if len(t) < 2 * n + 5:  # Need enough points for spline interpolation
        raise ValueError(f"Need at least {2 * n + 5} points for {n} moments with degree-5 spline")
    
    if not np.allclose(cf[0], cf[-1]) and abs(t[0] + t[-1]) > 1e-10:
        import warnings
        warnings.warn("Characteristic function should be symmetric around t=0 for best results")
    

    k = 5  # 5th degree spline for numerical stability
    
    # Compute derivatives at t=0 for real and imaginary parts separately
    derivs_at_zero = [
        1j * UnivariateSpline(t, cf.imag, k=k, s=0).derivative(n=i)(0)
        + UnivariateSpline(t, cf.real, k=k, s=0).derivative(n=i)(0)
        for i in range(1, n + 1)
    ]
    
    # Apply the moment formula: E[X^k] = (-i)^k * d^k/dt^k CF(t) |_{t=0}
    return [(1j**-k * derivs_at_zero[k - 1]) for k in range(1, n + 1)]


def skewness(t, cf):
    """
    Compute skewness from a characteristic function.
    
    Skewness measures the asymmetry of a probability distribution.
    Computed as the third central moment divided by the cube of the standard deviation.
    
    Parameters
    ----------
    t : array_like
        Sample points where characteristic function is evaluated
    cf : array_like (complex)
        Characteristic function values at points t
        
    Returns
    -------
    float
        Skewness of the distribution
        
    Examples
    --------
    >>> t = np.linspace(-5, 5, 1000)
    >>> cf = np.exp(-0.5 * t**2)  # Standard normal CF
    >>> skew = skewness(t, cf)
    >>> print(f"Skewness: {skew:.6f}")  # Should be ~0 for normal distribution
    
    Notes
    -----
    For a normal distribution, skewness = 0.
    Positive skewness indicates a longer tail on the right side.
    """
    # Input validation
    t = np.asarray(t)
    cf = np.asarray(cf)
    
    if len(t) != len(cf):
        raise ValueError("t and cf must have the same length")
    
    first, second, third = nth_moment(3, t, cf)
    
    # Convert to real parts (moments should be real for real distributions)
    mu = first.real  # mean
    second_moment = second.real
    third_moment = third.real
    
    # Compute variance and third central moment
    sigma2 = second_moment - mu**2  # variance
    third_central = third_moment - 3 * mu * sigma2 - mu**3  # third central moment
    
    return third_central / np.sqrt(sigma2)**3