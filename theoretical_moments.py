"""
Statistical moments computation from characteristic functions as well as the analytical solution 
for the moments of two-point-function characteristic functions.

This module provides utilities for computing statistical moments (mean, variance,
skewness, etc.) from characteristic functions using spline interpolation and
numerical differentiation.
"""

import numpy as np
from scipy.interpolate import UnivariateSpline
import itertools
from jax import jit
import jax.numpy as jnp

__all__ = [
    'nth_moment',
    'skewness',
    'get_moments_from_combination_matrix_1d',
    'get_moments_from_combination_matrix_nd', 
    'convert_moments_to_cumulants_nd',
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


def get_moments_from_combination_matrix_1d(m, cov):
    """
    Compute moments from combination matrix and covariance (1D case).
    
    This function computes moments for correlation functions derived from
    combination matrices (used in cosmic shear pseudo-C_l analysis).
    
    Parameters
    ----------
    m : array_like
        Combination matrix relating observables to underlying fields
    cov : array_like  
        Covariance matrix of underlying fields
        
    Returns
    -------
    list
        [mean, second_moment, third_moment]
    """
    if np.array_equal(m, np.diag(np.diag(m))):
        m_diag = np.diag(m)
        mean = np.dot(m_diag, np.diag(cov))
        prod = m_diag[:, None] * cov
    else:
        prod = m @ cov
        mean = np.trace(prod)

    var = 2 * np.sum(prod * np.transpose(prod))
    second = var + mean**2

    prod2 = prod @ prod
    third = 8 * np.sum(prod2 * np.transpose(prod)) + mean**3 + 3 * mean * var

    moments = [mean, second, third]

    return moments


def get_moments_from_combination_matrix_nd(m, cov, ndim, n_moments=3):
    """
    Calculate first three moments for an n-dimensional likelihood.

    Parameters
    ----------
    m : list of 2d arrays
        list of combination matrices, order defines order of dimensions
    cov : 2d array
        covariance matrix for all pseudo alm needed for all dimensions

    Returns
    -------
    list
        list of lists of first, second and third moments. second moments are given out in the shape of a covariance matrix.
    """
    assert ndim == m.shape[0], "list of matrices m length {} does not align with ndim={}".format(
        m.shape[0], ndim
    )
    if ndim > 100:
        print("Warning: ndim > 100")
    dims = jnp.arange(ndim)

    prods = jnp.einsum("dij,jk->dik", m, cov)

    def first_moments():

        return jnp.einsum("dii->d", prods)

    def second_moments(firsts):
        seconds = jnp.zeros((ndim, ndim), dtype="float64")
        combs = jnp.array(list(itertools.combinations_with_replacement(dims, 2)))

        one, two = combs[:, 0], combs[:, 1]

        second_moment_values = firsts[one] * firsts[two] + 2 * jnp.einsum(
            "dij,dji->d", prods[one], prods[two]
        )
        seconds = seconds.at[one, two].set(second_moment_values)
        seconds = seconds.at[two, one].set(second_moment_values)

        return seconds

    def third_moments(firsts):
        combs = jnp.array(list(itertools.combinations_with_replacement(dims, 3)))

        i, j, k = combs[:, 0], combs[:, 1], combs[:, 2]

        third_moment_values = (
            jnp.prod(jnp.array([firsts[i], firsts[j], firsts[k]]), axis=0)
            + 2
            * (
                jnp.einsum("ijk,ikj->i", prods[j], prods[k]) * firsts[i]
                + jnp.einsum("ijk,ikj->i", prods[k], prods[i]) * firsts[j]
                + jnp.einsum("ijk,ikj->i", prods[i], prods[j]) * firsts[k]
            )
            + 8 * jnp.einsum("dij,djk,dki->d", prods[i], prods[j], prods[k])
        )

        return jnp.ravel(third_moment_values)

    def third_moments_old(firsts, seconds):
        thirds = []
        # define n-dim 3rd order hermite polynomials in exactly the same manner / order!
        for comb in itertools.combinations_with_replacement(dims, 3):
            i, j, k = comb
            prod2 = prods[j] @ prods[k]
            third_moment = (
                np.prod([firsts[m] for m in comb])
                + 2
                * (
                    np.sum(
                        [
                            firsts[one] * np.sum(prods[two] * np.transpose(prods[three]))
                            for (one, two, three) in [(i, j, k), (k, i, j), (j, i, k)]
                        ]
                    )
                )
                + 8 * np.sum(prods[i] * np.transpose(prod2))
            )
            thirds.append(third_moment)
        thirds = np.array(thirds)
        return thirds

    if n_moments == 1:
        return first_moments()
    elif n_moments == 2:
        firsts = first_moments()
        return firsts, second_moments(firsts)
    elif n_moments == 3:

        firsts = first_moments()
        seconds = second_moments(firsts)
        thirds = third_moments(firsts)
        return firsts, seconds, thirds
    else:
        raise ValueError("n > 3 not implemented")


moments_nd_jitted = jit(get_moments_from_combination_matrix_nd, static_argnums=(2, 3))

def convert_moments_to_cumulants_nd(moments):
    """
    Calculate cumulants from non central moments for an n-dimensional distribution.
    Up to third order cumulants are calculated (same as central moments).

    Parameters
    ----------
    moments : list
        list of lists of first, second and third moments. second moments are given in the shape of a covariance matrix.

    Returns
    -------
    list
        list of lists of first, second and third cumulants. second cumulants are given out in the shape of a covariance matrix.
    """
    n = len(moments)
    assert (
        n <= 3
    ), "ncmom2cum_nd: moments list too long, only implemented up to third order cumulants"
    conversion_functions = cumulant_generator(n)
    cumulants = [c_function(moments) for c_function in conversion_functions]

    return cumulants


def select_conversion_function(n):
    def first_cumulant(moments):
        return moments[0]

    def second_cumulant(moments):
        # cumulants and (central!) moments should be the same at second order - we use non-central moments
        first = moments[0]
        return moments[1] - np.outer(first, first)

    def third_cumulant(moments):
        first = moments[0]
        dims = np.arange(len(first))
        second = moments[1]
        third = moments[2]
        third_cumulants = []
        for n, comb in enumerate(itertools.combinations_with_replacement(dims, 3)):
            i, j, k = comb
            kappa_ijk = (
                third[n]
                - np.sum(
                    [
                        first[one] * second[two, three]
                        for (one, two, three) in [(i, j, k), (j, i, k), (k, i, j)]
                    ]
                )
                + 2 * np.prod([first[m] for m in comb])
            )
            third_cumulants.append(kappa_ijk)
        return third_cumulants

    if n == 0:
        return first_cumulant
    elif n == 1:
        return second_cumulant
    elif n == 2:
        return third_cumulant
    else:
        raise ValueError("n > 2 not implemented")


def cumulant_generator(N):
    n = 0
    while n < N:
        yield select_conversion_function(n)
        n += 1