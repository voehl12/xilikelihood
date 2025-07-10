"""
Pseudo-alm covariance calculation utilities.

This module provides optimized functions for computing covariances of pseudo
spherical harmonic coefficients (pseudo-alm) used in cosmological analyses
with incomplete sky coverage.

Key features:
- JAX-optimized tensor operations for performance
- Support for E/B mode decomposition
- Efficient einsum operations with precomputation
- Handles complex alm coefficient correlations

Mathematical background:
Pseudo-alm coefficients arise when computing spherical harmonic transforms
on masked sky maps. Their covariances differ from full-sky case due to
mode coupling induced by the mask.
"""

import numpy as np
import jax.numpy as jnp
import jax
from jax import lax

__all__ = [
    # Core covariance functions
    'cov_4D',
    'cov_4D_jit',
    'optimized_cov_4D',
    'optimized_cov_4D_jit',
    
    # Utility functions
    'match_alm_inds',
    'w_stack',
    'c_ell_stack',
    'delta_mppmppp',
    
    # Optimization utilities
    'precompute_xipm',
    'precompute_einsum',
]

# ============================================================================
# Index Mapping and Selection Utilities
# ============================================================================


def match_alm_inds(alm_keys):
    """
    Convert alm string keys to numerical indices.
    
    Parameters:
    -----------
    alm_keys : list
        List of alm mode strings (e.g., ['ReE', 'ImE', 'ReB'])
        
    Returns:
    --------
    list
        Sorted list of corresponding numerical indices
    """
    alm_keys = list(set(alm_keys))
    alm_dict = {"ReE": 0, "ImE": 1, "ReB": 2, "ImB": 3, "ReT": 4, "ImT": 5}
    alm_inds = [alm_dict[str(alm_keys[i])] for i in range(len(alm_keys))]
    # Check for invalid keys
    invalid_keys = [key for key in alm_keys if key not in alm_dict]
    if invalid_keys:
        valid_keys = list(alm_dict.keys())
        raise ValueError(
            f"Invalid alm mode strings: {invalid_keys}. "
            f"Valid modes are: {valid_keys}"
        )
    
    alm_inds.sort()
    return alm_inds


def sel_perm(I):
    """
    Select linear combination of full-sky alm for given pseudo-alm.
    
    Maps pseudo-alm indices to combinations of:
    - Sign (+/-)
    - Real/Imaginary part of W matrix
    - W matrix type (+/-/0)  
    - Real/Imaginary part of alm
    - Polarization type (E/B/T)
    
    Parameters:
    -----------
    I : int
        Pseudo-alm index (0-5 for ReE, ImE, ReB, ImB, ReT, ImT)
        
    Returns:
    --------
    jnp.ndarray
        Array of 5-tuples specifying W-alm combinations

    Notes:
    -----
    Currently only E/B modes (I=0-3) are fully implemented.
    T modes (I=4-5) are work in progress.
    """

    i = [
        [(0, 0, 0, 0, 0), (1, 1, 0, 1, 0), (0, 0, 1, 0, 1), (1, 1, 1, 1, 1)],
        [(0, 0, 0, 1, 0), (0, 1, 0, 0, 0), (0, 0, 1, 1, 1), (0, 1, 1, 0, 1)],
        [(0, 0, 0, 0, 1), (1, 1, 0, 1, 1), (1, 0, 1, 0, 0), (0, 1, 1, 1, 0)],
        [(0, 0, 0, 1, 1), (0, 1, 0, 0, 1), (1, 0, 1, 1, 0), (1, 1, 1, 0, 0)],
        [(0, 0, 2, 0, 2), (1, 1, 2, 1, 2)],
        [(0, 0, 2, 1, 2), (0, 1, 2, 0, 2)],
    ]

    def ere():
        return jnp.array(i[0])

    def eim():
        return jnp.array(i[1])

    def bre():
        return jnp.array(i[2])

    def bim():
        return jnp.array(i[3])

    def tre():
        return jnp.array(i[4])

    def tim():
        return jnp.array(i[5])

    return lax.switch(I, [ere, eim, bre, bim])

# ============================================================================
# Matrix Stacking and Manipulation
# ============================================================================


def part(j, arr):
    return jax.lax.cond(j == 0, lambda _: jnp.real(arr), lambda _: jnp.imag(arr), operand=None)


def w_stack(I, w_arr):
    """
    Stack W-arrays according to pseudo-alm permutation.
    
    Parameters:
    -----------
    I : int
        Pseudo-alm index
    w_arr : jnp.ndarray
        W-matrix array
        
    Returns:
    --------
    jnp.ndarray
        Stacked W-array with appropriate signs and parts
    """
    # 0: +/-, 1: Re/Im W, 2: W+/-/0, 3: Re/Im a, 4: E/B/0 a
    # i is going to be the first axis of W

    perm = sel_perm(I)

    def process_permutation(p):
        fac = (-1) ** p[0]
        wpart = part(p[1], w_arr)
        wpm0 = wpart[p[2]]
        return fac * wpm0

    w_s = jax.vmap(process_permutation)(perm)
    return w_s


def delta_mppmppp(I, len_m):
    """
    Create Kronecker delta matrix for alm covariance structure.
    
    Accounts for correlations between alm coefficients with same |m|
    but different signs, including special treatment for m=0.
    
    Parameters:
    -----------
    I : int
        Pseudo-alm index
    len_m : int
        Length of m-dimension
        
    Returns:
    --------
    jnp.ndarray
        Delta matrix array
    """
    m_max = int((len_m - 1) / 2)
    m0_ind = m_max
    perm = sel_perm(I)

    diag = jnp.arange(len_m)

    def fill_d_array(p):
        d_array = jnp.eye(len_m)
        d_array = jax.lax.cond(
            p[3] == 0,
            lambda _: d_array.at[diag, jnp.flip(diag)].set(
                (-1) ** (jnp.fabs(jnp.arange(-m_max, m_max + 1)))
            ),
            lambda _: d_array.at[diag, jnp.flip(diag)].set(
                (-1) ** (jnp.fabs(jnp.arange(-m_max, m_max + 1)) + 1)
            ),
            operand=None,
        )
        d_array = jax.lax.cond(
            p[3] == 0,
            lambda _: d_array.at[m0_ind, m0_ind].set(2),
            lambda _: d_array.at[m0_ind, m0_ind].set(0),
            operand=None,
        )
        return d_array

    def process_permutation(p):
        return fill_d_array(p)

    deltas = jax.vmap(process_permutation)(perm)

    return jnp.array(deltas)


def c_ell_stack(I, J, lmin, lmax, theory_cell):
    """
    Stack theory C_ell according to pseudo-alm combination order.
    
    Parameters:
    -----------
    I, J : int
        Pseudo-alm indices for correlation
    lmin, lmax : int
        Multipole range
    theory_cell : jnp.ndarray
        Theory power spectra [3, 3, n_ell] for E/B/T correlations
        
    Returns:
    --------
    jnp.ndarray
        Stacked C_ell array matching permutation structure
    """
    perm_i, perm_j = sel_perm(I), sel_perm(J)
    len_i = len(perm_i)
    len_j = len(perm_j)
    len_l = lmax - lmin + 1

    def fill_stack(i, j, ii, jj, cl_stack):
        cl_stack = jax.lax.cond(
            ii[3] != jj[3],
            lambda _: cl_stack.at[i, j].set(jnp.zeros(len_l)),
            lambda _: cl_stack.at[i, j].set(theory_cell[ii[4], jj[4]]),
            operand=None,
        )
        return cl_stack

    cl_stack = jnp.full((len_i, len_j, len_l), jnp.nan)
    for i, ii in enumerate(perm_i):
        for j, jj in enumerate(perm_j):
            cl_stack = fill_stack(i, j, ii, jj, cl_stack)

    return cl_stack

# ============================================================================
# Core Covariance Computation
# ============================================================================


def cov_4D(I, J, w_arr, lmax, lmin, theory_cell):
    """
    Calculate pseudo-alm covariances for given mode combination.
    
    Computes full 4D covariance tensor for all (ell,m) vs (ell',m') 
    combinations for specified pseudo-alm modes I and J.
    
    Parameters:
    -----------
    I, J : int
        Pseudo-alm mode indices to correlate
    w_arr : jnp.ndarray
        W-matrix array from mask
    lmax, lmin : int
        Multipole range
    theory_cell : jnp.ndarray
        Theory power spectra including noise
        
    Returns:
    --------
    jnp.ndarray
        Covariance matrix [n_ell, n_m, n_ell', n_m']
        
    Notes:
    ------
    Uses optimized einsum operations. Factor 0.5 accounts for
    cov(alm, alm) = 0.5 * C_ell for most cases.
    """
    if not (0 <= I <= 5 and 0 <= J <= 5):
        raise ValueError(f"Pseudo-alm indices must be 0-5, got I={I}, J={J}")
    if lmax < lmin:
        raise ValueError(f"lmax ({lmax}) must be >= lmin ({lmin})")
    
    w_arr = jnp.array(w_arr)
    theory_cell = jnp.array(theory_cell)
    wlm = w_stack(I, w_arr)  # i,l,m,lpp,mpp
    wlpmp = w_stack(J, w_arr)  # j,lp,mp,lpp,mppp

    # create "Kronecker" delta matrix for covariance structure of alm with m with same modulus but different sign (also accounting for mpp = mppp = 0)
    len_m = jnp.shape(w_arr)[4]
    delta = delta_mppmppp(
        J, len_m
    )  # j,mpp,mppp (only need to specify one permuation, as the equality of Re/Im of alm to correlate is already enforced by the C_l that are set to zero otherwise)

    # write theory C_ell and noise to matrix and then stack according to pseudo-alm combination (I,J)

    c_lpp = c_ell_stack(I, J, lmin, lmax, theory_cell)  # i,j,lpp

    # multiply and sum over lpp, mpp, mppp. Factor 0.5 is because cov(alm,alm) = 0.5 Cl for most cases (others are modified by delta_mppmppp).

    mid_ind = (wlm.shape[2] - 1) // 2
    wlm_pos = wlm[:, :, mid_ind:, :, :]
    wlpmp_pos = wlpmp[:, :, mid_ind:, :, :]
    # return 0.5 * jnp.einsum("ilmbc,ijb,jcd,jnabd->lmna", wlm_pos, c_lpp, delta, wlpmp_pos)
    step1 = jnp.einsum("ilmbc,ijb->lmbcj", wlm_pos, c_lpp)
    step2 = jnp.einsum("jcd,jnabd->jcnab", delta, wlpmp_pos)
    return 0.5 * jnp.einsum("jcnab,lmbcj->lmna", step2, step1)

# ============================================================================
# Optimization and Precomputation
# ============================================================================


def precompute_einsum(wlm, delta):
    """
    Precompute einsum operations for optimization.
    
    Parameters:
    -----------
    wlm : jnp.ndarray
        W-matrix stack
    delta : jnp.ndarray
        Delta matrix
        
    Returns:
    --------
    jnp.ndarray
        Precomputed tensor for reuse
    """
   
    return jnp.einsum("jcd,jnabd->jcnab", delta, wlm)


@jax.jit
def precompute_xipm(w_arr):
    """
    Precompute all xi+/- combinations for efficiency.
    
    Since xi+/- indices are fixed, precompute all W-stacks and
    delta matrices to avoid redundant computation.
    
    Parameters:
    -----------
    w_arr : jnp.ndarray
        W-matrix array
        
    Returns:
    --------
    tuple
        (precomputed_tensors, w_stacks) for optimized computation
    """
    w_arr = jnp.array(w_arr)
    len_m = jnp.shape(w_arr)[4]
    mid_ind = (len_m - 1) // 2
    alm_inds = jnp.arange(4)

    w_stacks = jax.vmap(lambda i: w_stack(i, w_arr)[:, :, mid_ind:, :, :])(alm_inds)
    deltas = jax.vmap(lambda i: delta_mppmppp(i, len_m))(alm_inds)
    batched_einsum = jax.vmap(precompute_einsum)

    precomputed = batched_einsum(w_stacks, deltas)
    return precomputed, w_stacks


def optimized_cov_4D(i, j, precomputed, lmax, lmin, theory_cell):
    """
    Optimized covariance calculation using precomputed tensors.
    
    Parameters:
    -----------
    i, j : int
        Pseudo-alm indices
    precomputed : tuple
        Output from precompute_xipm()
    lmax, lmin : int
        Multipole range
    theory_cell : jnp.ndarray
        Theory power spectra
        
    Returns:
    --------
    jnp.ndarray
        Covariance matrix
    """
    wd, w_stacks = precomputed
    c_lpp = c_ell_stack(i, j, lmin, lmax, theory_cell)
    step1 = jnp.einsum("ilmbc,ijb->lmbcj", w_stacks[i], c_lpp)
    step2 = wd[j]
    
    return 0.5 * jnp.einsum("jcnab,lmbcj->lmna", step2, step1)

# ============================================================================
# JIT-compiled versions for performance
# ============================================================================

cov_4D_jit = jax.jit(cov_4D, static_argnums=(0, 1, 3, 4))

optimized_cov_4D_jit = jax.jit(optimized_cov_4D, static_argnums=(0, 1, 3, 4))
