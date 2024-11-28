import numpy as np
import jax.numpy as jnp
import jax
from jax import lax


def match_alm_inds(alm_keys):
    alm_keys = list(set(alm_keys))
    alm_dict = {"ReE": 0, "ImE": 1, "ReB": 2, "ImB": 3, "ReT": 4, "ImT": 5}
    alm_inds = [alm_dict[str(alm_keys[i])] for i in range(len(alm_keys))]
    alm_inds.sort()
    return alm_inds


def sel_perm(I):
    """
    select the linear combination of full-sky alm corresponding to given pseudo-alm

    Parameters
    ----------
    I : int
        I,J : E/B/0, Re/Im of the pseudo alm. 0 -> E/Re, 1 -> E/Im, usw..

    Returns
    -------
    list of tuples
        each 5-tuple specifices a Wllpmmp-alpmp combination
        0: +/-, 1: Re/Im W, 2: W+/-/0, 3: Re/Im a, 4: E/B/0 a
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


def part(j, arr):
    return jax.lax.cond(j == 0, lambda _: jnp.real(arr), lambda _: jnp.imag(arr), operand=None)


def w_stack(I, w_arr):
    """
    stack the W-arrays according to a given permutation I
    each permutation sets which part (+/-/0, real/imaginary) of the W-array is needed
    and which sign it has attached
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
    cast theory C_ell into shape corresponding to the order of E/B/EB contributions
    given by the combination of pseudo-alm to correlate.
    theory_cell must be cube: 3D array of c_ell, first and second axis run over E/B/0, third is ell
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


def cov_4D(I, J, w_arr, lmax, lmin, theory_cell):
    """
    calculate covariances for given combination of pseudo-alm (I,J) for all m, ell, m', ell' at once.
    theory_cell already include noise
    """
    # stack parts of w-matrices in the right order:

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




def precompute_einsum(wlm, delta, wlpmp):
    return jnp.einsum("ilmxc,jcd,jnayd->ilmxjnay", wlm, delta, wlpmp)


@jax.jit
def precompute_xipm(w_arr):
    alm_inds = jnp.arange(
        2
    )  # indices are fixed for xi+/-, so we can precompute all of them directly
    w_arr = jnp.array(w_arr)
    len_m = jnp.shape(w_arr)[4]
    mid_ind = (len_m - 1) // 2
    alm_inds = jnp.arange(4)

    w_stacks = jax.vmap(lambda i: w_stack(i, w_arr)[:, :, mid_ind:, :, :])(alm_inds)
    deltas = jax.vmap(lambda i: delta_mppmppp(i, len_m))(alm_inds)
    print(w_stacks.shape)
    print(deltas.shape)
    batched_einsum = jax.vmap(precompute_einsum)
    
    precomputed = batched_einsum(w_stacks, deltas, w_stacks)
    return precomputed


def optimized_cov_4D(i, j, precomputed, lmax, lmin, theory_cell):
    # Perform the einsum operation with c_lpp

    c_lpp = c_ell_stack(i, j, lmin, lmax, theory_cell)
    return 0.5 * jnp.einsum("ilmbjnab,ijb->lmna", precomputed[i], c_lpp)


cov_4D_jit = jax.jit(cov_4D, static_argnums=(0, 1, 3, 4))

optimized_cov_4D_jit = jax.jit(optimized_cov_4D, static_argnums=(0, 1, 3, 4))
