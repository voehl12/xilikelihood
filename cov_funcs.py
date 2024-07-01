import numpy as np
import gc


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
    return i[I]


def part(j, arr):
    if j == 0:
        return np.real(arr)
    else:
        return np.imag(arr)


def w_stack(I, w_arr):
    """
    stack the W-arrays according to a given permutation I
    each permutation sets which part (+/-/0, real/imaginary) of the W-array is needed
    and which sign it has attached
    """
    # 0: +/-, 1: Re/Im W, 2: W+/-/0, 3: Re/Im a, 4: E/B/0 a
    # i is going to be the first axis of W
    w_s = []
    perm = sel_perm(I)
    len_m = np.shape(w_arr)[4]

    for p in perm:
        fac = (-1) ** p[0]
        wpart = part(p[1], w_arr)
        wpm0 = wpart[p[2]]
        w_s.append(fac * wpm0)

    return np.array(w_s)


def delta_mppmppp(I, len_m):
    m_max = int((len_m - 1) / 2)
    m0_ind = m_max
    perm = sel_perm(I)
    deltas = []
    diag = np.arange(len_m)

    for p in perm:
        d_array = np.eye(len_m)
        if p[3] == 0:
            d_array[diag, np.flip(diag)] = (-1) ** (np.fabs(np.arange(-m_max, m_max + 1)))
            d_array[m0_ind, m0_ind] = 2
        else:
            d_array[diag, np.flip(diag)] = (-1) ** (np.fabs(np.arange(-m_max, m_max + 1)) + 1)
            d_array[m0_ind, m0_ind] = 0
        deltas.append(d_array)

    return np.array(deltas)


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
    cl_stack = np.full((len_i, len_j, len_l), np.nan)
    for i, ii in enumerate(perm_i):
        for j, jj in enumerate(perm_j):

            if ii[3] != jj[3]:
                # checking whether covariance is between real and imaginary alm
                cl_stack[i, j] = np.zeros(len_l)
            else:
                cl_stack[i, j] = theory_cell[ii[4], jj[4]]

    return cl_stack


def cov_4D(I, J, w_arr, lmax, lmin, theory_cell, l_out=None, pos_m=False):
    from einsumt import einsumt as einsum
    from sys import getsizeof

    """
    calculate covariances for given combination of pseudo-alm (I,J) for all m, ell, m', ell' at once.
    theory_cell already include noise
    """
    # stack parts of w-matrices in the right order:
    wlm = w_stack(I, w_arr)  # i,l,m,lpp,mpp
    wlpmp = w_stack(J, w_arr)  # j,lp,mp,lpp,mppp

    # create "Kronecker" delta matrix for covariance structure of alm with m with same modulus but different sign (also accounting for mpp = mppp = 0)
    len_m = np.shape(w_arr)[4]
    delta = delta_mppmppp(
        J, len_m
    )  # j,mpp,mppp (only need to specify one permuation, as the equality of Re/Im of alm to correlate is already enforced by the C_l that are set to zero otherwise)

    # write theory C_ell and noise to matrix and then stack according to pseudo-alm combination (I,J)

    c_lpp = c_ell_stack(I, J, lmin, lmax, theory_cell)  # i,j,lpp

    # multiply and sum over lpp, mpp, mppp. Factor 0.5 is because cov(alm,alm) = 0.5 Cl for most cases (others are modified by delta_mppmppp).
    if l_out is not None:
        wl = wlm[:, l_out - lmin, :, :, :]  # i,m,lpp,mpp
        wlp = wlpmp[:, l_out - lmin, :, :, :]  # j, mp, lpp, mppp
        cov_l = 0.5 * einsum("ijb,imbc,jabd,jcd->ma", c_lpp, wl, wlp, delta)
        return cov_l
    else:
        # only need m >= 0 elements, maybe already enforce this here, save resources
        # assert np.allclose(wlm[:,:,mid_ind:,:,:],wlm[:,:,:mid_ind+1,:,:])
        # assert np.allclose(wlpmp[:,:,mid_ind:,:,:],wlpmp[:,:,:mid_ind+1,:,:])
        if pos_m == True:
            mid_ind = int((wlm.shape[2] - 1) / 2)
            wlm = wlm[:, :, mid_ind:, :, :]
            wlpmp = wlpmp[:, :, mid_ind:, :, :]

        # need to reduce all this to the m that are necessary (i.e. not go to m = lmax in the covariance matrix for each ell) possibly by slicing endresult for now?
        step1 = einsum("ilmbc,ijb->lmbcj", wlm, c_lpp)
        print("stepsize: {} mb".format(getsizeof(step1) / 1024**2))
        step2 = einsum("jcd,jnabd->jcnab", delta, wlpmp)
        cov_lmlpmp = 0.5 * einsum("jcnab,lmbcj->lmna", step2, step1)
        print("cov part size: {} mb".format(getsizeof(cov_lmlpmp) / 1024**2))
        del step1
        del step2
        gc.collect()
        # cov_lmlpmp = 0.5 * einsum('ijb,ilmbc,jnabd,jcd->lmna',c_lpp,wlm,wlpmp,delta)
        return cov_lmlpmp
