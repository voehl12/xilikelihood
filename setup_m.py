import numpy as np

import os.path


def mmatrix_xi(prefactors, kind="p", pos_m=False):

    # TODO: implement xi_minus (need to add minus sign to B mode pseudo alms)
    # take out bin_n and pass only one angular bin prefactors (more is never needed anyway)
    n_field = 2  # for xi+, there need to be two fields.
    lmax = len(prefactors[0]) - 1

    # diag = fac(l0) * len(sub), fac(l1) * len(sub), ...
    if kind == "p":
        fac_arr = prefactors[0]
    elif kind == "m":
        fac_arr = prefactors[1]

    if pos_m == True:
        len_sub = lmax + 1
        diag = 2 * np.repeat(fac_arr, len_sub)
        # every m = 0 entry is halved --> first of every l stretch.
        diag[::len_sub] *= 0.5
    else:
        len_sub = 2 * lmax + 1
        diag = np.repeat(fac_arr, len_sub)
        # diag[lmax :: len_sub] *= 0.5

    m = np.diag(np.tile(diag, n_field * 2))
    return m


def m_xi_cross(prefactors, combs=((1, 1), (0, 1)), kind=("p", "p"), pos_m=True):
    """
    Creates combination matrices according to the number of correlation functions for which we wish to
    calculate the joint likelihood.

    Parameters
    ----------
    prefactors : tuple of arrays
        each entry is a (2,l_max+1) array, where first dim: xip and xim, second dim: ell-dependence
    combs : tuple, optional
        each entry gives a redshift bin-combination as a tuple, by default ((1, 1), (0, 1))
        ordering and indexing according to the covariance matrix, which needs to be provided for all redshift bins jointly.
    kind : tuple, optional
        specifies xip or xim for each correlation function, by default ("p", "p")
    pos_m : bool, optional
        only positive m modes are considered, by default True

    Returns
    -------
    array
        stack of 2D combination matrices
    """
    m = []
    n_m = len(combs)
    listlist = [list(set(tuple(combs[t]))) for t in range(len(combs))]
    allbins = [x for xs in listlist for x in xs]
    if len(combs) != len(kind):
        raise RuntimeError(
            "m_xi_cross: number of redshift combinations needs to match number of kinds of correlation functions!"
        )
    n_bins = max(allbins) + 1

    for i in range(n_m):
        sub_m = mmatrix_xi(prefactors[i], kind[i], pos_m)
        len_sub_m = len(sub_m)
        m_i = np.zeros((len_sub_m * n_bins, len_sub_m * n_bins))

        comb = combs[i]

        m_i[
            comb[0] * len_sub_m : (comb[0] + 1) * len_sub_m,
            comb[1] * len_sub_m : (comb[1] + 1) * len_sub_m,
        ] = sub_m
        if comb[0] != comb[1]:
            m_i[
                comb[1] * len_sub_m : (comb[1] + 1) * len_sub_m,
                comb[0] * len_sub_m : (comb[0] + 1) * len_sub_m,
            ] = sub_m
            m_i *= 0.5
        m.append(m_i)

    return np.array(m)


def mmatrix_pcl():
    # add zeros depending on overall lmax so can be used for inter l cross correlations as well
    pass


def save_m(m, name):
    print("Saving M matrix.")
    np.savez(name, matrix=m)


def check_m(name):
    print("Checking for M matrix...")
    if os.path.isfile(name):
        print("Found.")
        return True
    else:
        print("Not found.")
        return False


def load_m(name):
    print("Loading M matrix.")
    mfile = np.load(name)
    return mfile["matrix"]
