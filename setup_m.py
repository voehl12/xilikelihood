# functions to set up pdf calculations
import numpy as np
import healpy as hp
import helper_funcs

import os.path


def mmatrix_xi(t_in_deg, lmax, wl, kind="p"):
    # idea: if theta is a tuple: call bin_prefactors
    # TODO: implement xi_minus (need to add minus sign to B mode pseudo alms)
    n_field = 2  # for xi+, there need to be two fields.
    len_sub = lmax + 1

    # diag = fac(l0) * len(sub), fac(l1) * len(sub), ...
    if type(t_in_deg) is tuple:
        fac_arr = helper_funcs.bin_prefactors(t_in_deg, wl, len(wl) - 1, kind)
        fac_arr = fac_arr[:len_sub]
    else:
        fac_arr = helper_funcs.ang_prefactors(t_in_deg, wl, lmax, kind)
    diag = 2 * np.repeat(fac_arr, len_sub)
    diag[::len_sub] *= 0.5
    # every m = 0 entry is halved --> first of every l stretch.
    m = np.diag(np.tile(diag, n_field * 2))
    return m


def mmatrix_pcl():
    # add zeros depending on overall lmax so can be used for inter l cross correlations as well
    pass


def save_m(m, name):
    print("saving M matrix...")
    np.savez(name, matrix=m)


def check_m(name):
    print("checking for M matrix...")
    return os.path.isfile(name)


def load_m(name):
    print("loading M matrix...")
    mfile = np.load(name)
    return mfile["matrix"]
