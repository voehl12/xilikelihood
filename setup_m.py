# functions to set up pdf calculations
import numpy as np
import healpy as hp
import helper_funcs

import os.path


def mmatrix_xi(t_in_deg, lmax, wl, kind="p", pos_m=True):
    # idea: if theta is a tuple: call bin_prefactors
    # TODO: implement xi_minus (need to add minus sign to B mode pseudo alms)
    n_field = 2  # for xi+, there need to be two fields.
    if pos_m == True:
        len_sub = lmax + 1
    else:
        len_sub = 2 * lmax + 1

    # diag = fac(l0) * len(sub), fac(l1) * len(sub), ...
    if type(t_in_deg) is tuple:
        fac_arr = helper_funcs.bin_prefactors(t_in_deg, wl, lmax, lmax, kind)

    else:
        fac_arr = helper_funcs.ang_prefactors(t_in_deg, wl, lmax, lmax, kind)
    diag = 2 * np.repeat(fac_arr, len_sub)
    if pos_m == True:
        diag[::len_sub] *= 0.5
    else:
        diag[lmax + 1 :: len_sub] *= 0.5

    # every m = 0 entry is halved --> first of every l stretch.
    m = np.diag(np.tile(diag, n_field * 2))
    return m


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
