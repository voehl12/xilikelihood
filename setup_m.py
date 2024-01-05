# functions to set up pdf calculations
import numpy as np
import healpy as hp
import helper_funcs

import os.path


def mmatrix_xi(prefactors, bin_n=0,kind="p", pos_m=False):
    
    # TODO: implement xi_minus (need to add minus sign to B mode pseudo alms)
    n_field = 2  # for xi+, there need to be two fields.
    lmax = len(prefactors[0,0]) - 1 
 

    # diag = fac(l0) * len(sub), fac(l1) * len(sub), ...
    if kind == 'p':
        fac_arr = prefactors[bin_n,0]
    elif kind == 'm':
        fac_arr = prefactors[bin_n,1]
    
    if pos_m == True:
        len_sub = lmax + 1
        diag = 2 * np.repeat(fac_arr, len_sub)
        # every m = 0 entry is halved --> first of every l stretch.
        diag[::len_sub] *= 0.5
    else:
        len_sub = 2 * lmax + 1
        diag = np.repeat(fac_arr, len_sub)
        #diag[lmax :: len_sub] *= 0.5

    
    m = np.diag(np.tile(diag, n_field * 2))
    return m

def m_xi_cross(prefactors, bin_n=(0,0),kind=("p","p"), pos_m=True):
    n_m = len(prefactors)
    
    # function to create two (or more) m matrices to crosscorrelate different redshift bins (i.e. different sets of pseudo alms). first m only selects first half of covariance matrix, second, the second ect.
    # prefactors is tuple of prefactors, so is kinds and bin_n 
    pass

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
