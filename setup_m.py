# functions to set up pdf calculations
import numpy as np
import healpy as hp
import helper_funcs

import os.path


def mmatrix_xi(prefactors, kind="p", pos_m=False):
    
    # TODO: implement xi_minus (need to add minus sign to B mode pseudo alms)
    # take out bin_n and pass only one angular bin prefactors (more is never needed anyway)
    n_field = 2  # for xi+, there need to be two fields.
    lmax = len(prefactors[0]) - 1 
 

    # diag = fac(l0) * len(sub), fac(l1) * len(sub), ...
    if kind == 'p':
        fac_arr = prefactors[0]
    elif kind == 'm':
        fac_arr = prefactors[1]
    
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

def m_xi_cross(prefactors, combs =((1,1),(0,1)),kind=("p","p"), pos_m=True):
    m = []
    n_m = len(combs)
    listlist = [list(set(tuple(combs[t]))) for t in range(len(combs))]
    allbins = [x for xs in listlist for x in xs]
   
    n_bins = max(allbins) + 1
    
    # function to create two (or more) m matrices to crosscorrelate different redshift bins (i.e. different sets of pseudo alms). first m only selects first half of covariance matrix, second, the second ect.
    # prefactors is tuple of prefactors, so is kinds and bin_n 
    # combs: bin combinations according to the ordering of the covariance matrix
    for i in range(n_m):
        sub_m = mmatrix_xi(prefactors[i],kind[i],pos_m)
        len_sub_m = len(sub_m)
        m_i = np.zeros((len_sub_m*n_bins,len_sub_m*n_bins))
        print(m_i.shape)
        comb = combs[i]
        print(comb)
        m_i[comb[0]*len_sub_m:(comb[0]+1)*len_sub_m,comb[1]*len_sub_m:(comb[1]+1)*len_sub_m] = sub_m
        if comb[0] != comb[1]:
            m_i[comb[1]*len_sub_m:(comb[1]+1)*len_sub_m,comb[0]*len_sub_m:(comb[0]+1)*len_sub_m] = sub_m
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
