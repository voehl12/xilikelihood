# functions to set up pdf calculations
import numpy as np 
import healpy as hp
import wigner
from scipy.special import eval_legendre
from scipy.integrate import quad_vec
import os.path


 

def ang_prefactors(t_in_deg,wl,kind='p'):
    
    # normalization factor automatically has same l_max as the pseudo-Cl summation
    
    t = np.radians(t_in_deg)
    norm_l = np.arange(len(wl))
    legendres = eval_legendre(norm_l,np.cos(t))
    norm = 1 / np.sum((2 * norm_l + 1) * legendres * wl) / (2 * np.pi)

    if kind == 'p':
        wigners = wigner.wigner_dl(norm_l[0],norm_l[-1],2,2,t)
    elif kind == 'm':
        wigners = wigner.wigner_dl(norm_l[0],norm_l[-1],2,-2,t)
    else: 
        raise RuntimeError('correlation function kind needs to be p or m')
    
    return 2 * np.pi * norm * wigners

def bin_prefactors(bin_in_deg,wl,kind='p'):
    t0, t1 = bin_in_deg
    t0, t1 = np.radians(t0), np.radians(t1)
    
    norm_l = np.arange(len(wl))
    legendres = lambda t_in_rad: eval_legendre(norm_l,np.cos(t_in_rad))
    norm = lambda t_in_rad: 1 / np.sum((2 * norm_l + 1) * legendres(t_in_rad) * wl) / (2 * np.pi)
    if kind == 'p':
        wigners = lambda t_in_rad: wigner.wigner_dl(norm_l[0],norm_l[-1],2,2,t_in_rad)
    elif kind == 'm':
        wigners = lambda t_in_rad: wigner.wigner_dl(norm_l[0],norm_l[-1],2,-2,t_in_rad)
    else: 
        raise RuntimeError('correlation function kind needs to be p or m')

    
    integrand = lambda t_in_rad: norm(t_in_rad) * wigners(t_in_rad) * t_in_rad
    # norm * d_l * weights 
    W = 0.5 * (t1**2 - t0**2) # weights already integrated
    A_ell = quad_vec(integrand,t0,t1)

    return 2 * np.pi * A_ell[0] / W

def mmatrix_xi(t_in_deg,mask_object,kind='p'):
    # idea: if theta is a tuple: call bin_prefactors
    n_field = 2 # for xi+, there need to be two fields. 
    len_sub = mask_object.lmax + 1
    wlm = mask_object.wlm
    wl = hp.sphtfunc.alm2cl(wlm)
    #diag = fac(l0) * len(sub), fac(l1) * len(sub), ...
    if type(t_in_deg) is tuple:
        fac_arr = bin_prefactors(t_in_deg,wl,kind)
    else:
        fac_arr = ang_prefactors(t_in_deg,wl,kind)
    diag = 2 * np.repeat(fac_arr,len_sub)
    diag[::len_sub] *= 0.5
    # every m = 0 entry is halved --> first of every l stretch. 
    m = np.diag(np.tile(diag,n_field * 2))
    return m


def mmatrix_pcl():
    # add zeros depending on overall lmax so can be used for inter l cross correlations as well
    pass


def save_m(m,name):
    print('saving M matrix...')
    np.savez(name,matrix=m)


def check_m(name):
    print('checking for M matrix...')
    return os.path.isfile(name)
    

def load_m(name):
    print('loading M matrix...')
    mfile = np.load(name)
    return mfile['matrix']