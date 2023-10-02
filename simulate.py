import numpy as np
import healpy as hp 

def create_maps(C_ell,NSIDE=256,lmax=None):
    """
    Create Gaussian random maps from C_l
    C_l need to be tuple of arrays in order:  TT, TE, EE, *BB
    * are zero for pure shear
    """
    np.random.seed()
    if lmax:
        lmax = lmax
    else: 
        lmax = 3*NSIDE - 1
    maps = hp.sphtfunc.synfast(C_ell, NSIDE,lmax=lmax,verbose=True)
    return maps

def cl2xi(pcl_22,mask,lmax,norm_lm=False):
    """ 
    function to convert pseudo-Cl to correlation function estimator for given angular separation
    right now takes the same l-limits for the normalization factor, but this might have to be adjusted 
    to taking the maximum lmax here, since it would be no problem computationally 
    (unlike for the correlation function summation and therefore covariance matrix)
    norm_lm: whether the lmax for the norm should be a different one. If True, lmax for the norm will be 3*nside - 1,
    regardless of lmax for the pseudo_Cl summation. Turns out, this only makes a difference if lmax is less than 20% of the
    3nside-1 lmax.
    """
    
    cl_e, cl_b, cl_eb = pcl_22[0], pcl_22[3], (pcl_22[1] + pcl_22[2]) / 2
    if lmax > len(cl_e):
        raise RuntimeError('lmax for correlation function estimator is too large!')
    else:
        cl_short = lambda lmin: (cl_e[lmin:lmax+1], cl_b[lmin:lmax+1], cl_eb[lmin:lmax+1])
        l = lambda lmin: np.arange(lmin,lmax+1)
    
    
    if norm_lm == True:
        w_lm = mask_to_wlm(mask,None)
        wl = hp.sphtfunc.alm2cl(w_lm)
        l_norm = np.arange(len(wl))
        
        norm = lambda t,lmin: 1 / np.sum((2 * l_norm + 1) * eval_legendre(l_norm,np.cos(np.radians(t))) * wl) / (2 * np.pi)

    else:
        w_lm = mask_to_wlm(mask,lmax)
        wl_arr = hp.sphtfunc.alm2cl(w_lm)
        wl = lambda lmin: wl_arr[lmin:]
        
        norm = lambda t,lmin: 1 / np.sum((2 * l(lmin) + 1) * eval_legendre(l(lmin),np.cos(np.radians(t))) * wl(lmin)) / (2 * np.pi)
        
    xi_p = lambda t,lmin: 2 * np.pi * norm(np.radians(t),lmin) * np.sum((2 * l(lmin) + 1) * wigner.wigner_dl(lmin,lmax,2,2,np.radians(t)) * (cl_short(lmin)[0] + cl_short(lmin)[1]))
    xi_m = lambda t,lmin: 2 * np.pi * norm(np.radians(t),lmin) * np.sum((2 * l(lmin) + 1) * wigner.wigner_dl(lmin,lmax,2,-2,np.radians(t)) * (cl_short(lmin)[0] - cl_short(lmin)[1] - 2j * cl_short(lmin)[2]))
    return xi_p,xi_m

def get_pseudoCl(maps_TQU,mask):
    
    
    #f_0 = nmt.NmtField(mask, [maps_TQU[0]])
    tic = time.perf_counter()
    f_2 = nmt.NmtField(mask, maps_TQU[1:])
    toc = time.perf_counter()
    dt = toc-tic
    #cl_00 = nmt.compute_coupled_cell(f_0, f_0)
    #cl_02 = nmt.compute_coupled_cell(f_2, f_0)
    cl_22 = nmt.compute_coupled_cell(f_2, f_2)
    return cl_22

    
def xi_sim(cell_object,mask_file,mode='namaster'):





def pcl_sim(cell_object,mask_file):
