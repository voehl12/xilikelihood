import cov_calc 
import numpy as np
from helper_funcs import get_sigma_n


def cell_cube(cl_object):
    c_all = np.zeros((3,3,cl_object.len_l))
    c_all[0,0] = cl_object.ee
    c_all[0,2] = cl_object.ne
    c_all[2,0] = cl_object.ne
    c_all[2,2] = cl_object.nn
    return c_all

def noise_cl(sigma2,len_l):
    # separately for e,b and n?
    c_all = np.zeros((3,3,len_l))
    for i in range(3):
        c_all[i,i] = np.ones(len_l)
    return sigma2*c_all


def cov_xi(cl_object=None,mask_object=None,pos_m=False,noise_sigma=None,lmax=None):
    """
    order of alm follows structure of cov_4D - meaning first sort by E/B Re/Im, then by l and then by m
    only positive m part of covariance is needed (not so sure about this - current version: testing with all m) - way to only calculate this, just like wlmlpmp are only plugged into the sum for given l as needed
    only valid for E/B alms so far. 
    could implement limitation of m to current l, would make n_cov smaller overall
    """
    alm_kinds = ['ReE', 'ImE', 'ReB', 'ImB']
    alm_inds = cov_calc.match_alm_inds(alm_kinds)
    n_alm = len(alm_inds)
    if cl_object is not None:
        theory_cell = cell_cube(cl_object)
        lmax = cl_object.lmax
    elif noise_sigma is None:
            raise RuntimeError('Specify either power spectrum or noise variance and lmax')
    else:
        theory_cell = np.zeros((3,3,lmax+1))
    

    # take lmax from c_ell object
    
    lmin = 0

    if noise_sigma is not None:
        if noise_sigma == 'default':
            noise_sigma = get_sigma_n()
        theory_cell += noise_cl(noise_sigma**2,lmax+1)

    if pos_m:
        n_cov = n_alm * (lmax - lmin + 1) * (lmax + 1)
    else:
        n_cov = n_alm * (lmax - lmin + 1) * (2 * lmax + 1)
    
    

    if mask_object:
        if mask_object.w_arr is None:
            mask_object.calc_w_arrs()
            del mask_object.wpm_arr
        w_arr = mask_object.w_arr
        cov_matrix = np.full((n_cov,n_cov),np.nan)

        for i in alm_inds:
            for j in alm_inds:
                if i > j:
                    continue
                else:
                    cov_part = cov_calc.cov_4D(i,j,w_arr,lmax,lmin,theory_cell,pos_m=pos_m)
                    len_2D = cov_part.shape[0]*cov_part.shape[1]
                    
                    cov_2D = np.reshape(cov_part,(len_2D,len_2D))
                    pos_y = (len_2D * i,len_2D * (i + 1))
                    pos_x = (len_2D * j,len_2D * (j + 1))
                    cov_matrix[pos_y[0]:pos_y[1],pos_x[0]:pos_x[1]] = cov_2D
    else:
        if pos_m == False:
            raise RuntimeError('No mask case covariance matrix only implemented for positive m')
        else:
            cov_matrix = np.zeros((n_cov,n_cov))
            diag = np.zeros(n_cov)
            for i in alm_inds:
                t = int(np.floor(i/2)) # same c_l for Re and Im
                len_sub = lmax + 1
                cell_ranges = [np.repeat(theory_cell[t,t,i],i+1) for i in range(lmax+1)]
                full_ranges = [np.append(cell_ranges[i],np.zeros(lmax+1-len(cell_ranges[i]))) for i in range(len(cell_ranges))]
                cov_part = 0.5*np.ndarray.flatten(np.array(full_ranges))
                if i % 2 == 0:
                    cov_part[::len_sub] *= 2
                else:
                    cov_part[::len_sub] *= 0
                # alm with same m but different sign dont have vanishing covariance. This is only relevant if pos_m = False.
                len_2D = len(cov_part)
                pos = (len_2D * i,len_2D * (i + 1))
                
                diag[pos[0]:pos[1]] = cov_part
            assert len(diag) == n_cov
            cov_matrix = np.diag(diag)
            

    cov_matrix = np.where(np.isnan(cov_matrix), cov_matrix.T, cov_matrix) 
    assert np.allclose(cov_matrix, cov_matrix.T), 'Covariance matrix not symmetric'   
    return cov_matrix


def save_cov():
    print('saving covariance matrix...')
    # np.savez('cov_xip_l{:d}_n{:d}_none_nomask_noise.npz'.format(lmax,nside),cov=cov_xip_noise)

    pass

def check_cov():
    print('checking for covariance matrix...')
    #name='cov_xip_l{:d}_n{:d}_none_nomask.npz'.format(lmax,nside)):
    """ try:
        covs = np.load(name)
        return True
    except:
        return None """
    pass

def load_cov(name):
    print('loading covariance matrix...')
    #covs = np.load(name)
    pass
