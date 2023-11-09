import cov_calc 
import numpy as np
from helper_funcs import get_noise_cl
import os.path
import healpy as hp
import wigner
from scipy import integrate
from scipy.integrate import quad_vec
import scipy.stats as stats


def cell_cube(cl_object):
    c_all = np.zeros((3,3,cl_object.len_l))
    c_all[0,0] = cl_object.ee.copy()
    c_all[0,2] = cl_object.ne.copy()
    c_all[2,0] = cl_object.ne.copy()
    c_all[2,2] = cl_object.nn.copy()
    return c_all

def noise_cl(sigma2,len_l):
    # separately for e,b and n?
    c_all = np.zeros((3,3,len_l))
    for i in range(3):
        c_all[i,i] = np.ones(len_l)
    return sigma2*c_all
    


def cov_alm_xi(cl_object=None,mask_object=None,pos_m=False,sigma_e=None,lmax=None):

    """
    always a covariance of pseudo alms - order and number depends on two point statistics considered
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
    elif sigma_e is None:
            raise RuntimeError('Specify either power spectrum or noise variance and lmax')
    else:
        theory_cell = np.zeros((3,3,lmax+1))
    

    # take lmax from c_ell object
    
    lmin = 0

    if sigma_e is not None:
        noise_sigma = get_noise_cl()
        #noise_sigma = get_pp_sigma_n(sigma_e=sigma_e)
        theory_cell += noise_cl(noise_sigma,lmax+1)

    if pos_m:
        n_cov = n_alm * (lmax - lmin + 1) * (lmax + 1)
    else:
        n_cov = n_alm * (lmax - lmin + 1) * (2 * lmax + 1)
    
    

    if mask_object is None or mask_object.name == 'fullsky':
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


    else:
        cov_matrix = cov_masked(mask_object,alm_inds,n_cov,theory_cell,lmin,lmax,pos_m)

        
     
            

    cov_matrix = np.where(np.isnan(cov_matrix), cov_matrix.T, cov_matrix) 
    assert np.allclose(cov_matrix, cov_matrix.T), 'Covariance matrix not symmetric'   
    return cov_matrix


def save_cov(cov,covname):
    print('saving covariance matrix...')
    np.savez(covname,cov=cov)


def check_cov(covname):
    print('checking for covariance matrix...')
    return os.path.isfile(covname)
    

def load_cov(name):
    print('loading covariance matrix...')
    covfile = np.load(name)
    return covfile['cov']


def cov_masked(mask_object,alm_inds,n_cov,theory_cell,lmin,lmax,pos_m):
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
    return cov_matrix



def cov_cl_gaussian(cl_object,sigma_e=None,apo=False):
    cl_e = cl_object.ee.copy()
    noise2 = np.zeros_like(cl_e)
    ell = np.arange(cl_object.lmax + 1)
    if sigma_e is not None:
        
        
        noise_sigma2 = get_noise_cl()
        apo_width = cl_object.lmax/3
        noise_B = noise_E = noise_sigma2*np.ones_like(cl_e)
        if apo:
            noise_B *= stats.norm.pdf(ell,0,apo_width)/np.max(stats.norm.pdf(ell,0,apo_width))
            noise_E *= stats.norm.pdf(ell,0,apo_width)/np.max(stats.norm.pdf(ell,0,apo_width))
        cl_e += noise_E
        cl2 = np.square(cl_e) + np.square(noise_B)
        noise2 += np.square(noise_E) + np.square(noise_B)
    else:
        cl2 = np.square(cl_e)
    
    
    
    diag = 2*cl2
    noise_diag = 2*noise2
    return diag, noise_diag

def cov_xi_gaussian(cl_object,fsky,binmin_in_arcmin,binmax_in_arcmin,sigma_e=None,lmin=0,noise_apo=False):
    # e.g. https://www.aanda.org/articles/aa/full_html/2018/07/aa32343-17/aa32343-17.html
    
    c_tot, c_sn = cov_cl_gaussian(cl_object,sigma_e,noise_apo)
    c_tot, c_sn = c_tot[lmin:], c_sn[lmin:]
    lmax = cl_object.lmax
    
    l  = 2 * np.arange(lmin,lmax+1) + 1
    
    norm = 1 / (4 * np.pi)
    upper = np.radians(binmax_in_arcmin/60)
    lower = np.radians(binmin_in_arcmin/60)
    

    wigner_int = lambda theta: theta * wigner.wigner_dl(lmin,lmax,2,2,theta)
    
    
    t_norm = 2 / (upper**2 - lower**2)
    cov_xi = 1 / fsky * t_norm**2 * norm**2 * np.sum((quad_vec(wigner_int,lower,upper)[0])**2 * c_tot * l)
    cov_sn = 1 / fsky * t_norm**2 * norm**2 * np.sum((quad_vec(wigner_int,lower,upper)[0])**2 * l * c_sn) 
    pure_noise_mean = t_norm * norm * np.sum((quad_vec(wigner_int,lower,upper)[0]) * l * np.sqrt(c_sn)) 
    mean = t_norm * norm * np.sum((quad_vec(wigner_int,lower,upper)[0]) * l * (cl_object.ee.copy()[lmin:])) + pure_noise_mean
    
    print('summation over wigners for noise yields {:.2f}'.format(pure_noise_mean))
    return mean,cov_xi, cov_sn

def cov_xi_gaussian_flat(cl_object,fsky,t1_in_deg,t2_in_deg,binmin_in_arcmin,binmax_in_arcmin,sigma_e=None,nside=None,lmin=0):
    from scipy.special import j0
    from scipy.interpolate import interp1d
    import matplotlib.pyplot as plt
    c_lm, c_sn = cov_cl_gaussian(cl_object,binmin_in_arcmin,binmax_in_arcmin,sigma_e,nside)
    lmax = cl_object.lmax 
    ell = np.arange(lmax+1)
    cl_interp = interp1d(ell,c_lm,kind='cubic')
    cl_sn_interp = interp1d(ell,c_sn,kind='cubic')
    fine_ell = np.linspace(0,lmax,1000)
    upper = np.radians(binmax_in_arcmin/60)
    lower = np.radians(binmin_in_arcmin/60)
    bessel = j0(ell*np.radians(t1_in_deg))
    bessel_fine = lambda theta_in_rad: j0(fine_ell*theta_in_rad)
    norm = 1 / (2*np.pi)
    t_norm = 2 / (upper**2 - lower**2)
    mean = norm * np.sum(cl_object.ee.copy()*bessel*ell)
    plt.figure()
    plt.plot(fine_ell,bessel_fine((upper-lower)/2) * bessel_fine((upper-lower)/2) * np.square(fine_ell) / (2 * fine_ell + 1))
    plt.xlabel(r'$\ell$')
    plt.title(r'$\theta = {:.2f}$'.format(np.degrees((upper-lower)/2)))
    plt.yscale('log')
    plt.show()
    cov_xi_all_int = lambda theta, thetap: theta * thetap * np.trapz(bessel_fine(theta) * bessel_fine(thetap) * cl_interp(fine_ell) * np.square(fine_ell) / (2 * fine_ell + 1),fine_ell)
    cov_xi_sn_int = lambda theta, thetap: theta * thetap * np.trapz(bessel_fine(theta) * bessel_fine(thetap) * cl_sn_interp(fine_ell) * np.square(fine_ell) / (2 * fine_ell + 1),fine_ell)

    cov_xi = 1 / fsky * t_norm**2 * norm**2 * integrate.dblquad(cov_xi_all_int,lower,upper,lower,upper)[0]
    cov_sn = 1 / fsky * t_norm**2 * norm**2 * integrate.dblquad(cov_xi_sn_int,lower,upper,lower,upper)[0]

    return mean, cov_xi, cov_sn
