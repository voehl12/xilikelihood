
import numpy as np 


def clnames(s8):
    s8str = str(s8)
    s8str = s8str.replace('.','p').lstrip('0')
    s8name = 'S8'+s8str
    clpath = "Cl_3x2pt_kids55_s8{}.txt".format(s8str)
    return clpath, s8name


def cosmof(s8):
    import pyccl as ccl
    """
    function to generate vanilla cosmology with ccl given a value of S8

    Parameters
    ----------
    s8 : float
        lensing amplitude, clustering parameter S8

    Returns
    -------
    object
        ccl vanilla cosmology
    """
    omega_m = 0.31
    omega_b = 0.046
    omega_c = omega_m - omega_b
    sigma8 = s8 * (omega_m / 0.3)**-0.5
    cosmo = ccl.Cosmology(Omega_c = omega_c, Omega_b = omega_b, h = 0.7, sigma8 = sigma8, n_s = 0.97)
    return cosmo
    
def get_cl(cosmo,ell,nz_path):
    import pyccl as ccl
    """
    generate 3x2pt c_ell given a ccl cosmology

    Parameters
    ----------
    cosmo : object
        ccl cosmology to be used
    ell : array
        array of integer ell values to be used for the cl
    nz_path : string
        path to a redshift distribution to be used

    Returns
    -------
    tuple
        cl in the order ee, ne, nn
    """
    rbin = np.loadtxt(nz_path)
    z, nz = rbin[:,0], rbin[:,1]
    b = 1.5*np.ones_like(z)
    lens = ccl.WeakLensingTracer(cosmo, dndz=(z, nz))
    clu = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z,nz), bias=(z,b))
    ell = np.arange(2, 100000)
    cl_ee = ccl.angular_cl(cosmo, lens, lens, ell)
    cl_ne = ccl.angular_cl(cosmo,lens,clu,ell)
    cl_nn = ccl.angular_cl(cosmo,clu,clu,ell)
    return cl_ee,cl_ne,cl_nn


def save_cl(cl,clpath):
    cl_ee,cl_ne,cl_nn = cl
    #np.savez(clpath, theory_ellmin=2,ee=cl_ee,ne=cl_ne,nn=cl_nn)
    np.savetxt(clpath,(cl_ee,cl_ne,cl_nn),header='EE, nE, nn')


def get_cl_s8(s8):
    cosmo = cosmof(s8)
    nzpath = 'data/KiDS_Nz_bin5.txt'
    cl = get_cl(cosmo,np.arange(2,2000),nzpath)
    return cl