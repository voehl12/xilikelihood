import numpy as np
import pyccl as ccl


def clnames(s8):
    s8str = str(s8)
    s8str = s8str.replace(".", "p").lstrip("0")
    s8name = "S8" + s8str
    clpath = "Cl_3x2pt_kids55_s8{}.txt".format(s8str)
    return clpath, s8name


def prep_cosmo(params):
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
    s8 = params["s8"]
    omega_m = params["omega_m"] #0.31
    omega_b = 0.046
    omega_c = omega_m - omega_b
    sigma8 = s8 * (omega_m / 0.3) ** -0.5
    cosmo = ccl.Cosmology(Omega_c=omega_c, Omega_b=omega_b, h=0.7, sigma8=sigma8, n_s=0.97)
    return cosmo



def calc_cl(cosmo, ell, z_bins):
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
    # rbin = np.loadtxt(nz_path)
    bin1, bin2 = z_bins
    z1, nz1 = bin1.z, bin1.nz
    z2, nz2 = bin2.z, bin2.nz
    # b = 1.5 * np.ones_like(z)
    lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z1, nz1))
    lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z2, nz2))
    # clu = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, nz), bias=(z, b))
    cl_ee = ccl.angular_cl(cosmo, lens1, lens2, ell)
    # cl_ne = ccl.angular_cl(cosmo, lens, clu, ell)
    # cl_nn = ccl.angular_cl(cosmo, clu, clu, ell)
    cl_ne = cl_nn = np.zeros_like(cl_ee)
    return cl_ee, cl_ne, cl_nn


def save_cl(cl, clpath):
    cl_ee, cl_ne, cl_nn = cl
    # np.savez(clpath, theory_ellmin=2,ee=cl_ee,ne=cl_ne,nn=cl_nn)
    np.savetxt(clpath, (cl_ee, cl_ne, cl_nn), header="EE, nE, nn")



def get_cl(params_dict, z_bins):
    cosmo = prep_cosmo(params_dict)
    ell = np.arange(2, 2000)
    cl = calc_cl(cosmo, ell, z_bins)
    return cl
