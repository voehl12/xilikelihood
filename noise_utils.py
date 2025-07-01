import numpy as np
import healpy as hp

def get_noise_xi_cov(
    survey_area_in_sqd, binmin_in_deg, binmax_in_deg, sigma_e=(0.282842712474619, 1.207829761642)
):
    if type(sigma_e) is tuple:
        sigma, n_gal = sigma_e
    else:
        raise RuntimeError("sigma_e must be tuple (sigma_single_component,n_gal_per_arcmin2)")
    pp_sigma = sigma * np.sqrt(2)
    A_annulus = np.pi * ((binmax_in_deg * 60) ** 2 - (binmin_in_deg * 60) ** 2)
    N_pair = (
        survey_area_in_sqd * 3600 * A_annulus * (n_gal) ** 2
    )  # this is 2 * N_pair but it comes from defining the shotnoise this way
    return (pp_sigma**2 / np.sqrt(N_pair)) ** 2


def get_noise_cl(sigma_e=(0.282842712474619, 1.207829761642)):
    # TODO: should be extended to mixed redshift bin C_ell with different shape noise parameters.
    if type(sigma_e) is tuple:
        sigma, n_gal_per_arcmin2 = sigma_e
    else:
        raise RuntimeError("sigma_e must be tuple (sigma_single_component,n_gal_per_arcmin2)")
    return (sigma) ** 2 / (n_gal_per_arcmin2 * 3600 * 41253 / (4 * np.pi))


def get_noise_pixelsigma(nside=256, sigma_e=(0.282842712474619, 1.207829761642)):
    if type(sigma_e) is tuple:
        sigma, n_gal_per_arcmin2 = sigma_e
    else:
        raise RuntimeError("sigma_e must be tuple (sigma_single_component,n_gal_per_arcmin2)")
    return sigma / np.sqrt(n_gal_per_arcmin2 * hp.nside2pixarea(nside, degrees=True) * 3600)


def noise_cl_cube(noise_cl):
    # TODO: separately for e,b and n?
    c_all = np.zeros((3, 3, len(noise_cl)))
    for i in range(3):
        c_all[i, i] = noise_cl
    return c_all


def get_noisy_cl(cl_objects, lmax):
    cl_es, cl_bs = [], []
    for cl_object in cl_objects:
        if cl_object is None:
            # assert that this is in place of a cross cl?
            cl_e = cl_b = np.zeros((lmax + 1))
        else:
            cl_e = cl_object.ee.copy()
            cl_b = cl_object.bb.copy()
            if hasattr(cl_object, "_noise_sigma"):
                noise_B = noise_E = cl_object.noise_cl

                cl_e += noise_E
                cl_b += noise_B
        cl_es.append(cl_e)
        cl_bs.append(cl_b)
    return tuple(cl_es), tuple(cl_bs)