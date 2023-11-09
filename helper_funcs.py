import numpy as np
import healpy as hp
from scipy.interpolate import UnivariateSpline


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
    sigma, n_gal_per_arcmin2 = sigma_e
    return (sigma) ** 2 / (n_gal_per_arcmin2 * 3600 * 41253 / (4 * np.pi))


def get_noise_pixelsigma(nside=256, sigma_e=(0.282842712474619, 1.207829761642)):
    sigma, n_gal_per_arcmin2 = sigma_e
    return sigma / np.sqrt(n_gal_per_arcmin2 * hp.nside2pixarea(nside, degrees=True) * 3600)


def noise_cl(sigma2, len_l):
    # separately for e,b and n?
    c_all = np.zeros((3, 3, len_l))
    for i in range(3):
        c_all[i, i] = np.ones(len_l)
    return sigma2 * c_all


def gaussian_cf(t, mu, sigma):
    return np.exp(1j * t * mu - 0.5 * sigma**2 * t**2)


def nth_moment(n, t, cf):
    k = 5  # 5th degree spline
    derivs_at_zero = [
        1j * UnivariateSpline(t, cf.imag, k=k, s=0).derivative(n=i)(0)
        + UnivariateSpline(t, cf.real, k=k, s=0).derivative(n=i)(0)
        for i in range(1, 3)
    ]
    return np.abs(1j**-n * derivs_at_zero[n - 1])
