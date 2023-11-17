import numpy as np
import healpy as hp
from scipy.interpolate import UnivariateSpline
from scipy.special import eval_legendre
from scipy.integrate import quad_vec
from scipy.optimize import fsolve
import wigner


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
        for i in range(1, 4)
    ]
    return [np.abs(1j**-k * derivs_at_zero[k - 1]) for k in range(1, n + 1)]


def skewness(t, cf):
    first, second, third = nth_moment(3, t, cf)
    sigma2 = second - first**2
    return (third - 3 * first * sigma2 - first**3) / np.sqrt(sigma2) ** 3


def ang_prefactors(t_in_deg, wl, kind="p"):
    # normalization factor automatically has same l_max as the pseudo-Cl summation

    t = np.radians(t_in_deg)
    norm_l = np.arange(len(wl))
    legendres = eval_legendre(norm_l, np.cos(t))
    norm = 1 / np.sum((2 * norm_l + 1) * legendres * wl) / (2 * np.pi)

    if kind == "p":
        wigners = wigner.wigner_dl(norm_l[0], norm_l[-1], 2, 2, t)
    elif kind == "m":
        wigners = wigner.wigner_dl(norm_l[0], norm_l[-1], 2, -2, t)
    else:
        raise RuntimeError("correlation function kind needs to be p or m")

    return 2 * np.pi * norm * wigners


def bin_prefactors(bin_in_deg, wl, lmax, kind="p"):
    t0, t1 = bin_in_deg
    t0, t1 = np.radians(t0), np.radians(t1)

    norm_l = np.arange(lmax + 1)
    legendres = lambda t_in_rad: eval_legendre(norm_l, np.cos(t_in_rad))
    # TODO: check whether this sum needs to be done after the integration as well (even possible?)
    norm = (
        lambda t_in_rad: 1
        / np.sum((2 * norm_l + 1) * legendres(t_in_rad) * wl[: lmax + 1])
        / (2 * np.pi)
    )
    if kind == "p":
        wigners = lambda t_in_rad: wigner.wigner_dl(0, lmax, 2, 2, t_in_rad)
    elif kind == "m":
        wigners = lambda t_in_rad: wigner.wigner_dl(0, lmax, 2, -2, t_in_rad)
    else:
        raise RuntimeError("correlation function kind needs to be p or m")

    integrand = lambda t_in_rad: norm(t_in_rad) * wigners(t_in_rad) * t_in_rad
    # norm * d_l * weights
    W = 0.5 * (t1**2 - t0**2)  # weights already integrated
    A_ell = quad_vec(integrand, t0, t1)

    return 2 * np.pi * A_ell[0] / W


def cl_decay(sigma, *args):
    lmax, thres = args
    return -lmax * (lmax + 1) * sigma**2 - 2 * np.log(2 * lmax + 1) - thres


def get_smoothing_sigma_weird(lmax, thres):
    s0 = 1 / 180 * np.pi
    res = fsolve(cl_decay, s0, args=(lmax, thres), maxfev=10000)
    print(res.x)
    return res


def get_smoothing_sigma(lmax, thres):
    return np.sqrt(-(thres) / (lmax * (lmax + 1)))
