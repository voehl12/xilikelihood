import wigner
import healpy as hp
import numpy as np
import time
from scipy.special import factorial


# what if the mask is a class containing information about spins, geometry and the wlmlpmp?


def w_factor(l, l1, l2):
    return np.sqrt((2 * l1 + 1) * (2 * l2 + 1) * (2 * l + 1) / (4 * np.pi))


def prepare_wigners(spin, L1, L2, M1, M2, lmax):
    # prepare an array of wigners (or products thereof) of allowed l to be summed for given m1,m2,l1,l2

    m = M1 - M2
    l_arr = np.arange(lmax + 1)
    len_wigners = len(l_arr)
    wigners1 = wigners_on_array(L1, L2, -M1, M2, lmax)

    if spin == 0:
        wigners0 = wigners_on_array(L1, L2, 0, 0, lmax)
        return l_arr, wigners1 * wigners0

    elif spin == 2:
        wigners2p, wigners2m = wigners_on_array(L1, L2, 2, -2, lmax), wigners_on_array(
            L1, L2, -2, 2, lmax
        )

        w2sum = wigners2p + wigners2m
        w2diff = wigners2p - wigners2m
        wp_l = w2sum * wigners1
        wm_l = w2diff * wigners1
        return l_arr, wp_l, wm_l

    else:
        raise RuntimeError("Wigner 3j-symbols can only be calculated for spin 0 or 2 fields.")


def get_wlm_l(wlm, m, lmax, allowed_l):
    if m < 0:
        wlm_l = (-1) ** -m * np.conj(wlm[hp.sphtfunc.Alm.getidx(lmax, allowed_l, -m)])
    else:
        wlm_l = wlm[hp.sphtfunc.Alm.getidx(lmax, allowed_l, m)]

    return wlm_l


def shorten_wigners(wigner_lmax, lmax, wigners_l):
    if wigner_lmax > lmax:
        diff = int(wigner_lmax) - lmax
        wigner_lmax = lmax
        return wigner_lmax, wigners_l[:-diff]
    else:
        return wigner_lmax, wigners_l


def wigners_on_array(l1, l2, m1, m2, lmax):
    l_array = np.arange(lmax + 1)
    wigners = np.zeros(len(l_array))
    wlmin, wlmax, wcof = wigner.wigner_3jj(l1, l2, m1, m2)
    wlmax, wcof = shorten_wigners(wlmax, lmax, wcof)
    if len(wcof) == 0:
        return wigners
    else:
        min_ind, max_ind = np.argmin(np.fabs(wlmin - l_array)), np.argmin(np.fabs(wlmax - l_array))
        wigners[min_ind : max_ind + 1] = wcof
        return wigners


def purified_wigner(l_arr, lp, lpp, purified=True):
    lmax = l_arr[-1]

    wigners2 = np.zeros(len(l_arr))

    wigners0 = wigners_on_array(lp, lpp, -2, 0, lmax)
    if purified == False:
        return wigners0
    else:
        if lpp >= 1:
            prefac1 = 2 * np.sqrt(
                (factorial(l_arr + 1) * factorial(l_arr - 1) * factorial(lpp + 1))
                / (factorial(l_arr - 1) * factorial(l_arr + 2) * factorial(lpp - 1))
            )

            wigners1 = wigners_on_array(lp, lpp, -2, 1, lmax)
            wigners1 *= prefac1
        if lpp >= 2:
            prefac2 = np.sqrt(
                (factorial(l_arr + 2) * factorial(lpp + 2))
                / (factorial(l_arr + 2) * factorial(lpp - 2))
            )

            wigners2 = wigners_on_array(lp, lpp, -2, 2)
            wigners2 *= prefac2

        return wigners0 + wigners1 + wigners2


def m_llp(wlm, exact_lmax):
    wl = hp.sphtfunc.alm2cl(wlm)
    len_l = len(wl)
    m_3d_pp = np.zeros((len_l, len_l, exact_lmax + 1))
    m_3d_mm = np.zeros_like(m_3d_pp)
    l = lp = np.arange(len_l)
    lpp = np.arange(exact_lmax + 1)

    for cp, i in enumerate(lp):
        for cpp, j in enumerate(lpp):
            if i < 2:
                continue
            else:
                wigners_l = purified_wigner(l, i, j, purified=False)
                m_3d = (2 * i + 1) * (2 * j + 1) * wl[cpp] * np.square(wigners_l) / (4 * np.pi)
                m_3d_pp[:, cp, cpp] = m_3d * 0.5 * (1 + (-1) ** (l + i + j))
                m_3d_mm[:, cp, cpp] = m_3d * 0.5 * (1 - (-1) ** (l + i + j))

    return np.sum(m_3d_pp, axis=2), np.sum(m_3d_mm, axis=2)


def smooth_gauss(l, l_smooth):
    sigma2 = l_smooth**2 / 5 * np.log10(np.e)
    return np.exp(-(l**2) / (2 * sigma2))


def smooth_alm(alm, l_smooth, lmax):
    l_array = [np.arange(i, lmax + 1) for i in range(lmax + 1)]
    l_array = np.concatenate(l_array, axis=0)
    smoothing_arr = smooth_gauss(l_array, l_smooth)
    return alm * smoothing_arr
