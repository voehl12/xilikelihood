import numpy as np
import healpy as hp
from scipy.integrate import quad
import pymaster as nmt
import treecorr
import wigner
from scipy.special import eval_legendre
import matplotlib.pyplot as plt
from numpy.random import default_rng
from helper_funcs import get_noise_pixelsigma, get_noise_cl


def create_maps(cl=None, nside=256, lmax=None):
    """
    Create Gaussian random maps from C_l
    C_l need to be tuple of arrays in order:  TT, TE, EE, *BB
    * are zero for pure shear
    """
    if cl is None:
        npix = hp.nside2npix(nside)
        zeromaps = np.zeros((3, npix))
        return zeromaps
    else:
        np.random.seed()
        if lmax:
            lmax = lmax
        else:
            lmax = 3 * nside - 1
        maps = hp.sphtfunc.synfast(cl, nside, lmax=lmax, verbose=True)
        return maps


def add_noise(maps, sigma_n=None, lmax=None, testing=False):
    nside = hp.get_nside(maps[0])
    if sigma_n is None:
        return maps
    elif sigma_n == "default":
        sigma_n = get_noise_pixelsigma(nside=nside)
    npix = hp.get_map_size(maps[0])
    rng = default_rng()
    if lmax is not None:
        noise_map_q = limit_noise(rng.normal(size=(npix), scale=sigma_n), lmax)
        noise_map_u = limit_noise(rng.normal(size=(npix), scale=sigma_n), lmax)
        # if i limit the noise, the measured C_ell will be smaller than the input variance, therefore the measured correlation function will be smaller than the predicted one.
    else:
        noise_map_q = rng.normal(size=(npix), scale=sigma_n)
        noise_map_u = rng.normal(size=(npix), scale=sigma_n)
    if testing == True:
        cl_t = hp.anafast(noise_map_q)
        assert np.allclose(np.mean(cl_t), get_noise_cl(), rtol=1e-01), (
            np.mean(cl_t),
            get_noise_cl(),
        )

    maps[1] += noise_map_q
    maps[2] += noise_map_u

    return maps


def limit_noise(noisemap, lmax):
    nside = hp.get_nside(noisemap)
    almq = hp.map2alm(noisemap)
    clq = hp.sphtfunc.alm2cl(almq)
    np.random.seed()
    return hp.sphtfunc.synfast(clq, nside, lmax=lmax)


def get_pcl(maps_TQU, mask=None):
    if mask is None:
        cl_t, cl_e, cl_b, cl_te, cl_eb, cl_tb = hp.anafast(maps_TQU)
        return cl_e, cl_b, cl_eb
    else:
        # f_0 = nmt.NmtField(mask, [maps_TQU[0]])
        f_2 = nmt.NmtField(mask, maps_TQU[1:])
        # cl_00 = nmt.compute_coupled_cell(f_0, f_0)
        # cl_02 = nmt.compute_coupled_cell(f_2, f_0)
        cl_22 = nmt.compute_coupled_cell(f_2, f_2)
        pcl_e, pcl_b, pcl_eb = cl_22[0], cl_22[3], (cl_22[1] + cl_22[2]) / 2
        return pcl_e, pcl_b, pcl_eb


def mask_to_wlm(mask, lmax=None):
    """
    calculate spherical harmonic transform of a given mask.
    """
    return hp.sphtfunc.map2alm(mask, lmax=lmax)


def cl2xi(pcl_22, lmax=None, mask=None, norm_lm=False):
    """
    function to convert pseudo-Cl to correlation function estimator for given angular separation
    right now takes the same l-limits for the normalization factor, but this might have to be adjusted
    to taking the maximum lmax here, since it would be no problem computationally
    (unlike for the correlation function summation and therefore covariance matrix)
    norm_lm: whether the lmax for the norm should be a different one. If True, lmax for the norm will be 3*nside - 1,
    regardless of lmax for the pseudo_Cl summation. Turns out, this only makes a difference if lmax is less than 20% of the
    3nside-1 lmax.
    returns xi_p and xi_m as functions of angle in degree and lmin.
    """

    cl_e, cl_b, cl_eb = pcl_22[0], pcl_22[1], pcl_22[2]
    if lmax is None:
        lmax = len(cl_e) - 1
    if lmax > len(cl_e):
        raise RuntimeError("lmax for correlation function estimator is too large!")
    else:
        cl_short = lambda lmin: (
            cl_e[lmin : lmax + 1],
            cl_b[lmin : lmax + 1],
            cl_eb[lmin : lmax + 1],
        )
        l = lambda lmin: np.arange(lmin, lmax + 1)

    wigners_p = lambda t_in_deg, lmin: wigner.wigner_dl(lmin, lmax, 2, 2, np.radians(t_in_deg))
    wigners_m = lambda t_in_deg, lmin: wigner.wigner_dl(lmin, lmax, 2, -2, np.radians(t_in_deg))

    if mask is None:
        mask = np.ones(hp.nside2npix(256))
        # more precise than setting
        # norm = lambda t_in_deg, lmin: 1 / (8 * np.pi**2)

    if norm_lm == True:
        w_lm = mask_to_wlm(mask)
        wl = hp.sphtfunc.alm2cl(w_lm)
        l_norm = np.arange(len(wl))
        legendres = lambda t_in_deg: eval_legendre(l_norm, np.cos(np.radians(t_in_deg)))
        norm = (
            lambda t_in_deg, lmin: 1
            / np.sum((2 * l_norm + 1) * legendres(t_in_deg) * wl)
            / (2 * np.pi)
        )

    else:
        w_lm = mask_to_wlm(mask, lmax)
        wl_arr = hp.sphtfunc.alm2cl(w_lm)
        wl = lambda lmin: wl_arr[lmin:]
        legendres = lambda t_in_deg, lmin: eval_legendre(l(lmin), np.cos(np.radians(t_in_deg)))
        norm = (
            lambda t_in_deg, lmin: 1
            / np.sum((2 * l(lmin) + 1) * legendres(t_in_deg, lmin) * wl(lmin))
            / (2 * np.pi)
        )

    xi_p = (
        lambda t_in_deg, lmin: 2
        * np.pi
        * norm(np.radians(t_in_deg), lmin)
        * np.sum(
            (2 * l(lmin) + 1) * wigners_p(t_in_deg, lmin) * (cl_short(lmin)[0] + cl_short(lmin)[1])
        )
    )
    xi_m = (
        lambda t_in_deg, lmin: 2
        * np.pi
        * norm(np.radians(t_in_deg), lmin)
        * np.sum(
            (2 * l(lmin) + 1)
            * wigners_m(t_in_deg, lmin)
            * (cl_short(lmin)[0] - cl_short(lmin)[1] - 2j * cl_short(lmin)[2])
        )
    )
    return xi_p, xi_m


def prep_cat_treecorr(nside, mask=None):
    """takes healpy mask, returns mask needed for treecorr simulation"""
    if mask is None:
        all_pix = np.arange(hp.nside2npix(nside))
        phi, thet = hp.pixelfunc.pix2ang(nside, all_pix, lonlat=True)
        treecorr_mask_cat = (None, phi, thet)
    else:
        all_pixs = np.arange(len(mask))
        mask = np.array(1 - mask, dtype=int)
        masked_pixs = np.ma.array(all_pixs, mask=mask).compressed()
        phi, thet = hp.pixelfunc.pix2ang(nside, masked_pixs, lonlat=True)
        tqu_mask = np.tile(mask, (3, 1))
        treecorr_mask_cat = (tqu_mask, phi, thet)
    return treecorr_mask_cat


def prep_angles_treecorr(seps_in_deg):
    if type(seps_in_deg[0]) is tuple:
        bin_size = seps_in_deg[0][1] - seps_in_deg[0][0]
        thetamin = seps_in_deg[0][0]
        thetamax = seps_in_deg[-1][1]
    else:
        raise RuntimeError("Angular separations need to be bins for treecorr!")
    return thetamin, thetamax, bin_size


def get_xi_treecorr(maps_TQU, treecorr_seps, cat_angles, mask_treecorr=None):
    """takes maps and mask, returns treecorr correlation function"""
    """need to run cat and angles prep once before simulating"""
    phi, thet = cat_angles
    if mask_treecorr is not None:
        tqu_mask = mask_treecorr
        masked_TQU = np.ma.array(maps_TQU, mask=tqu_mask)
        maps_TQU = np.ma.compress_cols(masked_TQU)
    thetamin, thetamax, bin_size = treecorr_seps
    cat = treecorr.Catalog(
        ra=phi, dec=thet, g1=-maps_TQU[1], g2=maps_TQU[2], ra_units="deg", dec_units="deg"
    )
    gg = treecorr.GGCorrelation(
        min_sep=thetamin,
        max_sep=thetamax,
        bin_size=bin_size,
        bin_type="Linear",
        sep_units="deg",
        bin_slop=0,
    )
    gg.process(cat)
    xi_p = gg.xip
    xi_m = gg.xim
    angsep = gg.rnom
    return xi_p, xi_m, angsep


def get_xi_namaster(maps_TQU, seps_in_deg, mask=None, lmax=None, lmin=[0], norm_lm=True):
    pcl_22 = get_pcl(maps_TQU, mask)
    xi_p, xi_m = cl2xi(
        pcl_22, lmax=lmax, mask=mask, norm_lm=norm_lm
    )  # as functions of angle [degrees] and minimum multipole moment

    # as xi_p is already a function of t: could implement binned version here
    # check whether theta is a list of angles or tuples of angles
    if type(seps_in_deg[0]) is tuple:
        t = np.empty(len(seps_in_deg), dtype=object)  # make an array of tuples
        t[:] = seps_in_deg  # put boundary tuples into array in order to make a meshgrid
        t_grid, l_grid = np.meshgrid(t, lmin)
        # xip/xim are defined as functions of angle in degrees, so integration bounds should be in degrees
        # found error: integration needs to be weighted by t.
        weight = lambda t: t
        xip_integrand = lambda t_in_deg, lm: xi_p(t_in_deg, lm) * weight(t_in_deg)
        xim_integrand = lambda t_in_deg, lm: xi_m(t_in_deg, lm) * weight(t_in_deg)
        xip_binned = lambda th, lm: quad(xip_integrand, th[0], th[1], args=(lm,))[0] / (
            0.5 * (th[1] ** 2 - th[0] ** 2)
        )
        xim_binned = lambda th, lm: quad(xim_integrand, th[0], th[1], args=(lm,))[0] / (
            0.5 * (th[1] ** 2 - th[0] ** 2)
        )
        xi_p_arr = list(map(xip_binned, t_grid.flat, l_grid.flat))
        xi_m_arr = list(map(xim_binned, t_grid.flat, l_grid.flat))
    else:
        t_grid, l_grid = np.meshgrid(seps_in_deg, lmin)
        xi_p_arr = list(map(xi_p, t_grid.flat, l_grid.flat))
        xi_m_arr = list(map(xi_m, t_grid.flat, l_grid.flat))
    # is reshape correct? yes.
    return np.real(np.array(xi_p_arr).reshape(len(lmin), len(seps_in_deg))), np.real(
        np.array(xi_m_arr).reshape(len(lmin), len(seps_in_deg))
    )


def set_cl(cl_object, nside):
    if cl_object is None:
        lmax = 3 * nside - 1
        cl_tup = None
    else:
        lmax = cl_object.lmax
        cl_tup = (cl_object.nn, cl_object.ne, cl_object.ee, cl_object.bb)
    return lmax, cl_tup


def xi_sim(
    j,
    cl,
    seps_in_deg,
    nside=256,
    mask=None,
    sigma_n=None,
    mode="namaster",
    lmin=[0],
    batchsize=1,
    path="simulations/",
    noiselmax=None,
    testing=False,
):
    # ='/cluster/scratch/veoehl/xi_sims/xi_lm60_circmask_kids55_binned_treecorr'):
    xip, xim = [], []
    lmax, cl_tup = set_cl(cl, nside)

    if noiselmax is not None:
        lmax = noiselmax

    if mode == "namaster":
        for i in range(batchsize):
            maps_TQU = create_maps(cl_tup, lmax=lmax, nside=nside)
            maps = add_noise(maps_TQU, sigma_n, lmax=lmax, testing=testing)

            res = get_xi_namaster(maps, seps_in_deg, mask)
            xi_p, xi_m = res
            xip.append(xi_p)
            xim.append(xi_m)

        np.savez(
            path + "/job{:d}.npz".format(j),
            mode=mode,
            theta=seps_in_deg,
            lmin=lmin,
            xip=np.array(xip),
            xim=np.array(xim),
        )

    elif mode == "treecorr":
        tqu_mask, phi, thet = prep_cat_treecorr(nside, mask)
        treecorr_seps = prep_angles_treecorr(seps_in_deg)
        for i in range(batchsize):
            maps_TQU = create_maps(cl_tup, lmax=lmax, nside=nside)
            maps = add_noise(maps_TQU, sigma_n, lmax=lmax, testing=testing)
            res = get_xi_treecorr(maps, treecorr_seps, (phi, thet), tqu_mask)
            xi_p, xi_m, angsep = res
            xip.append(xi_p)
            xim.append(xi_m)
        np.savez(
            path + "/job{:d}.npz".format(j),
            mode=mode,
            theta=angsep,
            xip=np.array(xip),
            xim=np.array(xim),
        )

    else:  # both estimators to compare.
        tqu_mask, phi, thet = prep_cat_treecorr(nside, mask)
        treecorr_seps = prep_angles_treecorr(seps_in_deg)
        xip_t, xim_t = [], []

        for i in range(batchsize):
            maps_TQU = create_maps(cl_tup, lmax=lmax, nside=nside)
            maps = add_noise(maps_TQU, sigma_n, lmax=lmax, testing=testing)
            # on full sky treecorr and namaster only agree if noise limitation is set to current lmax. why?! use_pixel_weights in healpy does not seem to matter.
            # actually, treecorr just does way better without the bandlimit - namaster returns the same result because the summation is always just done up to the bandlimit.
            # does the magnitude of the correlation function depend on this limit? it seems to be roughly halfing when doubling the nside
            # the magnitude of the noise correlation function definitely depends on the noise sigma.
            res = get_xi_namaster(
                maps, seps_in_deg, mask
            )  # treecorr and namaster agree much better if namaster is used with the maximum possible lmax. The implemented binning seems fine.
            xi_p_n, xi_m_n = res
            xip.append(xi_p_n)
            xim.append(xi_m_n)

            res = get_xi_treecorr(maps, treecorr_seps, (phi, thet), mask_treecorr=tqu_mask)
            xi_p, xi_m, angsep = res

            xip_t.append(xi_p)
            xim_t.append(xi_m)
        np.savez(
            path + "job{:d}.npz".format(j),
            mode=mode,
            theta=seps_in_deg,
            xip_n=np.array(xip),
            xim_n=np.array(xim),
            xip_t=np.array(xip_t),
            xim_t=np.array(xim_t),
        )


def pcl_sim(cell_object, ells, mask_file):
    lmax, cl_tup = set_cl(cl, nside)
    pcls_e, pcls_b, pcls_eb = [], [], []
    for i in range(batchsize):
        maps_TQU = create_maps(cl_tup, lmax=lmax, nside=nside)
        maps = add_noise(maps_TQU, sigma_n, lmax=lmax)
        pcl_e, pcl_b, pcl_eb = get_pcl(maps_TQU, mask)
    pass
