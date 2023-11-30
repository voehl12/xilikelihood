import numpy as np
import healpy as hp
from scipy.integrate import quad
import pymaster as nmt
import treecorr
import wigner
from scipy.special import eval_legendre
import matplotlib.pyplot as plt
from numpy.random import default_rng
from helper_funcs import get_noise_pixelsigma, get_noise_cl, pcl2xi
from cov_setup import Cov


class TwoPointSimulation(Cov):
    # inherits theory c_ell and mask stuff, use properties to set up simulations (not for single simulation)
    def __init__(
        self,
        seps_in_deg,
        ximode="namaster",
        batchsize=1,
        simpath="simulations/",
        exact_lmax=None,
        spins=[2],
        lmax=None,
        sigma_e=None,
        clpath=None,
        theory_lmin=2,
        clname="3x2pt_kids_55",
        maskpath=None,
        circmaskattr=None,
        lmin=None,
        maskname="mask",
        l_smooth=None,
        smooth_signal=False,
    ):
        super().__init__(
            exact_lmax,
            spins,
            lmax,
            sigma_e,
            clpath,
            theory_lmin,
            clname,
            maskpath,
            circmaskattr,
            lmin,
            maskname,
            l_smooth,
            smooth_signal,
        )

        self.ximode = ximode
        self.batchsize = batchsize
        self.simpath = simpath
        self.seps_in_deg = seps_in_deg

    def create_maps(self):
        """
        Create Gaussian random maps from C_l
        C_l need to be tuple of arrays in order:  TT, TE, EE, *BB
        * are zero for pure shear
        """
        if self.clpath is None:
            npix = hp.nside2npix(self.nside)
            zeromaps = np.zeros((3, npix))
            return zeromaps
        else:
            np.random.seed()
            maps = hp.sphtfunc.synfast(
                (self.nn, self.ne, self.ee, self.bb), self.nside, lmax=self.lmax, verbose=True
            )
            return maps

    def add_noise(self, maps, testing=False):
        if hasattr(self, "pixelsigma"):
            sigma = self.pixelsigma
        else:
            return maps
        rng = default_rng()

        noise_map_q = self.limit_noise(rng.normal(size=(self.npix), scale=sigma), self.lmax)
        noise_map_u = self.limit_noise(rng.normal(size=(self.npix), scale=sigma), self.lmax)

        if testing == True:
            cl_t = hp.anafast(noise_map_q)
            assert np.allclose(np.mean(cl_t), get_noise_cl(), rtol=1e-01), (
                np.mean(cl_t),
                get_noise_cl(),
            )

        maps[1] += noise_map_q
        maps[2] += noise_map_u

        return maps

    def limit_noise(self, noisemap):
        almq = hp.map2alm(noisemap)
        clq = hp.sphtfunc.alm2cl(almq)

        if self.smooth_signal:
            clq *= self.smooth_array
        np.random.seed()
        assert np.allclose(clq, self.noise_cl)
        return hp.sphtfunc.synfast(clq, self.nside, lmax=self.lmax)

    def get_pcl(self, maps_TQU):
        if self.maskname == "fullsky":
            cl_t, cl_e, cl_b, cl_te, cl_eb, cl_tb = hp.anafast(maps_TQU)
            return cl_e, cl_b, cl_eb
        else:
            # f_0 = nmt.NmtField(mask, [maps_TQU[0]])
            f_2 = nmt.NmtField(self.mask, maps_TQU[1:])
            # cl_00 = nmt.compute_coupled_cell(f_0, f_0)
            # cl_02 = nmt.compute_coupled_cell(f_2, f_0)
            cl_22 = nmt.compute_coupled_cell(f_2, f_2)
            pcl_e, pcl_b, pcl_eb = cl_22[0], cl_22[3], (cl_22[1] + cl_22[2]) / 2
            return pcl_e, pcl_b, pcl_eb

    def get_xi_namaster(self, maps_TQU, sep_in_deg, lmin=0):
        # pcl2xi should work for several angles at once.
        pcl_22 = self.get_pcl(maps_TQU)
        xi_p, xi_m = pcl2xi(pcl_22, sep_in_deg, self.wl, self._exact_lmax, self.lmax, lmin=lmin)
        return xi_p, xi_m

    def xi_sim(self, j, seps_in_deg, lmin=0):
        xip, xim = [], []

        if self.ximode == "namaster":
            for _i in range(self.batchsize):
                maps_TQU = self.create_maps()
                maps = self.add_noise(maps_TQU)
                xi_p, xi_m = self.get_xi_namaster(maps, seps_in_deg, lmin)
                xip.append(xi_p)
                xim.append(xi_m)

            np.savez(
                self.simpath + "/job{:d}.npz".format(j),
                mode=self.ximode,
                theta=seps_in_deg,
                lmin=lmin,
                xip=np.array(xip),
                xim=np.array(xim),
            )


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
