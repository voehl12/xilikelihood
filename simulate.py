import os

import numpy as np
import healpy as hp

# import pymaster as nmt
# import treecorr

import matplotlib.pyplot as plt
from numpy.random import default_rng
from helper_funcs import get_noise_cl, pcl2xi, prep_prefactors
from cov_setup import Cov
import helper_funcs

from typing import Any, Union, Tuple, Generator, Optional, Sequence, Callable, Iterable

import glass.fields

# glass          2023.8.dev10+g67a5721 /cluster/home/veoehl/glass
import time


class TwoPointSimulation:
    # inherits theory c_ell and mask stuff, use properties to set up simulations (not for single simulation)
    def __init__(
        self,
        seps_in_deg,
        mask,
        theorycl,
        ximode="namaster",
        batchsize=1,
        simpath=None,
        lmax=None,
        healpix_datapath=None,
    ):

        self.mask = mask
        self.theorycl = theorycl

        self.ximode = ximode
        self.batchsize = batchsize
        if simpath is None:
            current = os.getcwd()
            if not os.path.isdir(current + "/simulations"):
                command = "mkdir simulations"
                os.system(command)
            self.simpath = current + "/simulations"
        else:
            self.simpath = simpath
        self.seps_in_deg = seps_in_deg
        self.healpix_datapath = healpix_datapath
        self.lmax = lmax

    def create_maps(self, seed=None):
        """
        Create Gaussian random maps from C_l
        C_l need to be tuple of arrays in order:  TT, TE, EE, *BB
        * are zero for pure shear
        """
        if self.theorycl.clpath is None:
            npix = hp.nside2npix(self.mask.nside)
            zeromaps = np.zeros((3, npix))
            return zeromaps
        else:
            np.random.seed(seed=seed)
            maps = hp.sphtfunc.synfast(
                (self.theorycl.nn, self.theorycl.ne, self.theorycl.ee, self.theorycl.bb),
                self.mask.nside,
                lmax=self.mask.lmax,
            )
            return maps

    def add_noise(self, maps, testing=False):
        if self.theorycl._sigma_e is not None:
            self.pixelsigma = set_pixelsigma(self.theorycl, self.mask.nside)
        else:
            return maps
        rng = default_rng()

        noise_map_q = self.limit_noise(rng.normal(size=(self.mask.npix), scale=self.pixelsigma))
        noise_map_u = self.limit_noise(rng.normal(size=(self.mask.npix), scale=self.pixelsigma))

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

        if self.theorycl.smooth_signal is not None:
            clq *= self.theorycl.smooth_array

        # assert np.allclose(clq[2*self.smooth_signal], self.noise_cl[2*self.smooth_signal], rtol=1e-01),(clq,self.noise_cl)
        np.random.seed()
        return hp.sphtfunc.synfast(clq, self.nside, lmax=self.lmax)

    def get_pcl(self, maps_TQU):
        if self.mask.name == "fullsky":
            cl_t, cl_e, cl_b, cl_te, cl_eb, cl_tb = hp.anafast(maps_TQU)
            return cl_e, cl_b, cl_eb
        else:
            if hasattr(self.mask, "smooth_mask"):
                mask = self.mask.smooth_mask
            else:
                mask = self.mask.mask
            maps_TQU_masked = mask[None, :] * maps_TQU
            pcl_t, pcl_e, pcl_b, pcl_te, pcl_eb, pcl_tb = hp.anafast(
                maps_TQU_masked,
                iter=5,
                use_pixel_weights=True,
                datapath=self.healpix_datapath,
                # "/cluster/home/veoehl/2ptlikelihood/masterenv/lib/python3.8/site-packages/healpy/data/",
            )
            return pcl_e, pcl_b, pcl_eb

    def get_xi_namaster(self, maps_TQU, prefactors, lmin=0, pixwin_p=None):
        # pcl2xi should work for several angles at once.
        pcl_22 = self.get_pcl(maps_TQU)
        if pixwin_p is not None:
            pcl_22 = pcl_22 / pixwin_p[None, :] ** 2
        xi_p, xi_m = pcl2xi(pcl_22, prefactors, self.lmax, lmin=lmin)
        return pcl_22, xi_p, xi_m

    def xi_sim_1D(self, j, lmin=0, plot=False, save_pcl=False, pixwin=False):
        xip, xim, pcls = [], [], []
        foldername = "/{}_{}_{}_{}".format(self.clname, self.maskname, self.sigmaname, self.ximode)
        path = self.simpath + foldername
        if not os.path.isdir(path):
            command = "mkdir " + path
            os.system(command)
        self.simpath = path

        if self.ximode == "namaster":
            prefactors = prep_prefactors(
                self.seps_in_deg, self.mask.wl, self.mask.lmax, self.mask.lmax
            )  # exact lmax shoukld not be used here.
            if pixwin:
                pixwin_t, pixwin_p = hp.pixwin(self.mask.nside, pol=True)
                pixwin_p[:2] = np.ones(2)
            else:
                pixwin_p = None
            for _i in range(self.batchsize):
                print(
                    "Simulating xip and xim......{:4.1f}%".format(_i / self.batchsize * 100),
                    end="\r",
                )
                maps_TQU = self.create_maps()
                maps = self.add_noise(maps_TQU)
                pcl, xi_p, xi_m = self.get_xi_namaster(maps, prefactors, lmin, pixwin_p=pixwin_p)
                xip.append(xi_p)
                xim.append(xi_m)
                pcls.append(pcl)
            xip = np.array(xip)
            xim = np.array(xim)
            pcls = np.array(pcls)
            if plot:
                plt.figure()
                plt.hist(xip[:, 0], bins=30)
                plt.savefig("sim_demo_{:d}.png".format(j))
            np.savez(
                self.simpath + "/job{:d}.npz".format(j),
                mode=self.ximode,
                theta=self.seps_in_deg,
                lmax=self.lmax,
                lmin=lmin,
                xip=np.array(xip),
                xim=np.array(xim),
            )
            if save_pcl:
                np.savez(
                    self.simpath + "/pcljob{:d}.npz".format(j),
                    lmin=lmin,
                    lmax=self.mask.lmax,
                    pcl_e=np.array(pcls[:, 0]),
                    pcl_b=np.array(pcls[:, 1]),
                    pcl_eb=np.array(pcls[:, 2]),
                )

        elif self.ximode == "treecorr":

            cat_props = prep_cat_treecorr(self.mask.nside, self.mask.smooth_mask)
            for _i in range(self.batchsize):
                print(
                    "Simulating xip and xim......{:4.1f}%".format(_i / self.batchsize * 100),
                    end="\r",
                )
                maps_TQU = self.create_maps()
                maps = self.add_noise(maps_TQU)
                xi_p, xi_m, angsep = get_xi_treecorr(maps, self.seps_in_deg, cat_props)

                xip.append(xi_p)
                xim.append(xi_m)
            xip = np.array(xip)
            xim = np.array(xim)
            if plot:
                plt.figure()
                plt.hist(xip[:, 0], bins=10)
                plt.savefig("sim_demo1_{:d}.png".format(j))

            if os.path.isfile(path + "/job{:d}.npz".format(j)):
                xifile = np.load(path + "/job{:d}.npz".format(j))
                angs = xifile["theta"]
                for a, ang in enumerate(angsep):
                    if ang not in angs:
                        angsep = np.append(angs, [ang], axis=0)
                        xip = np.append(xifile["xip"], xip, axis=1)
                        xim = np.append(xifile["xim"], xim, axis=1)

            np.savez(
                path + "/job{:d}.npz".format(j),
                mode=self.ximode,
                theta=angsep,
                xip=np.array(xip),
                xim=np.array(xim),
            )

        elif self.ximode == "comp":
            cat_props = prep_cat_treecorr(self.mask.nside, self.mask.smooth_mask)
            prefactors = prep_prefactors(
                self.seps_in_deg, self.mask.wl, self.mask.lmax, self.mask.lmax
            )

            maps_TQU = self.create_maps(seed=7)
            maps = self.add_noise(maps_TQU)
            xi_p_t, xi_m_t, angsep = get_xi_treecorr(maps, self.seps_in_deg, cat_props)
            pcl, xi_p_n, xi_m_n = self.get_xi_namaster(maps, prefactors, lmin, pixwin_p=None)
            self.comp = [xi_p_t, xi_p_n]

        else:
            raise RuntimeError("Simulation mode can either be namaster, treecorr or comp")


def create_maps_nD(gls=[], nside=256, lmax=None):
    """
    Create Gaussian random maps from C_l
    C_l need to be list of cl-arrays in order:  11,22,12,33,32,31,... for cross-correlations
    only E-mode C_ells! (T and B are set to zero within glass)
    """
    if len(gls) == 0:
        npix = hp.nside2npix(nside)
        zeromaps = np.zeros((3, npix))
        return zeromaps
    else:
        fields = glass.fields.generate_gaussian(gls, nside=nside)
        field_list = []
        while True:
            try:
                mapsTQU = next(fields)
                field_list.append(mapsTQU)
            except StopIteration:
                break

        return np.array(field_list)


def set_pixelsigma(theorycl, nside):
    if theorycl.sigma_e is not None:
        if isinstance(theorycl.sigma_e, str):
            pixelsigma = helper_funcs.get_noise_pixelsigma(nside)

        elif isinstance(theorycl.sigma_e, tuple):
            pixelsigma = helper_funcs.get_noise_pixelsigma(nside, theorycl.sigma_e)
        else:
            raise RuntimeError("sigma_e needs to be string for default or tuple (sigma_e,n_gal)")
    else:
        pixelsigma = None
    return pixelsigma


def limit_noise(noisemap, nside: int, lmax: Optional[int] = None):
    almq = hp.map2alm(noisemap)
    clq = hp.sphtfunc.alm2cl(almq)

    # assert np.allclose(clq[2*self.smooth_signal], self.noise_cl[2*self.smooth_signal], rtol=1e-01),(clq,self.noise_cl)
    np.random.seed()
    return hp.sphtfunc.synfast(clq, nside, lmax=lmax)


def add_noise_nD(maps_TQU_list, nside: int, sigmas=None, lmax=None):
    if sigmas is None:
        return maps_TQU_list
    else:
        assert len(sigmas) == len(maps_TQU_list), (sigmas, maps_TQU_list)
        size = len(maps_TQU_list[0, 0])

        for i, maps_TQU in enumerate(maps_TQU_list):
            rng = default_rng()

            noise_map_q = limit_noise(rng.normal(size=size, scale=sigmas[i]), nside, lmax)
            noise_map_u = limit_noise(rng.normal(size=size, scale=sigmas[i]), nside, lmax)
            maps_TQU[1] += noise_map_q
            maps_TQU[2] += noise_map_u

        return maps_TQU_list


def get_pcl_nD(maps_TQU_list, smooth_masks, fullsky=False):
    """
    measures sets of pseudo_cl. if nD, the pcl are ordered as 00,11,10,22,21,20,...

    Parameters
    ----------
    maps_TQU_list : _type_
        _description_
    smooth_masks : _type_
        _description_
    fullsky : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """

    if fullsky:
        cl_s = []
        for i, field_i in enumerate(maps_TQU_list):
            for j, field_j in reversed(list(enumerate(maps_TQU_list[: i + 1]))):
                # cl_t, cl_e, cl_b, cl_te, cl_eb, cl_tb order of cl_s
                cl_s.append(hp.anafast(field_i, field_j))
        cl_s = np.array(cl_s)[:, [1, 2, 4]]
        return cl_s  # cl_e, cl_b, cl_eb
    else:
        pcl_s = []
        masked_fields = smooth_masks[:,None,:]*np.array(maps_TQU_list)
        for i, field_i in enumerate(masked_fields):
            for j, field_j in reversed(list(enumerate(masked_fields[: i + 1]))):

                
                pcl_t, pcl_e, pcl_b, pcl_te, pcl_eb, pcl_tb = hp.anafast(
                    field_i,
                    field_j,
                    iter=5,
                    use_pixel_weights=True,
                    datapath="/cluster/home/veoehl/2ptlikelihood/masterenv/lib/python3.8/site-packages/healpy/data/",
                )

                pcl_s.append([pcl_e, pcl_b, pcl_eb])  # (n_croco, 3, n_ell)

        return np.array(pcl_s)


def get_xi_namaster_nD(maps_TQU_list, smooth_masks, prefactors, lmax, lmin=0):
    # pcl2xi should work for several angles at once.
    assert len(maps_TQU_list) == len(smooth_masks), "need to provide as many masks as fields!"
    pcl_ij = get_pcl_nD(maps_TQU_list, smooth_masks)
    n_corr = len(pcl_ij)
    xi_all = np.zeros((n_corr, 2, len(prefactors)))
    for i, pcl in enumerate(pcl_ij):
        xi_all[i] = np.array(pcl2xi(pcl, prefactors, lmax, lmin=lmin))
    return pcl_ij, xi_all


def xi_sim_nD(
    # also pass only mask entity and theorycl entities, no need for Cov?
    theorycls,
    masks,
    j,
    seps_in_deg,
    lmax=None,
    lmin=0,
    plot=False,
    save_pcl=False,
    ximode="namaster",
    batchsize=1,
    simpath="simulations/",
):
    xis, pcls = [], []
    gls = np.array([cl.ee.copy() for cl in theorycls])
    if not helper_funcs.check_property_equal(
        masks, "nside"
    ) or not helper_funcs.check_property_equal(masks, "eff_area"):
        raise RuntimeError("Different masks not implemented yet.")

    mask = masks[0]
    if not hasattr(mask, "smooth_mask"):
        print("Warning: mask used for simulations is not smoothed!")
        sim_mask = mask.mask
    else:
        sim_mask = mask.smooth_mask
    
    hp.mollview(sim_mask)
    plt.savefig('mask.png')
    nside = mask.nside
    noises = [set_pixelsigma(cl, nside) for cl in theorycls if cl.sigma_e is not None]
    sigmaname = theorycls[0].sigmaname
    foldername = "/croco_{}_{}_{}_llim_{}".format(
        theorycls[0].name, mask.name, sigmaname, str(lmax)
    )
    path = simpath + foldername
    if not os.path.isdir(path):
        command = "mkdir " + path
        os.system(command)
    simpath = path

    if ximode == "namaster":
        if lmax is None:
            lmax = mask.lmax

        prefactors = prep_prefactors(seps_in_deg, mask.wl, mask.lmax, mask.lmax)
        times = []
        for _i in range(batchsize):
            tic = time.perf_counter()
            print(
                "Simulating xip and xim......{:4.1f}%".format(_i / batchsize * 100),
                end="\r",
            )
            maps_TQU_list = create_maps_nD(gls, nside)
            maps = add_noise_nD(maps_TQU_list, nside, sigmas=noises)
            all_masks = np.array([sim_mask for _ in range(len(maps))])
            pcl_all, xi_all = get_xi_namaster_nD(maps, all_masks, prefactors, lmax, lmin=0)
            xis.append(xi_all)
            pcls.append(pcl_all)  # (n_batch, n_corr, 3, n_ell)
            toc = time.perf_counter()
            times.append(toc - tic)
        print(times, np.mean(times))
        xis = np.array(xis) # (n_batch,n_corr,2,n_angbins)
        pcls = np.array(pcls)
        if plot:
            plt.figure()
            plt.hist(xis[:, 2, 0, 1], bins=30) 
            plt.savefig("sim_demo_{:d}.png".format(j))
            plt.figure()
            # plt.plot(np.arange(pcls.shape[-1]), np.array(pcls[:, 2, 0, :]).T)
            plt.plot(
                np.arange(pcls.shape[-1]),
                np.mean(np.array(pcls[:, 2, 0, :]).T, axis=-1),
                color="black",
            )
            plt.plot(
                np.arange(pcls.shape[-1]),
                np.mean(np.array(pcls[:, 1, 0, :]).T, axis=-1),
                color="gray",
            )
            theorypcl53 = np.load("pcls/pcl_n256_circ10000smoothl30_3x2pt_kids_53_nonoise.npz")
            theorypcl55 = np.load("pcls/pcl_n256_circ10000smoothl30_3x2pt_kids_55_noisedefault.npz")
            plt.plot(np.arange(pcls.shape[-1]), theorypcl53["pcl_ee"], color="red")
            plt.plot(np.arange(pcls.shape[-1]), theorypcl55["pcl_ee"], color="blue")
            plt.savefig("sim_demo_pcle_35_{:d}.png".format(j))

        np.savez(
            simpath + "/job{:d}.npz".format(j),
            mode=ximode,
            theta=seps_in_deg,
            lmin=lmin,
            lmax=lmax,
            xip=np.array(xis[:, :, 0, :]),
            xim=np.array(xis[:, :, 1, :]),
        )
        if save_pcl:
            np.savez(
                simpath + "/pcljob{:d}.npz".format(j),
                lmin=lmin,
                lmax=lmax,
                pcl_e=np.array(pcls[:, :, 0]),
                pcl_b=np.array(pcls[:, :, 1]),
                pcl_eb=np.array(pcls[:, :, 2]),
            )


def prep_cat_treecorr(nside, mask=None):
    """takes healpy mask, returns mask needed for treecorr simulation"""
    if mask is None:
        all_pix = np.arange(hp.nside2npix(nside))
        phi, thet = hp.pixelfunc.pix2ang(nside, all_pix, lonlat=True)
        treecorr_mask_cat = (None, phi, thet)
        sum_weights = None
    else:
        all_pix = np.arange(len(mask))
        # mask = np.array(1 - mask, dtype=int)
        # masked_pixs = np.ma.array(all_pixs, mask=mask).compressed()
        # phi, thet = hp.pixelfunc.pix2ang(nside, masked_pixs, lonlat=True)
        phi, thet = hp.pixelfunc.pix2ang(nside, all_pix, lonlat=True)
        treecorr_mask_cat = (mask, phi, thet)
    return treecorr_mask_cat


def prep_angles_treecorr(seps_in_deg):
    if type(seps_in_deg[0]) is tuple:
        bin_size = seps_in_deg[0][1] - seps_in_deg[0][0]
        thetamin = seps_in_deg[0][0]
        thetamax = seps_in_deg[-1][1]
    else:
        raise RuntimeError("Angular separations need to be bins for treecorr!")
    return thetamin, thetamax, bin_size


def get_xi_treecorr(maps_TQU, ang_bins_in_deg, cat_props):
    """takes maps and mask, returns treecorr correlation function"""
    """need to run cat and angles prep once before simulating"""
    mask, phi, thet = cat_props

    cat = treecorr.Catalog(
        ra=phi, dec=thet, g1=-maps_TQU[1], g2=maps_TQU[2], ra_units="deg", dec_units="deg", w=mask
    )

    xip, xim, angs = [], [], []
    for ang_bin in ang_bins_in_deg:
        thetamin, thetamax = ang_bin

        gg = treecorr.GGCorrelation(
            min_sep=thetamin,
            max_sep=thetamax,
            nbins=1,
            bin_type="Linear",
            sep_units="deg",
            bin_slop=0,
        )
        gg.process(cat)
        xip.append(gg.xip)
        xim.append(gg.xim)
        angs.append(gg.rnom)
    cat.clear_cache()
    return np.array(xip)[:, 0], np.array(xim)[:, 0], np.array(angs)


def pcl_sim(cell_object, ells, mask_file):
    lmax, cl_tup = set_cl(cl, nside)
    pcls_e, pcls_b, pcls_eb = [], [], []
    for i in range(batchsize):
        maps_TQU = create_maps(cl_tup, lmax=lmax, nside=nside)
        maps = add_noise(maps_TQU, sigma_n, lmax=lmax)
        pcl_e, pcl_b, pcl_eb = get_pcl(maps_TQU, mask)
    pass
