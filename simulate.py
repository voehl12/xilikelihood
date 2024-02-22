import os

import numpy as np
import healpy as hp

import pymaster as nmt
import treecorr

import matplotlib.pyplot as plt
from numpy.random import default_rng
from helper_funcs import get_noise_cl, pcl2xi, prep_prefactors
from cov_setup import Cov

from typing import (Any, Union, Tuple, Generator, Optional, Sequence, Callable,
                    Iterable)

import glass.fields
import time

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
        clname="test_cl",
        maskpath=None,
        circmaskattr=None,
        lmin=None,
        maskname="mask",
        l_smooth_mask=None,
        l_smooth_signal=None,
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
            l_smooth_mask,
            l_smooth_signal,
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
                (self.nn, self.ne, self.ee, self.bb), self.nside, lmax=self.lmax
            )
            return maps

    def add_noise(self, maps, testing=False):
        if hasattr(self, "pixelsigma"):
            sigma = self.pixelsigma
        else:
            return maps
        rng = default_rng()

        noise_map_q = self.limit_noise(rng.normal(size=(self.npix), scale=sigma))
        noise_map_u = self.limit_noise(rng.normal(size=(self.npix), scale=sigma))

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

        if self.smooth_signal is not None:
            clq *= self.smooth_array
        
        #assert np.allclose(clq[2*self.smooth_signal], self.noise_cl[2*self.smooth_signal], rtol=1e-01),(clq,self.noise_cl)
        np.random.seed()
        return hp.sphtfunc.synfast(clq, self.nside, lmax=self.lmax)

    def get_pcl(self, maps_TQU):
        if self.maskname == "fullsky":
            cl_t, cl_e, cl_b, cl_te, cl_eb, cl_tb = hp.anafast(maps_TQU)
            return cl_e, cl_b, cl_eb
        else:
            # f_0 = nmt.NmtField(mask, [maps_TQU[0]])
            if self.smooth_mask is None:
                self.wl
                print("get_pcl: calculating wl to establish smoothed mask.")
            maps_TQU_masked = self.smooth_mask[None,:]*maps_TQU
            pcl_t, pcl_e, pcl_b, pcl_te, pcl_eb, pcl_tb = hp.anafast(maps_TQU_masked,iter=10,use_weights=True)
            #f_2 = nmt.NmtField(self.smooth_mask, maps_TQU[1:].copy())
            # cl_00 = nmt.compute_coupled_cell(f_0, f_0)
            # cl_02 = nmt.compute_coupled_cell(f_2, f_0)
            #cl_22 = nmt.compute_coupled_cell(f_2, f_2)
            #pcl_e, pcl_b, pcl_eb = cl_22[0], cl_22[3], (cl_22[1] + cl_22[2]) / 2
            return pcl_e, pcl_b, pcl_eb

    def get_xi_namaster(self, maps_TQU, prefactors, lmin=0):
        # pcl2xi should work for several angles at once.
        pcl_22 = self.get_pcl(maps_TQU)
        xi_p, xi_m = pcl2xi(pcl_22, prefactors, self.lmax, lmin=lmin)
        return pcl_22,xi_p, xi_m

    def xi_sim_1D(self, j, lmin=0, plot=False,save_pcl=False):
        xip, xim, pcls = [], [], []
        foldername = "/{}_{}_{}".format(self.clname,self.maskname,self.sigmaname)
        path = self.simpath+foldername
        if not os.path.isdir(path):
            command = "mkdir "+path
            os.system(command)
        self.simpath = path
        if self.ximode == "namaster":
            prefactors = prep_prefactors(self.seps_in_deg, self.wl, self.lmax, self.lmax) # exact lmax shoukld not be used here.
            for _i in range(self.batchsize):
                print(
                    "Simulating xip and xim......{:4.1f}%".format(_i / self.batchsize * 100),
                    end="\r",
                )
                maps_TQU = self.create_maps()
                maps = self.add_noise(maps_TQU)
                pcl,xi_p, xi_m = self.get_xi_namaster(maps, prefactors, lmin)
                xip.append(xi_p)
                xim.append(xi_m)
                pcls.append(pcl)
            xip = np.array(xip)
            xim = np.array(xim)
            pcls = np.array(pcls)
            if plot:
                plt.figure()
                plt.hist(xip[:, 0], bins=30)
                plt.savefig('sim_demo_{:d}.png'.format(j))
            np.savez(
                self.simpath + "/job{:d}.npz".format(j),
                mode=self.ximode,
                theta=self.seps_in_deg,
                lmin=lmin,
                xip=np.array(xip),
                xim=np.array(xim),
            )
            if save_pcl:
                np.savez(
                self.simpath + "/pcljob{:d}.npz".format(j),
                lmin=lmin,
                lmax=self.lmax,
                pcl_e=np.array(pcls[:,0]),
                pcl_b=np.array(pcls[:,1]),
                pcl_eb=np.array(pcls[:,2])
            )

def create_maps_nD(gls=[],nside=256,lmax=None):
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

def limit_noise(noisemap,nside : int,lmax : Optional[int]=None):
    almq = hp.map2alm(noisemap)
    clq = hp.sphtfunc.alm2cl(almq)

    #assert np.allclose(clq[2*self.smooth_signal], self.noise_cl[2*self.smooth_signal], rtol=1e-01),(clq,self.noise_cl)
    np.random.seed()
    return hp.sphtfunc.synfast(clq, nside,lmax=lmax)



def add_noise_nD(maps_TQU_list,nside : int,sigmas=None,lmax=None):
    if sigmas is None:
        return maps_TQU_list
    else:
        assert len(sigmas) == len(maps_TQU_list), (sigmas,maps_TQU_list)
        size = len(maps_TQU_list[0,0])
    
        

        for i,maps_TQU in enumerate(maps_TQU_list):
            rng = default_rng()

            noise_map_q = limit_noise(rng.normal(size=size, scale=sigmas[i]),nside,lmax)
            noise_map_u = limit_noise(rng.normal(size=size, scale=sigmas[i]),nside,lmax)
            maps_TQU[1] += noise_map_q
            maps_TQU[2] += noise_map_u

        return maps_TQU_list


def get_pcl_nD(maps_TQU_list,smooth_masks,fullsky=False):
    n_field = len(maps_TQU_list)
    n_corr = int(n_field * ((n_field - 1) / 2 + 1))
    if fullsky:
        cl_s = []
        for i,field_i in enumerate(maps_TQU_list):
            for j,field_j in reversed(list(enumerate(maps_TQU_list[:i+1]))):
                # cl_t, cl_e, cl_b, cl_te, cl_eb, cl_tb order of cl_s
                cl_s.append(hp.anafast(field_i,field_j))
        cl_s = np.array(cl_s)[:,[1,2,4]]
        return cl_s # cl_e, cl_b, cl_eb
    else:
        pcl_s = []
        for i,field_i in enumerate(maps_TQU_list):
            for j,field_j in reversed(list(enumerate(maps_TQU_list[:i+1]))):
                f_i = nmt.NmtField(smooth_masks[i], field_i[1:].copy())
                f_j = nmt.NmtField(smooth_masks[j], field_j[1:].copy())
                cl_22 = nmt.compute_coupled_cell(f_i, f_j)
                pcl_e, pcl_b, pcl_eb = cl_22[0], cl_22[3], (cl_22[1] + cl_22[2]) / 2
                pcl_s.append([pcl_e, pcl_b, pcl_eb])
        
        return np.array(pcl_s)

def get_xi_namaster_nD(maps_TQU_list,smooth_masks, prefactors, lmax,lmin=0):
    # pcl2xi should work for several angles at once.
    pcl_ij = get_pcl_nD(maps_TQU_list,smooth_masks)
    n_corr = len(pcl_ij)
    xi_all = np.zeros((n_corr,2,len(prefactors)))
    for i,pcl in enumerate(pcl_ij):
        xi_all[i] = np.array(pcl2xi(pcl, prefactors, lmax, lmin=lmin))
    return pcl_ij,xi_all

def xi_sim_nD(covs, j, seps_in_deg,lmax=None,lmin=0, plot=False,save_pcl=False,ximode='namaster',batchsize=1,simpath="simulations/"):
    xis,pcls = [],[]
    gls = np.array([cov.ee for cov in covs])
    
    nsides = [int(cov.nside) for cov in covs]
    areas = [cov.eff_area for cov in covs]
    smooth_masks = [cov.smooth_mask for cov in covs if hasattr(cov, 'smooth_mask')]
    
    noises = [cov.pixelsigma for cov in covs if hasattr(cov,'pixelsigma')] # should return noise for auto-cl in right order
    assert np.all(np.array(nsides)-nsides[0] == 0),nsides
    assert np.all(np.isclose(areas, areas[0])),areas
    nside = nsides[0]

    foldername = "/croco_{}_{}_{}_llim_{}".format(covs[0].clname,covs[0].maskname,covs[0].sigmaname,str(lmax))
    path = simpath+foldername
    if not os.path.isdir(path):
        command = "mkdir "+path
        os.system(command)
    simpath = path


    if ximode == "namaster":
        if lmax is None:
            lmax = covs[0].lmax
        prefactors = prep_prefactors(seps_in_deg, covs[0].wl, covs[0].lmax, lmax) 
        times = []
        for _i in range(batchsize):
            tic = time.perf_counter()
            print(
                "Simulating xip and xim......{:4.1f}%".format(_i / batchsize * 100),
                end="\r",
            )
            maps_TQU_list = create_maps_nD(gls,nside)
            maps = add_noise_nD(maps_TQU_list,nside,sigmas=noises)
            pcl_all,xi_all = get_xi_namaster_nD(maps, smooth_masks,prefactors, lmax,lmin=0)
            xis.append(xi_all)
            pcls.append(pcl_all)
            toc = time.perf_counter()
            times.append(toc-tic)
        print(times,np.mean(times))
        xis = np.array(xis)
        pcls = np.array(pcls)
        if plot:
            plt.figure()
            plt.hist(xis[:, 1,0,1], bins=30)
            plt.savefig('sim_demo_{:d}.png'.format(j))
        np.savez(
            simpath + "/job{:d}.npz".format(j),
            mode=ximode,
            theta=seps_in_deg,
            lmin=lmin,
            lmax=lmax,
            xip=np.array(xis[:,:,0,:]),
            xim=np.array(xis[:,:,1,:]),
        )
        if save_pcl:
            np.savez(
            simpath + "/pcljob{:d}.npz".format(j),
            
            lmin=lmin,
            lmax=lmax,
            pcl_e=np.array(pcls[:,:,0]),
            pcl_b=np.array(pcls[:,:,1]),
            pcl_eb=np.array(pcls[:,:,2])
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
