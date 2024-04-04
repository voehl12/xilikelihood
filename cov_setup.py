import numpy as np
import cov_calc, helper_funcs
import scipy.stats as stats
from scipy.integrate import quad_vec
import wigner
import os.path
from grf_classes import TheoryCl, SphereMask
import matplotlib.pyplot as plt
from sys import getsizeof
import gc
import time

class Cov(SphereMask, TheoryCl):
    """
    Class to calculate and store covariances for masked Gaussian random fields.

    Attributes
    ----------
    cov_alm : 2D array
        covariance matrix of pseudo alm (currently E and B modes)
    ang_bins : list of tuples
        angular bins for the correlation function under consideration (so far only one)
    cov_xi : float
        total (sample and shot noise) Gaussian covariance of a correlation function
    cov_sn : float
        shot noise Gaussian covariance of a correlation function
    xi: float
        mean of correlation function considered (only if Gaussian covariances are calculated)
    exact_lmax : integer
        maximum multipole moment to which calculations are taken to
    __sigma_e : tuple or None
        (sigma_epsilon,n_gal_per_arcmin2), single component intrinstic ellipticity dispersion, number density of galaxies.
        if None: no shot noise. Private because it shouldn't be changed afterwards.

    Methods
    -------
    cov_alm_xi(exact_lmax=None,pos_m=False)
        Calculates 2D covariance matrix of pseudo alm cov_alm needed for likelihoods of spin-2 correlation functions or power spectra (E and B modes)
    cov_xi_gaussian(lmin=0, noise_apo=False)
        Calculates Gaussian (auto) covariance (shot noise and sample variance) of a xi_plus correlation function (so far no cross correlations implemented)

    Parameters
    ----------
    SphereMask : _type_
        _description_
    TheoryCl : _type_
        _description_
    """

    def __init__(
        self,
        exact_lmax,
        spins,
        lmax=None,
        sigma_e=None,
        clpath=None,
        theory_lmin=2,
        clname="3x2pt_kids_55",
        s8 = None,
        maskpath=None,
        circmaskattr=None,
        lmin=None,
        maskname="mask",
        l_smooth_mask=None,
        l_smooth_signal=None,
        cov_ell_buffer=0,
    ):
        SphereMask.__init__(
            self,
            exact_lmax=exact_lmax,
            spins=spins,
            maskpath=maskpath,
            circmaskattr=circmaskattr,
            lmin=lmin,
            maskname=maskname,
            l_smooth=l_smooth_mask,
        )
        if lmax is None:
            self.lmax = 3 * self.nside - 1
        else:
            self.lmax = lmax
        TheoryCl.__init__(
            self,
            lmax=self.lmax,
            clpath=clpath,
            theory_lmin=theory_lmin,
            clname=clname,
            smooth_signal=l_smooth_signal,
            s8=s8,
        )

        self._sigma_e = sigma_e
        self.set_noise_sigma()
        self.ang_bins_in_deg = None
        self.cov_ell_buffer = cov_ell_buffer

        self.set_covalmpath()

    @property
    def sigma_e(self):
        return self._sigma_e

    @sigma_e.setter
    def sigma_e(self, new_sigma):
        if new_sigma is None or new_sigma != self._sigma_e:
            self._sigma_e = new_sigma
            self.set_covalmpath()
            self.set_noise_sigma()
            if hasattr(self, "cov_alm"):
                print("Set new noise level, recalculate pseudo alm covariance.")
                self.cov_alm_xi(exact_lmax=self._exact_lmax, pos_m=True)

            if hasattr(self, "cov_xi"):
                print("Set new noise level, recalculate Gaussian covariance.")
                self.cov_xi_gaussian()

            else:
                print("Set new noise level, nothing to recalculate.")

    @property
    def exact_lmax(self):
        return self._exact_lmax

    @exact_lmax.setter
    def exact_lmax(self, new_lmax):
        if isinstance(new_lmax, int) and new_lmax != self._exact_lmax:
            print("Warning: Resetting exact lmax within a covariance instance is not supported at the moment (concerning recalculation of dependent quantities)")
            self._exact_lmax = new_lmax
            if self.l_smooth_auto:
                self.l_smooth = new_lmax
            self.set_covalmpath()
            if hasattr(self, "cov_alm"):
                self.cov_alm_xi(pos_m=True)

    def cell_cube(self, exact_lmax):
        c_all = np.zeros((3, 3, exact_lmax + 1))
        c_all[0, 0] = self.ee.copy()[: exact_lmax + 1]
        c_all[0, 2] = self.ne.copy()[: exact_lmax + 1]
        c_all[2, 0] = self.ne.copy()[: exact_lmax + 1]
        c_all[2, 2] = self.nn.copy()[: exact_lmax + 1]
        return c_all

    def cov_alm_xi(self, exact_lmax=None, pos_m=True):
        """
        calculates covariance of pseudo-alm needed for the correlation function xi

        always a covariance of pseudo alms - order and number depends on two point statistics considered
        order of alm follows structure of cov_4D - meaning first sort by E/B Re/Im, then by l and then by m
        only positive m part of covariance is needed (tests say yes, analytically to be confirmed) - way to only calculate this, just like wlmlpmp are only plugged into the sum for given l as needed
        only valid for E/B alms so far.
        could implement limitation of m to current l, would make n_cov smaller overall

        Parameters
        ----------
        exact_lmax : integer, optional
            maximum multipole moment for exact calculations (only maximum multipole moment relevant here), by default None
        pos_m : bool, optional
            only calculate contributions for positive m (seems to be sufficient for plus correlation function, couldn't show analytically why yet), by default False

        Returns
        -------
        2D array
            covariance matrix for real and imaginiary E and B mode pseudo alms

        Raises
        ------
        NotImplementedError
            if all m covariance (including negative) calculation is attempted for fullsky version
        """
        
        buffer = self.cov_ell_buffer
        
        if exact_lmax is None:
            exact_lmax = self._exact_lmax
        elif exact_lmax != self._exact_lmax:
            self.exact_lmax = exact_lmax
        theory_cell = self.cell_cube(exact_lmax + buffer)

        if self.check_cov():
            self.load_cov()
            return self.cov_alm
        else:
            alm_kinds = ["ReE", "ImE", "ReB", "ImB"]
            alm_inds = cov_calc.match_alm_inds(alm_kinds)
            n_alm = len(alm_inds)
            lmin = 0

            if hasattr(self, "_noise_sigma"):
                theory_cell += helper_funcs.noise_cl_cube(self.noise_cl[:self.exact_lmax+1+buffer])

            if pos_m:
               
                n_cov = n_alm * (exact_lmax - lmin + 1) * (exact_lmax + 1)
            else:
                
                n_cov = n_alm * (exact_lmax - lmin + 1) * (2 * (exact_lmax) + 1)

            if self.maskname == "fullsky":
                if pos_m == False:
                    raise NotImplementedError(
                        "No mask case covariance matrix only implemented for positive m"
                    )
                else:
                    cov_matrix = np.zeros((n_cov, n_cov))
                    diag = np.zeros(n_cov)
                    for i in alm_inds:
                        t = int(np.floor(i / 2))  # same c_l for Re and Im
                        len_sub = exact_lmax + 1
                        cell_ranges = [
                            np.repeat(theory_cell[t, t, i], i + 1) for i in range(len_sub)
                        ]
                        full_ranges = [
                            np.append(
                                cell_ranges[i],
                                np.zeros(len_sub - len(cell_ranges[i])),
                            )
                            for i in range(len(cell_ranges))
                        ]
                        cov_part = 0.5 * np.ndarray.flatten(np.array(full_ranges))
                        if i % 2 == 0: # true if alm part is real
                            cov_part[::len_sub] *= 2
                        else:
                            cov_part[::len_sub] *= 0
                        # alm with same m but different sign dont have vanishing covariance. This is only relevant if pos_m = False.
                        len_2D = len(cov_part)
                        pos = (len_2D * i, len_2D * (i + 1))

                        diag[pos[0] : pos[1]] = cov_part
                    assert len(diag) == n_cov
                    cov_matrix = np.diag(diag)

            else:
                cov_matrix = self.cov_masked(alm_inds, n_cov, theory_cell, lmin, pos_m)

            cov_matrix = np.where(np.isnan(cov_matrix), cov_matrix.T, cov_matrix)
            assert np.allclose(cov_matrix, cov_matrix.T), "Covariance matrix not symmetric"
            diag_alm = np.diag(cov_matrix)
            
            if pos_m == False:
                len_sub = 2*exact_lmax+1
                reps = int(len(diag_alm) / (len_sub))
                check_pcl_sub = np.array([np.sum(diag_alm[i*len_sub:(i+1)*len_sub]) for i in range(reps)])
            else:
                len_sub = exact_lmax+1
                reps = int(len(diag_alm) / (len_sub))
                check_pcl_sub = np.array([np.sum(2*diag_alm[i*len_sub + 1:(i+1)*len_sub+1]) + diag_alm[i*len_sub] for i in range(reps)])
            
            check_pcl = np.zeros((2,exact_lmax+1))
            check_pcl[0], check_pcl[1] = check_pcl_sub[:exact_lmax+1]+check_pcl_sub[exact_lmax+1:2*(exact_lmax+1)], check_pcl_sub[2*(exact_lmax+1):3*(exact_lmax+1)]+check_pcl_sub[3*(exact_lmax+1):4*(exact_lmax+1)]
            pcl = np.zeros_like(check_pcl)
            twoell= 2 * self.ell + 1
            self.cl2pseudocl()
            pcl[0], pcl[1] = (self.p_ee * twoell)[:exact_lmax+1], (self.p_bb * twoell)[:exact_lmax+1]
            assert np.allclose(pcl,check_pcl,rtol=1e-1), "Covariance diagonal does not agree with pseudo C_ell"
            self.cov_alm = cov_matrix
            self.save_cov()
            return self.cov_alm
    
    def cov_masked(self, alm_inds, n_cov, theory_cell, lmin, pos_m):
        tic = time.perf_counter()
        buffer = self.cov_ell_buffer
        w_arr = self.w_arr(cov_ell_buffer=buffer)
        cov_matrix = np.full((n_cov, n_cov), np.nan)
        print('Beginning to fill covariance matrix with size {} mb.'.format(getsizeof(cov_matrix)/1024**2))
        numparts =  (len(alm_inds)**2 - len(alm_inds)) / 2 + len(alm_inds)
        part = 1
        for i in alm_inds:
            for j in alm_inds:
                if i <= j:
                    cov_part = cov_calc.cov_4D(
                        i, j, w_arr, self._exact_lmax + buffer, lmin, theory_cell, pos_m=pos_m
                    )
                    if pos_m == False:
                        len_2D = (cov_part.shape[0]-buffer) * (cov_part.shape[1]-2*buffer)
                    
                        cov_2D = np.reshape(cov_part[:-buffer,buffer:-buffer,:-buffer,buffer:-buffer], (len_2D, len_2D))
                    else:
                        
                        len_2D = (cov_part.shape[0]-buffer) * (cov_part.shape[1]-buffer)
                    
                        cov_2D = np.reshape(cov_part[:-buffer,:-buffer,:-buffer,:-buffer], (len_2D, len_2D))

                    pos_y = (len_2D * i, len_2D * (i + 1))
                    pos_x = (len_2D * j, len_2D * (j + 1))
                    cov_matrix[pos_y[0] : pos_y[1], pos_x[0] : pos_x[1]] = cov_2D
                    del cov_2D
                    gc.collect()
                    print('Finished part {:d}/{:d}.'.format(int(part),int(numparts)))
                    part += 1
        toc = time.perf_counter()
        print('Covariance matrix calculation took {:.2f} minutes'.format((toc-tic)/60))
        return cov_matrix

    def cov_cl_gaussian(self):
        cl_e = self.ee.copy()
        cl_b = self.bb.copy()
        noise2 = np.zeros_like(cl_e)
        ell = np.arange(self.lmax + 1)
        if hasattr(self, "_noise_sigma"):
            noise_B = noise_E = self.noise_cl

            cl_e += noise_E
            cl2 = np.square(cl_e) + np.square(noise_B)
            noise2 += np.square(noise_E) + np.square(noise_B)
        else:
            cl2 = np.square(cl_e)

        diag = 2 * cl2
        noise_diag = 2 * noise2
        return diag, noise_diag

    def cov_xi_gaussian(self, lmin=0, lmax=None):
        # TODO: this function is deprecated, because it is very specific for just one alm covariance, use cov_xi_gaussian nD in calc_pdf instead
        raise DeprecationWarning
        """
        Calculates covariance of xip correlation function in Gaussian approximation.

        Using fsky factor to take mask into account

        Parameters
        ----------
        lmin : int, optional
            mimimum multipole moment, by default 0
        noise_apo : bool, optional
            enable noise apodization, by default False

        Raises
        ------
        NotImplementedError
            if more than one angular bin is given
        """
        # e.g. https://www.aanda.org/articles/aa/full_html/2018/07/aa32343-17/aa32343-17.html
        if self.ang_bins_in_deg is None:
            raise RuntimeError("need to set angular bin for xi covariance.")
        self.wlm_lmax
        fsky = self.eff_area / 41253
        c_tot, c_sn = self.cov_cl_gaussian()
        if lmax is None:
            lmax = self.lmax
        c_tot, c_sn = c_tot[lmin : lmax + 1], c_sn[lmin : lmax + 1]

        l = 2 * np.arange(lmin, lmax + 1) + 1
        wigner_int = lambda theta_in_rad: theta_in_rad * wigner.wigner_dl(
            lmin, lmax, 2, 2, theta_in_rad
        )
        norm = 1 / (4 * np.pi)
        if len(self.ang_bins_in_deg) > 1:
            raise NotImplementedError("Gaussian cross-correlation not implemented yet")
        else:
            bin1 = self.ang_bins_in_deg[0]
            binmin_in_deg = bin1[0]
            binmax_in_deg = bin1[1]

            upper = np.radians(binmax_in_deg)
            lower = np.radians(binmin_in_deg)

            t_norm = 2 / (upper**2 - lower**2)
            # much closer to cl-xi if the normalization in the bin prefactors is taken to lmax! (two orders of magnitude, even with apodized mask)

            integrated_wigners = quad_vec(wigner_int, lower, upper)[0]
            # t_norm and integrated wigners could come from cl prefactors helper function
            cov_xi = (
                1 / fsky * t_norm**2 * norm**2 * np.sum(integrated_wigners**2 * c_tot * l)
            )
            cov_sn = 1 / fsky * t_norm**2 * norm**2 * np.sum(integrated_wigners**2 * l * c_sn)
            pure_noise_mean = t_norm * norm * np.sum(integrated_wigners * l * np.sqrt(c_sn))

            cl_mean_p, cl_mean_m = helper_funcs.cl2xi((self.ee.copy(), self.bb.copy()), bin1, lmax, lmin=lmin)
            cl_mean_p += pure_noise_mean
            prefactors = helper_funcs.prep_prefactors(
                self.ang_bins_in_deg, self.wl, self.lmax, lmax
            )

            pcl_mean_p, pcl_mean_m = helper_funcs.pcl2xi(
                (self.p_ee.copy(), self.p_bb.copy(), self.p_eb.copy()),
                prefactors,
                lmax,
                lmin=lmin,
            )
            pcl_mean_p, pcl_mean_m = pcl_mean_p[0], pcl_mean_m[0]
        print("lmin: {:d}, lmax: {:d}, pCl mean: {:.5e}, Cl mean: {:.5e}".format(lmin,lmax,pcl_mean_p, cl_mean_p))
        assert np.allclose(pcl_mean_p, cl_mean_p, rtol=1e-1)
        
        self.cov_xi = cov_xi
        self.cov_sn = cov_sn
        self.xi_pcl = pcl_mean_p
        self.xi_cl = cl_mean_p
        return pcl_mean_p,cov_xi

    def set_noise_sigma(self):
        if self._sigma_e is not None:
            if isinstance(self._sigma_e, str):
                self._noise_sigma = helper_funcs.get_noise_cl()
                self.pixelsigma = helper_funcs.get_noise_pixelsigma(self.nside)

            elif isinstance(self._sigma_e, tuple):
                self._noise_sigma = helper_funcs.get_noise_cl(*self._sigma_e)
                self.pixelsigma = helper_funcs.get_noise_pixelsigma(self.nside, self._sigma_e)
            else:
                raise RuntimeError(
                    "sigma_e needs to be string for default or tuple (sigma_e,n_gal)"
                )
            self.noise_cl = np.ones(self.lmax + 1) * self._noise_sigma
            if self.smooth_signal is not None:
                self.noise_cl *= self.smooth_array
        else:
            try:
                del self._noise_sigma
            except:
                pass

    def save_cov(self):
        print("Saving covariance matrix.")
        np.savez(self.covalm_path, cov=self.cov_alm)

    def check_cov(self):
        print("Checking for covariance matrix... ", end="")
        print(self.covalm_path)
        if os.path.isfile(self.covalm_path):
            print("Found.")
            return True
        else:
            print("Not found.")
            return False

    def load_cov(self):
        print("Loading covariance matrix.")
        covfile = np.load(self.covalm_path)
        self.cov_alm = covfile["cov"]

    def set_char_string(self):
        if self._sigma_e is None:
            self.sigmaname = "nonoise"
        else:
            if isinstance(self._sigma_e, str):
                self.sigmaname = "noise" + self._sigma_e

            else:
                self.sigmaname = "noise" + str(self._sigma_e).replace(".", "")

        charstring = "_l{:d}_n{:d}_{}_{}_{}.npz".format(
            self._exact_lmax,
            self.nside,
            self.maskname,
            self.clname,
            self.sigmaname,
        )
        return charstring

    def set_covalmpath(self):
        charac = self.set_char_string()
        #covname = "covariances/cov_xi" + charac
        #covname = "/cluster/scratch/veoehl/covariances/cov_xi" + charac
        covname = "/cluster/work/refregier/veoehl/covariances/cov_xi" + charac
        self.covalm_path = covname

    def cl2pseudocl(self):
        # from namaster scientific documentation paper
        if not os.path.isdir('pcls'):
            command = "mkdir pcls"
            os.system(command)
        pclpath = "pcls/pcl" + "_n{:d}_{}_{}_{}.npz".format(
            self.nside,
            self.maskname,
            self.clname,
            self.sigmaname,
        )
        print(pclpath)
        if os.path.isfile(pclpath):
            pclfile = np.load(pclpath)
            self.p_ee = pclfile["pcl_ee"]
            self.p_bb = pclfile["pcl_bb"]
            self.p_eb = pclfile["pcl_eb"]
        else:
            if self.smooth_mask is None:
                self.wl
                print("pseudo_cl: calculating wl to establish smoothed mask.")
            m_llp_p, m_llp_m = self.m_llp
            if hasattr(self, "_noise_sigma"):
                cl_e = self.ee.copy() + self.noise_cl
                cl_b = self.bb.copy() + self.noise_cl
                
            else:
                cl_e = self.ee.copy()
                cl_b = self.bb.copy()
            cl_eb = cl_be = cl_b

            self.p_ee = np.einsum("lm,m->l", m_llp_p, cl_e) + np.einsum("lm,m->l", m_llp_m, cl_b)
            self.p_bb = np.einsum("lm,m->l", m_llp_m, cl_e) + np.einsum("lm,m->l", m_llp_p, cl_b)
            self.p_eb = np.einsum("lm,m->l", m_llp_p, cl_eb) - np.einsum("lm,m->l", m_llp_m, cl_be)

            print("saving pseudo_cl...")
            np.savez(pclpath, pcl_ee=self.p_ee, pcl_bb=self.p_bb, pcl_eb=self.p_eb)
