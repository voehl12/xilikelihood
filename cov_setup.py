import numpy as np
import cov_funcs, helper_funcs
import os.path
from sys import getsizeof
import gc
import time


class Cov:
    """
    Class to calculate and store pseudo-alm covariances for masked spherical Gaussian random fields.

    Attributes
    ----------
    cov_alm : 2D array
        covariance matrix of pseudo alm (currently E and B modes)
    exact_lmax : integer
        maximum multipole moment to which calculations are taken to
    cov_ell_buffer : int
        buffer in exact_lmax to ensure convergence (is cut to exact_lmax for final covariance matrix, just for computation)
    __sigma_e : tuple or None
        (sigma_epsilon,n_gal_per_arcmin2), single component intrinstic ellipticity dispersion, number density of galaxies.
        if None: no shot noise. Private because it shouldn't be changed afterwards.
    ee,bb,nn,... : 1D arrays
        theory power spectra
    p_ee,p_bb,... : 1D arrays
        pseudo Cl

    Methods
    -------
    cov_alm_xi(exact_lmax=None,pos_m=False)
        Calculates 2D covariance matrix of pseudo alm cov_alm needed for likelihoods of spin-2 correlation functions or power spectra (E and B modes)
    cl2pcl()
        Calculates pseudo-Cl based on mask and theory Cl

    Parameters
    ----------
    SphereMask : _type_
        _description_
    TheoryCl : _type_
        _description_
    """

    def __init__(
        self,
        mask,
        theorycl,
        exact_lmax,
        lmax=None,
        cov_ell_buffer=10,
        working_dir=None,
    ):

        if working_dir is None:
            working_dir = os.getcwd()
        self.working_dir = working_dir

        self.mask = mask
        self.theorycl = theorycl
        self._exact_lmax = exact_lmax
        if not helper_funcs.check_property_equal([self, self.mask], "_exact_lmax"):
            raise RuntimeError(
                "Cov: exact_lmax of mask and cov class not equal. Please check."
            )

        self._lmax = lmax if lmax is not None else 3 * self.mask.nside - 1

        self._cov_ell_buffer = cov_ell_buffer

        self._cov_ell_buffer = cov_ell_buffer
        # self.set_noise_pixelsigma()
        self.set_covalmpath()

    @property
    def exact_lmax(self):
        return self._exact_lmax

    def cell_cube(self, lmax):
        c_all = np.zeros((3, 3, lmax + 1))
        c_all[0, 0] = self.theorycl.ee.copy()[: lmax + 1]
        c_all[0, 2] = self.theorycl.ne.copy()[: lmax + 1]
        c_all[2, 0] = self.theorycl.ne.copy()[: lmax + 1]
        c_all[2, 2] = self.theorycl.nn.copy()[: lmax + 1]
        return c_all

    def cov_alm_xi(self, ischain=False):
        """
        Calculates covariance of pseudo-alm needed for the spin-2 correlation function xi+/-.

        Parameters
        ----------
        exact_lmax : int, optional
            maximum multipole moment ell to be used, None defaults to class instance exact_lmax set earlier, by default None
        pos_m : bool, optional
            only calculate covariances of pseudo-alm with positive m, by default True

        Returns
        -------
        2D array
            Covariance matrix of pseudo-alm: E, and B-modes, real and imaginary parts, sorted by ell, then m (see (cov_alm_gen))
        """

        alm_kinds = [
            "ReE",
            "ImE",
            "ReB",
            "ImB",
        ]
        self.cov_alm = self.cov_alm_gen(alm_kinds, pos_m=True, ischain=ischain)
        return self.cov_alm

    def cov_alm_gen(self, alm_kinds, pos_m=True, lmin=0, ischain=False):
        """
        calculates covariance of pseudo-alm

        always a covariance of pseudo alms - order and number depends on two point statistics considered
        order of alm follows structure of cov_4D - meaning first sort by E/B Re/Im, then by l and then by m
        only positive m part of covariance is needed (tests say yes, analytically to be confirmed) - way to only calculate this, just like wlmlpmp are only plugged into the sum for given l as needed
        only tested for E/B alms so far.
        could implement limitation of m to current l, would make n_cov smaller overall

        Parameters
        ----------
        alm_kinds : list of strings
            which pseudo-alm to calculate covariance for; real/imag and mode (e.g. ReE, ImE, ...)
        exact_lmax : integer, optional
            maximum multipole moment for exact calculations (only maximum multipole moment relevant here), by default None
        pos_m : bool, optional
            only calculate contributions for positive m (seems to be sufficient for plus correlation function, couldn't show analytically why yet), by default True
        lmin : int, optional
            minimum ell considered , by default 0

        Returns
        -------
        2D array
            covariance matrix for real and imaginiary E and B mode pseudo alms

        Raises
        ------
        NotImplementedError
            if all m covariance (including negative) calculation is attempted for fullsky version
        """

        buffer = self._cov_ell_buffer
        theory_cell = self.cell_cube(self._exact_lmax + buffer)
        if not ischain and self.check_cov():
            self.load_cov()
            return self.cov_alm
        else:
            alm_inds = cov_funcs.match_alm_inds(alm_kinds)
            n_alm = len(alm_inds)

            if hasattr(self.theorycl, "_noise_sigma"):
                theory_cell += helper_funcs.noise_cl_cube(
                    self.theorycl.noise_cl[: self._exact_lmax + 1 + buffer]
                )

            if pos_m:

                n_cov = n_alm * (self._exact_lmax - lmin + 1) * (self._exact_lmax + 1)
            else:

                n_cov = n_alm * (self._exact_lmax - lmin + 1) * (2 * (self._exact_lmax) + 1)

            if self.mask.name == "fullsky":
                if pos_m == False:
                    raise NotImplementedError(
                        "cov_alm_gen: no mask case covariance matrix only implemented for positive m"
                    )
                else:
                    cov_matrix = self.cov_fullsky(alm_inds, n_cov, theory_cell)

            else:
                cov_matrix = self.cov_masked(alm_inds, n_cov, theory_cell, pos_m=pos_m)

            cov_matrix = np.where(np.isnan(cov_matrix), cov_matrix.T, cov_matrix)
            assert np.allclose(
                cov_matrix, cov_matrix.T
            ), "cov_alm_gen: covariance matrix not symmetric"

            self.cov_alm = cov_matrix
            if not ischain:
                self.save_cov()
            return self.cov_alm

    def cov_fullsky(self, alm_inds, n_cov, theory_cell):
        """
        Sets up covariance matrix for alms (i.e. when no mask is present). So far only works for spin-0 if spin-2 are calculated as well.

        Parameters
        ----------
        alm_inds : list of int
            integers according to alm kinds from cov_funcs.match_alm_inds()
        n_cov : int
            side length of covariance matrix
        theory_cell : 3D array
            cube of theory c_ell. first & second dimension: pairings of modes, third dimension: ell (see self.cell_cube())

        Returns
        -------
        2D array
            covariance matrix of pseudo-alm
        """
        cov_matrix = np.zeros((n_cov, n_cov))
        diag = np.zeros(n_cov)
        for i in alm_inds:
            t = int(np.floor(i / 2))  # same c_l for Re and Im
            len_sub = self._exact_lmax + 1
            cell_ranges = [np.repeat(theory_cell[t, t, i], i + 1) for i in range(len_sub)]
            full_ranges = [
                np.append(
                    cell_ranges[i],
                    np.zeros(len_sub - len(cell_ranges[i])),
                )
                for i in range(len(cell_ranges))
            ]
            cov_part = 0.5 * np.ndarray.flatten(np.array(full_ranges))
            if i % 2 == 0:  # true if alm part is real
                cov_part[::len_sub] *= 2
            else:
                cov_part[::len_sub] *= 0
            # alm with same m but different sign dont have vanishing covariance. This is only relevant if pos_m = False.
            len_2D = len(cov_part)
            pos = (len_2D * i, len_2D * (i + 1))

            diag[pos[0] : pos[1]] = cov_part
        assert len(diag) == n_cov
        cov_matrix = np.diag(diag)
        return cov_matrix

    def cov_masked(self, alm_inds, n_cov, theory_cell, lmin=0, pos_m=True):
        """
        Sets up covariance matrix for pseudo-alms. So far only works for spin-0 if spin-2 are calculated as well.

        Parameters
        ----------
        alm_inds : list of int
            integers according to alm kinds from cov_funcs.match_alm_inds()
        n_cov : int
            side length of covariance matrix
        theory_cell : 3D array
            cube of theory c_ell. first & second dimension: pairings of modes, third dimension: ell (see self.cell_cube())
        lmin : int
            minimum ell considered , by default 0
        pos_m : bool, optional
            only calculate contributions for positive m (seems to be sufficient for plus correlation function, couldn't show analytically why yet), by default True


        Returns
        -------
        2D array
            covariance matrix of pseudo-alm
        """
        tic = time.perf_counter()
        buffer = self._cov_ell_buffer

        if self.mask._precomputed:
            w_arr = self.mask._w_arr
        else:
            w_arr = self.mask.w_arr(cov_ell_buffer=buffer)
        cov_matrix = np.full((n_cov, n_cov), np.nan)
        print(
            "cov_masked: beginning to fill covariance matrix with size {} mb.".format(
                getsizeof(cov_matrix) / 1024**2
            )
        )
        num_alm = len(alm_inds)
        numparts = (num_alm**2 - num_alm) / 2 + num_alm

        part = 1
        for i in alm_inds:
            for j in alm_inds:
                if i <= j:
                    # if else depending on precomputation
                    if self.mask._precomputed:
                        precomputed = self.mask._wpm_delta, self.mask._wpm_stack
                        cov_part = cov_funcs.optimized_cov_4D_jit(
                            i, j, precomputed, self._exact_lmax + buffer, lmin, theory_cell
                        )
                    else:
                        cov_part = cov_funcs.cov_4D_jit(
                            i, j, w_arr, self._exact_lmax + buffer, lmin, theory_cell
                        )
                    if pos_m == False:
                        len_2D = (cov_part.shape[0] - buffer) * (cov_part.shape[1] - 2 * buffer)

                        cov_2D = np.reshape(
                            cov_part[:-buffer, buffer:-buffer, :-buffer, buffer:-buffer],
                            (len_2D, len_2D),
                        )
                    else:

                        len_2D = (cov_part.shape[0] - buffer) * (cov_part.shape[1] - buffer)

                        cov_2D = np.reshape(
                            cov_part[:-buffer, :-buffer, :-buffer, :-buffer], (len_2D, len_2D)
                        )

                    pos_y = (len_2D * i, len_2D * (i + 1))
                    pos_x = (len_2D * j, len_2D * (j + 1))
                    cov_matrix[pos_y[0] : pos_y[1], pos_x[0] : pos_x[1]] = cov_2D
                    # clear some memory:
                    del cov_2D
                    #gc.collect()
                    # print("cov_masked: finished part {:d}/{:d}.".format(int(part), int(numparts)))
                    part += 1
        toc = time.perf_counter()
        print("cov_masked: covariance matrix calculation took {:.3f} seconds".format((toc - tic)))
        return cov_matrix

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

    def set_noise_pixelsigma(self):
        print(
            "pixelsigma in Cov: deprecated, setting pixelsigma is moved to simulations. and should not be needed here."
        )
        if self.theorycl._sigma_e is not None:
            if isinstance(self.theorycl._sigma_e, str):
                self.pixelsigma = helper_funcs.get_noise_pixelsigma(self.mask.nside)

            elif isinstance(self.theorycl._sigma_e, tuple):
                self.pixelsigma = helper_funcs.get_noise_pixelsigma(
                    self.mask.nside, self.theorycl._sigma_e
                )
            else:
                raise RuntimeError(
                    "sigma_e needs to be string for default or tuple (sigma_e,n_gal)"
                )

        else:
            try:

                del self.pixelsigma
            except:
                pass

    def set_char_string(self):

        charstring = "_l{:d}_n{:d}_{}_{}_{}.npz".format(
            self._exact_lmax,
            self.mask.nside,
            self.mask.name,
            self.theorycl.name,
            self.theorycl.sigmaname,
        )
        return charstring

    def set_covalmpath(self):
        charac = self.set_char_string()

        if not os.path.isdir(self.working_dir + "/covariances"):
            command = "mkdir covariances"
            os.system(command)
        covname = self.working_dir + "/covariances/cov_xi" + charac
        self.covalm_path = covname

    def cl2pseudocl(self, ischain=False):
        # from namaster scientific documentation paper
        if not ischain:
            if not os.path.isdir(self.working_dir + "/pcls"):
                command = "mkdir " + self.working_dir + "/pcls"
                os.system(command)
            pclpath = (
                self.working_dir
                + "/pcls/pcl"
                + "_n{:d}_{}_{}_{}.npz".format(
                    self.mask.nside,
                    self.mask.name,
                    self.theorycl.name,
                    self.theorycl.sigmaname,
                )
            )

            if os.path.isfile(pclpath):
                pclfile = np.load(pclpath)
                self.p_ee = pclfile["pcl_ee"]
                self.p_bb = pclfile["pcl_bb"]
                self.p_eb = pclfile["pcl_eb"]
                return

        
        if hasattr(self.theorycl, "_noise_sigma"):
            cl_e = self.theorycl.ee.copy() + self.theorycl.noise_cl
            cl_b = self.theorycl.bb.copy() + self.theorycl.noise_cl

        else:
            cl_e = self.theorycl.ee.copy()
            cl_b = self.theorycl.bb.copy()
        cl_eb = cl_be = cl_b
        if self.mask.spin0 == True:
            m_llp_p, m_llp_m, m_llp_z = self.mask.m_llp
            self.p_ee = np.einsum("lm,m->l", m_llp_p, cl_e) + np.einsum("lm,m->l", m_llp_m, cl_b)
            self.p_bb = np.einsum("lm,m->l", m_llp_m, cl_e) + np.einsum("lm,m->l", m_llp_p, cl_b)
            self.p_eb = np.einsum("lm,m->l", m_llp_p, cl_eb) - np.einsum("lm,m->l", m_llp_m, cl_be)
            self.p_tt = np.einsum("lm,m->l", m_llp_z, cl_e)
        else:
            m_llp_p, m_llp_m = self.mask.m_llp
            self.p_ee = np.einsum("lm,m->l", m_llp_p, cl_e) + np.einsum("lm,m->l", m_llp_m, cl_b)
            self.p_bb = np.einsum("lm,m->l", m_llp_m, cl_e) + np.einsum("lm,m->l", m_llp_p, cl_b)
            self.p_eb = np.einsum("lm,m->l", m_llp_p, cl_eb) - np.einsum("lm,m->l", m_llp_m, cl_be)


        if not ischain:
            print("pseudo_cl: saving pseudo_cl...")
            np.savez(pclpath, pcl_ee=self.p_ee, pcl_bb=self.p_bb, pcl_eb=self.p_eb)
