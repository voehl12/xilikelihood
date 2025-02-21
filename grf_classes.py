import numpy as np
import healpy as hp
import wpm_funcs, cov_funcs
import pickle
import os.path
from sys import getsizeof
import file_handling
import time
import scipy.stats
import helper_funcs


def save_maskobject(maskobject, dir=""):
    name = dir + maskobject.name + "_l" + str(maskobject.lmax) + "_n" + str(maskobject.nside)
    maskfile = open(name, "wb")
    pickle.dump(maskobject, maskfile)


class RedshiftBin:
    """
    A class to store and handle redshift bins
    """

    def __init__(self, z, nz, nbin, zmean=None, zsig=None):
        self.z = z
        self.nz = nz
        self.nbin = nbin
        if zmean is not None and zsig is not None:
            self.zmean = zmean
            self.zsig = zsig
            self.nz = scipy.stats.norm.pdf(self.z, loc=zmean, scale=zsig)
        self.name = "bin{:d}".format(nbin)


class TheoryCl:
    """
    A class to read, create, store and handle 2D theory power spectra
    """

    def __init__(
        self,
        lmax,
        clpath=None,
        sigma_e=None,
        theory_lmin=2,
        clname="test_cl",
        smooth_signal=None,
        s8=None,
        working_dir=None,
    ):
        self.lmax = lmax
        print("lmax has been set to {:d}.".format(self.lmax))
        self.smooth_signal = smooth_signal
        self.ell = np.arange(self.lmax + 1)
        self.len_l = len(self.ell)
        self.theory_lmin = theory_lmin
        self.name = clname
        self.clpath = clpath  # if clpath is set, s8 will be ignored.
        if working_dir is None:
            working_dir = os.getcwd()
        self.working_dir = working_dir
        self.s8 = s8
        self.nn = None
        self.ee = None
        self.ne = None
        self.bb = None
        self.eb = None
        self._sigma_e = sigma_e
        self.set_noise_sigma()

        if self.clpath is not None:
            self.read_clfile()
            self.load_cl()
            print("Loaded C_l with lmax = {:d}".format(self.lmax))

        elif self.s8 is not None:
            import theory_cl

            self.clpath, self.name = theory_cl.clnames(self.s8)
            if file_handling.check_for_file(self.clpath, kind="theory cl"):
                self.read_clfile()
                self.load_cl()
                print("Loaded C_l with lmax = {:d}".format(self.lmax))
            else:

                cl = theory_cl.get_cl_s8(self.s8)
                theory_cl.save_cl(cl, self.clpath)
                cl = np.array(cl)
                spectra = np.concatenate(
                    (
                        np.zeros((3, self.theory_lmin)),
                        cl[:, : self.lmax - self.theory_lmin + 1],
                    ),
                    axis=1,
                )

                self.ee = spectra[0]
                self.ne = spectra[1]
                self.nn = spectra[2]
                self.bb = np.zeros_like(self.ee)
                self.eb = np.zeros_like(self.ee)

        else:
            print("Warning: no theory Cl provided, calculating with Cl=0")
            self.set_cl_zero()

        if self.smooth_signal is not None:
            smooth_ell = self.smooth_signal
            self.smooth_array = wpm_funcs.smooth_cl(self.ell, smooth_ell)
            self.ee *= self.smooth_array
            self.nn *= self.smooth_array
            self.ne *= self.smooth_array
            self.bb *= self.smooth_array
            self.name += "_smooth{:d}".format(smooth_ell)
            print("Theory C_l smoothed to lsmooth = {:d}.".format(smooth_ell))

    @property
    def sigma_e(self):
        return self._sigma_e

    def set_noise_sigma(self):

        if self._sigma_e is not None:
            if isinstance(self._sigma_e, str):
                self._noise_sigma = helper_funcs.get_noise_cl()
                self.sigmaname = "noise" + self._sigma_e

            elif isinstance(self._sigma_e, tuple):
                self._noise_sigma = helper_funcs.get_noise_cl(*self._sigma_e)
                self.sigmaname = "noise" + str(self._sigma_e).replace(".", "")

            else:
                raise RuntimeError(
                    "sigma_e needs to be string for default or tuple (sigma_e,n_gal)"
                )
            self.noise_cl = np.ones(self.lmax + 1) * self._noise_sigma
            if self.smooth_signal is not None:
                self.noise_cl *= self.smooth_array
        else:
            self.sigmaname = "nonoise"
            try:
                del self._noise_sigma

            except:
                pass

    def read_clfile(self):
        self.raw_spectra = np.loadtxt(self.clpath)

    def load_cl(self):
        # cl files should also eventually become npz files with ee, ne, nn saved seperately, ell
        spectra = np.concatenate(
            (
                np.zeros((3, self.theory_lmin)),
                self.raw_spectra[:, : self.lmax - self.theory_lmin + 1],
            ),
            axis=1,
        )
        self.ee = spectra[0]
        self.ne = spectra[1]
        self.nn = spectra[2]
        self.bb = np.zeros_like(self.ee)
        self.eb = np.zeros_like(self.ee)

    def set_cl_zero(self):
        self.name = "none"
        self.ee = self.ne = self.nn = np.zeros(self.len_l)


class SphereMask:
    """
    A class used to store and calculate properties of a survey mask on a sphere.
    Maybe split this class into a mask and a field class or have the mask inherit from the field.
    Attributes
    ----------
    mask : array
        a healpy map of the mask
    nside: integer
        healpy nside parameter of the mask
    exact_lmax: integer
        maximum multipole moment to which calculations are taken to

    Methods
    -------
    calc_w_arrs(verbose=True)
        Calculates 4D coupling matrices for a given mask, +,- and 0 depending on spins required. Saves to w_arr stacked as w_p, w_m,w_0


    """

    def __init__(
        self,
        spins=[0, 2],
        maskpath=None,
        circmaskattr=None,
        lmin=0,
        exact_lmax=None,
        maskname="mask",
        l_smooth=None,
        working_dir=None,
    ) -> None:
        """

        Parameters
        ----------
        spins : list, optional
            spins of the fields under consideration, by default [0, 2]
        maskpath : string, optional
            path to a fits-file for a mask, by default None
        circmaskattr : tuple, optional
            (area_in_deg,nside) for a circular mask, by default None
        prep_wlm : bool, optional
            calculate spherical harmonic coefficients of the mask on initialization, by default True
        lmin : integer, optional
            minimum multipole moment used, by default 0
        exact_lmax : integer, optional
            maximum multipole moment used for exact calculations, by default None, then defaults to bandlimit of mask resolution
        name : str, optional
            name of the mask used for saving covariance matrices, by default "mask"

        Raises
        ------
        RuntimeError
            If no maskfile or specifications for a circular mask (which can also be ('fullsky',nside) are provided)
        RuntimeError
            If neither spin 0 or spin 2 are specified
        """
        if maskpath is not None:
            self.name = maskname
            self.maskpath = maskpath
            self.read_maskfile()

        elif circmaskattr is not None:
            if circmaskattr[0] == "fullsky":
                self.nside = circmaskattr[1]
                self.fullsky_mask()
                self.area = 41253

            else:
                self.area, self.nside = circmaskattr
                self.get_circmask()
        else:
            raise RuntimeError(
                "Please specify either a mask path or attributes for a circular mask"
            )
        if working_dir is None:
            working_dir = os.getcwd()
        self.working_dir = working_dir
        self._precomputed = False
        self.npix = hp.nside2npix(self.nside)
        self.spins = spins
        self.spin0 = None
        self.spin2 = None
        self.n_field = 0
        if 0 in spins:
            self.spin0 = True
            self.n_field += 1
        if 2 in spins:
            self.spin2 = True
            self.n_field += 2
        if not self.spin0 and not self.spin2:
            raise RuntimeError("Spin needs to be 0 and/or 2")
        self.lmax = 3 * self.nside - 1
        if exact_lmax is not None:
            self._exact_lmax = exact_lmax
        else:
            self._exact_lmax = 3 * self.nside - 1
            print("Warning: exact lmax has been set to {:d}.".format(self._exact_lmax))

        self.lmin = lmin

        self.L = None
        self.M = None
        self.w0_arr = None
        self.wpm_arr = None
        self._w_arr = None
        if l_smooth is not None:
            if l_smooth == "auto":
                self.l_smooth_auto = True
                self.l_smooth = self._exact_lmax

            else:
                self.l_smooth_auto = False
                self.l_smooth = l_smooth

            self.set_smoothed_mask()
            self.name += "smoothl{}".format(str(self.l_smooth))

    def read_maskfile(self):
        """Reads a fits file and sets mask properties accordingly"""
        self.mask = hp.fitsfunc.read_map(self.maskpath)
        self.nside = hp.pixelfunc.get_nside(self.mask)
        self.area = hp.nside2pixarea(self.nside, degrees=True) * np.sum(self.mask)

    def get_circmask(self):
        """Sets mask properties to a circular mask and saves to file"""
        self.maskpath = "circular_{:d}sqd_nside{:d}.fits".format(self.area, self.nside)
        self.name = "circ{:d}".format(self.area)
        if os.path.isfile(self.maskpath):
            self.mask = hp.fitsfunc.read_map(self.maskpath)
            assert np.allclose(
                self.area, hp.nside2pixarea(self.nside, degrees=True) * np.sum(self.mask), rtol=0.1
            ), (self.area, hp.nside2pixarea(self.nside, degrees=True) * np.sum(self.mask))
        else:
            npix = hp.nside2npix(self.nside)
            m = np.zeros(npix)
            vec = hp.ang2vec(np.pi / 2, 0)
            r = np.sqrt(self.area / np.pi)
            disc = hp.query_disc(nside=self.nside, vec=vec, radius=np.radians(r))
            m[disc] = 1
            self.mask = m
            hp.fitsfunc.write_map(self.maskpath, m, overwrite=True)

    def fullsky_mask(self):
        """Sets mask properties to full sky (i.e. no mask)"""
        npix = hp.nside2npix(self.nside)
        m = np.ones(npix)
        self.mask = m
        self.maskpath = "fullsky_nside{:d}.fits".format(self.nside)
        self.name = "fullsky"
        self.area = hp.nside2pixarea(self.nside, degrees=True) * np.sum(self.mask)
        hp.fitsfunc.write_map(self.maskpath, m, overwrite=True)

    @property
    def smooth_alm(self):
        self._smooth_alm = wpm_funcs.smooth_alm(self.l_smooth, self._exact_lmax)
        return self._smooth_alm

    @property
    def smooth_alm_lmax(self):
        self._smooth_alm_lmax = wpm_funcs.smooth_alm(self.l_smooth, self.lmax)
        return self._smooth_alm_lmax

    @property
    def wlm(self):
        """Calculates spherical harmonic coefficients of the mask

        Returns
        -------
        array
            spin 0 alm of mask in healpix ordering
        """

        if hasattr(self, "l_smooth"):
            self._wlm = hp.sphtfunc.map2alm(self.smooth_mask, lmax=self._exact_lmax)
        else:
            self._wlm = hp.sphtfunc.map2alm(self.mask, lmax=self._exact_lmax)
        return self._wlm

    @property
    def wlm_lmax(self):
        """Calculates spherical harmonic coefficients of the mask to bandlimit of mask

        Returns
        -------
        array
            spin 0 alm of mask in healpix ordering
        """

        if hasattr(self, "l_smooth"):
            self._wlm_lmax = hp.sphtfunc.map2alm(self.smooth_mask, lmax=self.nside // 2)
            if self._exact_lmax > self.nside // 2:
                ValueError(
                    "Exact lmax exceeds mask bandlimit after smoothing, requires careful reconsideration of mask calculations (mixing matrices need to be filled up with zeros)."
                )

            return self._wlm_lmax
        else:
            self._wlm_lmax = hp.sphtfunc.map2alm(self.mask, lmax=self.lmax)
            return self._wlm_lmax

    def set_smoothed_mask(self):

        sigma = 1 / self.l_smooth * 300
        smooth_mask = hp.sphtfunc.smoothing(
            self.mask, sigma=np.abs(sigma), iter=50, use_pixel_weights=True
        )
        self.smooth_mask = smooth_mask
        self.wl
        self.wlm

    @property
    def wl(self):
        wlm = self.wlm_lmax
        self._wl = hp.sphtfunc.alm2cl(wlm)

        return self._wl

    @property
    def eff_area(self):
        if hasattr(self, "smooth_mask"):
            self._eff_area = hp.nside2pixarea(self.nside, degrees=True) * np.sum(self.smooth_mask)
            return self._eff_area
        else:
            print("Warning: attempting effective area of unsmoothed mask, returning actual area...")
            return self.area

    def initiate_w_arrs(self, cov_ell_buffer):

        buffer = cov_ell_buffer
        self.L = np.arange(self._exact_lmax + buffer + 1)
        print(
            "4D W_llpmmp will be calculated for lmax = {:d} + {:d}".format(self._exact_lmax, buffer)
        )
        self.M = np.arange(-self._exact_lmax - buffer, self._exact_lmax + buffer + 1)
        Nl = len(self.L)
        Nm = len(self.M)
        if self.spin0:
            self.w0_arr = np.zeros((Nl, Nm, Nl, Nm), dtype=complex)
        if self.spin2:
            self.wpm_arr = np.zeros((2, Nl, Nm, Nl, Nm), dtype=complex)

    def calc_w_element(
        self, L1, L2, M1, M2
    ):  # move this to wpm_funcs, since it does not attach anything to the object? but it heavily relies on wlm
        m = M1 - M2
        m1_ind = np.argmin(np.fabs(M1 - self.M))
        m2_ind = np.argmin(np.fabs(M2 - self.M))
        l1_ind = np.argmin(np.fabs(L1 - self.L))
        l2_ind = np.argmin(np.fabs(L2 - self.L))
        inds = (l1_ind, m1_ind, l2_ind, m2_ind)
        self.buffer_lmax = self._exact_lmax  # buffer in these sums does not change anything

        if self.spin0 is None:
            w0 = 0
        else:
            allowed_l, wigners0 = wpm_funcs.prepare_wigners(0, L1, L2, M1, M2, self.buffer_lmax)

            wlm_l = wpm_funcs.get_wlm_l(
                self._wlm_lmax, m, allowed_l
            )  # needs the lmax the wlm are calculated for.
            prefac = wpm_funcs.w_factor(allowed_l, L1, L2)
            w0 = (-1) ** np.abs(M1) * np.sum(wigners0 * prefac * wlm_l)

        if self.spin2 is None or np.logical_or(L1 < 2, L2 < 2):
            wp, wm = 0, 0
        else:
            allowed_l, wp_l, wm_l = wpm_funcs.prepare_wigners(2, L1, L2, M1, M2, self.buffer_lmax)

            prefac = wpm_funcs.w_factor(allowed_l, L1, L2)
            wlm_l = wpm_funcs.get_wlm_l(self._wlm_lmax, m, allowed_l)
            wlm_l_large = np.where(np.abs((wlm_l)) > 1e-17, wlm_l, 0)
            wp = 0.5 * (-1) ** np.abs(M1) * np.sum(prefac * wlm_l * wp_l)
            """ assert np.allclose(
                np.sum(prefac * wlm_l * wp_l).real - np.cumsum(prefac * wlm_l * wp_l).real[-10:],
                np.zeros(10),
            )
            assert np.allclose(
                np.sum(prefac * wlm_l * wp_l).imag - np.cumsum(prefac * wlm_l * wp_l).imag[-10:],
                np.zeros(10),
            ) """
            wm = 0.5 * 1j * (-1) ** np.abs(M1) * np.sum(prefac * wlm_l * wm_l)

        return (inds, w0, wp, wm)

    def save_w_element(self, result):
        inds, w0, wp, wm = result
        if w0:
            self.w0_arr[inds] = w0
        if wp or wm:
            inds = (slice(0, 2), *inds)
            self.wpm_arr[inds] = [wp, wm]

    def w_arr(self, cov_ell_buffer=0, verbose=True, path=None):
        if path is None:
            self.set_wpmpath(cov_ell_buffer)
        else:
            self.wpm_path = path
        if self.check_w_arr():
            self.load_w_arr()
            return self._w_arr
        if self.w0_arr is None and self.wpm_arr is None:
            self.initiate_w_arrs(cov_ell_buffer)

        arglist = []
        if verbose:
            print("Preparing list of l, m arguments...")
        for l1, L1 in enumerate(self.L):
            # TODO: implement pos_m stuff here, so m1 is only calculated for positive m. Check, whether positive m also suffice for m2.
            M1_arr = np.arange(-L1, L1 + 1)
            for l2, L2 in enumerate(self.L):
                M2_arr = np.arange(-L2, L2 + 1)

                for m1, M1 in enumerate(M1_arr):
                    for m2, M2 in enumerate(M2_arr):
                        arglist.append((L1, L2, M1, M2))

        self.wlm
        self.wlm_lmax
        print("Starting computation of 4D W_llpmmp arrays... ")
        for i, arg in enumerate(arglist):
            print(
                "Computing 4D W_llpmmp arrays......{:4.1f}%".format(i / len(arglist) * 100),
                end="\r",
            )
            result = self.calc_w_element(*arg)
            self.save_w_element(result)
            # pool.apply_async(self.calc_w_element, args=arg,callback=self.save_w_element)
            # pool.close()
            # pool.join()
        print()
        print("Finished.")
        if self.spin0 and self.spin2:
            self._w_arr = np.append(self.wpm_arr, self.w0_arr, axis=0)
            self.w0_arr = (
                None  # could also delete these attributes from the instance itself to make space
            )
            self.wpm_arr = None
            self.save_w_arr()
            return self._w_arr

        elif self.spin0:
            helper = np.empty_like(self.w0_arr)[None, :, :, :, :]
            self._w_arr = np.append(np.append(helper, helper, axis=0), self.w0_arr, axis=0)
            self.w0_arr = (
                None  # could also delete these attributes from the instance itself to make space
            )
            self.wpm_arr = None
            self.save_w_arr()
            return self._w_arr

        elif self.spin2:
            self._w_arr = self.wpm_arr
            self.w0_arr = None
            self.wpm_arr = None
            self.save_w_arr()
            return self._w_arr

    def precompute_for_cov_masked(self, cov_ell_buffer=0):
        tic1 = time.perf_counter()
        w_arr = self.w_arr(cov_ell_buffer=cov_ell_buffer)
        self._wpm_delta, self._wpm_stack = cov_funcs.precompute_xipm(w_arr)
        _ = self.m_llp
        self._precomputed = True
        toc1 = time.perf_counter()
        print("cov_masked: precomputation took {:.2f} minutes".format((toc1 - tic1) / 60))

    @property
    def m_llp(self):
        self.set_mllppath()
        if file_handling.check_for_file(self.mllp_path):
            self.load_mllp_arr()
        else:
            m_llp_p, m_llp_m = wpm_funcs.m_llp(self.wl, self.lmax)
            self._m_llp = m_llp_p, m_llp_m
            self.save_mllp_arr()
        return self._m_llp

    def save_w_arr(self):
        print("Saving Wpm0 arrays.")
        np.savez(self.wpm_path, wpm0=self._w_arr)

    def save_mllp_arr(self):
        np.savez(self.mllp_path, m_llp_p=self._m_llp[0], m_llp_m=self._m_llp[1])

    def check_w_arr(self):
        print("Checking for Wpm0 arrays... ", end="")
        print(self.wpm_path)
        if os.path.isfile(self.wpm_path):
            print("Found.")
            return True
        else:
            print("Not found.")
            return False

    def load_w_arr(self):
        print("Loading Wpm0 arrays.")
        wpmfile = np.load(self.wpm_path)
        print("Loaded with size {} mb.".format(getsizeof(wpmfile["wpm0"]) / 1024**2))
        self._w_arr = wpmfile["wpm0"]

    def load_mllp_arr(self):
        print("Loading Mllp arrays.")
        mllpfile = np.load(self.mllp_path)
        m_llp_p, m_llp_m = mllpfile["m_llp_p"], mllpfile["m_llp_m"]
        self._m_llp = m_llp_p, m_llp_m

    def set_wpm_string(self, cov_ell_buffer):

        charstring = "_l{:d}_n{:d}_{}.npz".format(
            self._exact_lmax + cov_ell_buffer,
            self.nside,
            self.name,
        )
        return charstring

    def set_wpmpath(self, cov_ell_buffer):
        charac = self.set_wpm_string(cov_ell_buffer)
        # covname = "covariances/cov_xi" + charac
        if not os.path.isdir(self.working_dir + "/wpm_arrays"):
            command = "mkdir " + self.working_dir + "/wpm_arrays"
            os.system(command)
        wpm_name = self.working_dir + "/wpm_arrays/wpm" + charac
        self.wpm_path = wpm_name

    def set_mllppath(self):
        charstring = "_l{:d}_n{:d}_{}.npz".format(
            self._exact_lmax,
            self.nside,
            self.name,
        )

        if not os.path.isdir(self.working_dir + "/mllp_arrays"):
            command = "mkdir " + self.working_dir + "/mllp_arrays"
            os.system(command)
        mllp_name = self.working_dir + "/mllp_arrays/mllp" + charstring
        self.mllp_path = mllp_name
