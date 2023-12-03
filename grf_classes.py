import numpy as np
import healpy as hp
import wpm_funcs
import pickle
import os.path


def save_maskobject(maskobject, dir=""):
    name = dir + maskobject.name + "_l" + str(maskobject.lmax) + "_n" + str(maskobject.nside)
    maskfile = open(name, "wb")
    pickle.dump(maskobject, maskfile)


class TheoryCl:
    """
    A class to read, store and handle 2D theory power spectra
    """

    def __init__(
        self, lmax=30, clpath=None, theory_lmin=2, clname="3x2pt_kids", smooth_signal=False
    ):
        self.lmax = lmax
        print("lmax has been set to {:d}.".format(self.lmax))
        self.smooth_signal = smooth_signal
        self.ell = np.arange(self.lmax + 1)
        self.len_l = len(self.ell)
        self.theory_lmin = theory_lmin
        self.clname = clname
        self.clpath = clpath

        self.nn = None
        self.ee = None
        self.ne = None
        self.bb = None

        if self.clpath is not None:
            self.read_clfile()
            self.load_cl()
            print("Loaded C_l with lmax = {:d}".format(self.lmax))
            if self.smooth_signal == True:
                smooth_ell = self.lmax
                self.smooth_array = wpm_funcs.smooth_cl(self.ell, smooth_ell)
                self.ee *= self.smooth_array
                self.nn *= self.smooth_array
                self.ne *= self.smooth_array
                self.bb *= self.smooth_array
                self.clname += "_smooth{:d}".format(smooth_ell)
                print("Theory C_l smoothed to lsmooth = {:d}.".format(smooth_ell))

        else:
            print("Warning: no theory Cl provided, calculating with Cl=0")
            self.set_cl_zero()

    def read_clfile(self):
        self.raw_spectra = np.loadtxt(self.clpath)

    def load_cl(self):
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

    def set_cl_zero(self):
        self.clname = "none"
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
        maskname : str, optional
            name of the mask used for saving covariance matrices, by default "mask"

        Raises
        ------
        RuntimeError
            If no maskfile or specifications for a circular mask (which can also be ('fullsky',nside) are provided)
        RuntimeError
            If neither spin 0 or spin 2 are specified
        """
        if maskpath is not None:
            self.maskname = maskname
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
        if l_smooth == "auto":
            self.l_smooth_auto = True
            self.l_smooth = self._exact_lmax
            self.maskname += 'smoothl{}'.format(str(self.l_smooth))
        else:
            self.l_smooth_auto = False
            self.l_smooth = l_smooth
            self.maskname += 'smoothl{}'.format(str(self.l_smooth))

    def read_maskfile(self):
        """Reads a fits file and sets mask properties accordingly"""
        self.mask = hp.fitsfunc.read_map(self.maskpath, verbose=True)
        self.nside = hp.pixelfunc.get_nside(self.mask)
        self.area = hp.nside2pixarea(self.nside, degrees=True) * np.sum(self.mask)

    def get_circmask(self):
        """Sets mask properties to a circular mask and saves to file"""
        self.maskpath = "circular_{:d}sqd_nside{:d}.fits".format(self.area, self.nside)
        self.maskname = "circ{:d}".format(self.area)
        if os.path.isfile(self.maskpath):
            self.mask = hp.fitsfunc.read_map(self.maskpath, verbose=True)
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
        self.maskname = "fullsky"
        self.area = hp.nside2pixarea(self.nside, degrees=True) * np.sum(self.mask)
        hp.fitsfunc.write_map(self.maskpath, m, overwrite=True)

    @property
    def wlm(self):
        """Calculates spherical harmonic coefficients of the mask

        Returns
        -------
        array
            spin 0 alm of mask in healpix ordering
        """

        if self.l_smooth is None:
            self._wlm = hp.sphtfunc.map2alm(self.mask, lmax=self._exact_lmax)
        elif isinstance(self.l_smooth, int):
            self._wlm = hp.sphtfunc.map2alm(self.mask, lmax=self._exact_lmax)
            self._wlm *= wpm_funcs.smooth_alm(self._wlm, self.l_smooth, self._exact_lmax)
        elif self.l_smooth == "auto":
            self.l_smooth = self._exact_lmax
            self._wlm = hp.sphtfunc.map2alm(self.mask, lmax=self._exact_lmax)
            self._wlm *= wpm_funcs.smooth_alm(self._wlm, self._exact_lmax, self._exact_lmax)
        else:
            raise RuntimeError("l_smooth needs to be None, integer or auto")

        return self._wlm

    @property
    def wlm_lmax(self):
        """Calculates spherical harmonic coefficients of the mask to bandlimit of mask

        Returns
        -------
        array
            spin 0 alm of mask in healpix ordering
        """
        if self.l_smooth is None:
            self._wlm_lmax = hp.sphtfunc.map2alm(self.mask)
        elif isinstance(self.l_smooth, int):
            self._wlm_lmax = hp.sphtfunc.map2alm(self.mask)
            self._wlm_lmax *= wpm_funcs.smooth_alm(
                self._wlm_lmax, self.l_smooth, 3 * self.nside - 1
            )
        
        elif self.l_smooth == "auto":
            self.l_smooth = self._exact_lmax
            self._wlm = hp.sphtfunc.map2alm(self.mask)
            self._wlm *= wpm_funcs.smooth_alm(self._wlm_lmax, self._exact_lmax, 3 * self.nside - 1)
        else:
            raise RuntimeError("l_smooth needs to be None or integer")
        return self._wlm_lmax

    @property
    def wl(self):
        wlm = self.wlm_lmax
        self._wl = hp.sphtfunc.alm2cl(wlm)
        mask_smooth = hp.sphtfunc.alm2map(wlm, self.nside)
        self.smooth_mask = mask_smooth
        
        return self._wl

    @property
    def eff_area(self):
        mask_smooth = hp.sphtfunc.alm2map(self.wlm_lmax, self.nside)
        self._eff_area = hp.nside2pixarea(self.nside, degrees=True) * np.sum(mask_smooth)
        return self._eff_area

    def initiate_w_arrs(self):
        self.L = np.arange(self._exact_lmax + 1)
        print("4D W_llpmmp will be calculated for lmax = {:d}".format(self._exact_lmax))
        self.M = np.arange(-self._exact_lmax, self._exact_lmax + 1)
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
        self.buffer_lmax = self._exact_lmax + 0

        if self.spin0 is None:
            w0 = 0
        else:
            allowed_l, wigners0 = wpm_funcs.prepare_wigners(0, L1, L2, M1, M2, self.buffer_lmax)

            wlm_l = wpm_funcs.get_wlm_l(
                self._wlm_lmax, m, self.lmax, allowed_l
            )  # needs the lmax the wlm are calculated for.
            prefac = wpm_funcs.w_factor(allowed_l, L1, L2)
            w0 = (-1) ** np.abs(M1) * np.sum(wigners0 * prefac * wlm_l)

        if self.spin2 is None or np.logical_or(L1 < 2, L2 < 2):
            wp, wm = 0, 0
        else:
            allowed_l, wp_l, wm_l = wpm_funcs.prepare_wigners(2, L1, L2, M1, M2, self.buffer_lmax)

            prefac = wpm_funcs.w_factor(allowed_l, L1, L2)
            wlm_l = wpm_funcs.get_wlm_l(self._wlm_lmax, m, self.lmax, allowed_l)
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

    @property
    def w_arr(self, verbose=True):
        if self.w0_arr is None and self.wpm_arr is None:
            self.initiate_w_arrs()

        arglist = []
        if verbose:
            print("Preparing list of l, m arguments")
        for l1, L1 in enumerate(self.L):
            M1_arr = np.arange(-L1, L1 + 1)
            for l2, L2 in enumerate(self.L):
                M2_arr = np.arange(-L2, L2 + 1)

                for m1, M1 in enumerate(M1_arr):
                    for m2, M2 in enumerate(M2_arr):
                        arglist.append((L1, L2, M1, M2))

        # n_proc = mup.cpu_count() - 1

        """ if verbose:
            print(f'Computing W_lmlpmp with {n_proc} cores')     """

        # pool = mup.Pool(processes=n_proc)
        # with mup.Pool(processes=n_proc) as pool:
        self.wlm
        self.wlm_lmax
        print("Starting computation of 4D W_llpmmp arrays... ", end="")
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
            return self._w_arr

        elif self.spin0:
            helper = np.empty_like(self.w0_arr)[None, :, :, :, :]
            self._w_arr = np.append(np.append(helper, helper, axis=0), self.w0_arr, axis=0)
            self.w0_arr = (
                None  # could also delete these attributes from the instance itself to make space
            )
            self.wpm_arr = None
            return self._w_arr

        elif self.spin2:
            self._w_arr = self.wpm_arr
            self.w0_arr = None
            self.wpm_arr = None
            return self._w_arr

    @property
    def m_llp(self):
        m_llp_p, m_llp_m = wpm_funcs.m_llp(self.wlm_lmax, self._exact_lmax)
        self._m_llp = m_llp_p, m_llp_m
        return self._m_llp
