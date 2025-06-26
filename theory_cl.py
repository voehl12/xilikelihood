import numpy as np
import pyccl as ccl
import os
import helper_funcs
import wpm_funcs


class RedshiftBin:
    """
    A class to store and handle redshift bins
    """

    def __init__(self, nbin, z=None, nz=None, zmean=None, zsig=None, filepath=None):
        self.nbin = nbin
        self.z = z
        self.nz = nz
        if filepath is not None:
            zbin = np.loadtxt(filepath)
            self.z, self.nz = zbin[:, 0], zbin[:, 1]
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
        cosmo=None,
        z_bins=None,
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
        self.cosmo = cosmo
        self.z_bins = z_bins
        self.nn = None
        self.ee = None
        self.ne = None
        self.bb = None
        self.eb = None
        self._sigma_e = sigma_e
        

        if self.clpath is not None:
            self.read_clfile()
            self.load_cl()
            print("Loaded C_l with lmax = {:d}".format(self.lmax))

        elif self.cosmo is not None:
            import theory_cl

            if self.z_bins is None:
                print("Warning: no redshift bins provided, using default bins.")
                bin1 = RedshiftBin(
                    z=np.linspace(0, 1, 100), nz=np.ones(100), zmean=0.5, zsig=0.1, nbin=1
                )
                bin2 = RedshiftBin(
                    z=np.linspace(0, 1, 100), nz=np.ones(100), zmean=0.5, zsig=0.1, nbin=2
                )
                self.z_bins = (bin1, bin2)

            cl = theory_cl.get_cl(self.cosmo, self.z_bins)

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
            self.name = None
            self.clpath = 'fromscratch'

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

        self.set_noise_sigma()

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
        self.bb = np.zeros_like(self.ee)
        self.eb = np.zeros_like(self.ee)



def clnames(s8):
    s8str = str(s8)
    s8str = s8str.replace(".", "p").lstrip("0")
    s8name = "S8" + s8str
    clpath = "Cl_3x2pt_kids55_s8{}.txt".format(s8str)
    return clpath, s8name


def prep_cosmo(params):
    import pyccl as ccl

    """
    function to generate vanilla cosmology with ccl given a value of S8

    Parameters
    ----------
    s8 : float
        lensing amplitude, clustering parameter S8

    Returns
    -------
    object
        ccl vanilla cosmology
    """
    s8 = params["s8"]
    omega_m = params["omega_m"] #0.31
    omega_b = 0.046
    omega_c = omega_m - omega_b
    sigma8 = s8 * (omega_m / 0.3) ** -0.5
    cosmo = ccl.Cosmology(Omega_c=omega_c, Omega_b=omega_b, h=0.7, sigma8=sigma8, n_s=0.97)
    return cosmo



def calc_cl(cosmo, ell, z_bins):
    """
    generate 3x2pt c_ell given a ccl cosmology

    Parameters
    ----------
    cosmo : object
        ccl cosmology to be used
    ell : array
        array of integer ell values to be used for the cl
    nz_path : string
        path to a redshift distribution to be used

    Returns
    -------
    tuple
        cl in the order ee, ne, nn
    """
    # rbin = np.loadtxt(nz_path)
    bin1, bin2 = z_bins
    z1, nz1 = bin1.z, bin1.nz
    z2, nz2 = bin2.z, bin2.nz
    # b = 1.5 * np.ones_like(z)
    lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z1, nz1))
    lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z2, nz2))
    # clu = ccl.NumberCountsTracer(cosmo, has_rsd=False, dndz=(z, nz), bias=(z, b))
    cl_ee = ccl.angular_cl(cosmo, lens1, lens2, ell)
    # cl_ne = ccl.angular_cl(cosmo, lens, clu, ell)
    # cl_nn = ccl.angular_cl(cosmo, clu, clu, ell)
    cl_ne = cl_nn = np.zeros_like(cl_ee)
    return cl_ee, cl_ne, cl_nn


def save_cl(cl, clpath):
    cl_ee, cl_ne, cl_nn = cl
    # np.savez(clpath, theory_ellmin=2,ee=cl_ee,ne=cl_ne,nn=cl_nn)
    np.savetxt(clpath, (cl_ee, cl_ne, cl_nn), header="EE, nE, nn")



def get_cl(params_dict, z_bins):
    cosmo = prep_cosmo(params_dict)
    ell = np.arange(2, 2000)
    cl = calc_cl(cosmo, ell, z_bins)
    return cl

class BinCombinationMapper:
    def __init__(self, max_n):
        """
        Initialize the mapper with a maximum value for combinations.
        """
        self.index_to_comb = {}
        self.comb_to_index = {}
        self.combinations = []
        index = 0
        for i in range(max_n):
            for j in range(i, -1, -1):
                self.index_to_comb[index] = (i, j)
                self.comb_to_index[(i, j)] = index
                self.combinations.append([i, j])
                index += 1
        self.combinations = np.array(self.combinations)

    def get_combination(self, index):
        """
        Retrieve the combination corresponding to the given index.
        """
        return self.index_to_comb.get(index, None)

    def get_index(self, combination):
        """
        Retrieve the index corresponding to the given combination.
        """
        return self.comb_to_index.get(combination, None)
    
   





def prepare_theory_cl_inputs(redshift_bins, noise):
    """
    Prepares redshift bin combinations, cross-/autocorrelation flags, and noise values.

    Parameters:
    - redshift_bins: List of redshift bins.
    - noise: Default noise value or None.

    Returns:
    - numerical_combinations: Array of numerical redshift bin combinations.
    - redshift_bin_combinations: List of redshift bin combinations as tuples.
    - is_cov_cross: Boolean array indicating cross-correlations.
    - shot_noise: List of noise values for each redshift bin combination.
    """
    n_redshift_bins = len(redshift_bins)
    mapper = BinCombinationMapper(n_redshift_bins)
    numerical_combinations = mapper.combinations
    # Create redshift bin combinations
    redshift_bin_combinations = [
        (redshift_bins[comb[0]], redshift_bins[comb[1]])
        for comb in numerical_combinations
    ]

    # Determine cross-/autocorrelations
    is_cov_cross = numerical_combinations[:, 0] != numerical_combinations[:, 1]

    # Assign noise values
    shot_noise = [None if val else noise for val in is_cov_cross]

    return numerical_combinations, redshift_bin_combinations, is_cov_cross, shot_noise, mapper

def generate_theory_cl(lmax, redshift_bin_combinations, shot_noise, cosmo):
    """
    Generate TheoryCl instances for given redshift bins and cosmology.

    Parameters:
    - lmax: Maximum multipole moment.
    - redshift_bins: List of redshift bins.
    - noise: List of noise values for each redshift bin combination.
    - cosmo: Cosmology dictionary.

    Returns:
    - List of TheoryCl instances.
    """
       
    # Generate TheoryCl instances
    theory_cl = [
        TheoryCl(lmax, cosmo=cosmo, z_bins=bin_comb, sigma_e=noise_val)
        for bin_comb, noise_val in zip(redshift_bin_combinations, shot_noise)
    ]
    return theory_cl