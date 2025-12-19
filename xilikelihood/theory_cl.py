"""
Theoretical power spectra computation and handling.

This module provides classes and functions for working with theoretical
angular power spectra, including C_l computation from cosmology and
redshift bin handling.

Classes
-------
RedshiftBin : Redshift distribution container
TheoryCl : Theoretical power spectrum handler  
BinCombinationMapper : Maps redshift bin combinations to indices

Examples
--------
>>> # Create redshift bins
>>> z = np.linspace(0, 2, 100)
>>> bin1 = RedshiftBin(1, z=z, zmean=0.5, zsig=0.1)
>>> bin2 = RedshiftBin(2, z=z, zmean=1.0, zsig=0.1)

>>> # Create cosmology
>>> cosmo_params = {'s8': 0.8, 'omega_m': 0.3}
>>> theory_cl = TheoryCl(lmax=1000, cosmo=cosmo_params, z_bins=(bin1, bin2))
"""


import numpy as np
import healpy as hp
import re
import scipy.stats 
from pathlib import Path
import logging
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

# Optional dependencies with graceful degradation
try:
    import pyccl as ccl
    HAS_CCL = True
except ImportError:
    HAS_CCL = False
    ccl = None

# Local imports (fix based on your package structure)
try:
    from . import noise_utils
    from . import wpm_funcs
except ImportError:
    # For backward compatibility during development
    import noise_utils
    import wpm_funcs

__all__ = [
    # Main classes
    'RedshiftBin',
    'TheoryCl', 
    'BinCombinationMapper',
    
    # Utility functions
    'create_cosmo',
    'compute_angular_power_spectra',
    'get_cl',
    'prepare_theory_cl_inputs',
    'generate_theory_cl',
    'load_kids_redshift_bins',
    
    # Legacy functions (if needed)
    'clnames',
    'save_cl',
]

class RedshiftBin:
    """
    A class to store and handle redshift bins.
    
    Parameters
    ----------
    nbin : int
        Bin number identifier
    z : array_like, optional
        Redshift values
    nz : array_like, optional
        Number density at each redshift
    zmean : float, optional
        Mean redshift for Gaussian distribution
    zsig : float, optional
        Standard deviation for Gaussian distribution
    filepath : str, optional
        Path to file containing z, nz data
        
    Attributes
    ----------
    z_eff : float
        Effective redshift computed as sum(z * nz) / sum(nz), computed once and cached
    """

    def __init__(self, nbin, z=None, nz=None, zmean=None, zsig=None, filepath=None):
        self.nbin = nbin
        self.name = f"bin{nbin:d}"
        self._z_eff = None  # Cache for effective redshift
        
        if filepath is not None:
            self._load_from_file(filepath)
        elif zmean is not None and zsig is not None and z is not None:
            self._create_gaussian(z, zmean, zsig)
        elif z is not None and nz is not None:
            self.z = np.array(z)
            self.nz = np.array(nz)
        else:
            raise ValueError("Must provide either filepath, (z,nz), or (z,zmean,zsig)")

        self.z_min = np.min(self.z)
        self.z_max = np.max(self.z)
    @property
    def z_eff(self):
        """
        Effective redshift computed as the mean of the distribution weighted by nz.
        
        Returns
        -------
        float
            Effective redshift: sum(z * nz) / sum(nz)
        """
        if self._z_eff is None:
            self._z_eff = np.sum(self.z * self.nz) / np.sum(self.nz)
        return self._z_eff
    
    def _load_from_file(self, filepath):
        """Load redshift distribution from file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Redshift file not found: {filepath}")
        zbin = np.loadtxt(filepath)
        self.z, self.nz = zbin[:, 0], zbin[:, 1]
    
    def _create_gaussian(self, z, zmean, zsig):
        """Create Gaussian redshift distribution."""
        self.z = np.array(z)
        self.zmean = zmean
        self.zsig = zsig
        self.nz = scipy.stats.norm.pdf(self.z, loc=zmean, scale=zsig)


def load_kids_redshift_bins(directory=None):
    """
    Load KiDS tomographic redshift bins from directory.
    
    Parameters
    ----------
    directory : str, optional
        Directory containing KiDS TOMO*.asc files. If None, uses package default.
        
    Returns
    -------
    list of RedshiftBin
        Sorted list of RedshiftBin objects (sorted by bin number)
        
    Examples
    --------
    >>> # Load default KiDS bins
    >>> bins = load_kids_redshift_bins()
    >>> 
    >>> # Load from custom directory
    >>> bins = load_kids_redshift_bins("/path/to/kids/bins")
    >>> 
    >>> # Use in analysis
    >>> bin1, bin2 = bins[0], bins[1]
    """
    # Default to package-relative path
    if directory is None:
        package_dir = Path(__file__).parent.parent
        directory = package_dir / "redshift_bins" / "KiDS"
    else:
        directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Redshift directory not found: {directory}")
    
    # Find all TOMO files
    pattern = re.compile(r"TOMO(\d+)")
    nbins = []
    matched_files = []
    
    for filepath in directory.iterdir():
        if not filepath.is_file():
            continue
        match = pattern.search(filepath.name)
        if match:
            nbins.append(int(match.group(1)))
            matched_files.append(filepath)
    
    if not nbins:
        raise ValueError(f"No TOMO files found in {directory}")
    
    # Create RedshiftBin objects
    redshift_bins = [
        RedshiftBin(nbin=nbin, filepath=str(filepath))
        for nbin, filepath in zip(nbins, matched_files)
    ]
    
    # Sort by bin number
    redshift_bins_sorted = sorted(redshift_bins, key=lambda x: x.nbin)
    
    logger.info(f"Loaded {len(redshift_bins_sorted)} KiDS redshift bins from {directory}")
    return redshift_bins_sorted


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
        nside=None,
        cosmo=None,
        z_bins=None,
    ):
        
        self.lmax = lmax
        self.theory_lmin = theory_lmin
        self.name = clname
        self.clpath = clpath # if clpath is set, cosmo will be ignored.
        self.cosmo = cosmo
        self.z_bins = z_bins
        self.smooth_signal = smooth_signal
        self.nside = nside
        self._sigma_e = sigma_e
        
        # Initialize arrays
        self.ell = np.arange(self.lmax + 1)
        self.len_l = len(self.ell)
        self._initialize_spectra()

        # Apply smoothing if specified
        if self.smooth_signal is not None:
            self._apply_smoothing()
        
        if self.nside is not None:
            self.ell_window = hp.pixwin(self.nside, lmax=self.lmax)
            for attr in ['ee', 'nn', 'ne', 'bb']:
                if hasattr(self, attr):
                    setattr(self, attr, getattr(self, attr) * self.ell_window)
    

        # Set up noise
        self.set_noise_sigma()

        logger.info(f"Initialized TheoryCl with lmax = {self.lmax}")

    def _initialize_spectra(self):
        """Initialize power spectra from various sources."""
        if self.clpath is not None:
            self._load_from_file()
        elif self.cosmo is not None:
            self._compute_from_cosmology()
        else:
            self._set_zero_spectra()

    def _load_from_file(self):
        """Load spectra from file."""
        self.read_clfile()
        self.load_cl()

    def _compute_from_cosmology(self):
        """Compute spectra from cosmology."""
        if self.z_bins is None:
            self.z_bins = self._get_default_bins()
        
        logger.debug("Computing C_l from cosmology parameters")
        cl = get_cl(self.cosmo, self.z_bins)
        cl = np.array(cl)
        
        spectra = np.concatenate((
            np.zeros((3, self.theory_lmin)),
            cl[:, :self.lmax - self.theory_lmin + 1]
        ), axis=1)
        
        self.ee = spectra[0]
        self.ne = spectra[1] 
        self.nn = spectra[2]
        self.bb = np.zeros_like(self.ee)
        self.eb = np.zeros_like(self.ee)
        self.clpath = 'fromscratch'
        logger.debug(f"Computed C_l with shape {cl.shape}")
    
    def _set_zero_spectra(self):
        """Set all spectra to zero."""
        logger.warning("No theory C_l provided, calculating with C_l=0")
        self.set_cl_zero()

    def _get_default_bins(self):
        """Create default redshift bins."""
        logger.warning("No redshift bins provided, using default bins")
        z = np.linspace(0, 1, 100)
        bin1 = RedshiftBin(nbin=1, z=z, zmean=0.5, zsig=0.1)
        bin2 = RedshiftBin(nbin=2, z=z, zmean=0.5, zsig=0.1)
        return (bin1, bin2)
    
    def _apply_smoothing(self):
        """Apply smoothing to spectra."""
        self.smooth_array = wpm_funcs.smooth_cl(self.ell, self.smooth_signal)
        for attr in ['ee', 'nn', 'ne', 'bb']:
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr) * self.smooth_array)
        self.name += f"_smooth{self.smooth_signal:d}"
        logger.info(f"Theory C_l smoothed to lsmooth = {self.smooth_signal:d}")


    @property
    def sigma_e(self):
        return self._sigma_e
    
    @sigma_e.setter
    def sigma_e(self, value):
        """Set sigma_e and update noise calculations."""
        self._sigma_e = value
        self.set_noise_sigma()  # Recalculate noise when sigma_e changes

    def set_noise_sigma(self):
        """Set noise sigma and related attributes based on sigma_e type."""
        if self._sigma_e is None:
            self._set_no_noise()
            return
            
        self._compute_noise_sigma()
        self._create_noise_cl()
        
    def _set_no_noise(self):
        """Configure for no noise case."""
        self.sigmaname = "nonoise"
        if hasattr(self, '_noise_sigma'):
            delattr(self, '_noise_sigma')
            
    def _compute_noise_sigma(self):
        """Compute noise sigma based on sigma_e type."""
        if isinstance(self._sigma_e, str):
            self._noise_sigma = noise_utils.get_noise_cl()
            self.sigmaname = f"noise{self._sigma_e}"
            
        elif isinstance(self._sigma_e, tuple):
            self._noise_sigma = noise_utils.get_noise_cl(self._sigma_e)
            sigma_str = str(self._sigma_e).replace(".", "")
            self.sigmaname = f"noise{sigma_str}"
            
        else:
            raise ValueError(
                "sigma_e must be either a string for default noise or "
                "a tuple (sigma_e, n_gal) for custom noise parameters"
            )
            
    def _create_noise_cl(self):
        """Create the noise power spectrum array."""
        self.noise_cl = np.ones(self.lmax + 1) * self._noise_sigma
        if self.smooth_signal is not None:
            self.noise_cl *= self.smooth_array

    def read_clfile(self):
        clpath = Path(self.clpath)
        if not clpath.exists():
            raise FileNotFoundError(f"C_l file not found: {clpath}")
        logger.debug(f"Loading C_l from {clpath}")
        self.raw_spectra = np.loadtxt(clpath)
        logger.debug(f"Loaded C_l with shape {self.raw_spectra.shape}")
        

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



class BinCombinationMapper:
    """
    Maps combinations of redshift bins to indices and vice versa.
    
    Generates all unique ordered combinations (i,j) where i >= j.
    """
        
    def __init__(self, max_n):
        """Initialize mapper for combinations up to max_n bins."""
        self.max_n = max_n
        self._build_mappings()

    def _build_mappings(self):
        """Build the index-combination mappings."""
        self.index_to_comb = {}
        self.comb_to_index = {}
        self.combinations = []
        index = 0
        for i in range(self.max_n):
            for j in range(i, -1, -1):
                combination = (i, j)
                self.index_to_comb[index] = combination
                self.comb_to_index[combination] = index
                combination_rev = (j,i)
                self.comb_to_index[combination_rev] = index
                self.combinations.append([i, j])
                index += 1
        self.combinations = np.array(self.combinations)

    @property
    def n_combinations(self):
        """Number of combinations."""
        return len(self.combinations)
    
    def get_combination(self, index):
        """Get combination for given index."""
        return self.index_to_comb.get(index)
    
    def get_index(self, combination):
        """Get index for given combination."""
        return self.comb_to_index.get(tuple(combination))




def create_cosmo(params):
    

    """
    Create CCL cosmology from parameter dictionary.
    
    Parameters:
    -----------
    params : dict
        Dictionary containing 's8' and 'omega_m' keys
        
    Returns:
    --------
    ccl.Cosmology
        CCL cosmology object
    """
    if not HAS_CCL:
        raise ImportError("pyccl is required for cosmology calculations. Install with: pip install pyccl")
    
    s8 = params.get("s8", 0.8)
    #omega_m = params.get("omega_m", 0.3)
    h = params.get("h", 0.7)
    n_s = params.get("ns", 0.97)
    w_b = params.get("w_b", 0.021)
    w_c = params.get("w_c", 0.12)
    omega_b = w_b / (h ** 2)
    omega_c = w_c / (h ** 2)
    omega_m = omega_c + omega_b
    sigma8 = s8 * (omega_m / 0.3) ** -0.5
    ccl.gsl_params.LENSING_KERNEL_SPLINE_INTEGRATION = False
    return ccl.Cosmology(
        Omega_c=omega_c, 
        Omega_b=omega_b, 
        h=h, 
        sigma8=sigma8, 
        n_s=n_s
    )



def compute_angular_power_spectra(cosmo, ell, z_bins, ia_bias=(0.0, 0.0), z_bias=(0.0, 0.0)):
    """
    Compute angular power spectra for given cosmology and redshift bins.
    Returns 3x2pt power spectra: (cl_ee, cl_ne, cl_nn).
    
    Parameters:
    -----------
    cosmo : ccl.Cosmology
        CCL cosmology object
    ell : array_like
        Multipole moments
    z_bins : tuple
        Tuple of RedshiftBin objects
        
    Returns:
    --------
    tuple
        (cl_ee, cl_ne, cl_nn) power spectra
    """
    if not HAS_CCL:
        raise ImportError("pyccl is required for power spectrum calculations")
    
    bin1, bin2 = z_bins

    z1 = bin1.z
    nz1 = bin1.nz if z_bias[0] == 0.0 else add_n_of_z_bias(bin1.z, bin1.nz, z_bias[0])
    z2 = bin2.z
    nz2 = bin2.nz if z_bias[1] == 0.0 else add_n_of_z_bias(bin2.z, bin2.nz, z_bias[1])
    
    lens1 = ccl.WeakLensingTracer(cosmo, dndz=(z1, nz1),ia_bias=(z1, np.ones_like(z1) * ia_bias[0]))
    lens2 = ccl.WeakLensingTracer(cosmo, dndz=(z2, nz2),ia_bias=(z2, np.ones_like(z2) * ia_bias[1]))
    
    cl_ee = ccl.angular_cl(cosmo, lens1, lens2, ell)
    cl_ne = cl_nn = np.zeros_like(cl_ee)
    
    return cl_ee, cl_ne, cl_nn

def get_cl(params_dict, z_bins):
    """
    Get angular power spectra for given cosmology parameters and redshift bins.

    Parameters:
    -----------
    params_dict : dict
        Dictionary containing cosmology parameters including A_IA and delta_z
    z_bins : tuple
        Tuple of RedshiftBin objects
        
    Returns:
    --------
    tuple
        (cl_ee, cl_ne, cl_nn) power spectra
    """
    cosmo = create_cosmo(params_dict)
    ell = np.arange(2, 2000)
    ia_all = params_dict.get('A_IA', None)
    delta_z_all = params_dict.get('delta_z',None)

    bin1, bin2 = z_bins

    if ia_all is not None:
        ia_bias = (ia_all[bin1.nbin - 1], ia_all[bin2.nbin - 1])
    else:
        ia_bias = (0.0, 0.0)

    if delta_z_all is not None:
        z_bias = (delta_z_all[bin1.nbin - 1], delta_z_all[bin2.nbin - 1])
    else:
        z_bias = (0.0, 0.0)
    
    cl = compute_angular_power_spectra(cosmo, ell, z_bins, ia_bias=ia_bias, z_bias=z_bias)
    return cl



def clnames(s8):
    s8str = str(s8)
    s8str = s8str.replace(".", "p").lstrip("0")
    s8name = "S8" + s8str
    clpath = "Cl_3x2pt_kids55_s8{}.txt".format(s8str)
    return clpath, s8name

def save_cl(cl, clpath):
    cl_ee, cl_ne, cl_nn = cl
    # np.savez(clpath, theory_ellmin=2,ee=cl_ee,ne=cl_ne,nn=cl_nn)
    np.savetxt(clpath, (cl_ee, cl_ne, cl_nn), header="EE, nE, nn")





def prepare_theory_cl_inputs(redshift_bins, noise='default'):
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

def generate_theory_cl(lmax, redshift_bin_combinations, shot_noise, cosmo,nside=None):
    """
    Generate TheoryCl instances for given redshift bins and cosmology.

    Parameters:
    - lmax: Maximum multipole moment.
    - redshift_bins: List of redshift bins.
    - noise: List of noise values for each redshift bin combination.
    - cosmo: Cosmology dictionary.
    - nside: needed when pixel window function is applied.

    Returns:
    - List of TheoryCl instances.
    """
       
    # Generate TheoryCl instances
    theory_cl = [
        TheoryCl(lmax, cosmo=cosmo, z_bins=bin_comb, sigma_e=noise_val,nside=nside)
        for bin_comb, noise_val in zip(redshift_bin_combinations, shot_noise)
    ]
    return theory_cl


def n_combs_to_n_bins(n_comb):
    """
    Convert number of unique bin combinations to number of redshift bins.

    Parameters:
    - n_comb: Number of unique bin combinations.

    Returns:
    - n_bins: Number of redshift bins.
    """
    n_bins = int((np.sqrt(8 * n_comb + 1) - 1) / 2)
    return n_bins

def add_n_of_z_bias(z,nz,bias):
    f = interp1d(z, nz, fill_value=0.0, bounds_error=False)
    nz_biased = f(z - bias)
    
    
        
    nz_biased /= np.trapz(nz_biased, z)
    return nz_biased