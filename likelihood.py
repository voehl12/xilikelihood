from theory_cl import prepare_theory_cl_inputs, generate_theory_cl, RedshiftBin
from pseudo_alm_cov import Cov
import characteristic_functions
import cl2xi_transforms
import os, re

#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import copula_funcs
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import norm, multivariate_normal
import jax.numpy as jnp
import jax
import logging
#print(jax.devices())
from multiprocessing import Pool
import gc
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("likelihood.log"),  # Log to a file
        logging.StreamHandler()  # Also log to the console
    ]
)
logger = logging.getLogger(__name__)


class XiLikelihood:
    # should in the end have a function that takes in cosmological parameters and a datapoint and returns the likelihood
    # initiate by computing mask specific quantities that can also be stored
    # should remove the inheritance of Cov from TheoryCl, then this class can inherit from Cov

    def __init__(
        self,
        mask,
        redshift_bins,
        ang_bins_in_deg,
        exact_lmax=None,
        lmax=None,
        noise='default',
        working_dir=None,
    ):
        if working_dir is None:
            working_dir = os.getcwd()
        self.working_dir = working_dir

        self.mask = mask
        if lmax is None:
            self.lmax = 3 * self.mask.nside - 1
        else:
            self.lmax = lmax
        # decide whether to get these from a config file or to set them on initialization
        self._ang_bins_in_deg = ang_bins_in_deg
        self._n_ang_bins = len(ang_bins_in_deg)  # Make n_ang_bins private
        self._redshift_bins = redshift_bins
        self._n_redshift_bins = len(self._redshift_bins)
        (
            self._numerical_redshift_bin_combinations,
            self.redshift_bin_combinations,
            self._is_cov_cross,
            self._shot_noise,
            self._n_to_bin_comb_mapper,
        ) = prepare_theory_cl_inputs(redshift_bins, noise)

        self._n_redshift_bin_combs = len(self._numerical_redshift_bin_combinations)

        
        self.noise = noise
        if exact_lmax is None:
            self._exact_lmax = mask.exact_lmax
        else:
            self._exact_lmax = exact_lmax
            if not file_handling.check_property_equal([self, self.mask], "_exact_lmax"):
                raise RuntimeError("Exact lmax does not align for mask and likelihood.")
        self._highell = False
        self.gaussian_covariance = None

    @property
    def ang_bins_in_deg(self):
        return self._ang_bins_in_deg

    @property
    def redshift_bins(self):
        return self._redshift_bins

    @property
    def n_redshift_bins(self):
        return self._n_redshift_bins

    @property
    def shot_noise(self):
        return self._shot_noise

    def initiate_mask_specific(self):
        # initiate the mask specific quantities

        self.mask.precompute_for_cov_masked(cov_ell_buffer=10)
        self._eff_area = self.mask.eff_area

    def prep_data_array(self):
        return np.zeros((self._n_redshift_bin_combs, len(self._ang_bins_in_deg)))

    def check_pdfs(self):
        """
        Wrapper for PDF validation. Delegates the actual checks to a utility function.
        """
        copula_funcs.validate_pdfs(self._pdfs, self._xs, self._cdfs)

    def precompute_combination_matrices(self):
        prefactors = cl2xi_transforms.prep_prefactors(
            self._ang_bins_in_deg, self.mask.wl, self.lmax, self.lmax
        )
        self._prefactors = prefactors
        len_sub_m = self._exact_lmax + 1
        m = 2 * np.repeat(prefactors[:, :, : self._exact_lmax + 1], len_sub_m, axis=2)
        m[:, :, ::len_sub_m] *= 0.5
        self._m_1d = np.tile(m, (1, 1, 4))
        # prefactors is a 3D array with shape (len(angles), 2, out_lmax)
        # we need all 2dim M to capture all covariances for the copula
        # it might make sense to work with the prefactors alone though
        # because the matrix multiplicatinons with the covariance matrix will be simplified
        """ m = setup_m.m_xi_cross(
            (prefactors[0, :, : self._exact_lmax + 1],), combs=((0, 0), (1, 0)), kind=("p",)
        )[0] """
        # Ms for 2d likelihoods, all angular bin combinations

    def initiate_theory_cl(self, cosmo):
        # get the theory Cl for the given cosmology
        self._theory_cl = generate_theory_cl(self.lmax, self.redshift_bin_combinations, self._shot_noise, cosmo)
    
        return self._theory_cl

    def _get_pseudo_alm_covariances(self):
        # get the pseudo alm covariances for the given mask
        # all redshift bin combinations, organized as in GLASS
        # this loop could be mpi parallelized

        pseudo_alm_covs = [
            Cov(self.mask, theory_cl, self._exact_lmax).cov_alm_xi(ischain=True)
            for theory_cl in self._theory_cl
        ]
        return pseudo_alm_covs

    def _prepare_matrix_products(self):
        # prepare the matrix products for moments and marginals
        # put together every combination of two different redshift bins n_redshift over 2 -> 10 combinations
        covs = self._get_pseudo_alm_covariances()
        m_diags_xiplus = self._m_1d[:, 0]
        self._products = np.array(
            [m_diags_xiplus[:, :, None] * covs[i] for i in range(len(covs))]
        )  # all angular bins eacn, all covariances
        # products is a 4D array with shape (n_redshift_bin_combs, len(angular_bins), len(cov), len(cov))
        # these means are not right yet, need to take care of the cross terms
        logger.info("Calculating 1D means...")
        einsum_means = np.einsum("cbll->cb", self._products.copy())
        self.pseudo_cl = cl2xi_transforms.cl2pseudocl(self.mask.m_llp, self._theory_cl)
        self._means_lowell = characteristic_functions.mean_xi_gaussian_nD(
            self._prefactors, self.pseudo_cl, lmin=0, lmax=self._exact_lmax
        )
        diff = einsum_means - self._means_lowell
      
        if np.any(np.abs(diff) > 1e-10):
            logger.error("Means do not match: %s, %s", einsum_means, self._means_lowell)
            # raise RuntimeError("Means do not match")
        
        del covs, einsum_means, diff
        #gc.collect()
        # mean for each redshift bin combination and angular bin, shape (n_redshift_bin_combs, len(angular_bins))
        # var = 2 * np.sum(prod * np.transpose(prod))

    def _compute_variances(self, auto_prods, cross_prods, cross_combs):
        logger.info("Computing variances...")
        auto_transposes = np.transpose(auto_prods, (0, 1, 3, 2))
        variances = np.zeros((self._n_redshift_bin_combs, len(self._ang_bins_in_deg)))

        # Auto terms
        variances[~self._is_cov_cross] = 2 * np.sum(
            auto_prods * auto_transposes, axis=(-2, -1)
        )

        # Cross terms
        cross_transposes = np.transpose(cross_prods, (0, 1, 3, 2))
        auto_normal, auto_transposed = cross_combs[:, 0], cross_combs[:, 1]
        variances[self._is_cov_cross] = np.sum(
            cross_prods * cross_transposes, axis=(-2, -1)
        ) + np.sum(
            auto_prods[auto_normal] * auto_transposes[auto_transposed], axis=(-2, -1)
        )

        return variances
    
    def _compute_auto_eigenvalues(self, auto_prods):
        logger.info("Computing auto eigenvalues...")
        # shape (n_redshift_bins, len(angular_bins),len(cov))
        auto_prods = jnp.array(auto_prods)
        eigvals_auto, _ = jnp.linalg.eig(auto_prods)
        eigvals_auto_padded = jnp.pad(
            eigvals_auto, ((0, 0), (0, 0), (0, eigvals_auto.shape[-1])), "constant"
        )
        return eigvals_auto_padded

    def _compute_cross_eigenvalues(self, cross_prods, auto_prods, cross_combs):
        logger.info("Computing cross eigenvalues...")
        # shape (n_redshift_bin_cross_combs, len(angular_bins), 2*len(cov), 2*len(cov))
        cross_eigvals = []
        for c, comb in enumerate(cross_combs):
            diag_elem = cross_prods[c]
            off_diag_elem_1 = auto_prods[comb[0]]
            off_diag_elem_2 = auto_prods[comb[1]]
            mat = 0.5 * np.block(
                [[diag_elem, off_diag_elem_2], [off_diag_elem_1, diag_elem]]
            )
            eigvals = np.linalg.eigvals(mat)
            cross_eigvals.append(eigvals)
        return jnp.array(cross_eigvals)
    
    def _compute_cfs(self):
        logger.info("Computing CDFs...")
        t_lowell, cfs_lowell = characteristic_functions.batched_cf_1d_jitted(
            self._eigvals, self._ximax, steps=4096
        )
        return np.array(t_lowell), np.array(cfs_lowell)

    def _get_cfs_1d_lowell(self):
        logger.info("Starting _get_cfs_1d_lowell...")
        self._variances = np.zeros((self._n_redshift_bin_combs, len(self._ang_bins_in_deg)))
        self._eigvals = jnp.zeros(
            (self._n_redshift_bin_combs, len(self._ang_bins_in_deg), 2 * len(self._products[0, 0])),
            dtype=complex,
        )
        products = self._products.view()
        products.flags.writeable = False
        cross_prods = products[self._is_cov_cross]
        auto_prods = products[~self._is_cov_cross]
        del products

        # Compute variances
        cross_combs = self._numerical_redshift_bin_combinations[self._is_cov_cross]
        self._variances = self._compute_variances(auto_prods, cross_prods, cross_combs)

        # Compute eigenvalues
        self._ximax = jnp.array(self._means_lowell + 40 * jnp.sqrt(self._variances))
        self._ximin = self._means_lowell - 5 * jnp.sqrt(self._variances)
        eigvals_auto_padded = self._compute_auto_eigenvalues(auto_prods)
        self._eigvals = self._eigvals.at[~self._is_cov_cross].set(eigvals_auto_padded)

        if len(cross_combs) > 0:
            cross_eigvals = self._compute_cross_eigenvalues(cross_prods, auto_prods, cross_combs)
            self._eigvals = self._eigvals.at[self._is_cov_cross].set(cross_eigvals)

        # Compute CFs
        self._t_lowell, self._cfs_lowell = self._compute_cfs()

        # Cleanup
        del auto_prods, cross_prods, cross_combs
        gc.collect()
    
    
    def get_covariance_matrix_lowell(self):
        # get the covariance matrix for the full data vector exact part
        # use products of combination matrix and pseudo alm covariance
        cov_length = self._n_redshift_bin_combs * len(self._ang_bins_in_deg)
        self._cov_lowell = np.full((cov_length, cov_length), np.nan)

        i = 0
        products = self._products.copy()
        for rcomb1 in self._numerical_redshift_bin_combinations:
            for k in range(len(self._ang_bins_in_deg)):
                j = 0
                for rcomb2 in self._numerical_redshift_bin_combinations:
                    for l in range(len(self._ang_bins_in_deg)):
                        if i <= j:
                            all_combs = [
                                (rcomb1[0], rcomb2[0]),
                                (rcomb1[1], rcomb2[1]),
                                (rcomb1[1], rcomb2[0]),
                                (rcomb1[0], rcomb2[1]),
                            ]
                            sorted = [tuple(np.sort(comb)[::-1]) for comb in all_combs]
                            # fix with combination mapper:
                            cov_pos = [self._n_to_bin_comb_mapper.get_index(comb) for comb in sorted]
                            # make this into a jax function? is quite slow...
                            self._cov_lowell[i, j] = np.sum(
                                [
                                    np.sum(
                                        products[cov_pos[idx], k] * products[cov_pos[idx + 1], l].T
                                    )
                                    for idx in [0, 2]
                                ]
                            )
                        j += 1
                i += 1
            # xi_cov = np.where(np.isnan(xi_cov), xi_cov.T, xi_cov)
        self._cov_lowell = np.where(
            np.isnan(self._cov_lowell), self._cov_lowell.T, self._cov_lowell
        )
        return self._cov_lowell

    def get_covariance_matrix_highell(self):
        # get the covariance matrix for the full data vector Gaussian part
        # use C_ell approximation
        self._cov_highell = characteristic_functions.cov_xi_gaussian_nD(
            self._theory_cl,
            self._numerical_redshift_bin_combinations,
            self._ang_bins_in_deg,
            self._eff_area,
            lmin=self._exact_lmax + 1,
            lmax=self.lmax,
        )

    def _get_means_highell(self):
        # get the mean for the full data vector Gaussian part
        self._means_highell = characteristic_functions.mean_xi_gaussian_nD(
            self._prefactors, self.pseudo_cl, lmin=self._exact_lmax + 1, lmax=self.lmax
        )

    def _get_cfs_1d_highell(self):
        # get the Gaussian cf for the high ell part
        vars = np.diag(self._cov_highell)
        vars = vars.reshape((self._n_redshift_bin_combs, len(self._ang_bins_in_deg)))

        return characteristic_functions.high_ell_gaussian_cf_1d(self._t_lowell, self._means_highell, vars)

    def marginals(self):
        # get the marginal pdfs and potentially cdfs
        self._get_cfs_1d_lowell()

        if self._highell:
            print("Adding high-ell contribution to CFs...")
            self._cfs = self._cfs_lowell * self._get_cfs_1d_highell()
        else:
            self._cfs = self._cfs_lowell
        # need to build in a test here checking that the boundaries of the pdfs are converged to zero
        self._marginals = characteristic_functions.cf_to_pdf_1d(self._t_lowell, self._cfs)
        return self._marginals

    def gauss_compare(self, data, data_subset=None):
        # should always use a fixed covariance, produce on initialization?
        mean = self._mean
        if self.gaussian_covariance is None:
            logger.warning("Using cosmology-dependent covariance for Gaussian likelihood!")
            cov = self._cov
        else:
            cov = self.gaussian_covariance
        if data_subset is not None:
            data = copula_funcs.data_subset(data, data_subset)
            cov = copula_funcs.cov_subset(cov, data_subset, self._n_ang_bins)  # Use _n_ang_bins
            mean = copula_funcs.data_subset(mean, data_subset)

        mean = mean.flatten()
        mvn = multivariate_normal(mean=mean, cov=cov)
        data_flat = data.flatten()
        return mvn.logpdf(data_flat)
    
    def _prepare_likelihood_components(self, cosmology, highell=True):
        """
        Prepare the components needed for likelihood computation.

        Parameters
        ----------
        cosmology : dict
            Cosmological parameters.
        highell : bool, optional
            Whether to include high-ell contributions, by default True.

        Returns
        -------
        tuple
            xs, pdfs, cdfs, cov, mean
        """
        if highell:
            self._highell = True
        else:  # Highell is False
            self._highell = False

        self.initiate_theory_cl(cosmology)
        self._prepare_matrix_products()

        self._cov = self.get_covariance_matrix_lowell()
        self._mean = self._means_lowell

        if self._highell:
            self.get_covariance_matrix_highell()
            self._get_means_highell()
            self._cov = self._cov_lowell + self._cov_highell
            self._mean = self._means_lowell + self._means_highell

        xs, pdfs = self.marginals()
        self._cdfs, self._pdfs, self._xs = copula_funcs.pdf_to_cdf(xs, pdfs)  # Interpolated xs and pdfs
        self.check_pdfs()


    def loglikelihood(self, data, cosmology, highell=True, gausscompare=False, data_subset=None):
        """
        Compute the log-likelihood for a given cosmology and data point.

        Parameters
        ----------
        data : ndarray
            Data point to evaluate the likelihood at.
        cosmology : dict
            Cosmological parameters.
        highell : bool, optional
            Whether to include high-ell contributions, by default True.
        gausscompare : bool, optional
            Whether to compare with Gaussian likelihood, by default False.
        data_subset : list, optional
            Subset of data to evaluate, by default None.

        Returns
        -------
        float or tuple
            Log-likelihood value, optionally with Gaussian comparison.
        """
        self._prepare_likelihood_components(cosmology, highell)

        # Quick check to ensure data and self._xs have matching first two dimensions
        if data.shape[:2] != self._xs.shape[:2]:
            raise ValueError("Mismatch in dimensions: data and self._xs must have the same first two dimensions. But got: {} and {}".format(data.shape, self._xs.shape))

        likelihood = copula_funcs.evaluate(data, self._xs, self._pdfs, self._cdfs, self._cov, subset=data_subset)

        if gausscompare:
            return likelihood, self.gauss_compare(data, data_subset=data_subset)
        else:
            return likelihood
    
    def likelihood_function(self, cosmology, highell=True):
        """
        Return the likelihood as a joint PDF for the entire data space at a given cosmology.

        Parameters
        ----------
        cosmology : dict
            Cosmological parameters.
        highell : bool, optional
            Whether to include high-ell contributions, by default True.

        Returns
        -------
        ndarray
            Joint PDF values for the entire data space.
        """
        # Prepare the likelihood components
        self._prepare_likelihood_components(cosmology, highell)

        # Compute the joint PDF using copula_funcs.joint_pdf
        joint_pdf_values = copula_funcs.joint_pdf(self._cdfs, self._pdfs, self._cov)

        return self._xs, joint_pdf_values

    def gaussian_2d(self,xs,mean,cov):
        mean_flat = mean.flatten()
        n_dim = mean_flat.shape[0]
        n_points_per_dim = xs.shape[-1]
        xs_flat = xs.reshape(-1, n_points_per_dim)

        # Create a meshgrid for the xs_flat
        x_grid = np.meshgrid(*xs_flat)
        x_points = np.stack(x_grid, axis=-1)
        x_points = x_points.reshape(-1, x_points.shape[-1])

        # Compute Gaussian PDF on the meshgrid
        gaussian_pdf = multivariate_normal.logpdf(x_points, mean=mean_flat, cov=cov)
        shape = (n_points_per_dim,) * n_dim
        # Reshape Gaussian PDF to match the 2D subset
        gaussian_pdf_reshaped = gaussian_pdf.reshape(shape)
        return gaussian_pdf_reshaped


    def likelihood_function_2d(self, data_subset=None,gausscompare=False):
        """
        Return a 2D subset of the likelihood as a joint PDF at a given cosmology.

        Parameters
        ----------
        cosmology : dict
            Cosmological parameters.
        
        data_subset : list
            Subset of data to evaluate (e.g., 2D indices).

        Returns
        -------
        tuple
            xs and joint PDF values for the specified 2D subset.
        """
        if data_subset is None:
            raise ValueError("data_subset must be specified for likelihood_function_2d.")
        
        # Ensure data_subset selects exactly 2 data points
        if len(data_subset) != 2:
            raise ValueError("data_subset must select exactly 2 data dimensions.")

        #self._prepare_likelihood_components(cosmology, highell)
        
        cov_2d = copula_funcs.cov_subset(self._cov, data_subset, self._n_ang_bins)  # Use _n_ang_bins
        xs_subset = copula_funcs.data_subset(self._xs, data_subset)
        pdfs_subset = copula_funcs.data_subset(self._pdfs, data_subset)
        cdfs_subset = copula_funcs.data_subset(self._cdfs, data_subset)
        log_pdf_2d = copula_funcs.joint_pdf(cdfs_subset, pdfs_subset, cov_2d,copula_type='student-t',df=100)
        
        
        if gausscompare:
            means_subset = copula_funcs.data_subset(self._mean, data_subset)
            gaussian_log_pdf = self.gaussian_2d(xs_subset,means_subset,cov_2d)
            return xs_subset, log_pdf_2d, gaussian_log_pdf
        else:
            return xs_subset, log_pdf_2d


def fiducial_dataspace():
    rs_directory = "redshift_bins/KiDS/"
    redshift_filepaths = os.listdir(rs_directory)
    pattern = re.compile(r"TOMO(\d+)")
    nbins = [int(pattern.search(f).group(1)) for f in redshift_filepaths]
    redshift_bins = [
        RedshiftBin(nbin=i, filepath=rs_directory + f) for i, f in zip(nbins, redshift_filepaths)
    ]
    redshift_bins_sorted = sorted(redshift_bins, key=lambda x: x.nbin)

    # Define the initial log-spaced array between 0.5 and 300 arcminutes
    initial_ang_bins_in_arcmin = np.logspace(np.log10(0.5), np.log10(300), num=9, endpoint=True)
    # Filter out bins smaller than 15 arcminutes
    filtered_ang_bins_in_arcmin = initial_ang_bins_in_arcmin[initial_ang_bins_in_arcmin >= 15]
    # Add one more bin on the larger side according to the same pattern
    last_bin_ratio = filtered_ang_bins_in_arcmin[-1] / filtered_ang_bins_in_arcmin[-2]
    new_bin = filtered_ang_bins_in_arcmin[-1] * last_bin_ratio
    extended_ang_bins_in_arcmin = np.append(filtered_ang_bins_in_arcmin, new_bin)
    # Convert to degrees
    ang_bins_in_deg = extended_ang_bins_in_arcmin / 60

    # Create tuples representing the bin edges
    ang_bins_in_deg = [
        (ang_bins_in_deg[i], ang_bins_in_deg[i + 1]) for i in range(len(ang_bins_in_deg) - 1)
    ]

    return redshift_bins_sorted, ang_bins_in_deg
