from grf_classes import SphereMask, TheoryCl, RedshiftBin
from cov_setup import Cov
import calc_pdf, helper_funcs, setup_m
import os
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import copula_funcs
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
import postprocess_nd_likelihood
from scipy.stats import norm, multivariate_normal


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
        self._redshift_bins = redshift_bins
        self._n_redshift_bins = len(self._redshift_bins)
        self._numerical_redshift_bin_combinations = calc_pdf.generate_combinations(
            self._n_redshift_bins
        )
        self._n_redshift_bin_combs = len(self._numerical_redshift_bin_combinations)

        self._is_cov_cross = (
            self._numerical_redshift_bin_combinations[:, 0]
            != self._numerical_redshift_bin_combinations[:, 1]
        )

        self.redshift_bin_combinations = [
            (self._redshift_bins[comb[0]], self._redshift_bins[comb[1]])
            for comb in self._numerical_redshift_bin_combinations
        ]
        self._shot_noise = None  # shot noise for each redshift bin
        if exact_lmax is None:
            self._exact_lmax = mask.exact_lmax
        else:
            self._exact_lmax = exact_lmax
            if not helper_funcs.check_property_equal([self, self.mask], "_exact_lmax"):
                raise RuntimeError("Exact lmax does not align for mask and likelihood.")
        self._highell = False

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

    def precompute_combination_matrices(self):
        prefactors = helper_funcs.prep_prefactors(
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

    def initiate_theory_cl(self, paths, names, noises):
        # get the theory Cl for the given cosmology
        # this is the part that needs to be changed for different likelihoods
        # should return a list of theoryCl instances
        # redshift bin combinations is a list of two tuples of reshift bins, each redshift bin is a tuple z,nz
        """theory_cl = [
            TheoryCl(cosmology, redshift_bin_combination)
            for redshift_bin_combination in self.redshift_bin_combinations
        ]"""
        theory_cl = [
            TheoryCl(self.lmax, path, noise, clname=name)
            for path, name, noise in zip(paths, names, noises)
        ]
        self._theory_cl = theory_cl
        return self._theory_cl

    def _get_pseudo_alm_covariances(self):
        # get the pseudo alm covariances for the given mask
        # all redshift bin combinations, organized as in GLASS
        #

        pseudo_alm_covs = [
            Cov(self.mask, theory_cl, self._exact_lmax).cov_alm_xi()
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
        einsum_means = np.einsum("cbll->cb", self._products.copy())
        self._means_lowell = calc_pdf.mean_xi_gaussian_nD(
            self._prefactors, self._theory_cl, self.mask, lmin=0, lmax=self._exact_lmax
        )
        diff = einsum_means - self._means_lowell
        assert np.all(np.abs(diff) < 1e-10), ("Means do not match", einsum_means, self._means_lowell)
        # mean for each redshift bin combination and angular bin, shape (n_redshift_bin_combs, len(angular_bins))
        # var = 2 * np.sum(prod * np.transpose(prod))

    def _get_cfs_1d_lowell(self):
        # get the marginals for the full data vector
        # use products of combination matrix and pseudo alm covariance
        self._variances = np.zeros((self._n_redshift_bin_combs, len(self._ang_bins_in_deg)))
        self._eigvals = np.zeros(
            (self._n_redshift_bin_combs, len(self._ang_bins_in_deg), 2 * len(self._products[0, 0])),
            dtype=complex,
        )
        products = self._products.copy()
        cross_prods = products[self._is_cov_cross]
        auto_prods = products[~self._is_cov_cross]
        auto_transposes = np.transpose(auto_prods, (0, 1, 3, 2))
        self._variances[~self._is_cov_cross] = 2 * np.sum(
            auto_prods * auto_transposes, axis=(-2, -1)
        )
        cross_transposes = np.transpose(cross_prods, (0, 1, 3, 2))
        cross_combs = self._numerical_redshift_bin_combinations[self._is_cov_cross]
        auto_normal, auto_transposed = cross_combs[:, 0], cross_combs[:, 1]
        self._variances[self._is_cov_cross] = np.sum(
            cross_prods * cross_transposes, axis=(-2, -1)
        ) + np.sum(auto_prods[auto_normal] * auto_transposes[auto_transposed], axis=(-2, -1))
        # no factor 2 because of the cross terms
        self._ximax = self._means_lowell + 10 * np.sqrt(self._variances)
        self._ximin = self._means_lowell - 5 * np.sqrt(self._variances)
        eigvals_auto = np.linalg.eigvals(
            auto_prods
        )  # shape (n_redshift_bins, len(angular_bins),len(cov))
        eigvals_auto_padded = np.pad(
            eigvals_auto, ((0, 0), (0, 0), (0, eigvals_auto.shape[-1])), "constant"
        )
        self._eigvals[~self._is_cov_cross] = eigvals_auto_padded
        cross_matrices = []
        for c, comb in enumerate(cross_combs):
            diag_elem = cross_prods[c]  # shape (len(angular_bins),len(cov),len(cov))
            off_diag_elem_1 = auto_prods[comb[0]]
            off_diag_elem_2 = auto_prods[comb[1]]
            mat = 0.5 * np.block(
                [[diag_elem, off_diag_elem_2], [off_diag_elem_1, diag_elem]]
            )  # shape (len(angular_bins),2*len(cov), 2*len(cov))
            cross_matrices.append(mat)
        self._cross_matrices = np.array(
            cross_matrices
        )  # shape (n_redshift_bin_cross_combs, len(angular_bins), 2*len(cov), 2*len(cov))
        eigvals_cross = np.linalg.eigvals(self._cross_matrices)
        self._eigvals[self._is_cov_cross] = eigvals_cross
        self._t_lowell, self._cfs_lowell = calc_pdf.batched_cf_1d(
            self._eigvals, self._ximax, steps=4096
        )
        # shape (n_redshift_bin_cross_combs, len(angular_bins), 2*len(cov))
        # also introduce ximin? need symmetry in t?

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
                            sorted = [np.sort(comb)[::-1] for comb in all_combs]
                            cov_pos = [calc_pdf.get_cov_n(comb) for comb in sorted]
                            # it does not make sense that idx is the same in both products here. This is inconsitent with the variances.
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
        self._cov_highell = calc_pdf.cov_xi_gaussian_nD(
            self._theory_cl,
            self._numerical_redshift_bin_combinations,
            self._ang_bins_in_deg,
            self._eff_area,
            lmin=self._exact_lmax + 1,
            lmax=self.lmax,
        )

    def _get_means_highell(self):
        # get the mean for the full data vector Gaussian part
        self._means_highell = calc_pdf.mean_xi_gaussian_nD(
            self._prefactors, self._theory_cl, self.mask, lmin=self._exact_lmax + 1, lmax=self.lmax
        )

    def _get_cfs_1d_highell(self):
        # get the Gaussian cf for the high ell part
        vars = np.diag(self._cov_highell)
        vars = vars.reshape((self._n_redshift_bin_combs, len(self._ang_bins_in_deg)))

        return calc_pdf.high_ell_gaussian_cf_1d(self._t_lowell, self._means_highell, vars)

    @property
    def marginals(self):
        # get the marginal pdfs and potentially cdfs
        self._get_cfs_1d_lowell()

        if self._highell:
            self.get_covariance_matrix_highell()
            self._get_means_highell()
            self._cfs = self._cfs_lowell * self._get_cfs_1d_highell()
        else:
            self._cfs = self._cfs_lowell

        self._marginals = calc_pdf.cf_to_pdf_1d(self._t_lowell, self._cfs)
        return self._marginals

    def gauss_compare(self):
        mean = self._mean.flatten()[1:]
        mvn = multivariate_normal(mean=mean, cov=self._cov[1:, 1:])
        print('Calculated mean and cov: ')
        print(mean, self._cov[1:, 1:])
        return mvn

    def likelihood(self, data, cosmology, highell=True):
        # compute the likelihood for a given cosmology
        # don't forget to build in the factor 1/2 for the cross terms where always two m-cov products are used
        if highell:
            self._highell = True
        cl_paths, names, noises = cosmology
        self.initiate_theory_cl(cl_paths, names, noises)
        self._prepare_matrix_products()
        xs, pdfs = self.marginals

        self._cov = self.get_covariance_matrix_lowell()
        self._mean = self._means_lowell
        highell_moms = None
        assert (
            np.fabs(np.diag(self._cov_lowell) - self._variances).all() < 1e-10
        ), "Variances do not match"
        if self._highell:
            self._cov = self._cov_lowell + self._cov_highell
            self._mean = self._means_lowell + self._means_highell
            highell_moms = [self._means_highell[1:], self._cov_highell[1:, 1:]]
        self._cdfs, self._pdfs, self._xs = copula_funcs.pdf_to_cdf(xs, pdfs) # new xs and pdfs are interpolated
        
        
        copula = copula_funcs.joint_pdf(
        self._cdfs[1:],
        self._pdfs[1:],
        self._cov[1:,1:],
        )
        
        
        
        
        fig, ((ax00, ax01, ax02), (ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(
            3, 3, gridspec_kw=dict(width_ratios=[1, 1, 1]), figsize=(11, 11)
        )
        #bincenters, mean, errors, mu_estimate, cov_estimate
        configpath = "config_adjusted.ini"
        simspath = "/cluster/work/refregier/veoehl/xi_sims/croco_3x2pt_kids_33_circ10000smoothl30_noisedefault_llim_30_newwpm/"
        config = postprocess_nd_likelihood.load_config(configpath)
        
        
        
        diag_fig,diag_ax = plt.subplots()
        sims_lmax = self.lmax if highell else self._exact_lmax
        bincenters, mean, errors, mu_estimate, cov_estimate = (
            postprocess_nd_likelihood.load_and_bootstrap_sims_nd(
                config, simspath, sims_lmax,axes=(ax00, ax1, ax3), vmax=None,n_bootstrap=1000,diagnostic_ax=diag_ax
            )
        )
        x_vals = self._xs[1, 0]
        y_vals = self._xs[2, 0]
        diag_ax.plot(x_vals,self._pdfs[1, 0],label='xi55_analytic')
        diag_ax.plot(y_vals,self._pdfs[2, 0],label='xi53_analytic')
        diag_ax.legend()
        diag_fig.savefig('marginal_diagnostics_10000sqd_lowell_newwpm.png')

     
        
        
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        test_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
        
        # x_exact, pdf_exact = postprocess_nd_likelihood.convert_nd_cf_to_pdf(config,highell_moms=highell_moms)
        vmax = np.max(copula)
        copula_grid = copula.reshape(x_grid.shape).T
        interp = RegularGridInterpolator((x_vals[1:-1], y_vals[1:-1]), copula_grid[1:-1,1:-1], method="cubic")
        # interp_exact = RegularGridInterpolator((x_exact[:,0,0],x_exact[0,:,1]),pdf_exact,method='cubic')
        # marginals_exact = postprocess_nd_likelihood.get_marginal_likelihoods([x_exact[:,0,0],x_exact[0,:,1]],pdf_exact)
        # marginals_copula = postprocess_nd_likelihood.get_marginal_likelihoods([x_vals,y_vals],copula_grid)

        
        

        # grid_z_copula = griddata(test_points, copula, (x_grid, y_grid), method="cubic")
        gauss = self.gauss_compare().pdf(test_points)
        gauss_est = multivariate_normal(mean=mu_estimate, cov=cov_estimate)
        gauss_est = gauss_est.pdf(test_points)
        gauss_grid = gauss_est.reshape(x_grid.shape).T
        interp_gauss = RegularGridInterpolator((x_vals, y_vals), gauss_grid, method="cubic")
        (ax1, ax2, ax5), res_plot = postprocess_nd_likelihood.compare_to_sims_2d(
            [ax1, ax2, ax5], bincenters, mean, errors, interp, vmax
        )
        (ax3, ax4, ax6), gauss_res = postprocess_nd_likelihood.compare_to_sims_2d(
            [ax3, ax4, ax6], bincenters, mean, errors, interp_gauss, vmax
        )
        # (ax00,ax01,ax02), exact_res = postprocess_nd_likelihood.compare_to_sims_2d([ax00,ax01,ax02],bincenters,mean,errors,interp_exact,vmax)

        # fig, ax4 = plt.subplots()
        # c2 = ax4.contourf(x_grid, y_grid, grid_z_copula, levels=100, vmax=np.max(grid_z_copula))
        # ax4.set_title("Copula")

        fig.colorbar(res_plot, ax=ax5)
        fig.colorbar(gauss_res, ax=ax6)
        # fig.colorbar(exact_res, ax=ax02)
        fig.savefig("comparison_copula_sims_10000deg2_lowell_newwpm.png")

    # copula.evaluate(self._marginals, data)
    # pass


paths = ["Cl_3x2pt_kids33.txt", "Cl_3x2pt_kids55.txt", "Cl_3x2pt_kids53.txt"]
names = ["3x2pt_kids_33", "3x2pt_kids_55", "3x2pt_kids_53"]
noises = ["default", "default", None]
mask = SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=30, l_smooth=30)
z = np.linspace(0, 2, 100)
nz = scipy.stats.norm.pdf(z, loc=1, scale=0.5)
redshift_bins = [RedshiftBin(z, nz, 3), RedshiftBin(z, nz, 5)]
ang_bins_in_deg = [(4, 6)]
xi_likelihood = XiLikelihood(mask, redshift_bins, ang_bins_in_deg=ang_bins_in_deg, exact_lmax=30)

xi_likelihood.initiate_mask_specific()
xi_likelihood.precompute_combination_matrices()
xi_likelihood.likelihood(None, (paths, names, noises), highell=False)
