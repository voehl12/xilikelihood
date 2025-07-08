import numpy as np
import scipy
import scipy.optimize
from statsmodels.stats.moment_helpers import mnc2cum
from statsmodels.distributions.edgeworth import ExpandedNormal
from scipy.special import kv, gamma
import distributions
import itertools
import time
import jax
from jax import jit
import jax.numpy as jnp
from jax import config

class GeneralizedLaplace:
    def __init__(self, moments=None, params=None):
        if moments is not None:
            self.moments = moments
            self.ndim = 1 if isinstance(self.moments[0], (int, float)) else len(self.moments[0])
            self.moment_matching()
            self.s, self.mu, self.sigma = self.s_matched, self.mu_matched, self.sigma_matched
        elif params is not None:
            self.params = params
            self.s, self.mu, self.sigma = self.params
            self.ndim = 1 if isinstance(self.mu, (int, float)) else len(self.mu)
        else:
            raise ValueError("Either moments or params must be provided")

    def get_moments(self):
        return self.moments

    def get_pdf(self, x):
        return self.pdf(x)

    def pdf(self, x):

        s, mu, sigma = self.s, self.mu, self.sigma
        self.c_sigma_mu = self.get_c_sigma_mu(mu, sigma)
        if self.ndim == 1:
            q_x = np.fabs(x) / np.sqrt(sigma)
            c_sigma_mu = self.c_sigma_mu
            bessel = kv(s - 1 / 2, c_sigma_mu * q_x)
            prefactor = (
                2
                * (q_x / c_sigma_mu) ** (s - 1 / 2)
                / ((2 * np.pi) ** (1 / 2) * gamma(s) * np.sqrt(sigma))
            )
            exp_part = np.exp(x * mu / sigma)

        else:
            sigma_inv = np.linalg.inv(sigma)
            q_x = np.sqrt(np.einsum("ij,ni,nj->n", sigma_inv, x, x))
            c_sigma_mu = self.c_sigma_mu
            bessel = kv(s - self.ndim / 2, c_sigma_mu * q_x)
            prefactor = (
                2
                * (q_x / c_sigma_mu) ** (s - self.ndim / 2)
                / ((2 * np.pi) ** (self.ndim / 2) * gamma(s) * np.sqrt(np.linalg.det(sigma)))
            )
            exp_part = np.exp(np.einsum("ni,ij,j->n", x, sigma_inv, mu))

        return prefactor * bessel * exp_part

    def sample(self, n):
        gamma = scipy.stats.gamma(a=self.s)
        gamma_samples = gamma.rvs(n)
        if self.ndim == 1:
            normal = scipy.stats.norm(loc=0, scale=np.sqrt(self.sigma))
            normal_samples = normal.rvs(n)
            return np.sqrt(gamma_samples) * normal_samples + self.mu * gamma_samples
        else:
            normal = scipy.stats.multivariate_normal(mean=np.zeros(self.ndim), cov=self.sigma)
            normal_samples = normal.rvs(n)
            return (
                np.sqrt(gamma_samples)[:, None] * normal_samples + self.mu * gamma_samples[:, None]
            )

    def parametrized_moments(self, s, mu, sigma):
        def first_moment():
            return s * mu

        def second_moment():
            if self.ndim == 1:
                seconds = s * ((s + 1) * mu**2 + sigma)
                seconds_central = s * (mu**2 + sigma)
            else:
                seconds = s * (np.outer(mu, mu) + sigma)
                seconds_central = s * (mu**2 + sigma)
            return seconds

        def third_moment():
            # rewrite
            dims = np.arange(self.ndim)
            if self.ndim == 1:
                return s * (s + 1) * (s + 2) * mu**3 + 3 * s * (s + 1) * mu * sigma

            else:
                combs = jnp.array(list(itertools.combinations_with_replacement(dims, 3)))

                i, j, k = combs[:, 0], combs[:, 1], combs[:, 2]

                third_moment_values = (
                    s
                    * (s + 1)
                    * (
                        (s + 2) * np.prod(jnp.array([mu[i], mu[j], mu[k]]), axis=0)
                        + sigma[i, j] * mu[k]
                        + sigma[i, k] * mu[j]
                        + sigma[j, k] * mu[i]
                    )
                )

                return np.ravel(third_moment_values)

        all_moments = [first_moment(), second_moment(), third_moment()]

        return all_moments

    def convert_1d_params(self, params):
        s = params[0]
        mu = params[1 : 1 + self.mu_shape[0]].reshape(self.mu_shape)
        sigma = params[1 + self.mu_shape[0] :].reshape(self.sigma_shape)
        return s, mu, sigma

    def get_c_sigma_mu(self, mu, sigma):
        if self.ndim == 1:
            return np.sqrt(2 + mu**2 / sigma)
        else:
            return np.sqrt(2 + np.einsum("ij,i,j", np.linalg.inv(sigma), mu, mu))

    def moment_matching(self):
        self.mu_shape = self.moments[0].shape
        self.sigma_shape = self.moments[1].shape

        def equations_to_solve_1d(params):
            s, mu, sigma = params
            firsts, seconds, thirds = self.parametrized_moments(s, mu, sigma)
            return np.array(
                [
                    firsts - self.moments[0],
                    seconds - self.moments[1],
                    thirds - self.moments[2],
                ]
            )

        def equations_to_solve(params):
            s, mu, sigma = self.convert_1d_params(params)

            firsts, seconds, thirds = self.parametrized_moments(s, mu, sigma)
            first_diff = firsts - self.moments[0]
            second_diff = np.ravel(seconds - self.moments[1])
            third_diff = np.ravel(thirds - self.moments[2])

            return np.concatenate(
                [
                    first_diff,
                    second_diff,
                    third_diff,
                ]
            )

        def loss_function_1d(params):
            s, mu, sigma = params
            firsts, seconds, thirds = self.parametrized_moments(s, mu, sigma)
            diffs = np.array(
                [
                    firsts - self.moments[0],
                    seconds - self.moments[1],
                    thirds - self.moments[2],
                ]
            )
            weights = np.array(
                [1.0, 10.0, 0.1]
            )  # Weights to penalize first and second moments more
            weighted_diffs = weights * np.array(
                [
                    firsts - self.moments[0],
                    seconds - self.moments[1],
                    thirds - self.moments[2],
                ]
            )
            return np.sum(weighted_diffs**2)

        if self.ndim == 1:
            initial_guess_1d = np.array([3, self.moments[0] / 3, self.moments[1] / 9])
            solution = scipy.optimize.minimize(
                loss_function_1d,
                initial_guess_1d,
                method="Nelder-Mead",
                bounds=[(0.1, 10), (0, np.infty), (1e-15, np.infty)],
            )

        else:
            initial_guess = np.concatenate(
                [
                    np.array([3]),  # Initial guess for s
                    self.moments[0].ravel() / 3,  # Initial guess for mu
                    self.moments[1].ravel() / 9,  # Initial guess for sigma
                ]
            )
            solution = scipy.optimize.least_squares(
                equations_to_solve,
                initial_guess,
                ftol=1e-30,
                method="dogbox",
                bounds=(
                    np.hstack(
                        [
                            1.5,
                            initial_guess[1 : self.ndim + 1].ravel() * 0.5,
                            initial_guess[self.ndim + 1 :].ravel() * 0.5,
                        ]
                    ),
                    np.hstack(
                        [
                            10,
                            initial_guess[1 : self.ndim + 1].ravel() * 2,
                            initial_guess[self.ndim + 1 :].ravel() * 2,
                        ]
                    ),
                ),
            )
        if solution.success:
            self.params = solution.x
        else:
            raise ValueError("Root finding did not converge")
        self.params_1d = solution.x
        self.params = "matched"
        s_matched, mu_matched, sigma_matched = (
            self.convert_1d_params(self.params_1d) if self.ndim > 1 else self.params_1d
        )
        self.s_matched, self.mu_matched, self.sigma_matched = s_matched, mu_matched, sigma_matched
        self.param_moments = self.parametrized_moments(s_matched, mu_matched, sigma_matched)


class MultiNormalExpansion:

    def __init__(self, cumulants) -> None:
        self.cumulants = cumulants

    def normalize_third_cumulant(self):
        covariance = self.cumulants[1]
        inv_cov = np.linalg.inv(covariance)
        normalized_third_cumulant = []
        for cum3, (i, j, k) in zip(
            self.cumulants[2], itertools.combinations_with_replacement(range(len(covariance)), 3)
        ):
            normalized_third_cumulant.append(
                cum3 / np.sqrt(inv_cov[i, i] * inv_cov[j, j] * inv_cov[k, k])
            )

        return np.array(normalized_third_cumulant)

    def hermite_nd(self, x: np.ndarray, order=3):
        # x: point in R^n
        # should be extended to take any number of cumulants (just like the moment conversion function)
        ndim = len(self.cumulants[0])
        dims = np.arange(len(self.cumulants[0]))
        assert len(x[-1]) == ndim, "h_1: x has wrong dimensions"

        lambda_inv = np.linalg.inv(self.cumulants[1])

        def h_1(x):
            """
            Calculate first order hermite polynomials for n-dimensional distribution.

            Parameters
            ----------
            x : np.2darray of size Nxn
                points in R^n

            Returns
            -------
            np.array 2d of size Nxn
                first order hermite polynomials, evaluated at x
            """

            return np.einsum("ij,kj->ki", lambda_inv, x - self.cumulants[0])

        def h_2(x):
            """
            Calculate second order hermite polynomials for n-dimensional distribution.

            Parameters
            ----------
            x : np.2darray of size Nxn
                points in R^n

            Returns
            -------
            np.array 3d of size Nxnxn
                second order hermite polynomials, evaluated at x

            """
            # second order hermite polynomials arranged in a 2d square matrix, evaluated at x
            h1 = h_1(x)
            return np.einsum("ik,il->ikl", h1, h1) - lambda_inv

        def h_3(x):
            """
            Calculate third order hermite polynomials for n-dimensional distribution.

            Parameters
            ----------
            x : np.2darray of size Nxn
                points in R^n

            Returns
            -------
            np.array 2d of size NxK, where K = n(n+1)(n+2)/6
                third order hermite polynomials, evaluated at x

            """
            # third order hermite polynomials arranged in a list according to the order of itertools.combinations_with_replacement
            h_3 = []
            h1 = h_1(x)
            for comb in itertools.combinations_with_replacement(dims, 3):
                i, j, k = comb
                h_ijk = np.prod([h1[:, m] for m in comb], axis=0) - np.sum(
                    [
                        h1[:, one] * lambda_inv[two, three]
                        for (one, two, three) in [(i, j, k), (j, i, k), (k, i, j)]
                    ],
                    axis=0,
                )
                h_3.append(h_ijk)
            return np.array(h_3)

        if order == 1:
            return h_1(x)
        elif order == 2:
            return h_2(x)
        elif order == 3:
            return h_3(x)
        else:
            raise ValueError("order > 3 not implemented")

    def pdf(self, x: np.ndarray):
        assert len(self.cumulants) >= 2
        gaussian = scipy.stats.multivariate_normal(mean=self.cumulants[0], cov=self.cumulants[1])
        if len(self.cumulants) == 2:
            print("No higher order cumulants given, returning Gaussian...")
            extension = 1
        else:
            extension = 1 + np.einsum("ji,j->i", self.hermite_nd(x), self.cumulants[2]) / 6

        return gaussian.pdf(x) * extension


def get_edgeworth_1d(moments):
    cumulants = mnc2cum(moments)
    print(cumulants)
    edgw = ExpandedNormal(cumulants, name="edgw", momtype=0)
    return edgw

