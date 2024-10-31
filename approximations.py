import numpy as np
import scipy
import scipy.optimize
from statsmodels.stats.moment_helpers import mnc2cum
from statsmodels.distributions.edgeworth import ExpandedNormal
from scipy.special import kv, gamma
import calc_pdf
import itertools
import time
import jax
from jax import jit
import jax.numpy as jnp
from jax import config

config.update("jax_enable_x64", True)


def get_moments_1d(m, cov):
    if np.array_equal(m, np.diag(np.diag(m))):
        m_diag = np.diag(m)
        mean = np.dot(m_diag, np.diag(cov))
        prod = m_diag[:, None] * cov
    else:
        prod = m @ cov
        mean = np.trace(prod)

    var = 2 * np.sum(prod * np.transpose(prod))
    second = var + mean**2

    prod2 = prod @ prod
    third = 8 * np.sum(prod2 * np.transpose(prod)) + mean**3 + 3 * mean * var

    moments = [mean, second, third]

    return moments


def moments_nd(m, cov, ndim):
    """
    Calculate first three moments for an n-dimensional likelihood.

    Parameters
    ----------
    m : list of 2d arrays
        list of combination matrices, order defines order of dimensions
    cov : 2d array
        covariance matrix for all pseudo alm needed for all dimensions

    Returns
    -------
    list
        list of lists of first, second and third moments. second moments are given out in the shape of a covariance matrix.
    """
    assert ndim == m.shape[0], "list of matrices m length {} does not align with ndim={}".format(
        m.shape[0], ndim
    )
    if ndim > 100:
        print("Warning: ndim > 100")
    dims = jnp.arange(ndim)

    prods = jnp.einsum("dij,jk->dik", m, cov)

    def first_moments():

        return jnp.einsum("dii->d", prods)

    def second_moments(firsts):
        seconds = jnp.zeros((ndim, ndim), dtype="float64")
        combs = jnp.array(list(itertools.combinations_with_replacement(dims, 2)))

        one, two = combs[:, 0], combs[:, 1]

        second_moment_values = firsts[one] * firsts[two] + 2 * jnp.einsum(
            "dij,dji->d", prods[one], prods[two]
        )
        seconds = seconds.at[one, two].set(second_moment_values)
        seconds = seconds.at[two, one].set(second_moment_values)

        return seconds

    def third_moments(firsts):
        combs = jnp.array(list(itertools.combinations_with_replacement(dims, 3)))

        i, j, k = combs[:, 0], combs[:, 1], combs[:, 2]

        third_moment_values = (
            jnp.prod(jnp.array([firsts[i], firsts[j], firsts[k]]), axis=0)
            + 2
            * (
                jnp.einsum("ijk,ikj->i", prods[j], prods[k]) * firsts[i]
                + jnp.einsum("ijk,ikj->i", prods[k], prods[i]) * firsts[j]
                + jnp.einsum("ijk,ikj->i", prods[i], prods[j]) * firsts[k]
            )
            + 8 * jnp.einsum("dij,djk,dki->d", prods[i], prods[j], prods[k])
        )

        return jnp.ravel(third_moment_values)

    def third_moments_old(firsts, seconds):
        thirds = []
        # define n-dim 3rd order hermite polynomials in exactly the same manner / order!
        for comb in itertools.combinations_with_replacement(dims, 3):
            i, j, k = comb
            prod2 = prods[j] @ prods[k]
            third_moment = (
                np.prod([firsts[m] for m in comb])
                + 2
                * (
                    np.sum(
                        [
                            firsts[one] * np.sum(prods[two] * np.transpose(prods[three]))
                            for (one, two, three) in [(i, j, k), (k, i, j), (j, i, k)]
                        ]
                    )
                )
                + 8 * np.sum(prods[i] * np.transpose(prod2))
            )
            thirds.append(third_moment)
        thirds = np.array(thirds)
        return thirds

    firsts = first_moments()
    seconds = second_moments(firsts)
    thirds = third_moments(firsts)

    return firsts, seconds, thirds


moments_nd_jitted = jit(moments_nd, static_argnums=(2,))


class GeneralizedLaplace:
    def __init__(self, moments):
        self.moments = moments
        self.ndim = len(moments[0])

    def get_moments(self):
        return self.moments

    def get_pdf(self, x):
        return self.pdf(x)

    def pdf(self, x):
        if not hasattr(self, "params"):
            self.moment_matching()
        sigma_inv = np.linalg.inv(self.sigma_matched)
        q_x = np.sqrt(np.einsum("ij,ni,nj->n", sigma_inv, x, x))
        c_sigma_mu = self.c_sigma_mu
        bessel = kv(self.s_matched - self.ndim / 2, c_sigma_mu * q_x)
        prefactor = (
            2
            * (q_x / c_sigma_mu) ** (self.s_matched - self.ndim / 2)
            / ((2 * np.pi) ** (self.ndim / 2) * gamma(self.s_matched))
        )
        exp_part = np.exp(np.einsum("ni,ij,j->n", x, sigma_inv, self.mu_matched))

        return prefactor * bessel * exp_part

    def parametrized_moments(self, s, mu, sigma):
        def first_moment():
            return s * mu

        def second_moment(firsts):
            seconds = s * ((s + 1) * np.outer(firsts, firsts) + sigma)
            return seconds

        def third_moment(firsts, seconds):
            # rewrite
            dims = np.arange(len(firsts))
            combs = jnp.array(list(itertools.combinations_with_replacement(dims, 3)))

            i, j, k = combs[:, 0], combs[:, 1], combs[:, 2]

            third_moment_values = (
                s
                * (s + 1)
                * (
                    (s + 2) * np.prod(jnp.array([firsts[i], firsts[j], firsts[k]]), axis=0)
                    + seconds[i, j] * firsts[k]
                    + seconds[i, k] * firsts[j]
                    + seconds[j, k] * firsts[i]
                )
            )

            return np.ravel(third_moment_values)

        firsts = first_moment()
        seconds = second_moment(firsts)
        all_moments = [firsts, seconds, third_moment(firsts, seconds)]

        return all_moments

    def convert_1d_params(self, params):
        s = params[0]
        mu = params[1 : 1 + self.mu_shape[0]].reshape(self.mu_shape)
        sigma = params[1 + self.mu_shape[0] :].reshape(self.sigma_shape)
        return s, mu, sigma

    def moment_matching(self):
        self.mu_shape = self.moments[0].shape
        self.sigma_shape = self.moments[1].shape

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

        initial_guess = np.concatenate(
            [
                np.array([1]),  # Initial guess for s
                self.moments[0].ravel(),  # Initial guess for mu
                self.moments[1].ravel(),  # Initial guess for sigma
            ]
        )
        solution = scipy.optimize.least_squares(equations_to_solve, initial_guess, ftol=1e-20)
        if solution.success:
            self.params = solution.x
        else:
            raise ValueError("Root finding did not converge")
        self.params = solution.x
        s_matched, mu_matched, sigma_matched = self.convert_1d_params(self.params)
        self.s_matched, self.mu_matched, self.sigma_matched = s_matched, mu_matched, sigma_matched
        self.c_sigma_mu = np.sqrt(
            2 + np.einsum("ij,i,j", np.linalg.inv(sigma_matched), mu_matched, mu_matched)
        )
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


def ncmom2cum_nd(moments):
    """
    Calculate cumulants from moments for an n-dimensional distribution.
    Up to third order cumulants are calculated.

    Parameters
    ----------
    moments : list
        list of lists of first, second and third moments. second moments are given in the shape of a covariance matrix.

    Returns
    -------
    list
        list of lists of first, second and third cumulants. second cumulants are given out in the shape of a covariance matrix.
    """
    n = len(moments)
    assert (
        n <= 3
    ), "ncmom2cum_nd: moments list too long, only implemented up to third order cumulants"
    conversion_functions = cumulant_generator(n)
    cumulants = [c_function(moments) for c_function in conversion_functions]

    return cumulants


def select_conversion_function(n):
    def first_cumulant(moments):
        return moments[0]

    def second_cumulant(moments):
        # is this correct? does this cause the wrong edgeworth expansion because cumulants and moments should be the same at second order?
        first = moments[0]
        return moments[1] - np.outer(first, first)

    def third_cumulant(moments):
        first = moments[0]
        dims = np.arange(len(first))
        second = moments[1]
        third = moments[2]
        third_cumulants = []
        for n, comb in enumerate(itertools.combinations_with_replacement(dims, 3)):
            i, j, k = comb
            kappa_ijk = (
                third[n]
                - np.sum(
                    [
                        first[one] * second[two, three]
                        for (one, two, three) in [(i, j, k), (j, i, k), (k, i, j)]
                    ]
                )
                + 2 * np.prod([first[m] for m in comb])
            )
            third_cumulants.append(kappa_ijk)
        return third_cumulants

    if n == 0:
        return first_cumulant
    elif n == 1:
        return second_cumulant
    elif n == 2:
        return third_cumulant
    else:
        raise ValueError("n > 2 not implemented")


def cumulant_generator(N):
    n = 0
    while n < N:
        yield select_conversion_function(n)
        n += 1


def get_exact(m, cov, steps=4096):
    mean_trace, second, _ = get_moments_1d(m, cov)
    var_trace = second - mean_trace**2
    ximax = mean_trace + 100 * np.sqrt(var_trace)
    m = np.diag(m)
    t, cf = calc_pdf.calc_quadcf_1D(ximax, steps, cov, m, is_diag=True)
    x_low, pdf_low = calc_pdf.cf_to_pdf_1d(t, cf)
    return x_low, pdf_low, t, cf
