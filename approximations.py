import numpy as np
import scipy
from statsmodels.stats.moment_helpers import mnc2cum
from statsmodels.distributions.edgeworth import ExpandedNormal
import calc_pdf
import itertools
import time


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


def moments_nd(m, cov):
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
    ndim = m.shape[0]
    if ndim > 100:
        print("Warning: ndim > 100")
    dims = np.arange(ndim)
    prods = [m[i] @ cov for i in range(ndim)]

    def first_moments():
        return [np.trace(prods[dim]) for dim in dims]

    def second_moments(firsts):
        seconds = np.full(((ndim, ndim)), np.nan)
        for comb in itertools.combinations_with_replacement(dims, 2):
            one, two = comb
            second_moment = firsts[one] * firsts[two] + 2 * np.sum(
                prods[one] * np.transpose(prods[two])
            )
            seconds[one, two] = second_moment
        seconds = np.where(np.isnan(seconds), seconds.T, seconds)
        assert np.allclose(seconds, seconds.T), "moments_nd: covariance matrix not symmetric"
        return seconds

    def third_moments(firsts, seconds):
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
    thirds = third_moments(firsts, seconds)

    return firsts, seconds, thirds


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
