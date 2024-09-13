import numpy as np
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

    firsts = []
    for dim in dims:
        first_moment = np.trace(prods[dim])
        firsts.append(first_moment)

    seconds = np.full(((ndim, ndim)), np.nan)
    for comb in itertools.combinations_with_replacement(dims, 2):
        one, two = comb
        second_moment = np.trace(prods[one]) * np.trace(prods[two]) + 2 * np.sum(
            prods[one] * np.transpose(prods[two])
        )
        seconds[one, two] = second_moment

    seconds = np.where(np.isnan(seconds), seconds.T, seconds)
    assert np.allclose(seconds, seconds.T), "moments_nd: covariance matrix not symmetric"

    thirds = []
    # define n-dim 3rd order hermite polynomials in exactly the same manner / order!
    for comb in itertools.combinations_with_replacement(dims, 3):
        one, two, three = comb
        prod2 = prods[two] @ prods[three]
        third_moment = (
            np.prod([np.trace(prods[j]) for j in comb])
            + 2
            * (
                np.sum(
                    [
                        np.trace(prods[k]) * np.sum(prods[l] * np.transpose(prods[n]))
                        for (k, l, n) in [(one, two, three), (three, one, two), (two, three, one)]
                    ]
                )
            )
            + 8 * np.sum(prods[one] * np.transpose(prod2))
        )
        thirds.append(third_moment)

    return [firsts, seconds, thirds]


class MultiNormalExpansion:

    def __init__(self, cumulants) -> None:
        self.cumulants = cumulants

    def hermite_nd(self, x: np.ndarray):
        # x: point in R^n
        # should be extended to take any number of cumulants (just like the moment conversion function)
        ndim = len(self.cumulants[0])
        dims = np.arange(len(self.cumulants[0]))
        lambda_inv = np.linalg.inv(self.cumulants[1])

        def h_1(x):
            """
            Calculate first order hermite polynomials for n-dimensional distribution.

            Parameters
            ----------
            x : np.1darray of length n
                point in R^n

            Returns
            -------
            np.array 1d
                first order hermite polynomial, evaluated at x
            """
            return np.dot(lambda_inv, x - self.cumulants[0])

        def h_2(x):
            # second order hermite polynomials arranged in a 2d square matrix, evaluated at x
            return -lambda_inv + np.outer(h_1(x), h_1(x))

        def h_3(x):
            # third order hermite polynomials arranged in a list according to the order of itertools.combinations_with_replacement
            h_3 = []
            for comb in itertools.combinations_with_replacement(dims, 3):
                i, j, k = comb
                h_ijk = np.prod([h_1(x)[m] for m in comb]) - np.sum(
                    [
                        h_1(x)[one] * lambda_inv[two, three]
                        for (one, two, three) in [(i, j, k), (j, i, k), (k, i, j)]
                    ]
                )
                h_3.append(h_ijk)
            return h_3

        return [h_1(x), h_2(x), h_3(x)]

    def pdf(self, x: np.ndarray):

        return pdf


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

    assert (
        len(moments) <= 3
    ), "ncmom2cum_nd: moments list too long, only implemented up to third order cumulants"
    conversion_functions = cumulant_generator()
    cumulants = [conversion_functions[i](moments) for i in range(len(moments))]

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
        for comb in itertools.combinations_with_replacement(dims, 3):
            i, j, k = comb
            kappa_ijk = (
                third[i, j, k]
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


def cumulant_generator():
    n = 0
    while True:
        yield select_conversion_function(n)
        n += 1


def get_cumulants(moments):
    return mnc2cum(moments)


def get_exact(m, cov, steps=4096):
    mean_trace, second, _ = get_moments_1d(m, cov)
    var_trace = second - mean_trace**2
    ximax = mean_trace + 100 * np.sqrt(var_trace)
    m = np.diag(m)
    t, cf = calc_pdf.calc_quadcf_1D(ximax, steps, cov, m, is_diag=True)
    x_low, pdf_low = calc_pdf.cf_to_pdf_1d(t, cf)
    return x_low, pdf_low, t, cf
