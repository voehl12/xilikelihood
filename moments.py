import numpy as np
from scipy.interpolate import UnivariateSpline

def nth_moment(n, t, cf):
    k = 5  # 5th degree spline
    derivs_at_zero = [
        1j * UnivariateSpline(t, cf.imag, k=k, s=0).derivative(n=i)(0)
        + UnivariateSpline(t, cf.real, k=k, s=0).derivative(n=i)(0)
        for i in range(1, n + 1)
    ]
    return [(1j**-k * derivs_at_zero[k - 1]) for k in range(1, n + 1)]


def skewness(t, cf):
    first, second, third = nth_moment(3, t, cf)
    sigma2 = second - first**2
    return (third - 3 * first * sigma2 - first**3) / np.sqrt(sigma2) ** 3