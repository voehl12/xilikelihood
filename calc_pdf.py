import setup_cov, setup_m, helper_funcs
from cov_setup import Cov
import numpy as np
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt


def pdf_xi_1D(
    angular_bin_in_deg,
    cov_object,
    kind="p",
    steps=2048,
    savestuff=True,
    recalc_cov=False,
    high_ell_extension=True,
):
    exact_lmax = cov_object.exact_lmax
    maskname = cov_object.maskname
    cov_object.ang_bins_in_deg = angular_bin_in_deg

    if type(angular_bin_in_deg[0]) is tuple:
        ang = "{:.0f}_{:.0f}".format(angular_bin_in_deg[0][0], angular_bin_in_deg[0][1])
    else:
        raise RuntimeError("Specify angular bin as tuple of two values in degrees.")
    if cov_object.l_smooth is None:
        smoothstring = ""
    else:
        smoothstring = "_lapodize{:d}".format(cov_object.l_smooth)
    mname = "m_xi{}_l{:d}_t{}_{}{}.npz".format(kind, exact_lmax, ang, maskname, smoothstring)

    try:
        cov = cov_object.cov_alm
    except:
        cov = cov_object.cov_alm_xi(exact_lmax=exact_lmax, pos_m=True)

    if setup_m.check_m(mname):
        m = setup_m.load_m(mname)
    else:
        m = setup_m.mmatrix_xi(angular_bin_in_deg[0], exact_lmax, cov_object.wl, kind=kind)
        if savestuff:
            setup_m.save_m(m, mname)

    xip_max = 5.0e-6

    t, cf = calc_quadcf_1D(xip_max, steps, cov, m)

    x_low, pdf_low = cf_to_pdf_1d(t, cf)
    mean_lowell_cf = helper_funcs.nth_moment(1, t, cf)
    mean_lowell_pdf = np.trapz(x_low * pdf_low, x=x_low)

    assert np.allclose(mean_lowell_cf, mean_lowell_pdf, rtol=1e-5), (
        mean_lowell_cf,
        mean_lowell_pdf,
    )

    if high_ell_extension:
        cf *= high_ell_gaussian_cf(t, cov_object)
        mean_highell = cov_object.xi_pcl

    skewness = helper_funcs.skewness(t, cf)

    x, pdf = cf_to_pdf_1d(t, cf)

    mean = np.trapz(x * pdf, x=x)
    first, second = helper_funcs.nth_moment(2, t, cf)
    assert np.allclose(mean, first, rtol=1e-5), (mean, first)

    std = second - first**2
    norm = np.trapz(pdf, x=x)

    return x, pdf, norm, mean, std, skewness, mean_lowell_pdf, mean_highell


def high_ell_gaussian_cf(t_lowell, cov_object):
    cov_object.cov_xi_gaussian(lmin=cov_object.exact_lmax + 1)
    xip_max = max(
        np.fabs(cov_object.xi_pcl - 1500 * np.sqrt(cov_object.cov_xi)),
        np.fabs(cov_object.xi_pcl + 1500 * np.sqrt(cov_object.cov_xi)),
    )
    dt_xip = 0.45 * 2 * np.pi / xip_max
    steps = 4096
    t0 = -0.5 * dt_xip * (steps - 1)
    t = np.linspace(t0, -t0, steps - 1)

    gauss_cf = helper_funcs.gaussian_cf(t, cov_object.xi_pcl, np.sqrt(cov_object.cov_xi))
    print(helper_funcs.skewness(t, gauss_cf))
    assert np.allclose(helper_funcs.skewness(t, gauss_cf), 0, atol=1e-3), helper_funcs.skewness(
        t, gauss_cf
    )
    interp_to_lowell = 1j * UnivariateSpline(t, gauss_cf.imag, k=5, s=0)(
        t_lowell
    ) + UnivariateSpline(t, gauss_cf.real, k=5, s=0)(t_lowell)
    return interp_to_lowell


def pdf_pcl():
    pass


def calc_quadcf_1D(val_max, steps, cov, m):
    prod = cov @ m
    evals = np.linalg.eigvals(prod)
    dt = 0.45 * 2 * np.pi / val_max

    t0 = -0.5 * dt * (steps - 1)
    t = np.linspace(t0, -t0, steps - 1)

    t_evals = np.tile(
        evals, (steps - 1, 1)
    )  # number of t steps rows of number of eigenvalues columns; each column contains one eigenvalue
    tmat = np.repeat(t, len(evals))  # repeat each t number of eigenvalues times
    tmat = np.reshape(
        tmat, (steps - 1, len(evals))
    )  # recast to array with number of t steps rows and number of eigenvalues columns; each row contains one t step
    t_evals *= tmat  # each row is one set of eigenvalues times a given t step -> need to multiply along this row (axis 1)

    cf = np.prod(np.sqrt(1 / (1 - 2 * 1j * t_evals)), axis=1)

    return t, cf


def cf_to_pdf_1d(t, cf):
    """
    Converts a characteristic function phi(t) to a probability density function f(x).

    Parameters:
    - cf: The characteristic function as a function of t
    - t0: The first value of t at which the characteristic function has been evaluated
    - dt: The increment in t used

    Returns tuple of (x, pdf).
    """

    # Main part of the pdf is given by a FFT
    pdf = np.fft.fft(cf)
    dt = t[1] - t[0]
    t0 = t[0]
    # x is given by the FFT frequencies multiplied by some normalisation factor 2pi/dt
    x = np.fft.fftfreq(cf.size) * 2 * np.pi / dt

    # Multiply pdf by factor to account for the differences between numpy's FFT and a true continuous FT
    pdf *= dt * np.exp(1j * x * t0) / (2 * np.pi)

    # Take the real part of the pdf, and sort both x and pdf by x
    # x, pdf = list(zip(*sorted(zip(x, pdf.real))))
    x, pdf = list(
        zip(*sorted(zip(x, np.abs(pdf))))
    )  # I have found this tends to be more numerically stable

    return (np.array(x), np.array(pdf))
