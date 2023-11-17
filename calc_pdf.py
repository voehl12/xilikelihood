import setup_cov, setup_m, cf_to_pdf, helper_funcs
from cov_setup import Cov
import numpy as np
from scipy.interpolate import UnivariateSpline


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

    prod = cov @ m
    evals = np.linalg.eigvals(prod)

    xip_max = 5.0e-5
    dt_xip = 0.45 * 2 * np.pi / xip_max

    t0 = -0.5 * dt_xip * (steps - 1)
    t = np.linspace(t0, -t0, steps - 1)

    t_evals = np.tile(
        evals, (steps - 1, 1)
    )  # number of t steps rows of number of eigenvalues columns; each column contains one eigenvalue
    tmat = np.repeat(t, len(evals))  # repeat each t number of eigenvalues times
    tmat = np.reshape(
        tmat, (steps - 1, len(evals))
    )  # recast to array with number of t steps rows and number of eigenvalues columns; each row contains one t step
    t_evals *= tmat  # each row is one set of eigenvalues times a given t step -> need to multiply along this row (axis 1)

    cf = np.prod(
        np.sqrt(1 / (1 - 2 * 1j * t_evals)), axis=1
    )  # gives t steps different values for characteristic function
    mean_lowell = helper_funcs.nth_moment(1, t, cf)
    if high_ell_extension:
        cf *= high_ell_gaussian_cf(t, cov_object)
        mean_highell = cov_object.xi_pcl
    skewness = helper_funcs.skewness(t, cf)

    x, pdf = cf_to_pdf.cf_to_pdf_1d(cf, t0, dt_xip)
    mean = np.trapz(x * pdf, x=x)
    first, second = helper_funcs.nth_moment(2, t, cf)
    std = second - mean**2
    norm = np.trapz(pdf, x=x)

    return x, pdf, norm, mean, std, skewness, mean_lowell, mean_highell


def high_ell_gaussian_cf(t_lowell, cov_object):
    cov_object.cov_xi_gaussian(lmin=cov_object.exact_lmax)
    xip_max = max(
        np.fabs(cov_object.xi_pcl - 300 * np.sqrt(cov_object.cov_xi)),
        np.fabs(cov_object.xi_pcl + 300 * np.sqrt(cov_object.cov_xi)),
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
