import setup_cov, setup_m, cf_to_pdf, helper_funcs
from cov_setup import Cov
import numpy as np


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
    cov_object.ang_bins = angular_bin_in_deg

    if type(angular_bin_in_deg[0]) is tuple:
        ang = "{:.0f}_{:.0f}".format(angular_bin_in_deg[0][0], angular_bin_in_deg[0][1])
    else:
        raise RuntimeError("Specify angular bin as tuple of two values in degrees.")

    mname = "m_xi{}_l{:d}_t{}_{}.npz".format(kind, exact_lmax, ang, maskname)

    try:
        cov = cov_object.cov_alm
    except:
        cov = cov_object.cov_alm_xi(exact_lmax=exact_lmax, pos_m=True)

    if setup_m.check_m(mname):
        m = setup_m.load_m(mname)
    else:
        m = setup_m.mmatrix_xi(angular_bin_in_deg[0], exact_lmax, cov_object.wlm, kind=kind)
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

    if high_ell_extension:
        cf *= high_ell_gaussian_cf(t, cov_object)
    mean = helper_funcs.nth_moment(1, t, cf)
    x, pdf = cf_to_pdf.cf_to_pdf_1d(cf, t0, dt_xip)
    mean = np.trapz(x * pdf, x)
    norm = np.trapz(pdf, x=x)

    return x, pdf, norm, mean


def high_ell_gaussian_cf(t, cov_object):
    cov_object.cov_xi_gaussian(lmin=cov_object.exact_lmax)
    gauss_cf = helper_funcs.gaussian_cf(t, cov_object.xi, np.sqrt(cov_object.cov_xi))
    return gauss_cf


def pdf_pcl():
    pass
