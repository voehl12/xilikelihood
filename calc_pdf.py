import setup_m, helper_funcs
from cov_setup import Cov
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.linalg import block_diag
import matplotlib.pyplot as plt


def pdf_xi_1D(
    ang_bins_in_deg,
    cov_object,
    kind="p",
    steps=2048,
    savestuff=True,
    recalc_cov=False,
    high_ell_extension=True,
):
    exact_lmax = cov_object.exact_lmax
    maskname = cov_object.maskname
    cov_object.ang_bins_in_deg = ang_bins_in_deg

    if type(ang_bins_in_deg[0]) is tuple:
        ang = "{:.0f}_{:.0f}".format(ang_bins_in_deg[0][0], ang_bins_in_deg[0][1])
    else:
        raise RuntimeError("Specify angular bin as tuple of two values in degrees.")
 
    mname = "m_xi{}_l{:d}_t{}_{}.npz".format(kind, exact_lmax, ang, maskname)


    cov = cov_object.cov_alm_xi(exact_lmax=exact_lmax, pos_m=True)
    
    if setup_m.check_m(mname):
        m = setup_m.load_m(mname)
    else:
        prefactors = helper_funcs.prep_prefactors(ang_bins_in_deg,cov_object.wl, cov_object.lmax, cov_object.exact_lmax)
        m = setup_m.mmatrix_xi(prefactors, kind=kind, pos_m=True)
        if savestuff:
            setup_m.save_m(m, mname)

    xip_max = 5.0e-6

    t, cf = calc_quadcf_1D(xip_max, steps, cov, m)

    x_low, pdf_low = cf_to_pdf_1d(t, cf)
    mean_trace = np.trace(m@cov)
    mean_lowell_cf, var_lowell = helper_funcs.nth_moment(2, t, cf)
    var_trace = np.trace(m @ cov @ m @ cov)
    assert np.allclose(mean_trace,mean_lowell_cf)
    assert np.allclose(var_trace,var_lowell)
    mean_lowell_pdf = np.trapz(x_low * pdf_low, x=x_low)

    assert np.allclose(mean_lowell_cf, mean_lowell_pdf, rtol=1e-5), (
        mean_lowell_cf,
        mean_lowell_pdf,
    )

    if high_ell_extension:
        cf *= high_ell_gaussian_cf(t, cov_object)
        

    skewness = helper_funcs.skewness(t, cf)

    x, pdf = cf_to_pdf_1d(t, cf)

    mean = np.trapz(x * pdf, x=x)
    first, second = helper_funcs.nth_moment(2, t, cf)
    assert np.allclose(mean, first, rtol=1e-5), (mean, first)

    std = second - first**2
    
    
    norm = np.trapz(pdf, x=x)

    return x, pdf, norm, mean, std, skewness, mean_lowell_pdf


def cov_xi_nD(cov_objects):
    n = len(cov_objects)
    sidelen_almcov = int(0.5 * (-1 + np.sqrt(1 + 8*n)))
    covs = []
    # diagonal:
    for i,cov_object in enumerate(cov_objects[:sidelen_almcov]):
        sub_cov = cov_object.cov_alm_xi()
        len_sub = len(sub_cov)
        covs.append(sub_cov)
    covs = tuple(covs)
    cov = block_diag(*covs)
    # off-diagonal:
    c = sidelen_almcov
    for i in range(sidelen_almcov):
        for j in range(sidelen_almcov):
            if i < j:
                cov_object = cov_objects[c]
                sub_cov = cov_object.cov_alm_xi()
                cov[i*len_sub:(i+1)*len_sub,j*len_sub:(j+1)*len_sub] = sub_cov
                cov[j*len_sub:(j+1)*len_sub,i*len_sub:(i+1)*len_sub] = sub_cov
                c += 1
    assert np.allclose(cov, cov.T), "Covariance matrix not symmetric"
    return cov

def high_ell_gaussian_cf(t_lowell, cov_object):
    cov_object.cov_xi_gaussian(lmin=cov_object.exact_lmax + 1)
    xip_max = max(
        np.fabs(cov_object.xi_pcl - 500 * np.sqrt(cov_object.cov_xi)),
        np.fabs(cov_object.xi_pcl + 500 * np.sqrt(cov_object.cov_xi)),
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

def get_cf_nD(tset,mset,cov):
    # tset and mset can have any length, such that this cf element can be calculated for any number of dimensions
    big_m = np.einsum('i,ijk -> jk',tset,mset)
    evals = np.linalg.eigvals(big_m @ cov)
    return tset,np.prod((1 - 2j * evals) ** -0.5)


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

def cf_to_pdf_nd(cf_grid, t0, dt, verbose=True):
    """
    Converts an n-dimensional joint characteristic function to the corresponding joint probability density function.

    Arguments:

        cf_grid : n-dimensional grid of the characteristic function

        t0 : length-n vector of t0 values for each dimension

        dt : length-n vector of dt values for each dimension
    """

    # 1. Use an n-dimensional FFT to obtain the FFT grid
    if verbose:
        print('Performing FFT')
    fft_grid = np.fft.fftshift(np.fft.fftn(cf_grid))

    # 2. Obtain the range of each dimension of x and use this to form a grid of x
    if verbose:
        print('Forming x grid')

    # Form the x range in each dimension
    x_ranges = []
    n_dim = fft_grid.ndim
    grid_shape = fft_grid.shape
    for dim in range(n_dim):
        x_ranges.append(np.fft.fftshift(np.fft.fftfreq(grid_shape[dim])) * 2 * np.pi / dt[dim])

    # Form the grid
    x_grid = np.stack(np.meshgrid(*x_ranges, indexing='ij'), axis=-1)

    # 3. Use the grid of x to calculate the x-dependent phase factor and multiply this by the FFT grid
    if verbose:
        print('Applying phase factor')
    t0_dot_x = np.tensordot(t0, x_grid, axes=[0, n_dim])
    phase_factor_grid = np.exp(-1j * t0_dot_x)

    fft_grid_with_phase_factor = phase_factor_grid * fft_grid

    # 4. Uniformly multiply the entire grid by the normalisation factor and take the absolute value
    if verbose:
        print('Applying normalisation factor')
    norm_factor = np.prod(dt) / ((2 * np.pi) ** n_dim)
    pdf_grid = np.abs(fft_grid_with_phase_factor * norm_factor)

    return (x_grid, pdf_grid)
