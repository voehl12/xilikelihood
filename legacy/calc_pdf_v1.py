"""
Legacy PDF calculation functions from first paper, https://arxiv.org/abs/2407.08718.
Preserved for reproducibility and reference.
Use likelihood module for new work.
Module has been renamed to cf_pdf_cov.py for further use.
"""

import numpy as np

from file_handling import check_for_file
from file_handling_v1 import load_cfs, load_pdfs
import setup_m
import helper_funcs

def pdf_xi_1D(
    ang_bins_in_deg,
    cov_objects,
    comb=(0, 0),
    kind="p",
    steps=4096,
    savestuff=True,
    high_ell_extension=True,
    m_path="/cluster/work/refregier/veoehl/m_matrices/",
):
    """
    Calculates 1D likelihoods of a correlation function for several angular separation bins.

    Parameters
    ----------
    ang_bins_in_deg : array of tuples
        angular separation bins in degree
    cov_objects : tuple of Cov() objects
        objects of the class Cov() containing covariance of pseudo-alm; can be two if cross-correlation of two redshift bins is attempted
    comb : tuple, optional
        combination of redshift bins according to order of covariance objects, by default (0, 0)
    kind : str, optional
        kind of spin-2 correlation function, "p" or "m", by default "p"
    steps : int, optional
        number of steps for the characteristic function grid, by default 4096
    savestuff : bool, optional
        whether to save characteristic functions and pdfs, by default True
    high_ell_extension : bool, optional
        whether apply the Gaussian extension of the likleihood up to the bandlimit of the given mask, by default True
    m_path : str, optional
        path for the combination matrices, by default "/cluster/work/refregier/veoehl/m_matrices/"

    Returns
    -------
    arrays
       arrays containing correlation function values, corresponding likelihood and an array of summaries (mean, std and skewness)

    Raises
    ------
    RuntimeError
        _description_
    """
    exact_lmax = cov_objects[0].exact_lmax
    maskname = cov_objects[0].mask.name
    x_arr = np.zeros((len(ang_bins_in_deg), steps - 1))
    pdf_arr = np.zeros((len(ang_bins_in_deg), steps - 1))
    stat_arr = np.zeros(((len(ang_bins_in_deg), 3)))
    t_s, cf_res, cf_ims, ximax_s, cf_angs = [], [], [], [], []
    if cov_objects[0].theorycl.sigma_e is None:
        noise = 'nonoise'
    else:
        noise = ''
    if not high_ell_extension:
        ext = "_noext"
    else:
        ext = ""
    pdfname = "pdfs/pdfs_xi{}_{:d}_{:d}_l{:d}_{}_{}{}{}.npz".format(
        kind, *comb, exact_lmax, maskname, cov_objects[0].theorycl.name, noise,ext
    )
    print("Setting pdf name: {}".format(pdfname))
    # there might be a bug as to how pdfs are added to the pdf file when there was a file with extension before
    pdffile = False
    cfname = "cf_xi{}_{:d}_{:d}_l{:d}_{}_{}.npz".format(
        kind, *comb, exact_lmax, maskname, cov_objects[0].theorycl.name
    )
    cffile = False

    if check_for_file(pdfname, "pdf"):
        xs, pdfs, statss, file_angs = load_pdfs(pdfname)
        if len(xs[0]) == steps - 1:
            pdffile = True
            if np.array_equal(file_angs, np.array(ang_bins_in_deg)):
                print("direct return")
                return xs, pdfs, statss

    if check_for_file(cfname, "cf"):
        ts, cfs_real, cfs_imag, ximaxs, cf_file_angs = load_cfs(cfname)
        if len(ts[0]) == steps - 1:
            cffile = True

    cov_flag = False
    for b, bin_in_deg in enumerate(ang_bins_in_deg):
        print("Working on angular bin {}.".format(bin_in_deg))
        if pdffile and bin_in_deg in file_angs:
            angind = np.where(file_angs == bin_in_deg)[0]
            x_arr[b], pdf_arr[b] = xs[angind[0]], pdfs[angind[0]]
            stat_arr[b] = statss[angind[0]]
            print("found pdf directly")
        else:
            if cffile and bin_in_deg in cf_file_angs:

                angind = np.where(cf_file_angs == bin_in_deg)[0]
                t, cf_real, cf_imag, ximax = (
                    ts[angind[0]],
                    cfs_real[angind[0]],
                    cfs_imag[angind[0]],
                    ximaxs[angind[0]],
                )
                cf = cf_real + 1j * cf_imag
                print("using cf from file")
                if high_ell_extension:
                    cov_index = get_cov_pos(comb)
                    cov_ref = get_cov_triang(cov_objects)[cov_index[0]][cov_index[1]]
                    cov_ref.cl2pseudocl()

            else:
                if not cov_flag:
                    cov = cov_xi_nD(cov_objects)
                    cov_flag = True
                    prefactors_all = helper_funcs.prep_prefactors(
                        ang_bins_in_deg, cov_objects[0].mask.wl, cov_objects[0].mask.lmax, cov_objects[0].mask.lmax
                    )
                    for cov_object in cov_objects:
                        cov_object.cl2pseudocl()

                    cov_index = get_cov_pos(comb)
                    cov_ref = get_cov_triang(cov_objects)[cov_index[0]][cov_index[1]]
                    cov_estimate = cov_xi_gaussian_nD(
                        [cov_ref.theorycl], [(0, 0)], [bin_in_deg],cov_ref.mask.eff_area, lmax=cov_ref.mask.lmax
                    )
                    xip_estimate, _ = helper_funcs.pcl2xi(
                        (cov_ref.p_ee, cov_ref.p_bb, cov_ref.p_eb), prefactors_all, cov_ref.mask.lmax
                    )

                if type(bin_in_deg) is tuple:
                    ang = "{:.2f}_{:.2f}".format(*bin_in_deg)
                    ang = ang.replace(".", "p")
                else:
                    raise RuntimeError("Specify angular bin as tuple of two values in degrees.")

                mname = "m_xi{}_{:d}_{:d}_l{:d}_t{}_{}.npz".format(
                    kind, *comb, exact_lmax, ang, maskname
                )

                is_diag = False
                if setup_m.check_m(m_path + mname):
                    m = setup_m.load_m(m_path + mname)

                else:
                    m = setup_m.m_xi_cross(
                        (prefactors_all[b, :, : exact_lmax + 1],), combs=(comb,), kind=("p",)
                    )[0]
                    assert m.shape == cov.shape

                    if savestuff:
                        setup_m.save_m(m, m_path + mname)

                if np.array_equal(m, np.diag(np.diag(m))):
                    is_diag = True
                    m = np.diag(m)

                print("Starting characteristic function computation")
                ximax = np.fabs(xip_estimate[b]) * 50  # needs a bit of tweaking 12 * 1000/Aeff
                ximax = np.fabs(xip_estimate[b]) + 100 * np.sqrt(cov_estimate[0, 0])
                t, cf = calc_quadcf_1D(ximax, steps, cov, m, is_diag=is_diag)
                if cffile:
                    ts = np.append(ts, [t], axis=0)
                    cfs_real = np.append(cfs_real, [cf.real], axis=0)
                    cfs_imag = np.append(cfs_imag, [cf.imag], axis=0)
                    ximaxs = np.append(ximaxs, [ximax], axis=0)
                    cf_file_angs = np.append(cf_file_angs, [bin_in_deg], axis=0)
                # cf = np.array(cf)
                print("Converting to pdf")
                x_low, pdf_low = cf_to_pdf_1d(t, cf)
                mean_lowell_pdf = np.trapz(x_low * pdf_low, x=x_low)
                mean_lowell_cf, var_lowell = helper_funcs.nth_moment(2, t, cf)
                if is_diag:

                    mean_trace = np.trace(m[:, None] * cov)

                    # var_trace = 2 * np.trace(m[:,None]*cov @ m[:,None]*cov)

                    assert np.allclose(mean_trace, mean_lowell_cf), (mean_trace, mean_lowell_cf)
                    # assert np.allclose(var_trace,var_lowell),(var_trace,var_lowell)
                assert np.allclose(mean_lowell_cf, mean_lowell_pdf, rtol=1e-3), (
                    mean_lowell_cf,
                    mean_lowell_pdf,
                )
            t_s.append(t)
            cf_ims.append(cf.imag)
            cf_res.append(cf.real)
            ximax_s.append(ximax)
            cf_angs.append(bin_in_deg)

            if high_ell_extension:
                cf *= high_ell_gaussian_cf(t, cov_ref, bin_in_deg)

            skewness = helper_funcs.skewness(t, cf)

            x, pdf = cf_to_pdf_1d(t, cf)

            mean = np.trapz(x * pdf, x=x)
            first, second = helper_funcs.nth_moment(2, t, cf)
            assert np.isclose(mean, first, rtol=1e-2), (mean, first)

            std = second - first**2

            norm = np.trapz(pdf, x=x)
            assert np.isclose(norm, 1, rtol=1e-2), norm

            x_arr[b] = x
            pdf_arr[b] = pdf
            stat_arr[b] = [mean, std, skewness]

            if pdffile:
                file_angs = np.append(file_angs, [bin_in_deg], axis=0)
                xs = np.append(xs, [x], axis=0)
                pdfs = np.append(pdfs, [pdf], axis=0)
                statss = np.append(statss, [np.array([mean, std, skewness])], axis=0)

    if savestuff:
        if not pdffile:
            np.savez(pdfname, x=x_arr, pdf=pdf_arr, stats=stat_arr, angs=np.array(ang_bins_in_deg))
        else:
            np.savez(pdfname, x=xs, pdf=pdfs, stats=statss, angs=file_angs)
        if len(t_s) > 0 and not cffile:
            np.savez(
                cfname,
                t=np.array(t_s),
                cf_re=np.array(cf_res),
                cf_im=np.array(cf_ims),
                ximax=np.array(ximax_s),
                angs=np.array(cf_angs),
            )
        else:
            np.savez(cfname, t=ts, cf_re=cfs_real, cf_im=cfs_imag, ximax=ximaxs, angs=cf_file_angs)

    return x_arr, pdf_arr, stat_arr


def high_ell_gaussian_cf(t_lowell, cov_object, angbin):
    """
    calculates the characteristic function for a 1d Gaussian used as high mulitpole moment extension.

    Parameters
    ----------
    t_lowell : 1d array
        Fourier space t grid used for the low multipole moment part
    cov_object : Cov object
        covariance object (Cov) used for the low multipole moment part
    angbin : tuple
        angular separation bin for the correlation function

    Returns
    -------
    1d complex array
        characteristic function of the Gaussian extension on the same t grid as the low ell part
    """

    cov = cov_xi_gaussian_nD([cov_object.theorycl], [(0, 0)], [angbin], cov_object.mask.eff_area,lmin=cov_object.exact_lmax + 1)
    prefactors = helper_funcs.prep_prefactors(
            [angbin], cov_object.mask.wl, cov_object.mask.lmax, cov_object.mask.lmax
        )
    mean = mean_xi_gaussian_nD(prefactors,(cov_object.p_ee, cov_object.p_bb, cov_object.p_eb), lmin=cov_object.exact_lmax + 1,lmax=cov_object.exact_lmax, kind="p")
    mean = mean[0]
    cov = cov[0, 0]

    xip_max = np.fabs(3 * mean)
    xip_max = np.fabs(mean) + 500 * np.sqrt(cov)
    dt_xip = 0.45 * 2 * np.pi / xip_max
    steps = 4096
    t0 = -0.5 * dt_xip * (steps - 1)
    t = np.linspace(t0, -t0, steps - 1)

    gauss_cf = helper_funcs.gaussian_cf(t, mean, np.sqrt(cov))

    assert np.allclose(helper_funcs.skewness(t, gauss_cf), 0, atol=1e-3), helper_funcs.skewness(
        t, gauss_cf
    )
    print("Skewness of Gaussian extension: {}".format(helper_funcs.skewness(t, gauss_cf)))
    # test_cf2pdf(t, mean, np.sqrt(cov))
    interp_to_lowell = 1j * UnivariateSpline(t, gauss_cf.imag, k=5, s=0)(
        t_lowell
    ) + UnivariateSpline(t, gauss_cf.real, k=5, s=0)(t_lowell)

    return interp_to_lowell


def calc_quadcf_1D(val_max, steps, cov, m, is_diag=False):
    """
    Calculates a 1d characteristic function given a covariance matrix and combination matrix.

    Parameters
    ----------
    val_max : float
        positive boundary of the real space grid
    steps : int
        number of gridpoints
    cov : 2-d array
        covariance matrix of the Gaussian distributed variables
    m : 2-d array
        combination matrix combining Gaussiand distributed variables to quadratic form
    is_diag : bool, optional
        if the combination matrix is diagonal, by default False

    Returns
    -------
    tuple
        t, cf: the Fourier space grid t and the characteristic function
    """
    tic = time.perf_counter()
    print("multiplying matrices, N = {:d}...".format(len(cov)))
    if is_diag:
        prod = m[:, None] * cov
    else:
        prod = np.dot(m, cov)
    toc = time.perf_counter()
    print("N = {:d} multiplication, took {:.2f} seconds".format(len(prod), toc - tic))
    tic = time.perf_counter()
    print("getting eigenvalues...")
    evals = np.linalg.eigvals(prod)
    toc = time.perf_counter()
    print("N = {:d} eigenvalues, took {:.2f} seconds".format(len(prod), toc - tic))
    dt = 0.45 * 2 * np.pi / val_max

    t0 = -0.5 * dt * (steps - 1)
    t = np.linspace(t0, -t0, steps - 1)
    t_evals = t[:, None] * evals
    """ t_evals = np.tile(
        evals, (steps - 1, 1)
    )  # number of t steps rows of number of eigenvalues columns; each column contains one eigenvalue
    tmat = np.repeat(t, len(evals))  # repeat each t number of eigenvalues times
    tmat = np.reshape(
        tmat, (steps - 1, len(evals))
    )  # recast to array with number of t steps rows and number of eigenvalues columns; each row contains one t step
    t_evals *= tmat  # each row is one set of eigenvalues times a given t step -> need to multiply along this row (axis 1)
    """
    cf = np.prod(np.sqrt(1 / (1 - 2 * 1j * t_evals)), axis=1)

    return t, cf


def cf_to_pdf_nd(cf_grid, t0, dt, verbose=True):
    """
    Converts an n-dimensional joint characteristic function to the corresponding joint probability density function.
    n-dimensional characteristic functions are computationally super heavy, which is why this won't be used in practice.

    Arguments:

        cf_grid : n-dimensional grid of the characteristic function

        t0 : length-n vector of t0 values for each dimension

        dt : length-n vector of dt values for each dimension
    """

    # 1. Use an n-dimensional FFT to obtain the FFT grid
    if verbose:
        print("Performing FFT")
    fft_grid = np.fft.fftshift(np.fft.fftn(cf_grid))

    # 2. Obtain the range of each dimension of x and use this to form a grid of x
    if verbose:
        print("Forming x grid")

    # Form the x range in each dimension
    x_ranges = []
    n_dim = fft_grid.ndim
    grid_shape = fft_grid.shape
    for dim in range(n_dim):
        x_ranges.append(np.fft.fftshift(np.fft.fftfreq(grid_shape[dim])) * 2 * np.pi / dt[dim])

    # Form the grid
    x_grid = np.stack(np.meshgrid(*x_ranges, indexing="ij"), axis=-1)

    # 3. Use the grid of x to calculate the x-dependent phase factor and multiply this by the FFT grid
    if verbose:
        print("Applying phase factor")
    t0_dot_x = np.tensordot(t0, x_grid, axes=[0, n_dim])
    phase_factor_grid = np.exp(-1j * t0_dot_x)

    fft_grid_with_phase_factor = phase_factor_grid * fft_grid

    # 4. Uniformly multiply the entire grid by the normalisation factor and take the absolute value
    if verbose:
        print("Applying normalisation factor")
    norm_factor = np.prod(dt) / ((2 * np.pi) ** n_dim)
    pdf_grid = np.abs(fft_grid_with_phase_factor * norm_factor)

    return (x_grid, pdf_grid)

def get_cf_nD(tset, mset, cov):
    # tset and mset can have any length, such that this cf element can be calculated for any number of dimensions
    # computationally super heavy, as this needs to be called for every point in characteristic function space
    # won't be used in practice, just for the proof of principle paper
    big_m = np.einsum("i,ijk -> jk", tset, mset)
    evals = np.linalg.eigvals(big_m @ cov)
    return tset, np.prod((1 - 2j * evals) ** -0.5)

def high_ell_gaussian_cf_nD(t_sets, mu, cov):

    gauss_cf = helper_funcs.gaussian_cf_nD(t_sets, mu, cov)

    return gauss_cf