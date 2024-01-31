import setup_m, helper_funcs
from cov_setup import Cov
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
from scipy.integrate import quad_vec
import wigner


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
    var_trace = 2 * np.trace(m @ cov @ m @ cov)
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

def cov_cl_gaussian(cov_object):
    #TODO: design to take two cov_objects and give (mixed) sum of Cl^2
    cl_e = cov_object.ee.copy()
    cl_b = cov_object.bb.copy()
    noise2 = np.zeros_like(cl_e)
    ell = np.arange(cov_object.lmax + 1)
    if hasattr(cov_object, "_noise_sigma"):
        noise_B = noise_E = cov_object.noise_cl

        cl_e += noise_E
        cl_b += noise_B
        noise2 += np.square(noise_E) + np.square(noise_B)
    
    cl2 = np.square(cl_e) + np.square(cl_b)

    diag = 2 * cl2
    noise_diag = 2 * noise2
    return diag, noise_diag

def get_noisy_cl(cov_objects):
    cl_es,cl_bs = [],[]
    for cov_object in cov_objects:
        cl_e = cov_object.ee.copy()
        cl_b = cov_object.bb.copy()
        if hasattr(cov_object, "_noise_sigma"):
            noise_B = noise_E = cov_object.noise_cl

            cl_e += noise_E
            cl_b += noise_B
        cl_es.append(cl_e)
        cl_bs.append(cl_b)
    return tuple(cl_es),tuple(cl_bs)
        


def cov_cl_gaussian_mixed(mixed_cov_objects):
    cl_es, cl_bs = get_noisy_cl(mixed_cov_objects) # should return all cle and clb needed with noise added
    one_ee, two_ee, three_ee, four_ee = cl_es
    one_bb, two_bb, three_bb, four_bb = cl_bs
    #cl2 = one_ee * two_ee + three_ee * four_ee + one_ee * two_bb + three_ee * four_bb + one_bb * two_ee + three_bb * four_ee + one_bb * two_bb + three_bb * four_bb
    cl2 = one_ee * two_ee + three_ee * four_ee + one_bb * two_bb + three_bb * four_bb
    return cl2

def get_cov_triang(cov_objects):
    #order of cov_objects: as in GLASS gaussian fields creation
    n = len(cov_objects)
    sidelen_xicov = int(0.5 * (-1 + np.sqrt(1 + 8*n)))
    rowlengths = np.arange(1,sidelen_xicov+1)
    cov_triang = [cov_objects[np.sum(rowlengths[:i]):np.sum(rowlengths[:i+1])] for i in range(rowlengths[-1])]
    # for i in range(rowlengths[-1]):
    #    cov_triang[i,:rowlengths[i]] = cov_objects[np.sum(rowlengths[:i]):np.sum(rowlengths[:i+1])]
    return cov_triang

def get_cov_pos(comb):
    column = comb[1]-comb[0]
    row = comb[0]
    return (row,column)


def cov_cl_nD(cov_objects, xicombs=((1,1),(1,0))):
    #xicombs: number stands for row of auto correlation in the GLASS ordering of C_ell, cross-corr always with the larger number first
    n = len(cov_objects)
    sidelen_xicov = int(0.5 * (-1 + np.sqrt(1 + 8*n)))
    
    cov_triang = get_cov_triang(cov_objects)
    
    variances = []
    cov = np.zeros((sidelen_xicov,sidelen_xicov,cov_objects[0].lmax+1))
    # diagonal:
    for i in range(sidelen_xicov):
        inds = get_cov_pos(xicombs[i])
        cov_obj = cov_triang[inds[0]][inds[1]]
        var,noise = cov_cl_gaussian(cov_obj)
        cov[i,i] = var
    
    # off-diagonal:
    print(cov)
    for i in range(sidelen_xicov):
        for j in range(sidelen_xicov):
            if i < j:
                (k,l), (m,n) = xicombs[i], xicombs[j]
                # which cl are going to be involved with such a combination? 
                mix = [(k,m),(l,n),(k,n),(l,m)]
                mix_cov_objects = [cov_triang[get_cov_pos(comb)[0]][get_cov_pos(comb)[1]] for comb in mix]
                              
                sub_cov = cov_cl_gaussian_mixed(tuple(mix_cov_objects))
                cov[i,j] = sub_cov
                cov[j,i] = sub_cov
               
    assert np.allclose(cov[:,:,0], cov[:,:,0].T), "Covariance matrix not symmetric"
    return cov


def get_int_lims(bin_in_deg):
    binmin_in_deg,binmax_in_deg = bin_in_deg[0], bin_in_deg[1]
    upper = np.radians(binmax_in_deg)
    lower = np.radians(binmin_in_deg)
    return lower,upper

def get_integrated_wigners(lmin,lmax,bin_in_deg):
    wigner_int = lambda theta_in_rad: theta_in_rad * wigner.wigner_dl(lmin, lmax, 2, 2, theta_in_rad)
    lower,upper = get_int_lims(bin_in_deg)
    t_norm = 2 / (upper**2 - lower**2)
    integrated_wigners = quad_vec(wigner_int, lower, upper)[0]
    norm = 1 / (4 * np.pi)
    return norm * integrated_wigners * t_norm


def cov_xi_gaussian_nD(cov_objects,xi_combs,angbins_in_deg, lmin=0, lmax=None):
   
    # cov_objects order like in GLASS. assume mask and lmax is the same as for cov_object[0] for all

    # e.g. https://www.aanda.org/articles/aa/full_html/2018/07/aa32343-17/aa32343-17.html
    assert len(xi_combs) == len(angbins_in_deg)
    cov_cl2 = cov_cl_nD(cov_objects,xi_combs)
    if lmax is None:
        lmax = cov_objects[0].lmax
    print(lmax)
    areas = [cov.eff_area for cov in cov_objects]
    assert np.all(np.isclose(areas, areas[0])),areas
    prefactors = helper_funcs.prep_prefactors(angbins_in_deg, cov_objects[0].wl, cov_objects[0].lmax, lmax)
    fsky = areas[0] / 41253 # assume that all fields have at least similar enough fsky
    c_tot = cov_cl2[:,:,lmin : lmax + 1] / fsky
    l = 2 * np.arange(lmin, lmax + 1) + 1
    
    
    xi_cov = np.full(cov_cl2.shape[:2],np.nan)
    means = []
    for i in range(len(cov_cl2)):
        for j in range(len(cov_cl2)):
            if i <= j:
             
                wigners1 = get_integrated_wigners(lmin,lmax,angbins_in_deg[i])
                wigners2 = get_integrated_wigners(lmin,lmax,angbins_in_deg[j])
                xi_cov[i,j] = np.sum(wigners1*wigners2 * c_tot[i,j] * l)
       
                if i == j:
                    autocomb = xi_combs[i]
                    auto_cov_object = get_cov_triang(cov_objects)[get_cov_pos(autocomb)[0]][get_cov_pos(autocomb)[1]]
                    auto_cov_object.cl2pseudocl()
                    pcl_mean_p, pcl_mean_m = helper_funcs.pcl2xi((auto_cov_object.p_ee, auto_cov_object.p_bb, auto_cov_object.p_eb),prefactors,lmax,lmin=lmin)
                    cl_mean_p, cl_mean_m = helper_funcs.cl2xi((auto_cov_object.ee, auto_cov_object.bb), angbins_in_deg[i], lmax, lmin=lmin)
                    #assert np.allclose(pcl_mean_p[0], cl_mean_p, rtol=1e-2), (pcl_mean_p[0], cl_mean_p)
                    print("lmin: {:d}, lmax: {:d}, pCl mean: {:.5e}, Cl mean: {:.5e}".format(lmin,lmax,pcl_mean_p[0], cl_mean_p))
                    means.append(pcl_mean_p[0])

    xi_cov = np.where(np.isnan(xi_cov), xi_cov.T, xi_cov)
    assert np.all(np.linalg.eigvals(xi_cov) >= 0)
    assert np.allclose(xi_cov, xi_cov.T), "Covariance matrix not symmetric"

    return np.array(means), xi_cov

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
    #TODO: make nD
    cov = cov_xi_gaussian(cov_object,lmin=cov_object.exact_lmax + 1)
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

def high_ell_gaussian_cf_nD(t_sets,mu, cov):
    #TODO: make nD
    print(mu,cov)
    
    gauss_cf = helper_funcs.gaussian_cf_nD(t_sets,mu,cov)
    

    return gauss_cf




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
