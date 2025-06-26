import numpy as np
import healpy as hp
from scipy.interpolate import UnivariateSpline
from scipy.special import eval_legendre
from scipy.integrate import quad_vec
from scipy.optimize import fsolve
import wigner
import jax.numpy as jnp
import jax


def get_noise_xi_cov(
    survey_area_in_sqd, binmin_in_deg, binmax_in_deg, sigma_e=(0.282842712474619, 1.207829761642)
):
    if type(sigma_e) is tuple:
        sigma, n_gal = sigma_e
    else:
        raise RuntimeError("sigma_e must be tuple (sigma_single_component,n_gal_per_arcmin2)")
    pp_sigma = sigma * np.sqrt(2)
    A_annulus = np.pi * ((binmax_in_deg * 60) ** 2 - (binmin_in_deg * 60) ** 2)
    N_pair = (
        survey_area_in_sqd * 3600 * A_annulus * (n_gal) ** 2
    )  # this is 2 * N_pair but it comes from defining the shotnoise this way
    return (pp_sigma**2 / np.sqrt(N_pair)) ** 2


def get_noise_cl(sigma_e=(0.282842712474619, 1.207829761642)):
    # TODO: should be extended to mixed redshift bin C_ell with different shape noise parameters.
    if type(sigma_e) is tuple:
        sigma, n_gal_per_arcmin2 = sigma_e
    else:
        raise RuntimeError("sigma_e must be tuple (sigma_single_component,n_gal_per_arcmin2)")
    return (sigma) ** 2 / (n_gal_per_arcmin2 * 3600 * 41253 / (4 * np.pi))


def get_noise_pixelsigma(nside=256, sigma_e=(0.282842712474619, 1.207829761642)):
    if type(sigma_e) is tuple:
        sigma, n_gal_per_arcmin2 = sigma_e
    else:
        raise RuntimeError("sigma_e must be tuple (sigma_single_component,n_gal_per_arcmin2)")
    return sigma / np.sqrt(n_gal_per_arcmin2 * hp.nside2pixarea(nside, degrees=True) * 3600)


def noise_cl_cube(noise_cl):
    # TODO: separately for e,b and n?
    c_all = np.zeros((3, 3, len(noise_cl)))
    for i in range(3):
        c_all[i, i] = noise_cl
    return c_all


def gaussian_cf(t, mu, sigma):

    return np.exp(1j * t * mu - 0.5 * sigma**2 * t**2)


def gaussian_cf_nD(t_sets, mu, cov):
    tmu = np.dot(t_sets, mu)
    cov_t = np.einsum("ij,kj->ki", cov, t_sets)
    tct = np.einsum("ki,ki->k", t_sets, cov_t)
    return np.exp(1j * tmu - 0.5 * tct)


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


def ang_prefactors(t_in_deg, wl, norm_lmax, out_lmax, kind="p"):
    # normalization factor automatically has same l_max as the pseudo-Cl summation

    t = np.radians(t_in_deg)
    norm_l = np.arange(norm_lmax + 1)
    legendres = eval_legendre(norm_l, np.cos(t))
    norm = 1 / np.sum((2 * norm_l + 1) * legendres * wl) / (2 * np.pi)

    if kind == "p":
        wigners = wigner.wigner_dl(0, out_lmax, 2, 2, t)
    elif kind == "m":
        wigners = wigner.wigner_dl(0, out_lmax, 2, -2, t)
    else:
        raise RuntimeError("correlation function kind needs to be p or m")

    return 2 * np.pi * norm * wigners


def bin_prefactors(ang_bin_in_deg, wl, norm_lmax, out_lmax, kind="p"):
    lower, upper = ang_bin_in_deg
    lower, upper = np.radians(lower), np.radians(upper)
    buffer = 0
    norm_l = np.arange(norm_lmax + buffer + 1)
    legendres = lambda t_in_rad: eval_legendre(norm_l, np.cos(t_in_rad))
    # TODO: check whether this sum needs to be done after the integration as well (even possible?)
    # -> sum is the same, the 1/ x is a problem when lots of the wl are zero, so it's actually better to do it this way, the order does not seem to matter in this case.
    norm = (
        lambda t_in_rad: 1
        / np.sum((2 * norm_l + 1) * legendres(t_in_rad) * wl[: norm_lmax + buffer + 1])
        / (2 * np.pi)
    )
    if kind == "p":
        wigners = lambda t_in_rad: wigner.wigner_dl(0, out_lmax, 2, 2, t_in_rad)
    elif kind == "m":
        wigners = lambda t_in_rad: wigner.wigner_dl(0, out_lmax, 2, -2, t_in_rad)
    else:
        raise RuntimeError("correlation function kind needs to be p or m")

    integrand = lambda t_in_rad: norm(t_in_rad) * wigners(t_in_rad) * t_in_rad
    # norm * d_l * weights
    W = 0.5 * (upper**2 - lower**2)  # weights already integrated
    A_ell = quad_vec(integrand, lower, upper)

    return 2 * np.pi * A_ell[0] / W


def prep_prefactors(angs_in_deg, wl, norm_lmax, out_lmax):
    """
    Calculate the l-dependent prefactors that contain all angular information for the correlation function.

    Parameters:
        angs_in_deg (list): A list of angles in degrees.
        wl (numpy array): the C_ell of the mask
        norm_lmax (int): the lmax used for the calculation of the normalization.
        out_lmax (int): the lmax to which the prefactors will be needed and should be returned

    Returns:
        numpy array: An array with the prefactors for both xi+ and xi-, the given angles and lmax.
        Shape: (len(angs_in_deg),2,out_lmax+1)
    """
    prefactors_arr = np.zeros((len(angs_in_deg), 2, out_lmax + 1))
    if type(angs_in_deg[0]) is tuple:
        prefactors = bin_prefactors
    else:
        prefactors = ang_prefactors
    for i, ang_in_deg in enumerate(angs_in_deg):
        prefactors_arr[i, 0] = prefactors(ang_in_deg, wl, norm_lmax, out_lmax)
        prefactors_arr[i, 1] = prefactors(ang_in_deg, wl, norm_lmax, out_lmax, kind="m")

    return prefactors_arr



def compute_kernel(pcl, prefactors, out_lmax=None, lmin=0):
    
    pcl_e, pcl_b, pcl_eb = pcl
    if out_lmax is None:
        out_lmax = len(pcl_e) - 1
    l = 2 * np.arange(lmin, out_lmax + 1) + 1
    p_cl_prefactors_p, p_cl_prefactors_m = prefactors[:, 0], prefactors[:, 1]
    kernel_xip = p_cl_prefactors_p[:, lmin : out_lmax + 1] * l * (pcl_e[lmin : out_lmax + 1] + pcl_b[lmin : out_lmax + 1])
    kernel_xim = p_cl_prefactors_m[:, lmin : out_lmax + 1] * l * (pcl_e[lmin : out_lmax + 1] - pcl_b[lmin : out_lmax + 1] - 2j * pcl_eb[lmin : out_lmax + 1])
    return kernel_xip, kernel_xim


def pcl2xi(pcl, prefactors, out_lmax=None, lmin=0):
    """
    _summary_

    Parameters
    ----------
    pcl : tuple
        three numpy arrays: (ee,bb,eb)
    prefactors : np.array
        array with bin or angle prefactors from prep_prefactors (i.e. an (len(angles),2,out_lmax) array)
    out_lmax : int
        lmax to which sum over pcl is taken
    lmin : int, optional
        _description_, by default 0

    Returns
    -------
    _type_
        _description_
    """
    
    kernel_xip, kernel_xim = compute_kernel(pcl, prefactors, out_lmax, lmin)
    xip = np.sum(kernel_xip, axis=-1)
    xim = np.sum(kernel_xim, axis=-1)
    return xip, xim


def pcls2xis(pcls, prefactors, out_lmax=None, lmin=0):
    """
    Generate xi+ and xi- from pseudo-Cl data.

    Parameters
    ----------
    pcls : numpy array
        (3, batchsize, lmax+1) ; batchsize can also be number of correlations
        if array is 4-dim (3, batchsize, n_corr, lmax+1) it is assumed that the batchsize is the first dimension
        and the second dimension is the number of correlations.
    prefactors : np.array
        Array with bin or angle prefactors from prep_prefactors (i.e., an (len(angles), 2, out_lmax) array).
    out_lmax : int
        lmax to which sum over pcl is taken.
    lmin : int, optional
        Minimum l value, by default 0.

    Returns
    -------
    xips, xims: (batchsize, n_angbins) arrays
    """
    pcls_e, pcls_b, pcls_eb = pcls[0], pcls[1], pcls[2]
    if out_lmax is None:
        out_lmax = len(pcls_e[0]) - 1
    l = 2 * jnp.arange(lmin, out_lmax + 1) + 1
    p_cl_prefactors_p, p_cl_prefactors_m = jnp.array(prefactors[:, 0]), jnp.array(prefactors[:, 1])

    if pcls_e.ndim == 2:
        xips = jnp.sum(
            p_cl_prefactors_p[None, :, lmin : out_lmax + 1]
            * l
            * (pcls_e[:, None, lmin : out_lmax + 1] + pcls_b[:, None, lmin : out_lmax + 1]),
            axis=-1,
        )
        xims = jnp.sum(
            p_cl_prefactors_m[None, :, lmin : out_lmax + 1]
            * l
            * (
                pcls_e[:, None, lmin : out_lmax + 1]
                - pcls_b[:, None, lmin : out_lmax + 1]
                - 2j * pcls_eb[:, None, lmin : out_lmax + 1]
            ),
            axis=-1,
        )
    elif pcls_e.ndim == 3:
        xips = jnp.sum(
            p_cl_prefactors_p[None,None, :, lmin : out_lmax + 1]
            * l
            * (pcls_e[:, :, None, lmin : out_lmax + 1] + pcls_b[:, :, None, lmin : out_lmax + 1]),
            axis=-1,
        )
        xims = jnp.sum(
            p_cl_prefactors_m[None, None,:, lmin : out_lmax + 1]
            * l
            * (
                pcls_e[:, :, None, lmin : out_lmax + 1]
                - pcls_b[:, :, None, lmin : out_lmax + 1]
                - 2j * pcls_eb[:, :, None, lmin : out_lmax + 1]
            ),
            axis=-1,
        )
    else:
        print("pcls2xis: pcls has to be 2 or 3 dimensional, returning 1d pcl2xi")
        return pcl2xi(pcls, prefactors, out_lmax, lmin)
    

    return xips, xims


pcls2xis_jit = jax.jit(pcls2xis, static_argnums=(2, 3))


def cl2xi(cl, ang_bin_in_deg, out_lmax, lmin=0):
    cl_e, cl_b = cl
    l = 2 * np.arange(lmin, out_lmax + 1) + 1
    wigner_int_p = lambda theta_in_rad: theta_in_rad * wigner.wigner_dl(
        lmin, out_lmax, 2, 2, theta_in_rad
    )
    wigner_int_m = lambda theta_in_rad: theta_in_rad * wigner.wigner_dl(
        lmin, out_lmax, 2, -2, theta_in_rad
    )
    norm = 1 / (4 * np.pi)
    lower, upper = ang_bin_in_deg
    lower, upper = np.radians(lower), np.radians(upper)
    t_norm = 2 / (upper**2 - lower**2)
    integrated_wigners_p = quad_vec(wigner_int_p, lower, upper)[0]
    integrated_wigners_m = quad_vec(wigner_int_m, lower, upper)[0]
    xip = (
        t_norm
        * norm
        * np.sum(integrated_wigners_p * l * (cl_e[lmin : out_lmax + 1] + cl_b[lmin : out_lmax + 1]))
    )
    xim = (
        t_norm
        * norm
        * np.sum(integrated_wigners_m * l * (cl_e[lmin : out_lmax + 1] - cl_b[lmin : out_lmax + 1]))
    )

    return xip, xim


def check_property_equal(instances, property_name):
    """
    Check if a specific property of all instances is equal.

    Parameters:
        instances (list): A list of instances to check.
        property_name (str): The name of the property to check.

    Returns:
        bool: True if the property is equal for all instances, False otherwise.
    """
    if not instances:
        return True  # If the list is empty, return True

    first_value = getattr(instances[0], property_name)
    return all(getattr(instance, property_name) == first_value for instance in instances)


def cl2pseudocl(mllp, theorycls):
    # from namaster scientific documentation paper

    cl_es, cl_bs = [], []
    for theorycl in theorycls:
        if hasattr(theorycl, "_noise_sigma"):
            cl_e = theorycl.ee.copy() + theorycl.noise_cl
            cl_b = theorycl.bb.copy() + theorycl.noise_cl

        else:
            cl_e = theorycl.ee.copy()
            cl_b = theorycl.bb.copy()
        cl_es.append(cl_e)
        cl_bs.append(cl_b)

    mllp = jnp.array(mllp)
    cl_es, cl_bs = jnp.array(cl_es), jnp.array(cl_bs)
    p_ee, p_bb, p_eb = cl2pseudocl_einsum(mllp, cl_es, cl_bs)
    return jnp.array([p_ee, p_bb, p_eb])


@jax.jit
def cl2pseudocl_einsum(mllp, cl_e, cl_b):
    cl_eb = cl_be = cl_b
    p_ee = jnp.einsum("lm,nm->nl", mllp[0], cl_e) + jnp.einsum("lm,nm->nl", mllp[1], cl_b)
    p_bb = jnp.einsum("lm,nm->nl", mllp[1], cl_e) + jnp.einsum("lm,nm->nl", mllp[0], cl_b)
    p_eb = jnp.einsum("lm,nm->nl", mllp[0], cl_eb) - jnp.einsum("lm,nm->nl", mllp[1], cl_be)
    return p_ee, p_bb, p_eb


def setup_t(xi_max, steps):
    dts, t0s, ts = [], [], []

    for xi in xi_max:

        dt = 0.45 * 2 * np.pi / xi
        t0 = -0.5 * dt * (steps - 1)
        t = np.linspace(t0, -t0, steps - 1)
        dts.append(dt)
        t0s.append(t0)
        ts.append(t)

    t_inds = np.arange(len(t))
    t_sets = np.stack(np.meshgrid(ts[0], ts[1]), -1).reshape(-1, 2)

    return t_inds, t_sets, t0s, dts
