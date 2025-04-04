import numpy as np
import plotting
import calc_pdf
import configparser
import file_handling
import itertools
from scipy.stats import moment, multivariate_normal
from scipy.interpolate import griddata
import matplotlib.pyplot as plt


# class ExactLikelihood()


# class EdgeworthLikelihood()
def load_config(configpath):
    config = configparser.ConfigParser()
    config.read(configpath)
    return config


def convert_nd_cf_to_pdf(config, highell_moms=None):
    t0_2, dt_2, t_sets, ind_sets, cf_grid = file_handling.read_2D_cf(config)
    if highell_moms is not None:
        mu_high, cov_high = highell_moms
        mu_high = np.array(mu_high).flatten()
        high_ell_cf = np.full_like(cf_grid, np.nan, dtype=complex)
        vals = calc_pdf.high_ell_gaussian_cf_nD(t_sets, mu_high, cov_high)
        for i, inds in enumerate(ind_sets):
            high_ell_cf[inds[0], inds[1]] = vals[i]
        cf_grid *= high_ell_cf
    x_grid, pdf_grid = calc_pdf.cf_to_pdf_nd(cf_grid, t0_2, dt_2, verbose=True)
    return x_grid, pdf_grid


def load_sims(config, simpath,lmax,njobs=1000):

    params = config["Params"]
    exact_lmax = int(params["l_exact"])
    # fix angbins to be flexible like in the config setup
    angbins_in_deg = get_angbins(config)
    comb_n = get_comb_ns(config)
    # read_sims: fix to be flexible like in the config setup for more than 2 dimensions
    print("Loading simulations...")
    allxis = file_handling.read_sims_nd(simpath, comb_n, angbins_in_deg[0], njobs, lmax)

    return allxis


def get_angbins(config):
    angbins = config.items("Geometry")[2:]
    angbins_in_deg = (
        (int(angbins[0][1]), int(angbins[1][1])),
        (int(angbins[2][1]), int(angbins[3][1])),
    )
    return angbins_in_deg


def get_comb_ns(config):
    params = config["Params"]
    return [int(params["comb{:d}".format(i)]) for i in range(int(config["Run"]["ndim"]))]


def get_stats_from_sims(sims, orders=[1, 2, 3], axis=1):

    dims = np.arange(sims.shape[axis - 1])
    n_sims = sims.shape[axis]
    print("number of simulations: " + str(n_sims))

    # np.cov needs to have the data in the form of (n, m) where n is the number of variables and m is the number of samples
    def moments_nd(order, sims):
        if order == 1:
            return np.mean(sims, axis=axis)
        elif order == 2:
            sims = sims - np.mean(sims, axis=axis)[:,None]
            if axis == 0:
                sims = sims.T
            
            cov = sims @ sims.T.conj() / n_sims
            np_cov = np.cov(sims, ddof=1)
            assert np.allclose(cov, np_cov), np_cov
            return cov

        else:

            combs = np.array(list(itertools.combinations_with_replacement(dims, order)))
            sims = sims - np.mean(sims, axis=axis)
            if axis == 0:
                sims = sims.T
            higher_moments = np.mean(np.prod(sims[combs], axis=1), axis=1)
            return np.ravel(higher_moments)

    stats = [moments_nd(order, sims) for order in orders]
    if len(orders) == 1:
        return np.array(stats)
    else:
        return stats


def setup_covs(config):
    theory = config.items("Theory")
    covs = []
    for i in range(int(theory[0][1])):
        cl_path = theory[i + 4][1]
        cl_name = theory[i + 4][0]
        area, nside = int(geom["area"]), int(geom["nside"])
        new_cov = Cov(
            exact_lmax,
            [2],
            circmaskattr=(area, nside),
            clpath=cl_path,
            clname=cl_name,
            sigma_e=noises[i],
            l_smooth_mask=exact_lmax,
            l_smooth_signal=None,
            cov_ell_buffer=ell_buffer,
        )
        covs.append(new_cov)
    cov_objects = tuple(covs)
    return covs


def compare_to_gaussian(config):
    angbins_in_deg = get_angbins(config)
    params = config["Params"]
    exact_lmax = int(params["l_exact"])
    xi_combs = [calc_pdf.get_combs(comb_n) for comb_n in get_comb_ns(config)]
    cov_objects = setup_covs(config)
    mu, cov = calc_pdf.cov_xi_gaussian_nD(
        cov_objects, xi_combs=xi_combs, angbins_in_deg=angbins_in_deg, lmin=0, lmax=exact_lmax
    )
    pass


def load_and_bootstrap_sims_nd(config, simpath, lmax,axes=None, vmax=None,n_bootstrap=500,diagnostic_ax=None):
    # finish nd version of this function
    
    sims = load_sims(config, simpath, lmax,njobs=1000)
    #mu, cov = [2.46969325e-07, 2.36668073e-07], np.array([[2.74045020e-14, 1.31405333e-14], [1.31405333e-14, 1.12209365e-14]])
    #[2.46969325e-07 2.36668073e-07] [[2.74045020e-14, 1.31405333e-14], [1.31405333e-14, 1.12209365e-14]]
    #mvn = multivariate_normal(mean=mu, cov=cov)
    #sims = mvn.rvs(1000000).T
    dims = len(sims)
    mu_estimate, cov_estimate = get_stats_from_sims(sims, orders=[1, 2])
    print('Estimated mean and cov: ')
    print(mu_estimate, cov_estimate)
    if dims == 2 and axes is not None:
        for ax in axes:
            hist = ax.hist2d(
                sims[0], sims[1], bins=128, density=True, vmin=0, vmax=vmax
            )
            density = hist[0]
            density_x = np.sum(density, axis=0)
            density_y = np.sum(density,axis=1)
            
            binedges = np.array(hist[1:3])
    else:
        density, binedges = np.histogramdd(sims.T, bins=128, density=True, vmin=0, max=vmax)

    # bincenters = [(d[i + 1] + d[i]) / 2 for i in range(len(d) - 1)]
    bincenters = np.array([0.5 * (edges[1:] + edges[:-1]) for edges in binedges])
    if diagnostic_ax is not None:
    
        diagnostic_ax.hist(sims[0],density=True,label='xi55_sims',bins=256)
        diagnostic_ax.hist(sims[1],density=True,label='xi53_sims',bins=256)
        
        

    def bootstrap_statistic(data, axis=0, ddof=1):

        f = np.histogramdd(
            data,
            bins=binedges,
            density=True,
        )
        return f[0]

    data = sims.T
    res = bootstrap(data, func=bootstrap_statistic, n=n_bootstrap)
    errors = np.std(res, axis=0, ddof=1)
    errors = np.ma.masked_where(density == 0, errors)
    mean = density

    return bincenters, mean, errors, mu_estimate, cov_estimate


def compare_to_sims_2d(axes, bincenters, sim_mean, sim_std, interp, vmax):

    bincenters_x, bincenters_y = bincenters
    X, Y = np.meshgrid(bincenters_x, bincenters_y, indexing="ij")

    exact_grid = interp((X, Y))
    diff_hist = np.fabs(sim_mean - exact_grid)
    exact = axes[1].contourf(X, Y, exact_grid, levels=100, vmax=vmax)
    rel_res = diff_hist / sim_std
    print("Mean deviation from simulations: {} std".format(np.mean(rel_res)))
    rel_res_plot = axes[2].contourf(X, Y, rel_res, levels=100, vmax=10)

    """ for ax in axes:
        ax.set_xlim(0.3e-6, 3e-6)
        ax.set_ylim(0, 1.8e-6) """
    return axes, rel_res_plot


def get_marginal_likelihoods(x_grid, pdf_grid):
    # x_53, x_55 = x_grid[0, :, 1], x_grid[:, 0, 0]
    x_55, x_53 = x_grid[0], x_grid[1]
    pdf_53 = np.trapz(pdf_grid, x=x_55, axis=0)
    pdf_55 = np.trapz(pdf_grid, x=x_53, axis=1)

    mu_53 = np.trapz(x_53 * pdf_53, x=x_53)
    mu_55 = np.trapz(x_55 * pdf_55, x=x_55)
    return pdf_55, pdf_53


def normalize_pdfs():
    pass

def exp_norm_mean(x,posterior,reg=350):
    posterior = np.array(posterior) - reg
    posterior = np.exp(posterior)
    integral = np.trapz(posterior[~np.isnan(posterior)], x=x[~np.isnan(posterior)])
    normalized_post = posterior / integral
    mean = np.trapz(x[~np.isnan(posterior)] * normalized_post[~np.isnan(posterior)], x=x[~np.isnan(posterior)])
    return normalized_post, mean

def bootstrap(data, n, axis=0, func=np.var, func_kwargs={"ddof": 1}):
    """Produce n bootstrap samples of data of the statistic given by func.

    Arguments
    ---------
    data : numpy.ndarray
        Data to resample.
    n : int
        Number of bootstrap trails.
    axis : int, optional
        Axis along which to resample. (Default ``0``).
    func : callable, optional
        Statistic to calculate. (Default ``numpy.var``).
    func_kwargs : dict, optional
        Dictionary with extra arguments for func. (Default ``{"ddof" : 1}``).

    Returns
    -------
    samples : numpy.ndarray
        Bootstrap samples of statistic func on the data.
    """

    fiducial_output = func(data, axis=axis, **func_kwargs)

    if isinstance(data, list):
        if axis != 0:
            raise NotImplementedError("Only axis == 0 supported.")
        assert all([d.shape[1:] == data[0].shape[1:] for d in data])

    samples = np.zeros((n, *fiducial_output.shape), dtype=fiducial_output.dtype)

    for i in range(n):
        print(i / n * 100, end="\r")
        if isinstance(data, list):
            idx = [np.random.choice(d.shape[0], size=d.shape[0], replace=True) for d in data]
            samples[i] = func([d[i] for d, i in zip(data, idx)], axis=axis, **func_kwargs)
        else:
            axes = np.arange(len(data.shape))
            indices = (1, Ellipsis, 1)
            idx = np.random.choice(data.shape[axis], size=data.shape[axis], replace=True)
            idx_array = tuple(idx if ax == axis else Ellipsis for ax in axes)
            # [np.arange(data.shape[ax]) if ax != axis else idx for ax in axes]
            samples[i] = func(data[idx_array], axis=axis, **func_kwargs)
    print()
    return samples
