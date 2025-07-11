"""
Legacy configuration-based analysis functions.

These functions were designed for an earlier version of the package
that used configparser for matching simulations and theory calculations.
They are preserved for backward compatibility but are not actively maintained.

For new analysis, use the modern likelihood_analysis package instead.
"""

import warnings
import configparser
import numpy as np

warnings.warn(
    "Legacy config-based analysis functions are deprecated. "
    "Use the modern likelihood_analysis package instead.",
    DeprecationWarning,
    stacklevel=2
)

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
        vals = distributions.high_ell_gaussian_cf_nD(t_sets, mu_high, cov_high)
        for i, inds in enumerate(ind_sets):
            high_ell_cf[inds[0], inds[1]] = vals[i]
        cf_grid *= high_ell_cf
    x_grid, pdf_grid = distributions.cf_to_pdf_nd(cf_grid, t0_2, dt_2, verbose=True)
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




 

