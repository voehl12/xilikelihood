import numpy as np
import plotting
import calc_pdf
import configparser
import file_handling
import itertools
from scipy.stats import moment


def load_config(configpath):
    config = configparser.ConfigParser()
    config.read(configpath)
    return config


def convert_nd_cf_to_pdf(config):
    t0_2, dt_2, _, _, cf_grid = file_handling.read_2D_cf(config)
    x_grid, pdf_grid = calc_pdf.cf_to_pdf_nd(cf_grid, t0_2, dt_2, verbose=True)
    return x_grid, pdf_grid


def compare_to_sims(config, simpath):
    angbins = config.items("Geometry")[2:]
    params = config["Params"]
    exact_lmax = int(params["l_exact"])
    # fix angbins to be flexible like in the config setup
    angbins_in_deg = (
        (int(angbins[0][1]), int(angbins[1][1])),
        (int(angbins[2][1]), int(angbins[3][1])),
    )
    comb_n = [params["comb{:d}".format(i)] for i in range(int(config["Run"]["ndim"]))]
    # read_sims: fix to be flexible like in the config setup for more than 2 dimensions
    allxis = file_handling.read_sims_nd(simpath, comb_n, angbins_in_deg[0], 1000, exact_lmax)

    pass


def get_stats_from_sims(sims):

    dims = np.arange(sims.shape[0])
    n_sims = sims.shape[1]

    # np.cov needs to have the data in the form of (n, m) where n is the number of variables and m is the number of samples
    def moments_nd(order):
        if order == 1:
            return np.mean(sims, axis=1)
        elif order == 2:

            return sims @ sims.T.conj() / n_sims

        else:
            higher_moments = []
            for comb in itertools.combinations_with_replacement(dims, order):
                moment = np.mean(np.prod([sims[i] for i in comb], axis=0))
                higher_moments.append(moment)
            return np.array(higher_moments)


def compare_to_gaussian(cov_objects):
    mu, cov = calc_pdf.cov_xi_gaussian_nD(
        cov_objects, xi_combs=xi_combs, angbins_in_deg=angbins_in_deg, lmin=0, lmax=exact_lmax
    )
    pass


def get_marginal_likelihoods(x_grid, pdf_grid):
    pdf_53 = np.trapz(pdf_grid, x=x_grid[:, 0, 0], axis=0)
    pdf_55 = np.trapz(pdf_grid, x=x_grid[0, :, 1], axis=1)
    x_53, x_55 = x_grid[0, :, 1], x_grid[:, 0, 0]
    mu_53 = np.trapz(x_53 * pdf_53, x=x_53)
    mu_55 = np.trapz(x_55 * pdf_55, x=x_55)
    pass


def normalize_pdfs():
    pass
