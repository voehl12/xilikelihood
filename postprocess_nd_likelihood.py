import numpy as np
import plotting
import calc_pdf
import configparser
import file_handling
import itertools
from scipy.stats import moment

#class ExactLikelihood()

#class EdgeworthLikelihood()
def load_config(configpath):
    config = configparser.ConfigParser()
    config.read(configpath)
    return config


def convert_nd_cf_to_pdf(config):
    t0_2, dt_2, _, _, cf_grid = file_handling.read_2D_cf(config)
    x_grid, pdf_grid = calc_pdf.cf_to_pdf_nd(cf_grid, t0_2, dt_2, verbose=True)
    return x_grid, pdf_grid


def load_sims(config, simpath,njobs=1000):
    
    params = config["Params"]
    exact_lmax = int(params["l_exact"])
    # fix angbins to be flexible like in the config setup
    angbins_in_deg = get_angbins(config)
    comb_n = get_comb_ns(config)
    # read_sims: fix to be flexible like in the config setup for more than 2 dimensions
    print('Loading simulations...')
    allxis = file_handling.read_sims_nd(simpath, comb_n, angbins_in_deg[0], njobs, exact_lmax)

    
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
    return  [int(params["comb{:d}".format(i)]) for i in range(int(config["Run"]["ndim"]))]

def get_stats_from_sims(sims, orders=[1,2,3],axis=1):

    dims = np.arange(sims.shape[axis-1])
    n_sims = sims.shape[axis]

    # np.cov needs to have the data in the form of (n, m) where n is the number of variables and m is the number of samples
    def moments_nd(order,sims):
        if order == 1:
            return np.mean(sims, axis=axis)
        elif order == 2:
            if axis == 0:
                sims = sims.T
            return sims @ sims.T.conj() / n_sims

        else:
            
            combs = np.array(list(itertools.combinations_with_replacement(dims, order)))
            if axis == 0:
                sims = sims.T
            higher_moments = np.mean(np.prod(sims[combs], axis=1),axis=1)
            return np.ravel(higher_moments)

    stats = [moments_nd(order,sims) for order in orders]
    if len(orders) == 1:
        return np.array(stats)
    else:
        return stats



def setup_covs(config):
    theory = config.items('Theory')
    covs = []
    for i in range(int(theory[0][1])):
        cl_path = theory[i+4][1]
        cl_name = theory[i+4][0]
        area, nside = int(geom['area']), int(geom['nside'])
        new_cov = Cov(exact_lmax,
                [2],
                circmaskattr=(area,nside),
                clpath=cl_path,
                clname = cl_name,
                sigma_e=noises[i],
                l_smooth_mask=exact_lmax,
                l_smooth_signal=None,
                cov_ell_buffer=ell_buffer,)
        covs.append(new_cov)
    cov_objects = tuple(covs)
    return covs


def compare_to_gaussian(config):
    angbins_in_deg = get_angbins(config)
    params = config['Params']
    exact_lmax = int(params['l_exact'])
    xi_combs = [calc_pdf.get_combs(comb_n) for comb_n in get_comb_ns(config)]
    cov_objects = setup_covs(config)
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

    samples = np.zeros((n, *fiducial_output.shape),
                       dtype=fiducial_output.dtype)

    for i in range(n):
        print(i/n * 100,end='\r')
        if isinstance(data, list):
            idx = [np.random.choice(d.shape[0], size=d.shape[0], replace=True)
                   for d in data]
            samples[i] = func([d[i] for d, i in zip(data, idx)],
                              axis=axis, **func_kwargs)
        else:
            axes = np.arange(len(data.shape))
            indices = (1, Ellipsis, 1)
            idx = np.random.choice(data.shape[axis], size=data.shape[axis],
                                   replace=True)
            idx_array = tuple(idx if ax == axis else Ellipsis for ax in axes)
            #[np.arange(data.shape[ax]) if ax != axis else idx for ax in axes]
            samples[i] = func(data[idx_array], axis=axis, **func_kwargs)
    print()
    return samples