"""
S8 posterior analysis using copula likelihood method.

Functions for computing S8 posteriors from various data subsets:
- 1D autocorrelations  
- Cross-correlations
- Multi-dimensional data
- Individual measurements
"""

import numpy as np
import sys
from time import time
from pathlib import Path
import xilikelihood as xlh
from mock_data_generation import create_mock_data

# Add package root to path
package_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_root))


# Configuration
EXACT_LMAX = 30
FIDUCIAL_COSMO = {"omega_m": 0.31, "s8": 0.8}
DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "data" / "s8posts"



def setup_likelihood_1d_example(create_data=False):
    """Set up standard likelihood configuration."""
    ang_bin_in_deg = [(2,3)]
    mask = xlh.SphereMask(spins=[2], circmaskattr=(10000, 256), 
                     exact_lmax=EXACT_LMAX, l_smooth=30)

    redshift_bin = xlh.RedshiftBin(5, filepath='redshift_bins/KiDS/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO5_Nz.txt')

    rs_bins = [redshift_bin]

    likelihood = xlh.XiLikelihood(
        mask=mask, redshift_bins=rs_bins,
        ang_bins_in_deg=ang_bin_in_deg, noise=None
    )
    mock_data_path = DATA_DIR+"mock_data_10000sqd_nonoise_firstpaper.npz"
    gaussian_covariance_path = DATA_DIR+"gaussian_covariance_10000sqd_nonoise_firstpaper.npz"
    data_paths = mock_data_path, gaussian_covariance_path
    if create_data:
        create_mock_data(likelihood, mock_data_path, gaussian_covariance_path, random='frommap')

    return likelihood, data_paths

def setup_likelihood_nd_example(create_data=False):
    mask = xlh.SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=EXACT_LMAX, l_smooth=30)


    redshift_bins, ang_bins_in_deg = xlh.fiducial_dataspace()
    ang_bins_in_deg = ang_bins_in_deg[:-1]
    likelihood = xlh.XiLikelihood(
        mask=mask, redshift_bins=redshift_bins, ang_bins_in_deg=ang_bins_in_deg,noise=None)

    mock_data_path = DATA_DIR+"mock_data_10000sqd_nonoise.npz"
    gaussian_covariance_path = DATA_DIR+"gaussian_covariance_10000sqd_nonoise.npz"
    data_paths = mock_data_path, gaussian_covariance_path
    if create_data:
        create_mock_data(likelihood, mock_data_path, gaussian_covariance_path, random='frommap')
    return likelihood, data_paths




     

def posterior_from_1d_autocorr(jobnumber,likelihood,data):
    # jobnumber gives redshift-bin, angular separation bin pair, so as many jobs as there are redshift bins times angular separation bins are recommended.
    if jobnumber < 5:
        s8 = np.linspace(0.7, 0.9, 200)
    elif jobnumber < 10:
        s8 = np.linspace(0.6, 1.0, 200)
    else:
        s8 = np.linspace(0.4, 1.2, 200)
    
    redshift_bins = likelihood.redshift_bins
    ang_bins_in_deg = likelihood.ang_bins_in_deg
    mask = likelihood.mask
    num_redshift_bins = np.arange(len(redshift_bins))
    num_ang_bins = np.arange(len(ang_bins_in_deg))

    rs_grid,ang_grid = np.meshgrid(num_redshift_bins,num_ang_bins)
    pairs = np.vstack([rs_grid.ravel(), ang_grid.ravel()]).T
    this_pair = pairs[jobnumber]
    rs_bin = this_pair[0]
    ang_bin = this_pair[1]
    rs_bins = [redshift_bins[rs_bin]]
    ang_bin_in_deg = [ang_bins_in_deg[ang_bin]]
    likelihood = xlh.XiLikelihood(
            mask=mask, redshift_bins=rs_bins, ang_bins_in_deg=ang_bin_in_deg,noise=None)
    cosmology = FIDUCIAL_COSMO.copy()

    mock_data, gaussian_covariance = data
    likelihood.setup_likelihood()
    mapper = likelihood._n_to_bin_comb_mapper
    n_comb = mapper.get_index((rs_bin,rs_bin))
    mockdata = mock_data[n_comb,ang_bin].reshape(1,1)
    n_cov = n_comb*len(ang_bins_in_deg) + ang_bin
    likelihood.gaussian_covariance = gaussian_covariance[n_cov,n_cov].reshape(1,1)
    posts, gauss_posts = [], []
    post, gauss_post = likelihood.loglikelihood(mockdata, cosmology, gausscompare=True)
    assert np.allclose(likelihood._mean,mockdata,rtol=1e-6), (likelihood._mean, mockdata)
    for s in s8:
        start_time = time()
        print(s)
        cosmology["s8"] = s
        post, gauss_post = likelihood.loglikelihood(mockdata, cosmology, gausscompare=True)
        iteration_time = time() - start_time
        print(f"Iteration for s8={s} took {iteration_time:.2f} seconds")
        posts.append(post)
        gauss_posts.append(gauss_post)

    post, mean = xlh.distributions.exp_norm_mean(s8,posts,reg=20)
    post_gauss, mean_gauss = xlh.distributions.exp_norm_mean(s8,gauss_posts,reg=20)


    np.savez(
        OUTPUT_DIR+'/s8posts/s8post_10000sqd_fiducial_nonoise_1dcomb_{:d}_auto.npz'.format(jobnumber),
        exact=post,
        gauss=post_gauss,
        s8=s8,
        means=[mean,mean_gauss],
        comb=[rs_bin,ang_bin]
    )


def posterior_from_1d_croco(jobnumber,likelihood,data,ns8=200):
    num_crocos = np.argwhere(likelihood._is_cov_cross)
    print(num_crocos)
    n_crocos = len(num_crocos)
    print(n_crocos)
    """ if jobnumber < n_crocos:
        s8 = np.linspace(0.7, 0.9, ns8)
    elif jobnumber < 2*n_crocos:
        s8 = np.linspace(0.6, 1.0, ns8)
    else: """
    s8 = np.linspace(0.4, 1.2, ns8)
    # convert num_crocos to a 1d array
    num_crocos = num_crocos.flatten()
    redshift_bins = likelihood.redshift_bins
    ang_bins_in_deg = likelihood.ang_bins_in_deg
    mask = likelihood.mask
    num_ang_bins = np.arange(len(ang_bins_in_deg))
    rs_grid,ang_grid = np.meshgrid(num_crocos,num_ang_bins)
    pairs = np.vstack([rs_grid.ravel(), ang_grid.ravel()]).T
    this_pair = pairs[jobnumber]
    croco_bin = this_pair[0]
    ang_bin = this_pair[1]
    mapper = likelihood._n_to_bin_comb_mapper
    croco = mapper.get_combination(croco_bin)
    # fix all the occurrences of calc_pdf comb stuff to use the new mapper in theory_cl
    auto1, auto2 = (croco[0], croco[0]), (croco[1], croco[1])
    rs_combs = (mapper.get_index(auto1),mapper.get_index(auto2),mapper.get_index(croco))
    print(croco,rs_combs)
    rs_bins = [redshift_bins[croco[0]], redshift_bins[croco[1]]]
    ang_bin_in_deg = [ang_bins_in_deg[ang_bin]]
    print(rs_bins, ang_bin_in_deg)
    # create a new likelihood object with the new redshift and angular separation bins
    likelihood_local = xlh.XiLikelihood(
            mask=mask, redshift_bins=rs_bins, ang_bins_in_deg=ang_bin_in_deg,noise=None)
    
    
    
    likelihood_local.setup_likelihood()
    data_shape = likelihood_local.prep_data_array().shape
    cosmology = FIDUCIAL_COSMO.copy()
    mock_data, gaussian_covariance = data
    num_angs = len(ang_bins_in_deg)
    subset_from_full_datavector = [(rs_combs[i],ang_bin) for i in range(len(rs_combs))]
    mockdata = xlh.copula_funcs.data_subset(mock_data,subset_from_full_datavector)
    mockdata = mockdata.reshape(data_shape)
    likelihood_local.gaussian_covariance = xlh.copula_funcs.cov_subset(gaussian_covariance,subset_from_full_datavector,num_angs)
    print(mockdata.shape,mockdata)
    print(likelihood_local.gaussian_covariance.shape,likelihood_local.gaussian_covariance)
    subset = [(2, 0)] #always the croco, only one ang bin
    posts, gauss_posts = [], []
    post, gauss_post = likelihood_local.loglikelihood(mockdata, cosmology, gausscompare=True)
    
    assert np.allclose(likelihood_local._mean,mockdata,rtol=1e-6), (likelihood_local._mean, mockdata)
    for s in s8:
        start_time = time()
        cosmology["s8"] = s
        post, gauss_post = likelihood_local.loglikelihood(mockdata, cosmology, gausscompare=True, data_subset=subset)
        iteration_time = time() - start_time
        print(f"Iteration for s8={s} took {iteration_time:.2f} seconds")
        posts.append(post)
        gauss_posts.append(gauss_post)

    post, mean = xlh.distributions.exp_norm_mean(s8,posts,reg=20)
    post_gauss, mean_gauss = xlh.distributions.exp_norm_mean(s8,gauss_posts,reg=20)

    np.savez(
        OUTPUT_DIR+'/s8post_10000sqd_fiducial_nonoise_1dcomb_{:d}_croco.npz'.format(jobnumber),
        exact=post,
        gauss=post_gauss,
        s8=s8,
        means=[mean,mean_gauss],
        comb=[croco_bin,ang_bin]
    )

def posterior_from_subset(jobnumber,likelihood,data,ns8=200):
    # jobnumber gives prior region for s8, so as many jobs as there are splits are recommended.
    #  # get posterior from several subcombinations of data, need to determine which ones
    # focus on large angular scales, only large scales, only small scales?
    all_rs_combs = np.arange(likelihood._n_redshift_bin_combs)
    crocos = np.argwhere(likelihood._is_cov_cross).flatten()
    autos = np.argwhere(~likelihood._is_cov_cross).flatten()
    ang_bins_in_deg = likelihood.ang_bins_in_deg
    largest_bin = len(ang_bins_in_deg) - 1
    subset = [(rs,largest_bin) for rs in all_rs_combs]
    s8_prior = np.linspace(0.4, 1.2, ns8)
    split_prior = np.array_split(s8_prior, 100)
    this_s8_prior = split_prior[jobnumber]
    cosmology = FIDUCIAL_COSMO.copy()
    likelihood.setup_likelihood()
    mock_data, gaussian_covariance = data
    likelihood.gaussian_covariance = gaussian_covariance
    posts, gauss_posts = [], []
    for s8 in this_s8_prior:
        print(s8)
        cosmology["s8"] = s8
        post, gauss_post = likelihood.loglikelihood(mock_data, cosmology, gausscompare=True,data_subset=subset)
        posts.append(post)
        gauss_posts.append(gauss_post)
    
    np.savez(
        OUTPUT_DIR+'/s8post_10000sqd_fiducial_nonoise_largescales_{:d}.npz'.format(jobnumber),
        exact=posts,
        gauss=gauss_posts,
        s8=this_s8_prior,)


def posterior_from_nd(jobnumber,likelihood,data):
    s8_prior = np.linspace(0.75,0.85, 200)
    split_prior = np.array_split(s8_prior, 100)
    this_s8_prior = split_prior[jobnumber]
    cosmology = FIDUCIAL_COSMO.copy()
    likelihood.setup_likelihood()
    mock_data, gaussian_covariance = data
    likelihood.gaussian_covariance = gaussian_covariance
    posts, gauss_posts = [], []
    for s8 in this_s8_prior:
        print(s8)
        cosmology["s8"] = s8
        post, gauss_post = likelihood.loglikelihood(mock_data, cosmology, gausscompare=True)
        posts.append(post)
        gauss_posts.append(gauss_post)
    
    np.savez(
        OUTPUT_DIR+'/s8post_10000sqd_fiducial_nonoise_nd_{:d}.npz'.format(jobnumber),
        exact=posts,
        gauss=gauss_posts,
        s8=this_s8_prior,
    )


  
def posterior_from_1d_firstpaper(jobnumber,likelihood,gaussian_covariance):
    
    s8 = np.linspace(0.4, 1.2, 200, endpoint=True)
    
    #measurement = TwoPointSimulation(angbin,mask,theorycl,batchsize=n)
    simsnumber = 20
    angbin = likelihood.ang_bins_in_deg
    simpath = 'simulations/None_circ10000smoothl30_nonoise_namaster/'
    xisims = xlh.file_handling.read_xi_sims(simpath,simsnumber,angbin)
    cosmology = FIDUCIAL_COSMO.copy()
    xip_measured = xisims[0,jobnumber]
    xip_measured = xip_measured.reshape(1,1)
    likelihood.setup_likelihood()
    likelihood.gaussian_covariance = gaussian_covariance[0,0].reshape(1,1)
    
    posts, gauss_posts = [], []
    for s in s8:
        start_time = time()
        cosmology["s8"] = s
        post, gauss_post = likelihood.loglikelihood(xip_measured, cosmology, gausscompare=True)
        print(post, gauss_post)
        iteration_time = time() - start_time
        print(f"Iteration for s8={s} took {iteration_time:.2f} seconds")
        posts.append(post)
        gauss_posts.append(gauss_post)

    post = np.array(posts)
    post_gauss = np.array(gauss_posts)


    np.savez(
        OUTPUT_DIR+'/s8posts/s8post_firstpaper_10000sqd_nonoise_measurement{:d}.npz'.format(jobnumber),
        exact=post,
        gauss=post_gauss,
        s8=s8
        )

# use posterior_script for slurm submission

if __name__ == "__main__":
    jobnumber = int(sys.argv[1]) - 1
    likelihood, data_paths = setup_likelihood_1d_example(create_data=False)
    mock_data_path, gaussian_covariance_path = data_paths
    #data = xlh.load_arrays(mock_data_path,"data")["data"]
    covariance = xlh.load_arrays(gaussian_covariance_path,"cov")["cov"]
    posterior_from_1d_firstpaper(jobnumber, likelihood, covariance)