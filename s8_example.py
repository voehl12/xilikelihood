# %%
import numpy as np
import scipy
from likelihood import XiLikelihood, fiducial_dataspace
from mask_props import SphereMask
from theory_cl import RedshiftBin
import os
import matplotlib.pyplot as plt
import re
from simulate import xi_sim_nD
from scipy.integrate import cumulative_trapezoid as cumtrapz
import sys
from random import randint
from time import time, sleep
from postprocess_nd_likelihood import exp_norm_mean
from calc_pdf import get_cov_n,get_combs
from copula_funcs import data_subset, cov_subset
import file_handling
from simulate import TwoPointSimulation
#sleep(randint(1, 5))

jobnumber = int(sys.argv[1]) - 1
exact_lmax = 30
fiducial_cosmo = {
    
    "omega_m": 0.31,  # Matter density parameter
    "s8": 0.8,  # Amplitude of matter fluctuations
}

""" mask = SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=exact_lmax, l_smooth=30)


redshift_bins, ang_bins_in_deg = fiducial_dataspace()
ang_bins_in_deg = ang_bins_in_deg[:-1]
likelihood = XiLikelihood(
        mask=mask, redshift_bins=redshift_bins, ang_bins_in_deg=ang_bins_in_deg,noise=None)
 """
#mock_data_path = "mock_data_10000sqd_nonoise.npz"
#gaussian_covariance_path = "gaussian_covariance_10000sqd_nonoise.npz"

ang_bin_in_deg = [(2,3)]
mask = SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=30, l_smooth=30)
redshift_bin = RedshiftBin(5,filepath='redshift_bins/KiDS/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO5_Nz.txt')
rs_bins = [redshift_bin]
likelihood = XiLikelihood(
            mask=mask, redshift_bins=rs_bins, ang_bins_in_deg=ang_bin_in_deg,noise=None)

mock_data_path = "mock_data_10000sqd_nonoise_firstpaper.npz"
gaussian_covariance_path = "gaussian_covariance_10000sqd_nonoise_firstpaper.npz"

      

def posterior_from_1d_autocorr(jobnumber):
    # jobnumber gives redshift-bin, angular separation bin pair, so as many jobs as there are redshift bins times angular separation bins are recommended.
    if jobnumber < 5:
        s8 = np.linspace(0.7, 0.9, 200)
    elif jobnumber < 10:
        s8 = np.linspace(0.6, 1.0, 200)
    else:
        s8 = np.linspace(0.4, 1.2, 200)
    num_redshift_bins = np.arange(len(redshift_bins))
    num_ang_bins = np.arange(len(ang_bins_in_deg))

    rs_grid,ang_grid = np.meshgrid(num_redshift_bins,num_ang_bins)
    pairs = np.vstack([rs_grid.ravel(), ang_grid.ravel()]).T
    this_pair = pairs[jobnumber]
    rs_bin = this_pair[0]
    ang_bin = this_pair[1]
    rs_bins = [redshift_bins[rs_bin]]
    ang_bin_in_deg = [ang_bins_in_deg[ang_bin]]
    likelihood = XiLikelihood(
            mask=mask, redshift_bins=rs_bins, ang_bins_in_deg=ang_bin_in_deg,noise=None)
    cosmology = fiducial_cosmo.copy()


    likelihood.initiate_mask_specific()
    likelihood.precompute_combination_matrices()
    n_comb = get_cov_n((rs_bin,rs_bin))
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

    post, mean = exp_norm_mean(s8,posts,reg=20)
    post_gauss, mean_gauss = exp_norm_mean(s8,gauss_posts,reg=20)


    np.savez(
        "/cluster/home/veoehl/2ptlikelihood/s8posts/s8post_10000sqd_fiducial_nonoise_1dcomb_{:d}_auto.npz".format(jobnumber),
        exact=post,
        gauss=post_gauss,
        s8=s8,
        means=[mean,mean_gauss],
        comb=[rs_bin,ang_bin]
    )

def create_mock_data(likelihood,mock_data_path,gaussian_covariance_path,random=None):
    """Create mock data for the likelihood analysis."""
    
    theory_cls = likelihood.initiate_theory_cl(fiducial_cosmo)
    likelihood.initiate_mask_specific()
    likelihood.precompute_combination_matrices()
    likelihood._prepare_matrix_products()

    likelihood.get_covariance_matrix_lowell()
    likelihood.get_covariance_matrix_highell()
    likelihood._get_means_highell()

    gaussian_covariance = likelihood._cov_highell + likelihood._cov_lowell
    fiducial_mean = likelihood._means_highell + likelihood._means_lowell
    if random is None:
        mock_data = fiducial_mean
    elif random == 'gaussian':
        mock_data = np.random.multivariate_normal(
            mean=likelihood._means_highell + likelihood._means_lowell,
            cov=likelihood._cov_highell + likelihood._cov_lowell,
            size=1,
        )
    elif random == 'frommap':
        sim = xi_sim_nD(
        theory_cls,
        [mask],
        42,
        ang_bins_in_deg,
        lmin=0,
        lmax=mask.lmax,
        plot=False,
        save_pcl=True,
        ximode="namaster",
        batchsize=1,    
    )
        mock_data = sim[0, :, 0, :]

    else:
        raise ValueError("Invalid random option. Choose 'gaussian' or 'frommap'.")
    # Save the covariance matrix and mock data to files
    np.savez(gaussian_covariance_path,cov=gaussian_covariance,s8=fiducial_cosmo['s8'])
    np.savez(mock_data_path,data=mock_data,s8=fiducial_cosmo['s8'])        
    print("Mock data and covariance matrix saved to {} and {}.".format(mock_data_path,gaussian_covariance_path))

def posterior_from_1d_croco(jobnumber,ns8=200):
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
    likelihood_local = XiLikelihood(
            mask=mask, redshift_bins=rs_bins, ang_bins_in_deg=ang_bin_in_deg,noise=None)
    
    
    
    likelihood_local.initiate_mask_specific()
    likelihood_local.precompute_combination_matrices()
    data_shape = likelihood_local.prep_data_array().shape
    cosmology = fiducial_cosmo.copy()
    num_angs = len(ang_bins_in_deg)
    subset_from_full_datavector = [(rs_combs[i],ang_bin) for i in range(len(rs_combs))]
    mockdata = data_subset(mock_data,subset_from_full_datavector)
    mockdata = mockdata.reshape(data_shape)
    likelihood_local.gaussian_covariance = cov_subset(gaussian_covariance,subset_from_full_datavector,num_angs)
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

    post, mean = exp_norm_mean(s8,posts,reg=20)
    post_gauss, mean_gauss = exp_norm_mean(s8,gauss_posts,reg=20)

    np.savez(
        "/cluster/home/veoehl/2ptlikelihood/s8posts/s8post_10000sqd_fiducial_nonoise_1dcomb_{:d}_croco.npz".format(jobnumber),
        exact=post,
        gauss=post_gauss,
        s8=s8,
        means=[mean,mean_gauss],
        comb=[croco_bin,ang_bin]
    )

def posterior_from_subset(jobnumber,ns8=200):
    # jobnumber gives prior region for s8, so as many jobs as there are splits are recommended.
    #  # get posterior from several subcombinations of data, need to determine which ones
    # focus on large angular scales, only large scales, only small scales?
    all_rs_combs = np.arange(likelihood._n_redshift_bin_combs)
    crocos = np.argwhere(likelihood._is_cov_cross).flatten()
    autos = np.argwhere(~likelihood._is_cov_cross).flatten()
    largest_bin = len(ang_bins_in_deg) - 1
    subset = [(rs,largest_bin) for rs in all_rs_combs]
    s8_prior = np.linspace(0.4, 1.2, ns8)
    split_prior = np.array_split(s8_prior, 100)
    this_s8_prior = split_prior[jobnumber]
    cosmology = fiducial_cosmo.copy()
    likelihood.initiate_mask_specific()
    likelihood.precompute_combination_matrices()
    likelihood.gaussian_covariance = gaussian_covariance
    posts, gauss_posts = [], []
    for s8 in this_s8_prior:
        print(s8)
        cosmology["s8"] = s8
        post, gauss_post = likelihood.loglikelihood(mock_data, cosmology, gausscompare=True,data_subset=subset)
        posts.append(post)
        gauss_posts.append(gauss_post)
    
    np.savez(
        "/cluster/home/veoehl/2ptlikelihood/s8posts/s8post_10000sqd_fiducial_nonoise_largescales_{:d}.npz".format(jobnumber),
        exact=posts,
        gauss=gauss_posts,
        s8=this_s8_prior,)


def posterior_from_nd(jobnumber):
    s8_prior = np.linspace(0.75,0.85, 200)
    split_prior = np.array_split(s8_prior, 100)
    this_s8_prior = split_prior[jobnumber]
    cosmology = fiducial_cosmo.copy()
    likelihood.initiate_mask_specific()
    likelihood.precompute_combination_matrices()
    likelihood.gaussian_covariance = gaussian_covariance
    posts, gauss_posts = [], []
    for s8 in this_s8_prior:
        print(s8)
        cosmology["s8"] = s8
        post, gauss_post = likelihood.loglikelihood(mock_data, cosmology, gausscompare=True)
        posts.append(post)
        gauss_posts.append(gauss_post)
    np.savez(
        "/cluster/home/veoehl/2ptlikelihood/s8posts/s8post_10000sqd_fiducial_nonoise_nd_{:d}.npz".format(jobnumber),
        exact=posts,
        gauss=gauss_posts,
        s8=this_s8_prior,
    )


  
def posterior_from_1d_firstpaper(jobnumber):
    
    s8 = np.linspace(0.4, 1.2, 200, endpoint=True)
    
    #measurement = TwoPointSimulation(angbin,mask,theorycl,batchsize=n)
    simsnumber = 20
    angbin = ang_bin_in_deg
    simpath = 'simulations/None_circ10000smoothl30_nonoise_namaster/'
    xisims = file_handling.read_xi_sims(simpath,simsnumber,angbin)
    cosmology = fiducial_cosmo.copy()
    xip_measured = xisims[0,jobnumber]
    xip_measured = xip_measured.reshape(1,1)
    likelihood.initiate_mask_specific()
    likelihood.precompute_combination_matrices()
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
        "/cluster/home/veoehl/2ptlikelihood/s8posts/s8post_firstpaper_10000sqd_nonoise_measurement{:d}.npz".format(jobnumber),
        exact=post,
        gauss=post_gauss,
        s8=s8
    )
#create_mock_data(likelihood,mock_data_path,gaussian_covariance_path,random=None)

#mock_data = np.load("fiducial_data_10000sqd_nonoise.npz")["data"]
#gaussian_covariance = np.load("gaussian_covariance_nonoise.npz")["cov"]
#create_mock_data(likelihood,mock_data_path,gaussian_covariance_path,random=None)

#mock_data = np.load(mock_data_path)["data"]
#print(mock_data.shape, mock_data)
gaussian_covariance = np.load(gaussian_covariance_path)["cov"]
#create_mock_data(likelihood,mock_data_path,gaussian_covariance_path,random=None)
posterior_from_1d_firstpaper(jobnumber)

# use posterior_script for slurm submission