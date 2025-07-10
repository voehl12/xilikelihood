from simulate import simulate_correlation_functions
from pseudo_alm_cov import Cov
from mask_props import SphereMask
from theory_cl import generate_theory_cl, prepare_theory_cl_inputs
import matplotlib.pyplot as plt
import sys
from random import randint
from time import time, sleep
from likelihood import fiducial_dataspace

sleep(randint(1,10))

jobnumber = int(sys.argv[1])

rootfolder = '/cluster/home/veoehl/xilikelihood/'
""" rootfolder = '/cluster/home/veoehl/xilikelihood/'
cl_55 = rootfolder+"Cl_3x2pt_kids55.txt"
cl_53 = rootfolder+"Cl_3x2pt_kids53.txt"
cl_33 = rootfolder+"Cl_3x2pt_kids33.txt"
cl_paths = (cl_33,cl_55,cl_53)
cl_names = ('3x2pt_kids_33','3x2pt_kids_55','3x2pt_kids_53')
batchsize = 1000
noise_contribs = ('default','default',None)
seps_in_deg = [(1,2),(4, 6),(7,10)] """
redshift_bins, angular_separation_bins = fiducial_dataspace()
n_redshift_bins = len(redshift_bins)
cosmo = {'omega_m': 0.31, 's8': 0.8}

mask = SphereMask(spins=[2], circmaskattr=(10000, 256), l_smooth=30)

sim_lmax = mask.lmax
numerical_combinations, redshift_bin_combinations, is_cov_cross, shot_noise = prepare_theory_cl_inputs(redshift_bins, noise=None)

theory_cls = generate_theory_cl(lmax=sim_lmax, redshift_bin_combinations=redshift_bin_combinations,shot_noise=shot_noise,cosmo=cosmo)

xi_lmax = sim_lmax
batchsize = 1000
if jobnumber == 1:
    plot = True
else:
    plot = False

simulate_correlation_functions(theory_cls,[mask], angular_separation_bins,job_id=jobnumber,lmin=0,lmax=xi_lmax,plot_diagnostics=plot,save_pcl=True,method='pcl_estimator',n_batch=batchsize,run_name="KiDS_setup",save_path="/cluster/scratch/veoehl/xi_sims")