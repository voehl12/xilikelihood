from simulate import xi_sim_nD
from cov_setup import Cov
from grf_classes import SphereMask, TheoryCl
import matplotlib.pyplot as plt
import sys
from random import randint
from time import time, sleep

sleep(randint(1,10))

jobnumber = int(sys.argv[1])
rootfolder = '/cluster/home/veoehl/2ptlikelihood/'
cl_55 = rootfolder+"Cl_3x2pt_kids55.txt"
cl_53 = rootfolder+"Cl_3x2pt_kids53.txt"
cl_33 = rootfolder+"Cl_3x2pt_kids33.txt"
cl_paths = (cl_33,cl_55,cl_53)
cl_names = ('3x2pt_kids_33','3x2pt_kids_55','3x2pt_kids_53')
batchsize = 1000
noise_contribs = ('default','default',None)
seps_in_deg = [(1,2),(4, 6),(7,10)]

mask = SphereMask(spins=[2], circmaskattr=(10000, 256), l_smooth=30)

sim_lmax = mask.lmax


theory_cls = [
            TheoryCl(sim_lmax, path, noise, clname=name)
            for path, name, noise in zip(cl_paths, cl_names, noise_contribs)
        ]

xi_lmax = 30



if jobnumber == 1:
    # get pcl for comparison:
    pcls = [Cov(mask, theorycl,exact_lmax=xi_lmax).cl2pseudocl() for theorycl in theory_cls]
    plot = True
else:
    plot = False

xi_sim_nD(theory_cls,[mask], jobnumber, seps_in_deg,lmin=0,lmax=xi_lmax,plot=plot,save_pcl=True,ximode='namaster',batchsize=batchsize,simpath="/cluster/scratch/veoehl/xi_sims/")