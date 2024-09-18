# setup n dimensional exact likelihood: combination matrices, covariance matrix and t/x grids
# write config file with all info, saves covariance and combination matrices
# as well as xi range

import numpy as np
import helper_funcs
import calc_pdf
import setup_m
from cov_setup import Cov
import configparser

config = configparser.ConfigParser()
l_exact = 30
ell_buffer = 10
ang_bins_in_deg = [(4,6),(4,6)]
area = 1000
nside = 256
cov_path = '/cluster/home/veoehl/2ptlikelihood/cov_xip_55_53_l{:d}_n{:d}_circ1000.npz'.format(l_exact,nside)
m_path = '/cluster/work/refregier/veoehl/m_matrices/m_xip{:d}_{:d}_xip{:d}_{:d}_l{:d}_n{:d}_circ1000.npz'.format(*ang_bins_in_deg[0],*ang_bins_in_deg[1],l_exact,nside)
tset_path = '/cluster/home/veoehl/2ptlikelihood/tsets.npz'
save_path = '/cluster/scratch/veoehl/2Dcf/xip_5535bins/'

cl_55 = "Cl_3x2pt_kids55.txt"
cl_53 = "Cl_3x2pt_kids53.txt"
cl_33 = "Cl_3x2pt_kids33.txt"
cl_paths = (cl_33,cl_55,cl_53)
cl_names = ('3x2pt_kids_33','3x2pt_kids_55','3x2pt_kids_53')
config['Paths'] = {'cov': cov_path,'M': m_path,'t_sets' : tset_path, 'result': save_path}
noise_contribs = ('default','default',None)

covs = (Cov(l_exact,
            [2],
            circmaskattr=(area,nside),
            clpath=cl_33,
            clname = '3x2pt_kids_33',
            sigma_e=noise_contribs[0],
            l_smooth_mask=l_exact,
            l_smooth_signal=None,
            cov_ell_buffer=ell_buffer,
        ), Cov(
            l_exact,
            [2],
            circmaskattr=(area,nside),
            clpath=cl_55,
            clname = '3x2pt_kids_55',
            sigma_e=noise_contribs[1],
            l_smooth_mask=l_exact,
            l_smooth_signal=None,
            cov_ell_buffer=ell_buffer,
        ), Cov(
            l_exact,
            [2],
            circmaskattr=(area,nside),
            clpath=cl_53,
            clname = '3x2pt_kids_53',
            sigma_e=noise_contribs[2], #! no noise for cross-C_ell!!
            l_smooth_mask=l_exact,
            l_smooth_signal=None,
            cov_ell_buffer=ell_buffer,
        ))
combs=((1,1),(1,0))
cov_mat = calc_pdf.cov_xi_nD(covs)
setup_m.save_m(cov_mat,cov_path)
prefactors = helper_funcs.prep_prefactors(ang_bins_in_deg,covs[0].wl, covs[0].lmax, covs[0].lmax)

m = setup_m.m_xi_cross((prefactors[0,:,:l_exact+1],prefactors[1,:,:l_exact+1]),combs=combs)
setup_m.save_m(m,m_path)
cov_triang = calc_pdf.get_cov_triang(covs)
xi_max = []
for j,comb in enumerate(combs):
    inds = calc_pdf.get_cov_pos(comb)
    cov = cov_triang[inds[0]][inds[1]]
    cov.cl2pseudocl()
    xip_estimate, _ = helper_funcs.pcl2xi((cov.p_ee,cov.p_bb,cov.p_eb),prefactors,l_exact)
    xi_max.append(np.fabs(xip_estimate[0])*6)
steps = 1024
t_inds, t_sets, t0_set, dt_set = calc_pdf.setup_t(xi_max,steps)
np.savez(tset_path,t=t_sets)

jobnum = 1024
batchsize = int(steps*steps / jobnum)
config['Run'] = {'jobnum': str(jobnum),'batchsize' : str(batchsize)}
config['Params'] = {'steps': steps, 'ximax1': str(xi_max[0]),'ximax2': str(xi_max[1]), 'l_exact': l_exact,'l_buffer': ell_buffer}
config['Geometry'] = {'area': area, 'nside': nside}
config['Theory'] = {'n_cl' : len(covs)}
for i in range(len(noise_contribs)):
    config.set('Theory','noise{:d}'.format(i),str(noise_contribs[i]))
for i in range(len(cl_names)):
    config.set('Theory', cl_names[i], cl_paths[i])
for i in range(len(ang_bins_in_deg)):
    config.set('Geometry','lower{:d}'.format(i),str(ang_bins_in_deg[i][0]))
    config.set('Geometry','upper{:d}'.format(i),str(ang_bins_in_deg[i][1]))

with open('config_{:d}.ini'.format(int(steps)), 'w') as configfile:
    config.write(configfile)
