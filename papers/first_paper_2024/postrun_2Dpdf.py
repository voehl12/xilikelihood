import numpy as np
import calc_pdf
import matplotlib.pyplot as plt 
import configparser
import os
from pseudo_alm_cov import Cov
import plotting
from scipy.stats import multivariate_normal
from scipy.interpolate import RegularGridInterpolator
rng = np.random.default_rng()
import time
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath,amssymb}')

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

    if axis != 0:
        raise NotImplementedError("Only axis == 0 supported.")

    fiducial_output = func(data, axis=axis, **func_kwargs)

    if isinstance(data, list):
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
            idx = np.random.choice(data.shape[axis], size=data.shape[axis],
                                   replace=True)
            samples[i] = func(data[idx], axis=axis, **func_kwargs)
    print()
    return samples


fig, ((ax1,ax2,ax5),(ax3,ax4,ax6)) = plt.subplots(2,3,gridspec_kw=dict(width_ratios=[1,1,1]),figsize=(11,6.14))
axes = [ax1,ax2,ax3,ax4,ax5,ax6]



configpath = '/cluster/home/veoehl/2ptlikelihood/config_1024.ini'
config = configparser.ConfigParser()
config.read(configpath)

theory = config.items('Theory')
params = config['Params']


geom = config['Geometry']
angbins = config.items('Geometry')[2:]
angbins_in_deg = ((int(angbins[0][1]),int(angbins[1][1])),(int(angbins[2][1]),int(angbins[3][1])))
print(angbins_in_deg)
exact_lmax = int(params['l_exact'])
ell_buffer = int(params['l_buffer'])
noises = [theory[i+1][1] if theory[i+1][1] != 'None' else None for i in range(int(theory[0][1]))]

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


xi_combs=((1,1),(1,0))

filepath = "/cluster/work/refregier/veoehl/xi_sims/croco_3x2pt_kids_33_circ1000smoothl30_noisedefault/"
filepath_full = "/cluster/work/refregier/veoehl/xi_sims/croco_3x2pt_kids_33_circ1000smoothl30_noisedefault_llim_None/"

t0_2,dt_2,t_sets,ind_sets,cf_grid = plotting.read_2D_cf(configpath)


# calculate gaussian likelihoods:
mu,cov = calc_pdf.cov_xi_gaussian_nD(cov_objects,xi_combs=xi_combs,angbins_in_deg=angbins_in_deg, lmin=0,lmax=exact_lmax)
mu_all,cov_all = calc_pdf.cov_xi_gaussian_nD(cov_objects,xi_combs=xi_combs,angbins_in_deg=angbins_in_deg, lmin=0)
mu_high,cov_high = calc_pdf.cov_xi_gaussian_nD(cov_objects,xi_combs=xi_combs,angbins_in_deg=angbins_in_deg, lmin=exact_lmax+1)


# convert cf to pdf:
x_grid, pdf_grid = calc_pdf.cf_to_pdf_nd(cf_grid, t0_2, dt_2, verbose=True)
# marginal likelihoods:
pdf_53 = np.trapz(pdf_grid,x=x_grid[:,0,0],axis=0)
pdf_55 =  np.trapz(pdf_grid,x=x_grid[0,:,1],axis=1)
x_53, x_55 = x_grid[0,:,1], x_grid[:,0,0]
mu_53 = np.trapz(x_53 * pdf_53,x=x_53)
mu_55 = np.trapz(x_55 * pdf_55,x=x_55)


allxi1,allxi2 = plotting.read_sims_nd(filepath, [1,2],  (4, 6),1000,exact_lmax)
allxi1_full,allxi2_full = plotting.read_sims_nd(filepath_full, [1,2],  (4, 6),1024,767)

cov_estimate_low = np.cov(np.array([allxi1,allxi2]),ddof=1)
cov_estimate_full = np.cov(np.array([allxi1_full,allxi2_full]),ddof=1)

high_ell_cf = np.full_like(cf_grid,np.nan,dtype=complex)
vals = calc_pdf.high_ell_gaussian_cf_nD(t_sets,mu_high,cov_high)
for i,inds in enumerate(ind_sets):
    high_ell_cf[inds[0],inds[1]] = vals[i]
all_cf = cf_grid * high_ell_cf
x_grid_all, pdf_grid_all = calc_pdf.cf_to_pdf_nd(all_cf, t0_2, dt_2, verbose=True)
x_53_all, x_55_all = x_grid_all[0,:,1], x_grid_all[:,0,0]
pdf_53_all = np.trapz(pdf_grid_all,x=x_55_all,axis=0)
pdf_55_all =  np.trapz(pdf_grid_all,x=x_53_all,axis=1)
mu_53_all = np.trapz(x_53_all * pdf_53_all,x=x_53_all)
mu_55_all = np.trapz(x_55_all * pdf_55_all,x=x_55_all)

norm_2d_all = 1 / np.trapz(np.trapz(pdf_grid_all,x=x_55_all/mu_all[0],axis=0),x=x_53_all/mu_all[1])
norm_2d_low = 1 / np.trapz(np.trapz(pdf_grid,x=x_55/mu_all[0],axis=0),x=x_53/mu_all[1])

A = np.diag(1/mu_all)
cov = np.dot(A,np.dot(cov,A.T))
cov_all = np.dot(A,np.dot(cov_all,A.T))

cov_estimate = np.dot(A,np.dot(cov_estimate_low,A.T))
cov_estimate_all = np.dot(A,np.dot(cov_estimate_full,A.T))

var = multivariate_normal(mean=mu/mu_all, cov=cov)
var_all = multivariate_normal(mean=np.ones(2), cov=cov_all)

multvar_estimate = multivariate_normal(mean=mu/mu_all, cov=cov_estimate)
multvar_estimate_all = multivariate_normal(mean=np.ones(2), cov=cov_estimate_all)

sigmas = np.array([np.sqrt(cov[0,0]),np.sqrt(cov[1,1])])
sigmas_all = np.array([np.sqrt(cov_all[0,0]),np.sqrt(cov_all[1,1])])

sigmas_estimate = np.sqrt(np.diag(cov_estimate))
sigmas_all = np.sqrt(np.diag(cov_estimate_all))



vmax=4e12*norm_2d_low
vmax_all = 2.5e12*norm_2d_all
print(vmax,vmax_all)
vmax_comp = .1
lims = ((0.3e-6,2.2e-6),(0,1.2e-6))

colormap = plt.cm.get_cmap("twilight").copy()
#colormap = plt.get_cmap('twilight_shifted').copy()
colormap.set_bad('white')
#twilight = plt.get_cmap('twilight_shifted')
colormap = plotting.truncate_colormap(colormap,0.5,1.0,100)
maxcolor = plt.cm.twilight(np.linspace(0.5,1,6))

#fig, (ax1,ax2) = plt.subplots(1,2,figsize=(15,6))

f = ax4.hist2d(allxi1_full/mu_all[0],allxi2_full/mu_all[1],bins=256,density=True,vmin=0,vmax=vmax_all,cmap=colormap)


def bootstrap_statistic(data,axis=0,ddof=1):
    xi1,xi2 = data[:,0], data[:,1]
    d = np.histogram2d(xi1,xi2,bins=256,density=True,range=[[f[1][0], f[1][-1]], [f[2][0], f[2][-1]]])
    return d[0]
tic = time.perf_counter()
print('starting bootstrap')
n_bootstrap = 500
data = np.stack((allxi1/mu_all[0],allxi2/mu_all[1]),axis=-1)
res = bootstrap(data, func=bootstrap_statistic,n=n_bootstrap)

data_all = np.stack((allxi1_full/mu_all[0],allxi2_full/mu_all[1]),axis=-1)
res_all = bootstrap(data_all, func=bootstrap_statistic,n=n_bootstrap)

errors = np.std(res,axis=0,ddof=1)/(np.max(pdf_grid)*norm_2d_low)
errors_all = np.std(res_all,axis=0,ddof=1)/(np.max(pdf_grid_all)*norm_2d_all)
# plot these and the differences data-prediciton as 1d histograms
toc = time.perf_counter()
print(toc-tic)

d = ax2.hist2d(allxi1/mu_all[0],allxi2/mu_all[1],bins=256,density=True,vmin=0,vmax=vmax,range=[[f[1][0], f[1][-1]], [f[2][0], f[2][-1]]],cmap=colormap)


bincenters_x = [(d[1][i+1]+d[1][i])/2 for i in range(len(d[1])-1)]
bincenters_y = [(d[2][i+1]+d[2][i])/2 for i in range(len(d[2])-1)]

x_hist, y_hist = np.meshgrid(bincenters_x,bincenters_y)

pos_hist = np.dstack((x_hist,y_hist))

levels = np.flip(np.array([np.exp(-0.5*np.dot(i*sigmas,np.dot(np.linalg.inv(cov),i*sigmas))) for i in range(1,4)]) * np.max(var.pdf(pos_hist)))
levels_full = np.flip(np.array([np.exp(-0.5*np.dot(i*sigmas_all,np.dot(np.linalg.inv(cov_all),i*sigmas_all))) for i in range(1,4)]) * np.max(var_all.pdf(pos_hist)))

print(levels)
# plot levels from covariance estimates for simulated histograms
k = ax2.contour(x_hist, y_hist, multvar_estimate.pdf(pos_hist),levels=levels,vmin=0,vmax=vmax,colors='white',linestyles='dashed')
j = ax1.contour(x_hist, y_hist, var.pdf(pos_hist),levels=levels,vmin=0,vmax=vmax,colors='white',linestyles='dashed')
l = ax4.contour(x_hist, y_hist, multvar_estimate_all.pdf(pos_hist),levels=levels_full,vmin=0,vmax=vmax_all,colors='white',linestyles='dashed')
m = ax3.contour(x_hist, y_hist, var_all.pdf(pos_hist),levels=levels_full,vmin=0,vmax=vmax_all,colors='white',linestyles='dashed')

x, y = np.meshgrid(x_grid_all[:,0,0]/mu_all[0],x_grid_all[0,:,1]/mu_all[1])
pos_comp_all = np.dstack((x,y))
# absolute differences instead of relative, exact contour with gaussian contour
gauss_comp_all = (pdf_grid_all.T*norm_2d_all-multvar_estimate_all.pdf(pos_comp_all)) / (np.max(pdf_grid_all)*norm_2d_all)
#gauss_comp_all = np.ma.masked_where(var_all.pdf(pos_comp_all) < levels_full[0], gauss_comp_all)
#x, y = np.meshgrid(x_grid[:,0,0]/mu_all[0],x_grid[0,:,1]/mu_all[1])




pos_comp = np.dstack((x,y))
gauss_comp =  (pdf_grid.T*norm_2d_low) / var.pdf(pos_comp) 

u = ax3.contour(x,y,pdf_grid_all.T*norm_2d_all,vmax=vmax_all,levels=levels_full,colors=maxcolor[-2])
u1 = ax4.contour(x,y,pdf_grid_all.T*norm_2d_all,vmax=vmax_all,levels=levels_full,colors=maxcolor[-2])
x, y = np.meshgrid(x_grid[:,0,0]/mu_all[0],x_grid[0,:,1]/mu_all[1])
v = ax1.contour(x,y,pdf_grid.T*norm_2d_low,vmax=vmax,levels=levels,colors=maxcolor[-2])
v2 = ax2.contour(x,y,pdf_grid.T*norm_2d_low,vmax=vmax,levels=levels,colors=maxcolor[-2])
interp = RegularGridInterpolator((x_grid[:,0,0]/mu_all[0],x_grid[0,:,1]/mu_all[1]), pdf_grid*norm_2d_low)
interp_all = RegularGridInterpolator((x_grid_all[:,0,0]/mu_all[0],x_grid_all[0,:,1]/mu_all[1]), pdf_grid_all*norm_2d_all)
X, Y = np.meshgrid(bincenters_x, bincenters_y, indexing='ij')
im = interp((X,Y))
im_all = interp_all((X,Y))
diff_hist = np.fabs(d[0]-im)/(np.max(pdf_grid)*norm_2d_low)
diff_hist = np.ma.masked_where(d[0] == 0, diff_hist)

diff_hist_all = np.fabs(f[0]-im_all)/(np.max(pdf_grid_all)*norm_2d_all)
diff_hist_all = np.ma.masked_where(f[0] == 0, diff_hist_all)


errors = np.ma.masked_where(d[0] == 0, errors)
rel_res = diff_hist / errors

errors_all = np.ma.masked_where(f[0] == 0, errors_all)
rel_res_all = diff_hist_all / errors_all

# residual 1d histogram preparation 
errors = errors.compressed()
errors_all = errors_all.compressed()
diff_hist = diff_hist.compressed()
diff_hist_all = diff_hist_all.compressed()


twilight = plt.get_cmap('twilight_shifted')
twilight_half = plotting.truncate_colormap(twilight,0.5,1.0,100)
g = plotting.plot_2D(fig,ax1,x_grid[:,0,0]/mu_all[0],x_grid[0,:,1]/mu_all[1],pdf_grid.T*norm_2d_low,vmax=vmax,colormap=colormap)
colors = plt.cm.RdBu([0,0.2,0.9])
fig2, (ax21,ax22) = plt.subplots(1,2,figsize=(5,3))
print(len(diff_hist.flatten()),len(errors.flatten()))
plotting.add_data_1d(ax21,diff_hist.flatten(),colors[2],r'residuals',mean=True,density=False)
plotting.add_data_1d(ax21,3*errors.flatten(),colors[1],r'bootstrap $3 \sigma$',mean=True,range=(diff_hist.flatten().min(),diff_hist.flatten().max()),density=False)
plotting.add_data_1d(ax22,diff_hist_all.flatten(),colors[2],r'residuals',mean=True,density=False)
plotting.add_data_1d(ax22,3*errors_all.flatten(),colors[1],r'bootstrap $3 \sigma$',mean=True,range=(diff_hist_all.flatten().min(),diff_hist_all.flatten().max()),density=False)
#diff = plotting.plot_2D(fig,ax7,bincenters_x,bincenters_y,diff_hist.T,colormap=twilight_half,vmin=0.001,vmax=1,log=True)
#bootstrap_hist_std = plotting.plot_2D(fig,ax9,bincenters_x,bincenters_y,errors.T,colormap=twilight_half,vmin=0.001,vmax=1,log=True)
fig3, ax31 = plt.subplots(figsize=(5,4))
h = plotting.plot_2D(fig,ax3,x_grid_all[:,0,0]/mu_all[0],x_grid_all[0,:,1]/mu_all[1],pdf_grid_all.T*norm_2d_all,vmax=vmax_all,colormap=colormap)
diff_all = plotting.plot_2D(fig3,ax31,x_grid_all[:,0,0]/mu_all[0],x_grid_all[0,:,1]/mu_all[1],gauss_comp_all,colormap='twilight_shifted',vmin=-vmax_comp,vmax=vmax_comp,log=False)
rel_res_plot_all = plotting.plot_2D(fig,ax6,bincenters_x,bincenters_y,rel_res_all.T,vmax=3,colormap=colormap)
rel_res_plot = plotting.plot_2D(fig,ax5,bincenters_x,bincenters_y,rel_res.T,vmax=3,colormap=colormap)
#ax1.scatter(mu[0],mu[1],marker='*',color='C3')
#ax1.scatter(mu_55,mu_53,marker='*',color='C1')
#ax2.scatter(mu[0],mu[1],marker='*',color='C3')
#ax4.scatter(mu_all[0],mu_all[1],marker='*',color='C3')
#ax2.scatter(np.mean(allxi1),np.mean(allxi2),marker='*',color='C1')
#ax4.scatter(np.mean(allxi1_full),np.mean(allxi2_full),marker='*',color='C1')
#ax3.scatter(mu_all[0],mu_all[1],marker='*',color='C3')
#ax3.scatter(mu_55_all,mu_53_all,marker='*',color='C1')
mode_pos = np.unravel_index(pdf_grid_all.T.argmax(), pdf_grid_all.T.shape)

#ax3.scatter(x_grid_all[mode_pos[1],0,0],x_grid_all[0,mode_pos[0],1],marker='+',color='C3')
fig.subplots_adjust(wspace=0, hspace=0)
fig2.subplots_adjust(wspace=0)

#k = ax2.pcolormesh(d[1],d[2], var.pdf(pos),vmin=0)
fig.subplots_adjust(right=0.8)
axcolres = fig.add_axes([0.85, 0.12, 0.03, 0.75])
#ax6 = fig.add_axes([0.85, 0.11, 0.03, 0.35])
fig.colorbar(rel_res_plot_all, cax=axcolres,label=r'$\vert p_{\mathrm{analytic}} - p_{\mathrm{simulations}}\vert / \sigma_{\mathrm{bootstrap}}$')
#cbar = fig.colorbar(diff_all, cax=ax6)
#cbar.set_ticks([-.35,-.25,-.1,0,.1, .25, .35])
#cbar.set_ticklabels(['-35%','-25%','-10%','0%', '10%','25%','35%'])
#fig.subplots_adjust(top=0.8)
#ax9 = fig.add_axes([0.1,0.85,0.5,0.01])
#fig.colorbar(g,cax=ax9,orientation='horizontal')
#lims = ((-0.7e-6,2e-6),(-0.5e-6,1.2e-6))
lims_all = ((-5,8),(-3.7,6))
lims_low = ((-2,12),(-1,8))
low_ticks = [[0,2,4,6,8,10],[0,2,4,6]]
all_ticks = [[-4,-2,0,2,4,6],[-2,0,2,4]]
for ax_low in [ax1,ax2,ax5]:
    ax_low.set_yticks(low_ticks[1])
    ax_low.set_yticklabels(low_ticks[1])
    ax_low.set_xticks(low_ticks[0])
    ax_low.set_xticklabels(low_ticks[0])

plotting.set_xi_axes_2D(ax1,(4,6),((5,5),(3,5)),lims_low,x=False,binnum=2,islow=True)

plotting.set_xi_axes_2D(ax2,(4,6),((5,5),(3,5)),lims_low,y=False,x=False,binnum=2,islow=True)
plotting.set_xi_axes_2D(ax5,(4,6),((5,5),(3,5)),lims_low,y=False,x=False,binnum=2,islow=True)

#plotting.set_xi_axes_2D(ax7,(4,6),((5,5),(3,5)),lims_low,y=False,x=False)
#plotting.set_xi_axes_2D(ax9,(4,6),((5,5),(3,5)),lims_low,y=False,x=False)
for ax_all in [ax3,ax4,ax6]:
    ax_all.set_yticks(all_ticks[1])
    ax_all.set_yticklabels(all_ticks[1])
    ax_all.set_xticks(all_ticks[0])
    ax_all.set_xticklabels(all_ticks[0])
plotting.set_xi_axes_2D(ax3,(4,6),((5,5),(3,5)),lims_all,binnum=2)
plotting.set_xi_axes_2D(ax4,(4,6),((5,5),(3,5)),lims_all,y=False,binnum=2)
plotting.set_xi_axes_2D(ax6,(4,6),((5,5),(3,5)),lims_all,y=False,binnum=2)


fig3.subplots_adjust(right=0.8)

axcol = fig3.add_axes([0.85, 0.11, 0.03, 0.75])
cbar = fig3.colorbar(diff_all, cax=axcol)
plotting.set_xi_axes_2D(ax31,(4,6),((5,5),(3,5)),lims_all,binnum=2)
#plotting.rem_boundary_ticklabels(axes)
#plotting.set_xi_axes_2D(ax8,(4,6),((5,5),(3,5)),lims_all,y=False)$
ax21.set_ylim(1,1000)
#ax21.set_xlim(0,0.06)
ax21.set_yscale('log')

ax21.set_xlabel(r'normalized residuals, low $\ell$')
ax22.set_ylim(1,1000)
#ax22.set_xlim(0,0.1)
ax22.set_yscale('log')
ax22.legend(frameon=False)
ax22.set_yticklabels([])
ax22.set_xlabel(r'normalized residuals, full range')
fig2.savefig('residuals.pdf',bbox_inches='tight',format='pdf')
fig.savefig('joined_pdf_5535_res_newaxis.png',bbox_inches='tight',dpi=300)
fig3.savefig('gausscomp_2d_newaxis.png',bbox_inches='tight',dpi=300)


#np.savez('2D_pdf_2_3.npz'.format(exact_lmax),x_grid=x_grid,pdf_grid=pdf_grid)