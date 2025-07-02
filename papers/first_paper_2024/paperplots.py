from distributions import pdf_xi_1D, cf_to_pdf_nd,cov_xi_gaussian_nD, mean_xi_gaussian_nD
from pseudo_alm_cov import Cov
from mask_props import SphereMask
from theory_cl import TheoryCl, RedshiftBin
import cl2xi_transforms
import matplotlib.pyplot as plt
import numpy as np
import plotting
import os
import scipy.stats as scistats
import file_handling
from postprocess_nd_likelihood import exp_norm_mean
from matplotlib import rc
from legacy.file_handling_v1 import read_xi_sims

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')
figdir = 'plots_paperone/

cov_params_masks = [{'sigma_e' : 'default' , 'clpath' : 'Cl_3x2pt_kids55.txt','clname' : "3x2pt_kids_55",'circmaskattr' : (1000,256),'l_smooth_mask' : 30,'cov_ell_buffer' : 10,'l_smooth_signal' : None},{'sigma_e' : 'default' , 'clpath' : 'Cl_3x2pt_kids55.txt','clname' : "3x2pt_kids_55",'circmaskattr' : (10000,256),'l_smooth_mask' : 30,'cov_ell_buffer' : 10}, {'sigma_e' : 'default' , 'clpath' : 'Cl_3x2pt_kids55.txt','clname' : "3x2pt_kids_55",'maskpath' : 'singlet_lowres.fits', 'maskname' : 'kids_lowres', 'l_smooth_mask' : 30,'cov_ell_buffer' : 10}]
masknames = [r'$1 \ 000 \ \mathrm{{deg}}^2$',r'$10 \ 000\ \mathrm{{deg}}^2$',r'KiDS']
diff_lmax = [50,70,100]
masks = [SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=50, l_smooth=30),SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=30, l_smooth=30),SphereMask(spins=[2], maskpath='singlet_lowres.fits', maskname='kids_lowres', exact_lmax=30, l_smooth=30)]
spins = [2]
cosmo = {'omega_m':0.31,'s8':0.8}
redshift_bin = RedshiftBin(5,filepath='redshift_bins/KiDS/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO5_Nz.txt')
theorycl = TheoryCl(masks[0].lmax,cosmo=cosmo,z_bins=(redshift_bin,redshift_bin),sigma_e='default')
theorycl_fromfile = TheoryCl(masks[0].lmax,clpath='Cl_3x2pt_kids55.txt',clname='3x2pt_kids_55',sigma_e='default')
theorycl_fromfile_nonoise = TheoryCl(masks[0].lmax,clpath='Cl_3x2pt_kids55.txt',clname='3x2pt_kids_55',sigma_e=None)
def fig1():
    fig, ax = plt.subplots(figsize=(5,4))
    
    lims = [(5e-6,1.4e-5),(0, 5e-6)]
    ang_bins_in_deg = [(0.5,1.0),(4,6)]
    exact_lmax = 30
    args = (exact_lmax,spins)
    colors=plt.cm.GnBu(np.linspace(0.5, 1.0, len(cov_params_masks)))[::-1]
    colors=plt.cm.twilight(np.linspace(0.2, 0.6, len(cov_params_masks)))[::-1]
    colors=plt.cm.twilight([0.6,0.8,0.2])
    plotting.set_xi_axes_hist(ax,None,(5,5),lims[1],islow=True)
    linestyles = ['dashed','solid']
    for i,mask in enumerate(masks):
        color = colors[i]
        cov_object = Cov(theorycl=theorycl,mask=mask,exact_lmax=exact_lmax)
        prefactors = cl2xi_transforms.prep_prefactors(ang_bins_in_deg,mask.wl,norm_lmax=mask.lmax,out_lmax=cov_object.exact_lmax)
        filepath = "/cluster/work/refregier/veoehl/xi_sims/3x2pt_kids_55_{}_noisedefault".format(cov_object.maskname)
        sims = read_xi_sims(filepath, 1000, ang_bins_in_deg,prefactors=prefactors,lmax=exact_lmax)
        x, pdf, stats = pdf_xi_1D(ang_bins_in_deg,(cov_object,),high_ell_extension=False,steps=4096)
        print(x,pdf)
        labels = [r'$\bar{{\theta}}_1 = [ {:.1f}^{{\circ}}, {:.1f}^{{\circ}}]$'.format(*ang_bins_in_deg[0]),r'$\bar{{\theta}}_2 = [ {:.1f}^{{\circ}}, {:.1f}^{{\circ}}]$'.format(*ang_bins_in_deg[1])] if i == 0 else [None,None]
        
        for j in range(len(ang_bins_in_deg)):
            ax.plot(x[j],pdf[j],color=color,linestyle=linestyles[j],label=labels[j])
        #fig.savefig(figdir+'figure1_step1.pdf',bbox_inches='tight')
        plotting.add_data_1d(ax,sims[0],color,name=masknames[i])
        plotting.add_data_1d(ax,sims[1],color,name=None)
    
    
    ax.legend(frameon=False)
    fig.savefig(figdir+'figure1.pdf',bbox_inches='tight')

def fig4():
    l_exact = [10,30,70]
    cov_params = cov_params_masks[0]
    maskname = masknames[0]
    ang_bins_in_deg = [(4,6)]
    colors = plt.cm.coolwarm([0.75,0.9,1.0])
    colors=plt.cm.RdBu([0,0.2,0.9])
    lims = (-2e-6, 2.5e-6)
    fig, ax = plt.subplots(1,1,figsize=(5,4))
    # too full, extra plot for l_exact convergence, only one l_exact for low ell/high ell
    for i,ell in enumerate(l_exact):
        lowlabel = r'low $\ell$' if  i == 0 else None
        highlabel = r'high $\ell$' if i == 0 else None
        exactlabel = r"$\ell_{{\mathrm{{exact}}}} = {:d}$".format(ell)
        color = colors[i]
        args = (ell,spins)
        cov_object = Cov(*args,**cov_params)
        x, pdf, stats = pdf_xi_1D(ang_bins_in_deg,(cov_object,),high_ell_extension=True,steps=4096)
        #x_low, pdf_low, stats_low = pdf_xi_1D(ang_bins_in_deg,(cov_object,),high_ell_extension=False,steps=4096)
        #mu_high, cov_high = cov_xi_gaussian_nD((cov_object,),((0,0),),ang_bins_in_deg, lmin=ell+1)
        #ax.plot(x_low[0],pdf_low[0],color=color,linestyle='dotted',label=lowlabel,alpha=0.5)
        #plotting.plot_gauss(ax,x[0],mu_high[0],cov_high[0,0],color,label=highlabel)
        ax.plot(x[0],pdf[0],color=color,label=exactlabel)

    filepath = "/cluster/work/refregier/veoehl/xi_sims/3x2pt_kids_55_{}_noisedefault".format(cov_object.maskname)
    sims = read_xi_sims(filepath, 1000, ang_bins_in_deg)
    plotting.add_data_1d(ax,sims[0],'gray',name=None,mean=False)
    plotting.set_xi_axes_hist(ax,ang_bins_in_deg[0],(5,5),lims,labels=True,binnum=2)
    ax.legend(frameon=False)
    fig.savefig(figdir+'ell_convergence.pdf',bbox_inches='tight')

def fig5():
    l_exact = 50
    cov_params = cov_params_masks[0]
    maskname = masknames[0]
    ang_bins_in_deg = [(4,6)]
    colors = plt.cm.coolwarm([0.75,0.9,1.0])
    colors=plt.cm.RdBu([0,0.2,0.9])
    lims = (-2e-6, 2.5e-6)
    fig, ax = plt.subplots(1,1,figsize=(5,4))
    mask = masks[0]
    
    lowlabel = r'low $\ell$' 
    highlabel = r'high $\ell$' 
    
    color = colors[0]
    args = (l_exact,spins)
    #cov_object = Cov(*args,**cov_params)
    cov_object = Cov(theorycl=theorycl_fromfile,mask=masks[0],exact_lmax=l_exact)
    cov_object.cl2pseudocl()
    cov_nonoise = Cov(theorycl=theorycl_fromfile_nonoise,mask=masks[0],exact_lmax=l_exact)
    x, pdf, stats = pdf_xi_1D(ang_bins_in_deg,(cov_object,),high_ell_extension=True,steps=4096,savestuff=False)
    x_low, pdf_low, stats_low = pdf_xi_1D(ang_bins_in_deg,(cov_object,),high_ell_extension=False,steps=4096,savestuff=False)
    x_nonoise, pdf_nonoise, stats_nonoise = pdf_xi_1D(ang_bins_in_deg,(cov_nonoise,),high_ell_extension=True,steps=4096,savestuff=False)
    print(x_low,pdf_low)
    cov_high = cov_xi_gaussian_nD((theorycl_fromfile,),((0,0),),ang_bins_in_deg, mask.eff_area,lmin=l_exact+1)
    prefactors = cl2xi_transforms.prep_prefactors(ang_bins_in_deg,mask.wl,norm_lmax=mask.lmax,out_lmax=mask.lmax)
    mean_high = mean_xi_gaussian_nD(prefactors,(cov_object.p_ee,cov_object.p_bb, cov_object.p_eb),lmin=l_exact+1,lmax=mask.lmax)
    plotting.set_xi_axes_hist(ax,ang_bins_in_deg[0],(5,5),lims,labels=True,binnum=2)
    ax.plot(x_low[0],pdf_low[0],color=color,linestyle='dotted',label=lowlabel,alpha=0.5)
    fig.savefig(figdir+'low_high_ell_1_new.pdf',bbox_inches='tight')
    plotting.plot_gauss(ax,x[0],mean_high[0],cov_high[0,0],color,label=highlabel)
    fig.savefig(figdir+'low_high_ell_2_new.pdf',bbox_inches='tight')
    ax.plot(x[0],pdf[0],color=color,label='convolution')
    fig.savefig(figdir+'low_high_ell_3_new.pdf',bbox_inches='tight')
    ax.plot(x_nonoise[0],pdf_nonoise[0],color=color,alpha=0.5,label=r'no noise')
    fig.savefig(figdir+'low_high_ell_3_nonoise.pdf',bbox_inches='tight')
    filepath = "/cluster/work/refregier/veoehl/xi_sims/3x2pt_kids_55_{}_noisedefault".format(cov_object.mask.name)
    sims, angles = file_handling.read_sims_nd(filepath, 1000, lmax=mask.lmax)
    print(sims.shape, angles)
    plotting.add_data_1d(ax,sims[:,1],'gray',name=None,mean=False)
    
    ax.legend(frameon=False)
    fig.savefig(figdir+'low_high_ell_4_new.pdf',bbox_inches='tight')



def fig2():
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(11,6.14),gridspec_kw=dict(height_ratios=[3,2]))
    axes = [[ax1,ax2],[ax3,ax4]]
    #colors=plt.cm.GnBu(np.linspace(0.5, 1.0, len(cov_params_masks)))[::-1]
    colors=plt.cm.twilight([0.6,0.8,0.2])
    lims = [(-1e-6,5e-6),(-1.25e-6, 2.0e-6)]
    ang_bins_in_deg = [(2,3),(4,6)]
    #ang_bins_in_deg = [(2,3),(4,6)]
    for i,params in enumerate(cov_params_masks):
        color = colors[i]
        e_lmax = diff_lmax[i]
        args = (e_lmax,spins)
        cov_object = Cov(*args,**params)
        filepath = "/cluster/work/refregier/veoehl/xi_sims/3x2pt_kids_55_{}_noisedefault".format(cov_object.maskname)
        prefactors = cl2xi_transforms.prep_prefactors(ang_bins_in_deg,cov_object.wl,norm_lmax=cov_object.lmax,out_lmax=cov_object.lmax)
        sims = read_xi_sims(filepath, 1000, ang_bins_in_deg,prefactors=prefactors)
        #print(sims.shape)
        x, pdf, stats = pdf_xi_1D(ang_bins_in_deg,(cov_object,),high_ell_extension=True,steps=4096)
        mu_approx, cov_approx = cov_xi_gaussian_nD((cov_object,cov_object,None),((0,0),(1,1)),ang_bins_in_deg)
        for j in range(len(ang_bins_in_deg)):
            maskname = masknames[i] if j == 1 else None
            #lowlabel = r'low $\ell$' if j == 0 and i == 0 else None
            #highlabel = r'high $\ell$' if j == 0 and i == 0 else None
            gausslabel = r'Gaussian likelihood,' '\n' r'analytic cov.' if j == 1 and i == 0 else None
            gausslabel_est = r'Gaussian likelihood' '\n' r'simulation-based cov.' if j == 1 and i == 0 else None
            exactlabel = r'exact likelihood' if j == 1 and i == 0 else None
            #plotting.plot_gauss(axes[0][j],x[j],mu_high[j],cov_high[j,j],color,label=highlabel)
            
            #axes[0][j].plot(x_low[j],pdf_low[j],color=color,linestyle='dotted',label=lowlabel,alpha=0.5)
            gauss_comp = scistats.norm.pdf(x[j], mu_approx[j],np.sqrt(cov_approx[j,j]))
            axes[0][j].plot(x[j],pdf[j],color=color,label=exactlabel)
            axes[1][j].plot(x[j],pdf[j],color=color)
            plotting.plot_gauss(axes[0][j],x[j],mu_approx[j],cov_approx[j,j],color,label=gausslabel)
            plotting.plot_gauss(axes[1][j],x[j],mu_approx[j],np.std(sims[j])**2,color,label=gausslabel_est,linestyle='dotted')
            plotting.plot_gauss(axes[1][j],x[j],mu_approx[j],cov_approx[j,j],color,label=None)
            
            gauss_masked = np.ma.masked_where(gauss_comp < 1, gauss_comp)
            reldiff = (pdf[j]-gauss_comp)/np.max(pdf[j])
            
    
            #plotting.plot_gauss(axes[0][j],x[j],mu_approx[j],cov_approx[j,j],'black')
            plotting.add_data_1d(axes[0][j],sims[j],color,name=maskname,mean=False)
            
            #axes[2][j].axvline(stats[j,0],color=color)
            #axes[2][j].plot(x[j],reldiff,color=color)
            
            #axes[1][j].axhline(0,color='black',linestyle='dotted')
    
    plotting.set_xi_axes_hist(ax3,ang_bins_in_deg[0],(5,5),lims[0],labels=True,binnum=3)
    plotting.set_xi_axes_hist(ax4,ang_bins_in_deg[1],(5,5),lims[1],labels=True,binnum=2)
    plotting.set_xi_axes_hist(ax1,ang_bins_in_deg[0],(5,5),lims[0],labels=False,binnum=3)
    plotting.set_xi_axes_hist(ax2,ang_bins_in_deg[1],(5,5),lims[1],labels=False,binnum=2)
    ax3.set_yscale('log')
    ax4.set_yscale('log')
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax1.set_ylim(0,1.6e6)
    ax2.set_ylim(0,2.75e6)
    ax3.set_ylim(100,5e6)
    ax4.set_ylim(100,5e6)
    #ax5.set_ylim(-.2,.2)
    #ax6.set_ylim(-.2,.2)
    #ax5.set_ylabel('abs. difference [max(PDF)]')
    #ax3.set_yticks([-.75,-.5,-.25,0,.25, .5, .75])
    #ax3.set_yticklabels(['-75%','-50%','-25%','0%', '25%','50%','75%'])
    #ax4.set_yticks([-.75,-.5,-.25,0,.25, .5, .75])
    #ax4.set_yticklabels(['-75%','-50%','-25%','0%', '25%','50%','75%'])
    fig.subplots_adjust(wspace=0.05, hspace=0)
    ax2.legend(frameon=False)
    ax4.legend(frameon=False,loc='upper right')
    fig.savefig(figdir+'figure2.pdf',bbox_inches='tight')


def fig3():
    pass


def low_ell_likelihood(fig,ax,cov_params,args,ang_bins_in_deg):
    
    
    cov_objects = [Cov(*args,**params) for params in cov_params]
    
    x, pdf, stats = pdf_xi_1D(ang_bins_in_deg,cov_objects,comb=(1,1),high_ell_extension=False,steps=256)
    x2, pdf2, stats2 = pdf_xi_1D(ang_bins_in_deg,cov_objects,comb=(1,0),high_ell_extension=False,steps=256)
    
    ax.plot(x[0],pdf[0],color='C0')
    ax.axvline(stats[0][0],color='C3')
    ax.plot(x2[0],pdf2[0],color='C1')
    ax.axvline(stats2[0][0],color='C3')
    ax.set_xlim(0,3e-6)
    ax.set_xlabel((r'$\xi^+ ({:3.1f}-{:3.1f} \degree)$'.format(*ang_bins_in_deg[0])))

def full_likelihood(ax,ang_bins_in_deg,args,cov_params,ximax=(2e-4,5e-6)):
    linestyles = ['solid','dotted','dashed']
    for p,params in enumerate(cov_params):
        linestyle = linestyles[p]
        cov_object = Cov(*args,**params)
        xs, pdfs, stats = pdf_xi_1D(ang_bins_in_deg,cov_object,high_ell_extension=True,xi_max=ximax)
        print(stats)
        colors = plt.cm.Blues(np.linspace(0.5, 1.0, len(ang_bins_in_deg)))
        for b,ang_bin in enumerate(ang_bins_in_deg):
            if b == 1:
                ax.plot(xs[b],pdfs[b],color=colors[b],linestyle=linestyle,label=r'{}, skews = {:.2f}, {:.2f}'.format(cov_object.maskname,np.real(stats[1][2]),np.real(stats[0][2])))
            else:
                ax.plot(xs[b],pdfs[b],color=colors[b],linestyle=linestyle)
    
   
def plot_skewness():
    from pseudo_alm_cov import Cov
    import distributions, mask_props
    from matplotlib.gridspec import GridSpec
    import scipy.stats as stats
    import plotting
    
    # for all masks we use. C_ell can be the same-
    lmax = [10,20,30,50,70,100]
    #lmax = [120]
    color = plt.cm.viridis(np.linspace(0, 1, len(lmax)))
    lims = [(-4e-6,7e-6),(-1.8e-6, 2.5e-6)]
    angbins = [(2,3),(4,6)]
    for p,params in enumerate(cov_params_masks[2:]):
        statistics = np.zeros((len(lmax),len(angbins),3),dtype=complex)
        
        fig1 = plt.figure(figsize=(22, 14))
        fig2 = plt.figure(figsize=(22, 14))
        gs = GridSpec(2, 3)
        ax1 = fig1.add_subplot(gs[0, :])
        ax2 = fig2.add_subplot(gs[0, :])

        args = (10,[2])
        cov_object = Cov(*args,**params)
        prefactors = cl2xi_transforms.prep_prefactors(angbins,cov_object.wl,norm_lmax=cov_object.lmax,out_lmax=cov_object.lmax)
        filepath = "/cluster/scratch/veoehl/xi_sims/3x2pt_kids_55_{}_noisedefault".format(cov_object.maskname)
        sims = read_xi_sims(filepath, 10, angbins,prefactors=prefactors)
        mean_measured, std_measured, skew_measured = np.mean(sims,axis=1), np.std(sims,axis=1),stats.skew(sims,axis=1)
        stats_measured = np.array([mean_measured, std_measured, skew_measured])
        for i, el in enumerate(lmax):
            args = (el,[2])
            cov_object = Cov(*args,**params)
            cov_object.cl2pseudocl()
            # add possibility to load this
            x, pdf, stats1 = distributions.pdf_xi_1D(
                angbins, (cov_object,), savestuff=True
            )
            ax1.plot(x[0], pdf[0], color=color[i], label=r"$\ell_{{\mathrm{{exact}}}} = {:d}$".format(el))
            ax2.plot(x[1], pdf[1], color=color[i], label=r"$\ell_{{\mathrm{{exact}}}} = {:d}$".format(el))
            ax1.set_xlim(lims[0])
            ax2.set_xlim(lims[1])
            statistics[i] = stats1
        # save x, pdf and stats for repeated use
        means, covs = distributions.cov_xi_gaussian_nD((cov_object,cov_object,None),((0,0),(1,1)),angbins)
        print(means,covs)
        ax1.set_title(r"$\Delta \theta = {:.1f}^{{\circ}} - {:.1f}^{{\circ}}$, $A_{{\mathrm{{eff}}}} = {:.1f} \mathrm{{sqd}}$".format(*angbins[0],cov_object.eff_area))
        ax2.set_title(r"$\Delta \theta = {:.1f}^{{\circ}} - {:.1f}^{{\circ}}$, $A_{{\mathrm{{eff}}}} = {:.1f} \mathrm{{sqd}}$".format(*angbins[1],cov_object.eff_area))
        ax1.plot(
            x[0],
            stats.norm.pdf(x[0], means[0], np.sqrt(covs[0,0])),
            label="Gaussian approximation",
            color="black",
            linestyle="dotted",
        )
        ax1.axvline(mean_measured[0],color='C0')
        ax1.axvline(means[0],color='C3')
        ax2.plot(
            x[1],
            stats.norm.pdf(x[1], means[1], np.sqrt(covs[1,1])),
            label="Gaussian approximation",
            color="black",
            linestyle="dotted",
        )
        ax2.axvline(mean_measured[1],color='C0')
        ax2.axvline(means[1],color='C3')
        
        statistics = np.abs(np.array(statistics))
        print(statistics)
        ax1 = plotting.plot_hist(ax1, sims[0], "test")
        ax2 = plotting.plot_hist(ax2, sims[1], "test")
        ax1.legend(), ax2.legend()
        ax1.set_xlabel(r"$\xi^+ (\Delta \theta)$")
        ax2.set_xlabel(r"$\xi^+ (\Delta \theta)$")

        plotting.add_stats(fig1,gs,lmax,statistics[:,0],stats_measured[:,0],means[0],covs[0,0])
        plotting.add_stats(fig2,gs,lmax,statistics[:,1],stats_measured[:,1],means[1],covs[1,1])

        ang1 = "{:.2f}_{:.2f}".format(*angbins[0])
        ang1 = ang1.replace('.','p')
        ang2 = "{:.2f}_{:.2f}".format(*angbins[1])
        ang2 = ang2.replace('.','p')
        
        fig1.savefig(figdir+"skewness{}_{}.png".format(cov_object.set_char_string()[4:-4],ang1))
        fig2.savefig(figdir+"skewness{}_{}.png".format(cov_object.set_char_string()[4:-4],ang2))
        # measured pseudo cell are too low for small angles



def app1():
    from simulate import TwoPointSimulation
    angles = [(0.5,1.0),(1,2),(2,3),(4,6)]
    smoothings = [None,100,300,500,700]
    colors=plt.cm.RdBu([0,0.2,0.4,0.7,0.9])
    fig, (ax1,ax2) = plt.subplots(2,1,gridspec_kw=dict(height_ratios=[3,1]),figsize=(5,5),sharex=True)
    angs = [np.mean(abin) for abin in angles]
    args = (30,spins)
    for s,smooth in enumerate(smoothings):
        color=colors[s]
        treecorrlabel = r'configuration space' if s == 0 else None
        namasterlabel = r'spherical harmonic space' if s == 0 else None
        smoothlabel = r"$\ell_{{\mathrm{{smooth}}}} = {:d}$".format(smooth) if smooth is not None else 'no smoothing'
        sim = TwoPointSimulation(angles,circmaskattr=(1000,256),l_smooth_mask=30,l_smooth_signal=smooth,clname='3x2pt_kids_55', clpath="Cl_3x2pt_kids55.txt", sigma_e='default',ximode='comp')
        cov_params = cov_params_masks[0]
        cov_params['l_smooth_signal'] = smooth
        cov_object = Cov(*args,**cov_params)
        sim.xi_sim_1D(1,save_pcl=False,pixwin=False,plot=False)
        
        mu_approx, cov_approx = cov_xi_gaussian_nD((cov_object,cov_object,None,cov_object,None,None,cov_object,None,None,None),((0,0),(1,1),(2,2),(3,3)),angles)
        
        reldiff = np.fabs(sim.comp[0]-sim.comp[1]) / np.sqrt(np.diag(cov_approx))
        ax1.plot(angs,sim.comp[0],label=treecorrlabel,color=color,linestyle='solid')
        ax1.plot(angs,sim.comp[1],label=namasterlabel,color=color,linestyle='dotted')
        ax2.plot(angs,reldiff,color=color,label=smoothlabel)
    ax2.set_yscale('log')
    
    ax2.set_xlabel(r'$\theta$ [deg]')
    ax2.set_ylabel(r'$|\hat{{\xi}}^+_{C_{\ell}} - \hat{{ \xi}}^+_{\mathrm{TC}}|$ / $\sigma$')
    ax1.set_ylabel(r'$\hat{{\xi}}^+_{{S5-S5}}$')
    ax2.axvline(0.458,color='black',linestyle='dashed')
    for bin in angles:
        for ang in bin:
            ax1.vlines(x=ang, ymin=0.0, ymax=0.2e-6, color='black',linestyle='dotted')
    fig.subplots_adjust(hspace=0)
    ax1.legend(frameon=False)
    handles, labels = ax2.get_legend_handles_labels()
    ax1.legend(handles,labels,frameon=False)
    
    #ax2.set_xticklabels(['1','2','3','4'])
    #ax1.set_xticklabels([])
    fig.savefig(figdir+'app1.pdf',bbox_inches='tight')

def app2():
    # skewnesses
    from pseudo_alm_cov import Cov
    import distributions, mask_props
    from matplotlib.gridspec import GridSpec
    import scipy.stats as stats
    import plotting
    # for all masks we use. C_ell can be the same-
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(11,5.07),gridspec_kw=dict(height_ratios=[1,1]))
    axes = [[ax1,ax3],[ax2,ax4]]
    lmax = [10,20,30,50,70,100]
    #lmax = [120]
    lims = [(-4e-6,7e-6),(-1.8e-6, 2.5e-6)]
    angbins = [(2,3),(4,6)]
    colors = plt.cm.RdBu([0,0.2,0.9])
    colors_masks = plt.cm.twilight([0.6,0.8,0.2])
    for p,params in enumerate(cov_params_masks):
        statistics = np.zeros((len(lmax),len(angbins),3),dtype=complex)
        
        args = (10,[2])
        cov_object = Cov(*args,**params)
        prefactors = cl2xi_transforms.prep_prefactors(angbins,cov_object.wl,norm_lmax=cov_object.lmax,out_lmax=cov_object.lmax)
        filepath = "/cluster/work/refregier/veoehl/xi_sims/3x2pt_kids_55_{}_noisedefault".format(cov_object.maskname)
        sims = read_xi_sims(filepath, 300, angbins,prefactors=prefactors)
        mean_measured, std_measured, skew_measured = np.mean(sims,axis=1), np.std(sims,axis=1),stats.skew(sims,axis=1)
        
        res = plotting.bootstrap(sims,500,func=plotting.all_stats,func_kwargs={'myaxis':1})
        mean_bootstrap, std_bootstrap, skew_bootstrap = res[:,0], res[:,1],res[:,2]
        stats_measured = np.array([mean_measured, std_measured, skew_measured])
        stats_bootstrap = np.array([np.std(mean_bootstrap,axis=0,ddof=1),np.std(std_bootstrap,axis=0,ddof=1),np.std(skew_bootstrap,axis=0,ddof=1)])
        for i, el in enumerate(lmax):
            args = (el,[2])
            cov_object = Cov(*args,**params)
            cov_object.cl2pseudocl()
            
            x, pdf, stats1 = distributions.pdf_xi_1D(
                angbins, (cov_object,), savestuff=True
            )
         
            statistics[i] = stats1
       
        means, covs = distributions.cov_xi_gaussian_nD((cov_object,cov_object,None),((0,0),(1,1)),angbins)
        statistics = np.abs(np.array(statistics))
        ylabel=True
        for j in range(len(angbins)):
            maskname = masknames[p] if j == 0 else None
            plotting.add_stats(axes[j],lmax,statistics[:,j],stats_measured[:,j],means[j],covs[j,j],color=colors_masks[p],bootstraps=stats_bootstrap[:,j],maskname=maskname,ylabel=ylabel)
            ylabel=False
            if j == 0 and p == 0:
                ax3.legend(frameon=False,loc='upper right')
        

    ax1.legend(frameon=False,loc='lower right')
    ax4.yaxis.set_label_position("right")
    ax4.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    fig.subplots_adjust(hspace=0,wspace=0.05)
    fig.savefig(figdir+"stats.pdf",bbox_inches='tight')

def app3():
    mask = SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=30, l_smooth=30)
    ang_bins_in_deg = [(0.5,1.0),(2,3),(4,6)]
    #cosmo = {'omega_m':0.31,'s8':0.8}
    #redshift_bin = RedshiftBin(5,filepath='redshift_bins/KiDS/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO5_Nz.txt')
    #theorycl = TheoryCl(mask.lmax,cosmo=cosmo,z_bins=(redshift_bin,redshift_bin))
    prefactors = cl2xi_transforms.prep_prefactors(ang_bins_in_deg,mask.wl,norm_lmax=mask.lmax,out_lmax=mask.lmax)
    plotting.plot_kernels(prefactors=prefactors,save_path="app3.pdf",ang_bins=ang_bins_in_deg)

def app4():
    
    from likelihood import XiLikelihood, fiducial_dataspace
    from mask_props import SphereMask
    colors = plt.cm.RdBu([0,0.2,0.9])
    exact_lmax = 30
    fiducial_cosmo = {
        
        "omega_m": 0.31,  # Matter density parameter
        "s8": 0.8,  # Amplitude of matter fluctuations
    }

    mask = SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=exact_lmax, l_smooth=30)


    redshift_bins, ang_bins_in_deg = fiducial_dataspace()

    rs = np.array([2,4])
    ab = np.array([3])
    rs_selection = [redshift_bins[i] for i in rs]
    ab_selection = [(4,6)]
    print(ab_selection)
    
    likelihood_nonoise = XiLikelihood(
            mask=mask, redshift_bins=rs_selection, ang_bins_in_deg=ab_selection,noise=None)
    likelihood_noise = XiLikelihood(
            mask=mask, redshift_bins=rs_selection, ang_bins_in_deg=ab_selection,noise='default')
    likelihood_nonoise.initiate_mask_specific()
    likelihood_noise.initiate_mask_specific()
    likelihood_nonoise.precompute_combination_matrices()
    likelihood_noise.precompute_combination_matrices()
    likelihood_nonoise._prepare_likelihood_components(fiducial_cosmo,highell=True)
    likelihood_noise._prepare_likelihood_components(fiducial_cosmo,highell=True)
    n_per_dim = likelihood_nonoise._xs.shape[-1]
    print(likelihood_noise._mean)
    print(likelihood_nonoise._mean)
    xs_nonoise,pdfs_nonoise = likelihood_nonoise._xs.reshape(-1,n_per_dim),likelihood_nonoise._pdfs.reshape(-1,n_per_dim)
    xs_noise,pdfs_noise = likelihood_noise._xs.reshape(-1,n_per_dim),likelihood_noise._pdfs.reshape(-1,n_per_dim)
    fig, ax = plt.subplots(1,1,figsize=(5,4))
    corr_labels = ['S3-S3','S5-S5','S3-S5']
    for i in range(1,3):
        color = colors[i]
        if i == 1:
            nonoiselabel = r'no noise'
            
        else:
            nonoiselabel = None
        noislabel = corr_labels[i]
        ax.plot(xs_nonoise[i],pdfs_nonoise[i],color=color,label=nonoiselabel,linestyle='dotted')
        ax.plot(xs_noise[i],pdfs_noise[i],color=color,label=noislabel,linestyle='solid')
        plotting.set_xi_axes_hist(ax,angbin=ab_selection[0],rs_bin=None,lims=(-0.4e-6,0.8e-6),labels=True,binnum=2)
    ax.legend(frameon=False)
    fig.savefig(figdir+'app4.pdf',bbox_inches='tight')
    fig.savefig(figdir+'app4.png',bbox_inches='tight',dpi=300)
    

    


def high_low_s8(n=100):
    from pseudo_alm_cov import Cov
    import distributions
    from simulate import TwoPointSimulation   
    import scipy.stats as stats
    
    
    lh_filename = 'likelihoods_wideranges8.npz'

    colors = plt.cm.RdBu([0,0.2,0.9])
    
    s8_ref = 0.8
    angbin = [(2,3)]
    mask = masks[1]
    print(mask.eff_area)
    # set up likelihood instance with correct mask and redshift bins, and angular sep bin, precompute for fine s8 grid?
    theorycl = TheoryCl(mask.lmax,cosmo=cosmo,z_bins=(redshift_bin,redshift_bin),sigma_e=None)
    if not os.path.isfile(lh_filename):
        
        s8 = np.linspace(0.4,1.2,num=65,endpoint=True)
        colors = plt.cm.viridis(np.linspace(0, 1, len(s8)))
        likelihood = []
        likelihood_gauss = []
        
        cov_bestfit = Cov(mask,theorycl,exact_lmax=30)
         
        _, covs_bestfit = distributions.cov_xi_gaussian_nD((cov_bestfit,),((0,0),),angbin)
        steps=4096
        exact_likelihood =  np.zeros((len(s8),steps-1))
        gauss_likelihood = np.zeros((len(s8),steps-1))
        xs = np.zeros((len(s8),steps-1))
        for i,(param,color) in enumerate(zip(s8,colors)):
            cosmo['s8'] = param
            theorycl = TheoryCl(mask.lmax,cosmo=cosmo,z_bins=(redshift_bin,redshift_bin),sigma_e=None)
            cov_s8 = Cov(mask,theorycl=theorycl,exact_lmax=30,working_dir='/cluster/work/refregier/veoehl')
            cov_s8.cl2pseudocl(ischain=True)
            x, pdf, statistics = distributions.pdf_xi_1D(
            angbin, (cov_s8,), steps=steps, savestuff=True)
            exact_likelihood[i] = pdf[0]
            xs[i] = x[0]
            
            means, covs = distributions.cov_xi_gaussian_nD((cov_s8,),((0,0),),angbin)
        
            gauss_likelihood[i] = stats.norm.pdf(x[0], means[0], np.sqrt(covs_bestfit[0,0]))
            
        np.savez(lh_filename, s8=s8,xs=xs,exact=exact_likelihood,gauss=gauss_likelihood)
        
    else:
        lh = np.load(lh_filename)
        s8 = lh['s8']
        xs = lh['xs']
        gauss_likelihood = lh['gauss']
        exact_likelihood = lh['exact']
    
    fiducial_lh_ind = np.argmin(np.fabs(s8-s8_ref))
    fiducial_x, fiducial_lh = xs[fiducial_lh_ind],exact_likelihood[fiducial_lh_ind]
    plt.figure()
    plt.plot(fiducial_x,fiducial_lh)
    plt.plot(fiducial_x,gauss_likelihood[fiducial_lh_ind],linestyle='dashed')
    plt.xlim(0.8e-6,2.2e-6)
    plt.savefig(figdir+'fiducial_likelihood.png')
    print(theorycl.ee)
    measurement = TwoPointSimulation(angbin,mask,theorycl,batchsize=n)
    jobnumber = 20
    #measurement.xi_sim_1D(jobnumber,plot=True)
    #print(measurement.simpath + "/job{:d}.npz".format(jobnumber))
    xisims = read_xi_sims(measurement.simpath,jobnumber,angbin)
    colors = plt.cm.viridis(np.linspace(0, 1, len(s8)))
    color = colors[2]
    means_exact, means_gauss = [],[]
    stds_exact, stds_gauss = [], []
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(11,3))
    pn = 0
    axes = [ax1,ax2,ax3]
    pplots = [49,8,52]
    for k in range(n):
        
        # could also pick these from an already acquired dataset (have that for noise free?)
        xip_measured = xisims[0,k]
        print(k,xip_measured)
        inds = np.argmin(np.fabs(xs-xip_measured),axis=1)
        # replace here with new likelihood framework, evaluate directly
        likelihood = exact_likelihood[np.arange(len(s8)),inds]
        
        likelihood_gauss = gauss_likelihood[np.arange(len(s8)),inds]
        norm, norm_gauss = np.trapz(likelihood, x=s8), np.trapz(likelihood_gauss, x=s8)
        likelihood /= norm
        likelihood_gauss /= norm_gauss
        mean_gauss = np.trapz(s8 * likelihood_gauss, x=s8)
        mean_exact = np.trapz(s8 * likelihood, x=s8)
        std_gauss = np.sqrt(np.trapz((s8-mean_gauss)**2 * likelihood_gauss, x=s8))
        std_exact = np.sqrt(np.trapz((s8-mean_exact)**2 * likelihood, x=s8))
        means_exact.append(mean_exact)
        means_gauss.append(mean_gauss)
        stds_exact.append(std_exact)
        stds_gauss.append(std_gauss)
        if k in pplots:
            ax = axes[pn]
            ax.plot(s8,likelihood,label=r'exact',color=color)
            ax.plot(s8,likelihood_gauss,label=r'Gaussian',color=color,linestyle='dashed')
            ax.axvline(s8_ref,color='C3')
            ax.axvline(mean_exact,color=color)
            ax.axvline(mean_gauss,color=color,linestyle='dashed')
            ax.set_xlabel(r'$S_8$')
            ax.set_xlim(0.55,1.05)
            if pn == 0:
                ax.set_ylabel(r'Posterior')
            
                ax.legend(frameon=False,loc='upper left')
            ax.set_yticklabels([])
            ax.set_yticks([])
            pn += 1
    fig.subplots_adjust(wspace=0.05, hspace=0)
    plt.savefig(figdir+'s8plots/combindeds8plot_wideprior.pdf',bbox_inches='tight')
    
    means_exact, means_gauss =  np.array(means_exact), np.array(means_gauss)
    stds_exact, stds_gauss = np.array(stds_exact), np.array(stds_gauss)
    means_filename = 'posterior_means.npz'
    """ if file_handling.check_for_file(means_filename):
        prev = np.load(means_filename)
        prev_exact, prev_gauss = prev['exact'], prev['gauss']
        means_exact = np.concatenate((prev_exact,means_exact))
        means_gauss = np.concatenate((prev_gauss,means_gauss)) """
    np.savez('posterior_means.npz', exact=means_exact,gauss=means_gauss)
    colors = plt.cm.RdBu([0,0.2,0.9])
    fig, ax = plt.subplots(figsize=(5, 4))
    plotting.add_data_1d(ax,means_exact,colors[2],r'exact likelihood',mean=True,density=False,nbins=20)
    plotting.add_data_1d(ax,means_gauss,colors[1],r'Gaussian likelihood',mean=True,density=False,nbins=20)
    ax.legend(frameon=False)
    ax.axvline(0.8,color='C3')
    ax.set_xlabel(r'S8')
    fig.savefig(figdir+'posterior_histograms_wideprior.pdf',bbox_inches='tight')

    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(5, 4))
    plotting.add_data_1d(ax1,(means_exact-means_gauss) / means_gauss,colors[2],r'difference',mean=True,density=False,nbins=15)
    ax1.set_xlabel(r'$\Delta \mu_{S_8}$ / $\mu_{S_8}^{\mathrm{Gauss}}$')
    #fig.savefig(figdir+'posterior_mean_differences.pdf',bbox_inches='tight')
    #ax.set_xlim(lims)
    #ax.axvline(xip_measured,color='C3',label='measured')
    #ax.set_xlabel(r'$\xi^+$')
    #ax.legend(frameon=False)

    
    plotting.add_data_1d(ax2,(stds_exact-stds_gauss) / stds_gauss,colors[1],r'difference',mean=True,density=False,nbins=15)
    ax2.set_xlabel(r'$\Delta \sigma_{S_8}$ / $\sigma_{S_8}^{\mathrm{Gauss}}$')
    #ax2.set_xlim(-0.1,0.1)
    fig.savefig(figdir+'posterior_mean_variance_difference_wideprior.pdf',bbox_inches='tight')
    
    
    
    #plt.savefig(figdir+'varied_s8_{:d}.pdf'.format(k),bbox_inches='tight')  
    

def plot_1d_posteriors():
    colors = plt.cm.viridis(np.linspace(0, 1, 65))
    s8_ref = 0.8
    n = 1000
    color = colors[2]
    means_exact, means_gauss = [],[]
    stds_exact, stds_gauss = [], []
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(11,3))
    pn = 0
    axes = [ax1,ax2,ax3]
    pplots = [49,8,52]
    for k in range(n):
        filestring = "/cluster/home/veoehl/2ptlikelihood/s8posts/s8post_firstpaper_10000sqd_nonoise_measurement{:d}.npz".format(k)

        posterior = file_handling.read_posterior_files(filestring)
        s8 = posterior['s8']
        gauss_posterior = posterior['gauss_posteriors']
        exact_posterior = posterior['exact_posteriors']
        
        s8_exact, post, mean, std = exp_norm_mean(s8,exact_posterior,reg=5)
        s8_gauss, post_gauss, mean_gauss, std_gauss = exp_norm_mean(s8,gauss_posterior,reg=5)
        means_exact.append(mean)
        means_gauss.append(mean_gauss)
        stds_exact.append(std)
        stds_gauss.append(std_gauss)
        print(mean,mean_gauss)
        # debugging section:
        if mean > 1.0:
            print('plotting posterior for k={}'.format(k))
            
            debugfig, ax = plt.subplots(1,1)
            ax.plot(s8_exact,post,label='exact',color=color)
            ax.plot(s8_gauss,post_gauss,label='Gaussian',color=color,linestyle='dashed')
            ax.axvline(s8_ref,color='C3')
            ax.axvline(mean,color=color)
            ax.axvline(mean_gauss,color=color,linestyle='dashed')
            ax.set_xlabel(r'$S_8$')
            ax.set_xlim(0.55,1.05)
            ax.set_ylabel(r'Posterior')
            ax.legend(frameon=False,loc='upper left')
            debugfig.savefig(figdir+'debug_nan_posterior_k{:d}.png'.format(k),bbox_inches='tight')
            plt.close(debugfig)
            continue
        
        if k in pplots:
            ax = axes[pn]
            ax.plot(s8_exact,post,label=r'exact',color=color)
            ax.plot(s8_gauss,post_gauss,label=r'Gaussian',color=color,linestyle='dashed')
            ax.axvline(s8_ref,color='C3')
            ax.axvline(mean,color=color)
            ax.axvline(mean_gauss,color=color,linestyle='dashed')
            ax.set_xlabel(r'$S_8$')
            ax.set_xlim(0.55,1.05)
            if pn == 0:
                ax.set_ylabel(r'Posterior')
            
                ax.legend(frameon=False,loc='upper left')
            ax.set_yticklabels([])
            ax.set_yticks([])
            pn += 1
    fig.subplots_adjust(wspace=0.05, hspace=0)
    fig.savefig(figdir+'s8plots/combindeds8plot_wideprior_smooth.pdf',bbox_inches='tight')

    means_exact, means_gauss =  np.array(means_exact), np.array(means_gauss)
    stds_exact, stds_gauss = np.array(stds_exact), np.array(stds_gauss)
    fig2, (ax1,ax2) = plt.subplots(1,2,figsize=(5, 4))
    colors = plt.cm.RdBu([0,0.2,0.9])
    plotting.add_data_1d(ax1,(means_exact-means_gauss) / means_gauss,colors[2],r'difference',mean=True,density=False,nbins=15)
    ax1.set_xlabel(r'$\Delta \mu_{S_8}$ / $\mu_{S_8}^{\mathrm{Gauss}}$')
    plotting.add_data_1d(ax2,(stds_exact-stds_gauss) / stds_gauss,colors[1],r'difference',mean=True,density=False,nbins=15)
    ax2.set_xlabel(r'$\Delta \sigma_{S_8}$ / $\sigma_{S_8}^{\mathrm{Gauss}}$')
    
    fig2.savefig(figdir+'posterior_mean_variance_difference_wideprior_smooth.pdf',bbox_inches='tight')
    
    
        


#fig2()
#cov_params_masks = [{'sigma_e' : 'default' , 'clpath' : 'Cl_3x2pt_kids55.txt','clname' : "3x2pt_kids_55",'circmaskattr' : (1000,256),'l_smooth_mask' : 30,'cov_ell_buffer' : 10},{'sigma_e' : 'default' , 'clpath' : 'Cl_3x2pt_kids55.txt','clname' : "3x2pt_kids_55",'circmaskattr' : (10000,256),'l_smooth_mask' : 30,'cov_ell_buffer' : 10}, {'sigma_e' : 'default' , 'clpath' : 'Cl_3x2pt_kids55.txt','clname' : "3x2pt_kids_55",'maskpath' : 'singlet_lowres.fits', 'maskname' : 'kids_lowres', 'l_smooth_mask' : 30,'cov_ell_buffer' : 10}]
#cov_params_masks = [{'sigma_e' : 'default' , 'clpath' : 'Cl_3x2pt_kids55.txt','clname' : "3x2pt_kids_55",'circmaskattr' : (1000,256),'l_smooth_mask' : 30,'cov_ell_buffer' : 10}]
#cov_params_masks = [{'sigma_e' : 'default' , 'clpath' : 'Cl_3x2pt_kids55.txt','clname' : "3x2pt_kids_55",'maskpath' : 'singlet_lowres.fits', 'maskname' : 'kids_lowres', 'l_smooth_mask' : 30,'cov_ell_buffer' : 10}]

#cov_params_masks = [{'sigma_e' : 'default' , 'clpath' : 'Cl_3x2pt_kids55.txt','clname' : "3x2pt_kids_55",'maskpath' : 'singlet_lowres.fits', 'maskname' : 'kids_lowres', 'l_smooth_mask' : 30,'cov_ell_buffer' : 10}]
def all_paperplots():

    fig1()
    fig2()
    fig4()
    fig5()
    app2()
    app1()




plot_1d_posteriors()





