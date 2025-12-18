import matplotlib.pyplot as plt
import numpy as np
import os
import healpy as hp

from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')

cov_params_masks = [{'sigma_e' : 'default' , 'clpath' : 'Cl_3x2pt_kids55.txt','clname' : "3x2pt_kids_55",'circmaskattr' : (1000,256),'l_smooth_mask' : 30,'cov_ell_buffer' : 10,'l_smooth_signal' : None},{'sigma_e' : 'default' , 'clpath' : 'Cl_3x2pt_kids55.txt','clname' : "3x2pt_kids_55",'circmaskattr' : (10000,256),'l_smooth_mask' : 30,'cov_ell_buffer' : 10}, {'sigma_e' : 'default' , 'clpath' : 'Cl_3x2pt_kids55.txt','clname' : "3x2pt_kids_55",'maskpath' : 'singlet_lowres.fits', 'maskname' : 'kids_lowres', 'l_smooth_mask' : 30,'cov_ell_buffer' : 10}]
masknames = [r'$1 \ 000$ sqd',r'$10 \ 000$ sqd',r'KiDS']
diff_lmax = [50,70,100]

spins = [2]

def maps():
    import xilikelihood as xlh
    from xilikelihood.simulate import create_maps
    fig1, ax1 = plt.subplots()
    
    nside = 256
    area = 1000
    mask = xlh.SphereMask(spins=[2], circmaskattr=(area, nside), exact_lmax=30, l_smooth=30)
    rs_bins = xlh.theory_cl.load_kids_redshift_bins()
    noise='default'
    cosmology = {
        "omega_m": 0.31,  # Matter density parameter
        "s8": 0.8,  # Amplitude of matter fluctuations
    }
    c_ell = xlh.theory_cl.generate_theory_cl(767,[(rs_bins[4],rs_bins[4])],noise,cosmology)
    maps = create_maps([c_ell[0].ee],nside)    
    fig = hp.mollview(maps[1],cbar=False,cmap='RdBu',title=None,coord='GC',notext=True,rot=[-60,0,0])
    plt.savefig('map1000_55.png')
    maps_TQU_masked = mask.smooth_mask*maps[1]
    hp.mollview(maps_TQU_masked,title=None,fig=fig,cmap='RdBu',cbar=False,coord='GC',notext=True,rot=[-60,0,0])
    plt.savefig('map_masked1000_55.png')
    

def ell_convergence():
    l_exact = [10,30,70]
    cov_params = cov_params_masks[0]
    maskname = masknames[0]
    ang_bins_in_deg = [(4,6)]
    colors = plt.cm.coolwarm([0.75,0.9,1.0])
    colors=plt.cm.RdBu([0,0.2,0.9])
    lims = (-2e-6, 2.5e-6)
    fig, ax = plt.subplots(1,1,figsize=(4,3.2))
    filepath = "/cluster/work/refregier/veoehl/xi_sims/3x2pt_kids_55_circ1000smoothl30_noisedefault"
    sims = plotting.read_xi_sims(filepath, 1000, ang_bins_in_deg)
    plotting.add_data_1d(ax,sims[0],'gray',name=None,mean=False,nbins=512)
    plotting.set_xi_axes_hist(ax,ang_bins_in_deg[0],(5,5),lims,labels=True,binnum=2)
    fig.savefig('ell_convergence_1.pdf',bbox_inches='tight')
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
        ax.legend(frameon=False)
        fig.savefig('ell_convergence_{:d}.pdf'.format(i+2),bbox_inches='tight')


def low_high_ell():
    l_exact = 50
    cov_params = cov_params_masks[0]
    maskname = masknames[0]
    ang_bins_in_deg = [(4,6)]
    colors = plt.cm.coolwarm([0.75,0.9,1.0])
    colors=plt.cm.RdBu([0,0.2,0.9])
    lims = (-2e-6, 2.5e-6)
    fig, ax = plt.subplots(1,1,figsize=(4,3.2))
 
    
    lowlabel = r'low $\ell$' 
    highlabel = r'high $\ell$' 
    
    color = colors[0]
    args = (l_exact,spins)
    cov_object = Cov(*args,**cov_params)
    x, pdf, stats = pdf_xi_1D(ang_bins_in_deg,(cov_object,),high_ell_extension=True,steps=4096)
    x_low, pdf_low, stats_low = pdf_xi_1D(ang_bins_in_deg,(cov_object,),high_ell_extension=False,steps=4096)
    print(x_low,pdf_low)
    mu_high, cov_high = cov_xi_gaussian_nD((cov_object,),((0,0),),ang_bins_in_deg, lmin=l_exact+1)
    plotting.set_xi_axes_hist(ax,ang_bins_in_deg[0],(5,5),lims,labels=True,binnum=2)
    ax.plot(x_low[0],pdf_low[0],color=color,linestyle='dotted',label=lowlabel,alpha=0.5)
    ax.legend(frameon=False)
    fig.savefig('low_high_ell_1.pdf',bbox_inches='tight')
    plotting.plot_gauss(ax,x[0],mu_high[0],cov_high[0,0],color,label=highlabel)
    ax.legend(frameon=False)
    fig.savefig('low_high_ell_2.pdf',bbox_inches='tight')
    ax.plot(x[0],pdf[0],color=color,label='convolution')
    ax.legend(frameon=False)
    fig.savefig('low_high_ell_3.pdf',bbox_inches='tight')
    filepath = "/cluster/work/refregier/veoehl/xi_sims/3x2pt_kids_55_{}_noisedefault".format(cov_object.maskname)
    sims = plotting.read_xi_sims(filepath, 1000, ang_bins_in_deg)
    plotting.add_data_1d(ax,sims[0],'gray',name=None,mean=False)
    
    ax.legend(frameon=False)
    fig.savefig('low_high_ell_4.pdf',bbox_inches='tight')

def low_ell():
    fig, ax = plt.subplots(figsize=(4,3.2))
    
    lims = [(5e-6,1.4e-5),(0, 5e-6)]
    ang_bins_in_deg = [(0.5,1.0),(4,6)]
    exact_lmax = 30
    args = (exact_lmax,spins)
    colors=plt.cm.GnBu(np.linspace(0.5, 1.0, len(cov_params_masks)))[::-1]
    colors=plt.cm.twilight(np.linspace(0.2, 0.6, len(cov_params_masks)))[::-1]
    colors=plt.cm.twilight([0.6,0.8,0.2])
    plotting.set_xi_axes_hist(ax,None,(5,5),lims[1])
    linestyles = ['dotted','solid']
    for i,params in enumerate(cov_params_masks):
        color = colors[i]
        cov_object = Cov(*args,**params)
        prefactors = cl2xi_transforms.prep_prefactors(ang_bins_in_deg,cov_object.wl,norm_lmax=cov_object.lmax,out_lmax=cov_object.exact_lmax)
        filepath = "/cluster/work/refregier/veoehl/xi_sims/3x2pt_kids_55_{}_noisedefault".format(cov_object.maskname)
        sims = plotting.read_xi_sims(filepath, 1000, ang_bins_in_deg,prefactors=prefactors,lmax=exact_lmax)
        
        plotting.add_data_1d(ax,sims[0],color,name=masknames[i])
        plotting.add_data_1d(ax,sims[1],color,name=None)
        ax.legend(frameon=False)
        fig.savefig('figure1_step1_m{:d}.pdf'.format(i),bbox_inches='tight')
        
    
    for i,params in enumerate(cov_params_masks):
        color = colors[i]    
        cov_object = Cov(*args,**params)
        x, pdf, stats = pdf_xi_1D(ang_bins_in_deg,(cov_object,),high_ell_extension=False,steps=4096)
        labels = [r'$\bar{{\theta}}_1 = [ {:.1f}^{{\circ}}, {:.1f}^{{\circ}}]$'.format(*ang_bins_in_deg[0]),r'$\bar{{\theta}}_2 = [ {:.1f}^{{\circ}}, {:.1f}^{{\circ}}]$'.format(*ang_bins_in_deg[1])] if i == 0 else [None,None]
        for j in range(len(ang_bins_in_deg)):
            ax.plot(x[j],pdf[j],color=color,linestyle=linestyles[j],label=labels[j])
        ax.legend(frameon=False)
        fig.savefig('figure1_step2_m{:d}.pdf'.format(i),bbox_inches='tight')
        
    
def high_low_s8(n=100,plot_single=False):
    from pseudo_alm_cov import Cov
    import distributions
    from simulate import TwoPointSimulation   
    import scipy.stats as stats
    
    
        

    colors = plt.cm.RdBu([0,0.2,0.9])
    
    s8_ref = 0.8
    angbin = [(2,3)]
    lh_name = 'likelihoods_wideranges8.npz'
     
    lh = np.load(lh_name)
    s8 = lh['s8']
    xs = lh['xs']
    gauss_likelihood = lh['gauss']
    exact_likelihood = lh['exact']

    colors = plt.cm.viridis(np.linspace(0, 1, len(s8)))
    color = colors[2]
    
    s8_ex = np.linspace(0,len(s8)-1,5)
    s8_ex = [int(s8_ex[i]) for i in range(len(s8_ex))]
    s8_ex = [len(s8) // 2]

    fig, ax = plt.subplots(figsize=(4,3.2))
    ax.plot(s8,np.ones(len(s8)),color=color)
    ax.vlines([s8[0],s8[-1]],0,1,linestyles='dotted',colors=color)
    ax.set_xlabel(r'S8')
    ax.set_ylabel(r'Prior')
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_ylim(0,2)
    ax.set_xlim(0.5,1.1)
    fig.savefig('prior.pdf',bbox_inches='tight')

    measurement = TwoPointSimulation(angbin,circmaskattr=(10000,256),l_smooth_mask=30,s8=s8_ref,batchsize=100,simpath="/cluster/home/veoehl/2ptlikelihood/S8p8_circ10000smoothl30_nonoise_namaster",sigma_e=None )
    jobnumber = 10
    #measurement.xi_sim_1D(jobnumber)
    #print(measurement.simpath + "/job{:d}.npz".format(jobnumber))
    xisims = plotting.read_xi_sims(measurement.simpath,jobnumber,angbin)
    
    means_exact, means_gauss = [],[]
    stds_exact, stds_gauss = [], []
    
    for k in range(n):
        fig, ax = plt.subplots(figsize=(4,3.2))
        # could also pick these from an already acquired dataset (have that for noise free?)
        xip_measured = xisims[0,k]
        print(k,xip_measured)
        inds = np.argmin(np.fabs(xs-xip_measured),axis=1)
        
        likelihood = exact_likelihood[np.arange(len(s8)),inds]
        if k == 0:
            fig, ax = plt.subplots(figsize=(4,3.2))
            ax.set_xlim(5e-7,2e-6)
            for e in s8_ex:
                ax.plot(xs[e],exact_likelihood[e,:],color=color,label=s8[e])
                mean = np.trapz(xs[e] * exact_likelihood[e,:], x=xs[e])
                ax.axvline(mean,color=color)
            ax.set_xlabel(r'$\xi^+_{S5-S5} (\bar{\theta}_3))$')
            ax.set_ylabel(r'Likelihood')
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.legend(frameon=False)
            fig.savefig('exact_likelihood_s8_one_wideprior.pdf',bbox_inches='tight')
        
        likelihood_gauss = gauss_likelihood[np.arange(len(s8)),inds]
        if k == 0:
            fig, ax = plt.subplots(figsize=(4,3.2))
            ax.set_xlim(5e-7,2e-6)
            for e in s8_ex:
                ax.plot(xs[e],gauss_likelihood[e,:],color=color,label=s8[e])
                mean = np.trapz(xs[e] * gauss_likelihood[e,:], x=xs[e])
                ax.axvline(mean,color=color)
            ax.set_xlabel(r'$\xi^+_{S5-S5} (\bar{\theta}_3))$')
            ax.set_ylabel(r'Likelihood')
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.legend(frameon=False)
            fig.savefig('gauss_likelihood_s8_one_wideprior.pdf',bbox_inches='tight')
            
        
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
        if plot_single and k == 2:
            fig, ax = plt.subplots(figsize=(5,4.2))
            ax.set_xlim(0.55,1.1)
            ax.set_xlabel(r'S8')
            ax.set_ylabel(r'Posterior')
            
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.set_ylim(0,max(np.max(likelihood),np.max(likelihood_gauss)))
            ax.axvline(s8_ref,color='C3',label=r'fiducial')
            ax.legend(frameon=False,loc=1)
            fig.savefig('s8plots/varied_s8_step1_new.pdf',bbox_inches='tight')

            ax.plot(s8,likelihood,label=r'exact likelihood',color=color)
            ax.legend(frameon=False,loc=1)
            fig.savefig('s8plots/varied_s8_step2_new.pdf',bbox_inches='tight')

            ax.plot(s8,likelihood_gauss,label=r'Gaussian likelihood',color=color,linestyle='dashed')
            ax.legend(frameon=False,loc=1)
            fig.savefig('s8plots/varied_s8_step3_new.pdf',bbox_inches='tight')

            ax.axvline(mean_exact,color=color)
            ax.axvline(mean_gauss,color=color,linestyle='dashed')
            fig.savefig('s8plots/varied_s8_step4_new.pdf',bbox_inches='tight')

            #ax.get_legend().remove()
            #ax.set_xlabel('')
            #ax.set_ylabel('')
            fig.savefig('s8plots/varied_s8_{:d}.pdf'.format(k),bbox_inches='tight')
    
    means_exact, means_gauss =  np.array(means_exact), np.array(means_gauss)
    stds_exact, stds_gauss = np.array(stds_exact), np.array(stds_gauss)
    

    colors = plt.cm.RdBu([0,0.2,0.9])
    fig, ax = plt.subplots(figsize=(4, 3.2))
    plotting.add_data_1d(ax,means_exact,colors[2],r'exact likelihood',mean=True,density=False,nbins=20)
    plotting.add_data_1d(ax,means_gauss,colors[1],r'Gaussian likelihood',mean=True,density=False,nbins=20)
    ax.legend(frameon=False)
    ax.axvline(0.8,color='C3')
    ax.set_xlabel(r'S8')
    fig.savefig('posterior_histograms_talk_wideprior.pdf',bbox_inches='tight')

    fig, ax1 = plt.subplots(figsize=(4, 3.2))
    plotting.add_data_1d(ax1,(means_exact-means_gauss) / means_gauss,colors[2],r'difference',mean=True,density=False,nbins=20)
    ax1.set_xlabel(r'$\Delta \mu_{S_8}$ / $\mu_{S_8}^{\mathrm{Gauss}}$')
    fig.savefig('posterior_mean_differences_talk_wideprior.pdf',bbox_inches='tight')
    #ax.set_xlim(lims)
    #ax.axvline(xip_measured,color='C3',label='measured')
    #ax.set_xlabel(r'$\xi^+$')
    #ax.legend(frameon=False)

    
     
def gausscompare_1d():
    filepath = "/cluster/work/refregier/veoehl/xi_sims/croco_KiDS_setup_circ10000smoothl30_nonoise_llim_767"
    redshift_bins, ang_bins_in_deg = fiducial_dataspace()
    redshift_i = [2,4]
    ang_bin_i = -2
    comb = (4,4)
    
    redshift_bins = [redshift_bins[i] for i in redshift_i]
    ang_bins_in_deg = [ang_bins_in_deg[-2]]
    print(redshift_bins,ang_bins_in_deg)
    fiducial_cosmo = {
    "omega_m": 0.31,  # Matter density parameter
    "s8": 0.8,  # Amplitude of matter fluctuations
    }
    mask = SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=30, l_smooth=30)
    likelihood = XiLikelihood(mask=mask, redshift_bins=redshift_bins, ang_bins_in_deg=ang_bins_in_deg, noise=None)
    mapper = theory_cl.BinCombinationMapper(5)
    corr = mapper.get_index(comb)
    print('correlation:',corr)
    likelihood.initiate_mask_specific()
    likelihood.precompute_combination_matrices()
    likelihood._prepare_likelihood_components(fiducial_cosmo,highell=True)
    xs, pdfs = likelihood._xs, likelihood._pdfs
    exact_pdf = (xs[1,0], pdfs[1,0])
    data, _ = read_sims_nd(filepath, 1000, 767)
    print('loaded sims with shape:',data.shape)
    selected_data = data[:,corr,ang_bin_i]
    print('selected data shape:',selected_data.shape)
    fig, ax = plt.subplots(figsize=(4, 3.2))
    ax = plotting.plot_hist(ax, selected_data,name=None,exact_pdf=exact_pdf,fit_gaussian=True) 
    fig.savefig('gausscompare_1d_3.pdf',bbox_inches='tight')

    
def all_marginals():
    fiducial_cosmo = {
        "omega_m": 0.31,  # Matter density parameter
        "s8": 0.8,  # Amplitude of matter fluctuations
    }
    redshift_bins, ang_bins_in_deg = fiducial_dataspace()
    
    mask = SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=30, l_smooth=30)
    likelihood = XiLikelihood(mask=mask, redshift_bins=redshift_bins, ang_bins_in_deg=ang_bins_in_deg, noise=None)
    likelihood.initiate_mask_specific()
    likelihood.precompute_combination_matrices()
    likelihood._prepare_likelihood_components(fiducial_cosmo, highell=True)
    xs, pdfs = likelihood._xs, likelihood._pdfs
    mapper = BinCombinationMapper(likelihood._n_redshift_bins)
    # Plot all marginals
    n_corr, n_angs, _ = xs.shape
    for corr in range(n_corr):
        for i,ang in enumerate(ang_bins_in_deg):
            comb = mapper.get_combination(corr)
            fig, ax = plt.subplots(figsize=(4, 3.2))
            ax.plot(xs[corr, i], pdfs[corr, i], label=r'z: ({:d},{:d}), $\theta$: {:.2f}-{:.2f}'.format(*comb, *ang), color='C0')
            ax.set_xlabel(r'$\xi^+$')
            #ax.set_ylabel('PDF')
            ax.legend(frameon=False)
            #ax.set_title(f'Correlation {corr}, Angle {ang}')
            fig.savefig(f'marginal_corr{corr}_ang{i}.pdf', bbox_inches='tight')
            plt.close(fig)

maps()