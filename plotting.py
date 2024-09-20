import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import configparser
import calc_pdf
from helper_funcs import pcls2xis, prep_prefactors
import traceback
import matplotlib.colors as colors
import numpy as np
import scipy.stats as stats

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

    fiducial_output = func(data, **func_kwargs)

    if isinstance(data, list):
        assert all([d.shape[1:] == data[0].shape[1:] for d in data])

    samples = np.zeros((n, *fiducial_output.shape),
                       dtype=fiducial_output.dtype)

    for i in range(n):
        print(i)
        if isinstance(data, list):
            idx = [np.random.choice(d.shape[0], size=d.shape[0], replace=True)
                   for d in data]
            samples[i] = func([d[i] for d, i in zip(data, idx)],
                              **func_kwargs)
        else:
            idx = np.random.choice(data.shape[axis], size=data.shape[axis],
                                   replace=True)
            samples[i] = func(data[idx], **func_kwargs)

    return samples

def all_stats(sims,myaxis=0):
    return np.array([np.mean(sims,axis=myaxis), np.std(sims,axis=myaxis),stats.skew(sims,axis=myaxis)])


def rem_boundary_ticklabels(axes):
    
    for ax in axes:
        labels_x = ax.get_xticklabels()
        labels_y = ax.get_yticklabels()
        plt.setp(labels_x[0], visible=False)    
        plt.setp(labels_y[-1], visible=False)
        #ax.set_xticklabels(labels_x)
        #ax.set_yticklabels(labels_y)

def ticks_inside(ax):
    ax.tick_params(direction="in")

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap



def xi_sims_from_pcl(i,prefactors,filepath,lmax=None):
    
    pclfile = np.load(filepath+"/pcljob{:d}.npz".format(i))
    pcl_s = np.array([pclfile['pcl_e'],pclfile['pcl_b'],pclfile['pcl_eb']])
    xips,xims = pcls2xis(pcl_s,prefactors,out_lmax=lmax)
    
# make dictionary of initial xi-file and add this angbin and these xis
    return xips,xims




def read_xi_sims(filepath,njobs,angbins,kind="xip",prefactors=None,lmax=None):
    # should change order of going over files and angles
    allsims = []
    for j,angbin in enumerate(angbins):
        allxi=[]
        missing = []
        for i in range(1,njobs+1):
            
            if os.path.isfile(filepath+"/job{:d}.npz".format(i)):
                xifile = np.load(filepath+"/job{:d}.npz".format(i))
                angs = xifile["theta"]
                print(angs)
                if filepath[-2:] == 'rr':
                    angind = np.where(angs == np.mean(angbin))
                else:
                    angind = np.where(angs == angbin)
                print(angind)
                
                if lmax is not None or len(angind[0]) == 0:
                    try:
                        xip = xi_sims_from_pcl(i,prefactors,filepath,lmax=lmax)[0][:,j]
                    except FileNotFoundError:
                        print('Missing job number {:d}.'.format(i))
                        missing.append(i)
                        #traceback.print_exc()
                        xip =[]
                    except:
                        traceback.print_exc()
                        xip =[]
                else:
                    #xip = xifile[kind][:,angind[0][0]]
                    xip = xifile[kind][:,angind[0][0]]
                allxi += list(xip)
            else:
                print('Missing job number {:d}.'.format(i))
                missing.append(i)
        allxi = np.array(allxi)
        missing_string = ','.join([str(x) for x in missing])
        print(missing_string)
        allsims.append(allxi)
    #allxi = allxi.flatten()
    return np.array(allsims)

def read_pcl_sims(filepath,njobs):
    allpcl=[]
    missing = []
    for i in range(1,njobs+1):
        print(i)
        if os.path.isfile(filepath+"/pcljob{:d}.npz".format(i)):
            pclfile = np.load(filepath+"/pcljob{:d}.npz".format(i))
            pcl_s = np.array([pclfile['pcl_e'],pclfile['pcl_b'],pclfile['pcl_eb']])
            pcl_s = np.swapaxes(pcl_s,0,1)
            allpcl += list(pcl_s)
        else:
            print('Missing job number {:d}.'.format(i))
            missing.append(i)
    allpcl = np.array(allpcl)
    print(allpcl.shape)
    #allpcl.reshape(allpcl.shape[0]*allpcl.shape[1],3,768)
    return allpcl

def read_sims_nd(filepath,corr_num,angbin,njobs,lmax,kind='xip'):
    allxi1,allxi2=[],[]
    missing = []
    for i in range(1,njobs+1):
        if os.path.isfile(filepath+"/job{:d}.npz".format(i)):
            xifile = np.load(filepath+"/job{:d}.npz".format(i))
            assert lmax == int(xifile['lmax'])
            angs = xifile["theta"]
            angind = np.where(angs == angbin)
            xip1,xip2 = xifile[kind][:,corr_num[0],angind[0][0]], xifile[kind][:,corr_num[1],angind[0][0]]
            allxi1.append(xip1)
            allxi2.append(xip2)
        else:
            print('Missing job number {:d}.'.format(i))
            missing.append(i)
    allxi1 = np.concatenate(allxi1, axis=0)
    allxi2 = np.concatenate(allxi2, axis=0)
    
    missing_string = ','.join([str(x) for x in missing])
    print(missing_string)
    print(len(missing))
    return allxi1,allxi2

def set_xi_axes_2D(ax,angbin,rs_bins,lims,x=True,y=True,binnum=None,islow=False):

    if x:
        #ax.set_xlabel((r'$\xi^+_{{\mathrm{{S{:d}-S{:d}}}}} ({:3.1f}-{:3.1f} \degree)$'.format(*rs_bins[0],*angbin)))
        if binnum is not None:
            if islow:
                ax.set_xlabel((r'$\hat{{\xi}}^{{+,\mathrm{{low}}}}_{{\mathrm{{S{:d}-S{:d}}}}}$ / $\operatorname{{E}}[\hat{{\xi}}^{{+,\mathrm{{low}}}}_{{\mathrm{{S{:d}-S{:d}}}}}] (\bar{{\theta}}_{:d})$'.format(*rs_bins[0],*rs_bins[0],binnum)))  
                #ax.set_xlabel((r'$\hat{{\xi}}^{{+,\mathrm{{low}}}}_{{\mathrm{{S{:d}-S{:d}}}}} [\hat{{\xi}}^{{+}}] (\bar{{\theta}}_{:d})$'.format(*rs_bins[0],binnum)))   
            else:
                ax.set_xlabel((r'$\hat{{\xi}}^+_{{\mathrm{{S{:d}-S{:d}}}}}$ / $\operatorname{{E}}[\hat{{\xi}}^{{+}}_{{\mathrm{{S{:d}-S{:d}}}}}] (\bar{{\theta}}_{:d})$'.format(*rs_bins[0],*rs_bins[0],binnum)))   
                #ax.set_xlabel((r'$\hat{{\xi}}^+_{{\mathrm{{S{:d}-S{:d}}}}} [\hat{{\xi}}^{{+}}] (\bar{{\theta}}_{:d})$'.format(*rs_bins[0],binnum)))   
    if y:    
        #ax.set_ylabel((r'$\xi^+_{{\mathrm{{S{:d}-S{:d}}}}} ({:3.1f}-{:3.1f} \degree)$'.format(*rs_bins[1],*angbin)))
        if binnum is not None:
            if islow:
                ax.set_ylabel((r'$\hat{{\xi}}^{{+,\mathrm{{low}}}}_{{\mathrm{{S{:d}-S{:d}}}}}$ / $\operatorname{{E}}[\hat{{\xi}}^{{+,\mathrm{{low}}}}_{{\mathrm{{S{:d}-S{:d}}}}}] (\bar{{\theta}}_{:d})$'.format(*rs_bins[1],*rs_bins[1],binnum)))  
                #ax.set_xlabel((r'$\hat{{\xi}}^{{+,\mathrm{{low}}}}_{{\mathrm{{S{:d}-S{:d}}}}} [\hat{{\xi}}^{{+}}] (\bar{{\theta}}_{:d})$'.format(*rs_bins[1],binnum)))   
            else:
                ax.set_ylabel((r'$\hat{{\xi}}^+_{{\mathrm{{S{:d}-S{:d}}}}}$ / $\operatorname{{E}}[\hat{{\xi}}^{{+}}_{{\mathrm{{S{:d}-S{:d}}}}}] (\bar{{\theta}}_{:d})$'.format(*rs_bins[1],*rs_bins[1],binnum))) 
                #ax.set_ylabel((r'$\hat{{\xi}}^+_{{\mathrm{{S{:d}-S{:d}}}}} [\hat{{\xi}}^{{+}}] (\bar{{\theta}}_{:d})$'.format(*rs_bins[1],binnum)))   
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])
    if not x:
        #ax.set_xticklabels([])
        ax.xaxis.tick_top()
    if not y:
        ax.set_yticklabels([])

def set_xi_axes_hist(ax,angbin,rs_bin,lims,labels=True,binnum=None,islow=False):

    if not labels:
        ax.set_xticklabels([])
    elif angbin is None:
        if islow:
            ax.set_xlabel((r'$\hat{{\xi}}^{{+, \mathrm{{low}}}}_{{\mathrm{{S{:d}-S{:d}}}}}$'.format(*rs_bin)))
        else:
            ax.set_xlabel((r'$\hat{{\xi}}^+_{{\mathrm{{S{:d}-S{:d}}}}}$'.format(*rs_bin)))
    else:
        ax.set_xlabel((r'$\hat{{\xi}}^+_{{\mathrm{{S{:d}-S{:d}}}}} ({:3.1f}\degree-{:3.1f} \degree)$'.format(*rs_bin,*angbin)))
        if binnum is not None:
            if islow:
                ax.set_xlabel((r'$\hat{{\xi}}^{{+, \mathrm{{low}}}}_{{\mathrm{{S{:d}-S{:d}}}}} (\bar{{\theta}}_{:d})$'.format(*rs_bin,binnum)))
    
            else:
                ax.set_xlabel((r'$\hat{{\xi}}^+_{{\mathrm{{S{:d}-S{:d}}}}} (\bar{{\theta}}_{:d})$'.format(*rs_bin,binnum)))
    
    ax.set_xlim(*lims)
    



def plot_hist(ax,sims, name, color='C0', linecolor='C3', exact_pdf=None, label=False, fit_gaussian=False):

    
    #


    if label:
        n, bins, patches = ax.hist(
            sims, bins=500, density=True, facecolor=color, alpha=0.5, label=name, color=color
        )
    else:
        n, bins, patches = ax.hist(sims, bins=500, density=True, facecolor=color, alpha=0.5)

    if exact_pdf is not None:
        x, pdf = exact_pdf
        ax.plot(x, pdf, color=linecolor, linewidth=2)

    if fit_gaussian:
        (mu, sigma) = norm.fit(sims)

        x = np.linspace(0.1 * bins[0], bins[-1], 100)
        # add a 'best fit' line
        y = norm.pdf(x, mu, sigma)
        ax.plot(x, y, "black", linestyle="dotted")

    ax.set_xlabel((r"$\xi^+$"))
    ax.legend()
    ax.ticklabel_format(style="scientific", scilimits=(0, 0))
    return ax

def plot_2D(fig,ax,x1,x2,pdf_grid,vmax=None,vmin=0,sims=None,colormap=None,log=False):
   
    if log:
        import matplotlib.colors as colors
        print(pdf_grid.min(),pdf_grid.max())
        h = ax.pcolormesh(x1,x2,pdf_grid,shading='auto',cmap=colormap, norm=colors.LogNorm(vmin=vmin, vmax=vmax))
    else:
        h = ax.pcolormesh(x1,x2,pdf_grid,vmin=vmin,vmax=vmax,shading='auto',cmap=colormap)

   
    return h

def add_data(fig,ax,filepath,njobs,cl_num,angbin,lmax):

    allxi1,allxi2 = read_sims_nd(filepath, cl_num,angbin,njobs,lmax)

    
    ax.hist(allxi1,512,density=True,alpha=0.5,color='C0',label='S5-S5')
    ax.axvline(np.mean(allxi1),color='C3',linestyle='dashed')
    ax.hist(allxi2,512,density=True,alpha=0.5,color='C1',label='S5-S3')
    ax.axvline(np.mean(allxi2),color='C3',linestyle='dashed')
    ax.set_xlim(0,3e-6)

def add_data_1d(ax,sims,color,name,mean=False,density=True,range=None,nbins=512):
    ax.hist(sims,nbins,density=density,alpha=0.6,color=color,label=name,range=range)
    if mean:
        ax.axvline(np.mean(sims),color=color,linestyle='dashed')


def read_2D_cf(config):
    ndim = int(config["Run"]['ndim'])
    paths = config['Paths']
    params = config['Params']
    batchsize = int(config['Run']['batchsize'])
    numjobs = int(config['Run']['jobnum'])
    steps = int(params['steps'])
    t_sets = np.load(paths['t_sets'])['t']

    xi_max = [float(params['ximax{:d}'.format(i)]) for i in range(ndim)]
    
    t_inds, t_sets, t0_set, dt_set = calc_pdf.setup_t(xi_max,steps)
    
    
    cf_grid = np.full((steps-1,steps-1),np.nan,dtype=complex)
 
    resultpath = paths['result']
    #resultpath = '/cluster/scratch/veoehl/2Dcf/xip_5535bins_new/'

    t0_2 = np.array(t0_set)
    dt_2 = np.array(dt_set)
  

    ind_sets = np.stack(np.meshgrid(t_inds,t_inds),-1).reshape(-1,2)
    fail_list = []

    for i in range(numjobs):
        
        try:
            batch = np.load(resultpath + 'job{:d}.npz'.format(i))
            size = os.path.getsize(resultpath + 'job{:d}.npz'.format(i))
            #if size < 131560 and i != numjobs-1:
            #    print('removing file job{:d}.npz'.format(i))
            #    os.remove(resultpath + 'job{:d}.npz'.format(i)) 
            batch_t = batch['ts']
            batch_cf = batch['cf']
        except:
            fail_list.append(i)
            batch_t = t_sets[i*batchsize:(i+1)*batchsize]
            batch_cf = np.zeros(len(batch_t))
        
        inds = ind_sets[i*batchsize:(i+1)*batchsize]
        #high_ell_ext = calc_pdf.high_ell_gaussian_cf_nD(batch_t,mu,cov)
        
        #high_ell_ext[np.isnan(high_ell_ext)] = 0
        #high_ell_ext[np.isinf(high_ell_ext)] = 0
        
        for j,idx in enumerate(inds):
            try:
                cf_grid[tuple(idx)] = batch_cf[j]
            except:
                cf_grid[tuple(idx)] = 0

    missing_string = ','.join([str(x+1) for x in fail_list])
    print(missing_string)
    
    print(len(fail_list))
    np.savez('missing_jobs.npz',numbers=np.array(fail_list))
    return t0_2, dt_2, t_sets,ind_sets,cf_grid


def add_stats(axes,lmax,statistics,stats_measured,mean,cov,color='C0',maskname=None,ylabel=True,bootstraps=None):
    ax1, ax2 = axes[:2]
        
    #ax1.set_xlabel(r"$\ell_{{\mathrm{{exact}}}}$")
    ax1.plot(lmax, statistics[:,2],color=color,label=maskname)
    ax1.axhline(stats_measured[2],color=color,linestyle='dotted')
    ax1.set_xticklabels([])
    ax2.set_xlabel(r"$\ell_{{\mathrm{{exact}}}}$")
    if ylabel:
        ax1.set_ylabel(r"Skewness")
        ax2.set_ylabel(r"$\sigma / \sigma_{\mathrm{Gauss}}$")
            
    ax2.plot(lmax, statistics[:,1] /cov,color=color,label='predicted')
    ax2.axhline(stats_measured[1]**2 / cov,color=color,linestyle='dotted',label='measured')

    if bootstraps is not None:
        ax1.fill_between(lmax, stats_measured[2]-bootstraps[2], stats_measured[2]+bootstraps[2], alpha=0.5, facecolor=color)
        ax2.fill_between(lmax, stats_measured[1]**2 / cov -bootstraps[1]**2 / cov, stats_measured[1]**2 / cov+bootstraps[1]**2 / cov, alpha=0.5, facecolor=color)
        
        
         
    if len(axes) == 3:
        ax3 = axes[2]
        ax3.set_xlabel(r"$\ell_{{\mathrm{{exact}}}}$")
        ax3.set_ylabel(r"$\mathbb{E}(\xi^+)$ / $\hat{\xi}^+$")
        ax3.plot(lmax, statistics[:,0] / mean,color=color, label="predicted")
        ax3.axhline(1, color="black", linestyle="dotted")
        ax3.axhline(stats_measured[0]/ mean,color=color,linestyle='dotted',label='measured')
        
        

def plot_gauss(ax,x,mu,cov,color,label=None,linestyle='dashed'):
    import scipy.stats as stats
    ax.plot(
        x,
        stats.norm.pdf(x, mu, np.sqrt(cov)),
        color=color,
        linestyle=linestyle,label=label,alpha=0.5
    )



