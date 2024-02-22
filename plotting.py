import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import configparser
import calc_pdf


def read_xi_sims(filepath,njobs,angbin,kind="xip"):
    allxi=[]
    missing = []
    for i in range(1,njobs+1):
        if os.path.isfile(filepath+"/job{:d}.npz".format(i)):
            xifile = np.load(filepath+"/job{:d}.npz".format(i))
            angs = xifile["theta"]
            angind = np.where(angs == angbin)
            xip = xifile[kind][:,angind[0][0]]
            allxi.append(xip)
        else:
            print('Missing job number {:d}.'.format(i))
            missing.append(i)
    allxi = np.array(allxi)
    allxi = allxi.flatten()
    return allxi

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

def set_xi_axes_2D(ax,angbin,rs_bins,lims):
    ax.set_xlabel((r'$\xi^+_{{S{:d}-S{:d}}} ({:3.1f}-{:3.1f} \degree)$'.format(*rs_bins[0],*angbin)))
    ax.set_ylabel((r'$\xi^+_{{S{:d}-S{:d}}} ({:3.1f}-{:3.1f} \degree)$'.format(*rs_bins[1],*angbin)))
    ax.set_xlim(*lims[0])
    ax.set_ylim(*lims[1])

def set_xi_axes_hist(ax,angbin,rs_bin,lims):
    if angbin is None:
        ax.set_xlabel((r'$\xi^+_{{S{:d}-S{:d}}}$'.format(*rs_bin)))
    else:
        ax.set_xlabel((r'$\xi^+_{{S{:d}-S{:d}}} ({:3.1f}-{:3.1f} \degree)$'.format(*rs_bin,*angbin)))
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

def plot_2D(fig,ax,x1,x2,pdf_grid,vmax,sims=None):
    #h = ax.contourf(x1,x2,pdf_grid)
    h = ax.pcolormesh(x1,x2,pdf_grid,vmin=0,vmax=vmax,shading='auto')
    #h = ax.imshow(pdf_grid)
    #ax.axis('scaled')
    
    fig.colorbar(h, ax=ax)

def add_data(fig,ax,filepath,njobs,cl_num,angbin,lmax):

    allxi1,allxi2 = read_sims_nd(filepath, cl_num,angbin,njobs,lmax)

    
    ax.hist(allxi1,512,density=True,alpha=0.5,color='C0',label='S5-S5')
    ax.axvline(np.mean(allxi1),color='C3',linestyle='dashed')
    ax.hist(allxi2,512,density=True,alpha=0.5,color='C1',label='S5-S3')
    ax.axvline(np.mean(allxi2),color='C3',linestyle='dashed')
    ax.set_xlim(0,3e-6)

def read_2D_cf(configpath):
    config = configparser.ConfigParser()
    config.read(configpath)
    paths = config['Paths']
    params = config['Params']
    batchsize = int(config['Run']['batchsize'])
    numjobs = int(config['Run']['jobnum'])
    steps = int(params['steps'])
    t_sets = np.load(paths['t_sets'])['t']

    
    xip_max1 = float(params['ximax1'])
    xip_max2 = float(params['ximax2'])
    xi_max = [xip_max1,xip_max2]
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


def add_stats(fig,gs,lmax,statistics,stats_measured,mean,cov):
        ax6 = fig.add_subplot(gs[1, 0])
        ax6.set_xlabel(r"$\ell_{{\mathrm{{exact}}}}$")
        ax6.set_ylabel(r"Skewness")
        ax6.plot(lmax, statistics[:,2],label='predicted')
        ax6.axhline(stats_measured[2],color='C0',linestyle='dotted',label='measured')
        ax6.legend()
        ax7 = fig.add_subplot(gs[1, 1])
        ax7.plot(lmax, statistics[:,1] /cov,label='predicted')
        ax7.axhline(stats_measured[1]**2 / cov,color='C0',linestyle='dotted',label='measured')
        ax7.axhline(1, color="black", linestyle="dotted")
        ax7.legend()
        ax7.set_xlabel(r"$\ell_{{\mathrm{{exact}}}}$")
        ax7.set_ylabel(r"$\sigma / \sigma_{\mathrm{Gauss}}$")
        ax8 = fig.add_subplot(gs[1, 2])
         
    
        ax8.plot(lmax, statistics[:,0] / mean, label="predicted",color='C3')
       
        ax8.axhline(1, color="black", linestyle="dotted")
        ax8.axhline(stats_measured[0]/ mean,color='C0',linestyle='dotted',label='measured')
        ax8.set_xlabel(r"$\ell_{{\mathrm{{exact}}}}$")
        ax8.set_ylabel(r"$\mathbb{E}(\xi^+)$ / $\hat{\xi}^+$")
        #ax8.set_ylim(0.99,1.02)
        ax8.legend()



