import matplotlib.pyplot as plt
import numpy as np
import os
import logging
from scipy.stats import norm
import configparser
from .cl2xi_transforms import pcls2xis, prep_prefactors, compute_kernel
import traceback
import matplotlib.colors as colors
import numpy as np
import scipy.stats as stats
import seaborn as sns  # Add seaborn for better corner plot aesthetics
from .file_handling import read_sims_nd
from .theory_cl import BinCombinationMapper, n_combs_to_n_bins
import pickle  # Add import for loading cache files
from itertools import product
import random
from .data_statistics import bootstrap, bootstrap_statistic_2d
from scipy.interpolate import RegularGridInterpolator, griddata
from matplotlib import rc
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
import cmasher as cmr

# Set up logger for this module
logger = logging.getLogger(__name__)

rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')


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

def find_contour_levels_pdf(x, y, pdf, levels=[0.32, 0.68, 0.95]):
    # Compute cell areas
    dx = np.diff(x)
    dy = np.diff(y)
    # For meshgrid, get the area for each cell
    dx2d, dy2d = np.meshgrid(dx, dy)
    cell_areas = dx2d * dy2d
    # Remove last row/col to match pdf shape
    pdf_cells = pdf[:-1, :-1]
    cell_areas = cell_areas[:pdf_cells.shape[0], :pdf_cells.shape[1]]
    # Flatten
    pdf_flat = pdf_cells.flatten()
    areas_flat = cell_areas.flatten()
    # Sort by PDF value descending
    idx = np.argsort(pdf_flat)[::-1]
    pdf_sorted = pdf_flat[idx]
    areas_sorted = areas_flat[idx]
    # Compute cumulative probability mass
    cum_mass = np.cumsum(pdf_sorted * areas_sorted)
    total_mass = cum_mass[-1]
    cum_mass /= total_mass
    contour_levels = []
    for lev in levels:
        i = np.searchsorted(cum_mass, lev)
        contour_levels.append(pdf_sorted[i])
    return np.array(contour_levels)[::-1]


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

def set_xi_axes_hist(ax,angbin,lims,rs_bin=None,labels=True,binnum=None,islow=False):

    if not labels:
        ax.set_xticklabels([])
    elif angbin is None:
        if islow:
            ax.set_xlabel((r'$\hat{{\xi}}^{{+, \mathrm{{low}}}}_{{\mathrm{{S{}-S{}}}}}$'.format(*rs_bin)))
        else:
            ax.set_xlabel((r'$\hat{{\xi}}^+_{{\mathrm{{S{}-S{}}}}}$'.format(*rs_bin)))
    elif rs_bin is None:
        if islow:
            ax.set_xlabel((r'$\hat{{\xi}}^{{+, \mathrm{{low}}}} ({:3.1f}\degree-{:3.1f} \degree)$'.format(*angbin)))
        else:
            ax.set_xlabel((r'$\hat{{\xi}}^+ (\bar{{\theta}}_{:d})$'.format(binnum)))
    else:
        ax.set_xlabel((r'$\hat{{\xi}}^+_{{\mathrm{{S{}-S{}}}}} ({:3.1f}\degree-{:3.1f} \degree)$'.format(*rs_bin,*angbin)))
        if binnum is not None:
            if islow:
                ax.set_xlabel((r'$\hat{{\xi}}^{{+, \mathrm{{low}}}}_{{\mathrm{{S{}-S{}}}}} (\bar{{\theta}}_{:d})$'.format(*rs_bin,binnum)))
    
            else:
                ax.set_xlabel((r'$\hat{{\xi}}^+_{{\mathrm{{S{}-S{}}}}} (\bar{{\theta}}_{:d})$'.format(*rs_bin,binnum)))
    
    ax.set_xlim(*lims)
    



def plot_hist(ax,sims, name, color='C0', linecolor='C3', exact_pdf=None, label=False, fit_gaussian=False):

    
    #


    if label:
        n, bins, patches = ax.hist(
            sims, bins=500, density=True, facecolor=color, alpha=0.5, label=name, color=color
        )
    else:
        n, bins, patches = ax.hist(sims, bins=500, density=True, facecolor=color, alpha=0.5)
    ax.set_xlim(bins[0], bins[-1])
    if exact_pdf is not None:
        x, pdf = exact_pdf
        ax.plot(x, pdf, color=linecolor,label=r'exact likelihood')

    if fit_gaussian:
        (mu, sigma) = norm.fit(sims)

        x = np.linspace(0.1 * bins[0], bins[-1], 100)
        # add a 'best fit' line
        y = norm.pdf(x, mu, sigma)
        ax.plot(x, y, color='C0', linestyle="dotted",label='Gaussian approximation')

    ax.set_xlabel((r"$\xi^+$"))
    ax.legend(frameon=False)
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



def plot_kernels(prefactors,save_path=None,ang_bins=None):
    """
    Plot the elements of the sum against l.
    """
    print(prefactors.shape)
    out_lmax  = prefactors.shape[-1] - 1
    lmin = 0
    l = 2 * np.arange(0, out_lmax + 1) + 1
    p_cl_prefactors_p, p_cl_prefactors_m = prefactors[:, 0], prefactors[:, 1]
    kernel_xip = p_cl_prefactors_p[:, lmin : out_lmax + 1] * l
    kernel_xim = p_cl_prefactors_m[:, lmin : out_lmax + 1] * l
    # normalize by maximum value of each:
    kernel_xip = kernel_xip / np.max(kernel_xip, axis=-1)[:, None]
    kernel_xim = kernel_xim / np.max(kernel_xim, axis=-1)[:, None]
    l = np.arange(kernel_xip.shape[-1])
    # Plot each angular bin
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    colors = plt.cm.RdBu([0,0.2,0.9])
    # Plot xip kernel
    if ang_bins is not None:
        labels = [r'$\bar{{\theta}} = [ {:.1f}^{{\circ}}, {:.1f}^{{\circ}}]$'.format(*ang_bins[i]) for i in range(len(kernel_xip))]
    else:
        labels = [f"Angular bin {i}" for i in range(len(kernel_xip))]
        
    for i, element in enumerate(kernel_xip):
        axes[0].plot(l, element.T, label=labels[i], color=colors[i])
    axes[0].set_xlabel(r"$\ell$")
    axes[0].set_ylabel(r"$\xi^+$ Kernel")
    axes[0].legend()
    axes[0].legend(frameon=False)
    axes[0].set_xscale('log')
    axes[0].set_xlim(2,767)
    
    # Plot xim kernel
    for i, element in enumerate(kernel_xim):
        axes[1].plot(l, element.T, label=labels[i], color=colors[i])
        axes[1].set_xlabel(r"$\ell$")
        axes[1].yaxis.set_label_position("right")
        axes[1].yaxis.tick_right()
        axes[1].set_ylabel(r"$\xi^-$ Kernel")
        axes[1].set_xscale('log')
        axes[1].set_xlim(2,767)
    

    # Adjust layout and save/show the plot
    fig.subplots_adjust(hspace=0,wspace=0.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

        
def compare_to_sims_2d(axes, bincenters, sim_mean, sim_std, interp, vmax,log=False,colormap=None):

    bincenters_x, bincenters_y = bincenters
    X, Y = np.meshgrid(bincenters_x, bincenters_y)
    exact_grid = interp((Y,X))
    
    if log:
        diff_hist = sim_mean - exact_grid
        rel_res = diff_hist
    else:
        diff_hist = (sim_mean - exact_grid)
        diff_hist = np.ma.masked_where(sim_mean == 0, diff_hist)
        rel_res = diff_hist / sim_std
    print("Mean deviation from simulations: {} std".format(np.mean(np.fabs(rel_res))))
    alpha_values = np.ones_like(rel_res)
    alpha_values[np.abs(rel_res) < 1.0] = 0.0
    alpha_values[np.abs(rel_res) >= 1.0] = 0.3
    alpha_values[np.abs(rel_res) >= 3.0] = 1.0
    alpha_values[np.ma.getmask(rel_res)] = 0.0
    # Display values per pixel using a heatmap
    rel_res = diff_hist / np.max(exact_grid)
    
    
    im = axes.pcolormesh(bincenters_x, bincenters_y,
        rel_res, 
        shading="auto", 
        #extent=(bincenters_x[0], bincenters_x[-1], bincenters_y[0], bincenters_y[-1]),
        cmap=colormap, 
        vmin=-vmax, 
        vmax=vmax,
        alpha=alpha_values
    )
    #im2 = axes.contour(bincenters_x, bincenters_y,exact_grid,cmap='gray',levels=5)

    return im

def plot_corner(simspath, likelihoodpath,njobs, lmax, save_path=None, redshift_indices=[0, 1, 2], angular_indices=[0, 1],prefactors=None,theta=None,marginals=None,nbins=100, use_gaussian=False):
    """
    Create a corner plot with 2D marginals and 1D histograms for simulations,
    and overlay PDF contours. Additionally, compare 2D histograms to analytic PDFs.
    """
    # Read simulations and angles
    name = 'Gaussian' if use_gaussian else 'Exact'
    sims, angles = read_sims_nd(simspath, njobs, lmax,prefactors=prefactors,theta=theta)
    n_bins = n_combs_to_n_bins(sims.shape[1])
    mapper = BinCombinationMapper(n_bins)
    cmap = cmr.guppy_r 
    linecolor = cmr.take_cmap_colors(cmap, 3, return_fmt='hex')[1]
    print('loaded sims with shape:',sims.shape)
    ang_bins = [angles[i] for i in angular_indices]
    for a, ang_bin in enumerate(ang_bins):
        logger.info(r'$\theta_{:d} = [{:3.1f}, {:3.1f}] \degree$'.format(a, *ang_bin))

    if marginals is not None:
        x,pdf = marginals # shape (n_correlations, n_ang_bins, n_points_per_dim)
        print('loaded marginals with shape:',pdf.shape)
        x = x.reshape(-1,x.shape[-1])
        pdf = pdf.reshape(-1,x.shape[-1])
        print('loaded marginals with shape:',pdf.shape)
    if len(angles) == len(angular_indices):
        print('simulations are subset already, asserting that the angles are the same')
        # i.e. if the simulations don't contain more angles than requested
        assert angles == theta
        angular_indices = np.arange(len(angles))
    # Select two redshift bins and two angular bins
      # Two auto and one cross-correlation
      # Two angular bins
    selected_data = sims[:, redshift_indices, :][:, :, angular_indices].reshape(sims.shape[0], -1)
    # Create a corner plot with improved formatting
    n_dims = selected_data.shape[1]
    fig, axes = plt.subplots(n_dims, n_dims, figsize=(11, 10))
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    colorbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])  # Big colorbar axis
    im = None  # Initialize the colorbar reference
    data_subset = list(product(redshift_indices,angular_indices))
    n_dims = len(data_subset)
    

    
    for i, indices_i in enumerate(data_subset):
        for j, indices_j in enumerate(data_subset):
            redshift_idx_i, angular_idx_i = indices_i
            redshift_idx_j, angular_idx_j = indices_j
            ax = axes[j, i]
            # Improved axis labels
            if i == 0:
                comb = tuple(x + 1 for x in mapper.get_combination(redshift_idx_j))
                ax.set_ylabel((r'$\hat{{\xi}}^{{+}}_{{\mathrm{{S{}-S{}}}}} (\bar{{\theta}}_{:d})$'.format(*comb,angular_idx_j+1)))
            else:
                ax.set_yticklabels([])
            if j == n_dims - 1:
                comb = tuple(x + 1 for x in mapper.get_combination(redshift_idx_i))
                ax.set_xlabel((r'$\hat{{\xi}}^{{+}}_{{\mathrm{{S{}-S{}}}}} (\bar{{\theta}}_{:d})$'.format(*comb,angular_idx_i+1)))
                ax.xaxis.get_offset_text().set_x(1.2)
            else:
                ax.set_xticklabels([])
            
            
            if i == j:
                # 1D histogram with improved style
                n, bins, _ = ax.hist(selected_data[:, i], bins=nbins, density=True, alpha=0.5, color=linecolor, histtype='stepfilled', label=r'Simulations')
                ax.set_xlim(bins[0], bins[-1])
                # define integration axes for 1d exact marginals (always over the next dimension)
                if j == n_dims - 1:
                    # Last diagonal: integrate over previous dimension
                    redshift_idx_i, angular_idx_i = data_subset[i-1]
                    axis = 1
                else:
                    # Integrate over next dimension
                    redshift_idx_j, angular_idx_j = data_subset[j+1]
                    axis = 0
                marginal_data = load_2d_pdf(likelihoodpath, redshift_idx_i, angular_idx_i, redshift_idx_j, angular_idx_j, integrate_axis=axis, use_gaussian=use_gaussian)
                if marginal_data is not None:
                    x_marginal, marginal = marginal_data
                    ax.plot(x_marginal, marginal, color=linecolor, linewidth=1.5, label=r"{} Marginal".format(name))
                mean, std = selected_data[:, i].mean(), selected_data[:, i].std()
                ax.axvline(mean, color=linecolor, linestyle='--', alpha=0.8, linewidth=1)
                if i == 0:
                    handles, labels = ax.get_legend_handles_labels()
                    
            elif i < j:
                # 2D marginal
                
                
                plot_2d_from_cache(ax, likelihoodpath, redshift_idx_i, angular_idx_i, redshift_idx_j, angular_idx_j,color=linecolor,use_gaussian=use_gaussian)
                h1, xedges, yedges = np.histogram2d(
                    selected_data[:, i], selected_data[:, j],
                    bins=nbins, density=True
                )
                h1_smooth = gaussian_filter(h1, sigma=1.0)
                
                # Plot contours for first set
                X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, 
                                  (yedges[:-1] + yedges[1:]) / 2)
                levels_1 = find_contour_levels_pdf(
                (xedges[:-1] + xedges[1:]) / 2,  # x bin centers
                (yedges[:-1] + yedges[1:]) / 2,  # y bin centers  
                h1.T,  # Transpose to match expected shape
                levels=[0.32, 0.68, 0.95]
                )
                ax.contour(X, Y, h1_smooth.T, levels=levels_1, colors=linecolor, 
                          linewidths=1.5, alpha=0.5, linestyles='solid')
                

                
            else:
                # Upper triangle: plot deviations in std
                oldax = axes[i, j]
                hist = np.histogram2d(
                    selected_data[:, j], selected_data[:, i],
                    bins=nbins, density=True
                )
                
                bincenters_x = 0.5 * (hist[1][1:] + hist[1][:-1])
                bincenters_y = 0.5 * (hist[2][1:] + hist[2][:-1])
                density = hist[0].T

                # Bootstrap to calculate standard deviation of bin heights
                res = bootstrap(np.array([selected_data[:, j], selected_data[:, i]]), n=100, axis=1, func=bootstrap_statistic_2d, func_kwargs={
                    "binedges": [hist[1], hist[2]],
                })
                std_dev = np.std(res, axis=0,ddof=1) 
                sim_mean = np.mean(res, axis=0)
                assert sim_mean.shape == density.shape
                sim_mean = (np.ma.masked_where(density == 0, sim_mean))
                std_dev = (np.ma.masked_where(density == 0, std_dev))

                # Load analytic PDF
                pdf_data = load_2d_pdf(likelihoodpath, redshift_idx_j, angular_idx_j, redshift_idx_i, angular_idx_i, use_gaussian=use_gaussian)
                if pdf_data is not None:
                    x_pdf, y_pdf, exact_pdf, gauss_pdf = pdf_data
                    # Interpolate PDF to match histogram bin centers
                    pdf_grid = gauss_pdf if use_gaussian else exact_pdf
                    interp_pdf  = RegularGridInterpolator((y_pdf,x_pdf), pdf_grid,method='cubic')
                    # the y,x order is weird, but necessary due to the indexing convention of the interpolator vs the plotting convention used elsewhere!
                    im = compare_to_sims_2d(ax, (bincenters_x, bincenters_y), sim_mean, std_dev, interp_pdf, vmax=.05,log=False,colormap=cmap)
                    # Compute relative residuals
                ax.set_xlim(hist[1][0],hist[1][-1])
                ax.set_ylim(hist[2][0],hist[2][-1])
                oldax.set_xlim(hist[1][0],hist[1][-1])
                oldax.set_ylim(hist[2][0],hist[2][-1])
                
                    
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False
    )
    #fig.subplots_adjust(top=0.9)          
    
    if im is not None:
        fig.colorbar(im, cax=colorbar_ax, label=r'(Sims - {}) / max({})'.format(name,name))  # Add colorbar to the new axis

    # Adjust layout
    #plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)  # Adjust padding to reduce space
    plt.subplots_adjust(wspace=0.12, hspace=0.12)
    #plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Corner plot saved to {save_path}")
    else:
        plt.show()

def plot_2d_from_cache(ax, filepath, i, j, k, l, color=None, use_gaussian=False):
    """
    Load a 2D likelihood cache file and plot the PDF.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the 2D PDF.
    filepath : str
        Path to the directory containing the cache files.
    i, j : int
        Indices for the first dimension (e.g., redshift_bin, angular_bin).
    k, l : int
        Indices for the second dimension (e.g., redshift_bin, angular_bin).
    """
    try:
        data = load_2d_pdf(filepath, i, j, k, l)
        if data is not None:
            x, y, pdf_exact, pdf_gauss = data
            pdf = pdf_gauss if use_gaussian else pdf_exact
            levels = find_contour_levels_pdf(x, y, pdf, levels=[0.32, 0.68, 0.95])
            logger.info(f"Contour levels for PDF: {levels}")
            X, Y = np.meshgrid(x, y)
            c = ax.contour(X, Y, pdf, levels=levels, colors=color, linewidths=1.5)
            #c2 = ax.contour(X,Y,pdf_gauss,levels=levels_gauss, colors='midnightblue', linestyles='dashed', linewidths=1.5)
            #c = ax.pcolormesh(X,Y,pdf, shading="auto",cmap="Blues")
    except Exception as e:
        print(f"Error loading or plotting cache file: {e}")
    



def load_2d_pdf(filepath, i, j, k, l, integrate_axis=None, use_gaussian=False):
    """
    Load a 2D likelihood PDF from the cache file and optionally integrate it.

    Parameters
    ----------
    filepath : str
        Path to the directory containing the cache files.
    i, j : int
        Indices for the first dimension (e.g., redshift_bin, angular_bin).
    k, l : int
        Indices for the second dimension (e.g., redshift_bin, angular_bin).
    integrate_axis : int, optional
        Axis along which to integrate the 2D PDF to compute the 1D marginal.
        If None, the full 2D PDF is returned.

    Returns
    -------
    tuple
        If `integrate_axis` is None, returns (x, y, pdf), where:
        - x : ndarray
            x-axis values of the 2D PDF.
        - y : ndarray
            y-axis values of the 2D PDF.
        - pdf : ndarray
            2D PDF values.
        If `integrate_axis` is specified, returns (x, marginal), where:
        - x : ndarray
            Axis values corresponding to the marginal.
        - marginal : ndarray
            1D marginal PDF values.
    """
    cache_file = f"{filepath}/likelihood_2d_cache_{i}_{j}_{k}_{l}.npz"
    
    try:
        data = np.load(cache_file)
        print(f"Loaded cache file: {cache_file}")
        xs = data["x"]
        x = xs[0]
        y = xs[1]
        if use_gaussian:
            logpdf = data["gauss_loglikelihood"]
        else:
            logpdf = data["likelihood_2d"]
            
        pdf = np.exp(logpdf)
        
        # Also load the other PDF for return (to maintain backward compatibility)
        if use_gaussian:
            logpdf_other = data["likelihood_2d"]
        else:
            logpdf_other = data["gauss_loglikelihood"]
        pdf_other = np.exp(logpdf_other)

        print(f"Fraction of NaN values in PDF: {np.isnan(logpdf).sum() / logpdf.size:.4f}")
        print(f"Fraction of finite values in PDF: {np.isfinite(logpdf).sum() / logpdf.size:.4f}")
        for p in [pdf, pdf_other]:
            p = np.where(np.isfinite(p), p, np.nan)
            p_f = np.nan_to_num(p, nan=0.0)
            
        
        
        
        if integrate_axis is not None:
            # Integrate along the specified axis
            if integrate_axis == 0:
                marginal = np.trapz(pdf, x=y, axis=0)  # Integrate along y-axis
                return x, marginal
            elif integrate_axis == 1:
                marginal = np.trapz(pdf, x=x, axis=1)  # Integrate along x-axis
                return y, marginal
            else:
                raise ValueError("integrate_axis must be 0 (y-axis) or 1 (x-axis).")
        
        if use_gaussian:
            return x, y, pdf_other, pdf  # pdf_other is exact, pdf is gaussian
        else:
            return x, y, pdf, pdf_other 

    except FileNotFoundError:
        print(f"Cache file {cache_file} not found.")
        return None
    except Exception as e:
        print(f"Error loading or processing cache file: {e}")
        return None


def plot_corner_comparison(simspath_1, simspath_2, label_1="Simulation 1", label_2="Simulation 2",
                           njobs=1000, lmax=None, save_path=None, 
                           redshift_indices=[0, 1, 2], angular_indices=[0, 1],
                           prefactors=None, theta=None, nbins=50, alpha=0.8):
    """
    Create a corner plot comparing two sets of simulations.
    
    Parameters
    ----------
    simspath_1 : str
        Path to first simulation set
    simspath_2 : str
        Path to second simulation set
    label_1 : str
        Label for first simulation set
    label_2 : str
        Label for second simulation set
    njobs : int
        Number of jobs to load
    lmax : int
        Maximum multipole (for file naming)
    save_path : str, optional
        Path to save the figure
    redshift_indices : list
        Redshift bin pairs to plot
    angular_indices : list
        Angular bins to plot
    prefactors : array_like, optional
        Prefactors for transformation
    theta : array_like, optional
        Angular separations
    nbins : int
        Number of bins for histograms
    alpha : float
        Transparency for histograms
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    # Read both simulation sets
    logger.info(f"Loading {label_1} simulations from {simspath_1}...")
    sims_1, angles_1 = read_sims_nd(simspath_1, njobs, lmax, prefactors=prefactors, theta=theta)
    
    logger.info(f"Loading {label_2} simulations from {simspath_2}...")
    sims_2, angles_2 = read_sims_nd(simspath_2, njobs, lmax, prefactors=prefactors, theta=theta)
    
    n_bins = n_combs_to_n_bins(sims_1.shape[1])
    mapper = BinCombinationMapper(n_bins)
    combs = [tuple(x + 1 for x in mapper.get_combination(j)) for j in redshift_indices]

    logger.info(f"Loaded {len(sims_1)} {label_1} sims, {len(sims_2)} {label_2} sims")
    
    # Select data
    selected_1 = sims_1[:, redshift_indices, :][:, :, angular_indices].reshape(sims_1.shape[0], -1)
    selected_2 = sims_2[:, redshift_indices, :][:, :, angular_indices].reshape(sims_2.shape[0], -1)
    
    ang_bins = [angles_1[i] for i in angular_indices]
    for a, ang_bin in enumerate(ang_bins):
        logger.info(r'$\theta_{:d} = [{:3.1f}, {:3.1f}] \degree$'.format(a, *ang_bin))
    n_dims = selected_1.shape[1]
    
    # Create corner plot
    fig, axes = plt.subplots(n_dims, n_dims, figsize=(11, 10))
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    colorbar_ax = fig.add_axes([0.92, 0.2, 0.02, 0.6])
    im = None
    
    # Define colors
    color_1 = '#4477AA'  # Blue
    color_2 = '#CC6677'  # Red

    for i in range(n_dims):
        for j in range(n_dims):
            redshift_idx_i, angular_idx_i = divmod(i, len(angular_indices))
            redshift_idx_j, angular_idx_j = divmod(j, len(angular_indices))
            ax = axes[j, i] if n_dims > 1 else axes
            
            # Set labels
            if i == 0:
                ax.set_ylabel((r'$\hat{{\xi}}^{{+, \mathrm{{low}}}}_{{\mathrm{{S{}-S{}}}}} (\bar{{\theta}}_{:d})$'.format(*combs[redshift_idx_j],angular_idx_j+1)))
            else:
                ax.set_yticklabels([])
                
            if j == n_dims - 1:
                ax.set_xlabel((r'$\hat{{\xi}}^{{+, \mathrm{{low}}}}_{{\mathrm{{S{}-S{}}}}} (\bar{{\theta}}_{:d})$'.format(*combs[redshift_idx_i],angular_idx_i+1)))
                #ax.xaxis.set_offset_position('top')
                ax.xaxis.get_offset_text().set_x(1.2)  # Move offset right
                ax.xaxis.get_offset_text().set_y(10)
                #ax.xaxis.get_offset_text().set_fontsize(10)
            else:
                ax.set_xticklabels([])
            
            if i == j:
                # 1D histogram on diagonal
                range_min = min(selected_1[:, i].min(), selected_2[:, i].min())
                range_max = max(selected_1[:, i].max(), selected_2[:, i].max())
                bins = np.linspace(range_min, range_max, nbins)
                
                ax.hist(selected_1[:, i], bins=bins, density=True, alpha=alpha, 
                       color=color_1, label=label_1, histtype='step')
                ax.hist(selected_2[:, i], bins=bins, density=True, alpha=alpha,
                       color=color_2, label=label_2, histtype='step')
                
                #if i == n_dims - 1:
                #    ax.legend(frameon=False, loc='upper right')
                
                # Add mean and std info
                mean_1, std_1 = selected_1[:, i].mean(), selected_1[:, i].std()
                mean_2, std_2 = selected_2[:, i].mean(), selected_2[:, i].std()
                ax.axvline(mean_1, color=color_1, linestyle='--', alpha=alpha, linewidth=1)
                ax.axvline(mean_2, color=color_2, linestyle='--', alpha=alpha, linewidth=1)
                if i == 0:
                    handles, labels = ax.get_legend_handles_labels()
                
            elif i < j:
                # 2D histogram in lower triangle
                # Determine ranges
                xmin = min(selected_1[:, i].min(), selected_2[:, i].min())
                xmax = max(selected_1[:, i].max(), selected_2[:, i].max())
                ymin = min(selected_1[:, j].min(), selected_2[:, j].min())
                ymax = max(selected_1[:, j].max(), selected_2[:, j].max())
                
                # Plot first simulation set
                h1, xedges, yedges = np.histogram2d(
                    selected_1[:, i], selected_1[:, j],
                    bins=nbins, range=[[xmin, xmax], [ymin, ymax]], density=True
                )
                h1_smooth = gaussian_filter(h1, sigma=1.0)
                
                # Plot contours for first set
                X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, 
                                  (yedges[:-1] + yedges[1:]) / 2)
                levels_1 = np.percentile(h1[h1 > 0], [32, 68, 95])
                ax.contour(X, Y, h1_smooth.T, levels=levels_1, colors=color_1, 
                          linewidths=1.5, alpha=alpha)
                
                # Plot second simulation set
                h2, _, _ = np.histogram2d(
                    selected_2[:, i], selected_2[:, j],
                    bins=nbins, range=[[xmin, xmax], [ymin, ymax]], density=True
                )
                h2_smooth = gaussian_filter(h2, sigma=1.0)
                levels_2 = np.percentile(h2[h2 > 0], [32, 68, 95])
                ax.contour(X, Y, h2_smooth.T, levels=levels_2, colors=color_2,
                          linewidths=1.5, alpha=alpha)
                
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)

                
            
            else:
                # Upper triangle: plot difference normalized by bootstrap uncertainty
                if j == 0 and i == 1:  # Log once for the first upper triangle plot
                    logger.info(f"Computing bootstrap uncertainties for upper triangle plots (100 resamples per panel)...")
                
                xmin = min(selected_1[:, j].min(), selected_2[:, j].min())
                xmax = max(selected_1[:, j].max(), selected_2[:, j].max())
                ymin = min(selected_1[:, i].min(), selected_2[:, i].min())
                ymax = max(selected_1[:, i].max(), selected_2[:, i].max())
                
                # Create histograms for both sets
                h1, xedges, yedges = np.histogram2d(
                    selected_1[:, j], selected_1[:, i],
                    bins=nbins, range=[[xmin, xmax], [ymin, ymax]], density=True
                )
                
                h2, _, _ = np.histogram2d(
                    selected_2[:, j], selected_2[:, i],
                    bins=nbins, range=[[xmin, xmax], [ymin, ymax]], density=True
                )
                
                # Bootstrap the lognormal set to get uncertainty
                # do bootstrap on differences, not on one set of values?
                
                res = bootstrap(
                    np.array([selected_2[:, j], selected_2[:, i]]), 
                    n=100, 
                    axis=1, 
                    func=bootstrap_statistic_2d, 
                    func_kwargs={"binedges": [xedges, yedges]}
                )
                
                # Calculate standard deviation from bootstrap
                std_dev = np.std(res, axis=0, ddof=1)
                
                # Mask bins with no data
                std_dev = np.ma.masked_where(h1 == 0, std_dev)
                
                # Calculate difference normalized by bootstrap uncertainty
                diff = h2 - h1
                diff = np.ma.masked_where(h1 == 0, diff)
                normalized_diff = diff / std_dev
                
                # Plot normalized difference
                X, Y = np.meshgrid((xedges[:-1] + xedges[1:]) / 2,
                                    (yedges[:-1] + yedges[1:]) / 2)

                vmax = 2  # Show differences up to 4 sigma
                im = ax.pcolormesh(X, Y, normalized_diff.T, cmap='RdBu_r',
                                    vmin=-vmax, vmax=vmax, shading='auto',alpha=0.5)
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                
    if im is not None:            # Add colorbar to rightmost upper triangle plot 
                
        cbar = plt.colorbar(im, cax=colorbar_ax)
        cbar.set_label(r'({} - {}) / $\sigma_{{\mathrm{{Bootstrap}}}}$'.format(label_2,label_1))
    
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.02),
        frameon=False
    )  
    plt.subplots_adjust(wspace=0.12, hspace=0.12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    return fig


def print_comparison_statistics(simspath_1, simspath_2, label_1="Simulation 1", label_2="Simulation 2",
                                njobs=1000, lmax=None, redshift_indices=[0, 1, 2], 
                                angular_indices=[0, 1], prefactors=None, theta=None):
    """
    Print statistical comparison between two simulation sets.
    
    Parameters
    ----------
    simspath_1, simspath_2 : str
        Paths to simulation sets
    label_1, label_2 : str
        Labels for the sets
    njobs : int
        Number of jobs to load
    lmax : int
        Maximum multipole
    redshift_indices : list
        Redshift pairs to compare
    angular_indices : list
        Angular bins to compare
    prefactors, theta : optional
        Transformation parameters
    """
    # Read simulations
    logger.info(f"Loading {label_1} simulations for statistical comparison...")
    sims_1, angles_1 = read_sims_nd(simspath_1, njobs, lmax, prefactors=prefactors, theta=theta)
    logger.info(f"Loading {label_2} simulations for statistical comparison...")
    sims_2, angles_2 = read_sims_nd(simspath_2, njobs, lmax, prefactors=prefactors, theta=theta)
    logger.info(f"Loaded simulations, starting statistical tests...")
    
    # Select data
    selected_1 = sims_1[:, redshift_indices, :][:, :, angular_indices].reshape(sims_1.shape[0], -1)
    selected_2 = sims_2[:, redshift_indices, :][:, :, angular_indices].reshape(sims_2.shape[0], -1)
    
    logger.info("\n" + "="*70)
    logger.info(f"STATISTICAL COMPARISON: {label_1} vs {label_2}")
    logger.info("="*70)
    
    for i in range(selected_1.shape[1]):
        redshift_idx, angular_idx = divmod(i, len(angular_indices))
        
        sample_1 = selected_1[:, i]
        sample_2 = selected_2[:, i]
        
        mean_1, std_1 = sample_1.mean(), sample_1.std()
        mean_2, std_2 = sample_2.mean(), sample_2.std()
        
        # Two-sample t-test
        t_stat, p_value = stats.ttest_ind(sample_1, sample_2)
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p_value = stats.ks_2samp(sample_1, sample_2)
        
        logger.info(f"\nCorrelation {redshift_idx}, Angular bin {angular_idx}:")
        logger.info(f"  {label_1:20s}: mean = {mean_1:.6e}, std = {std_1:.6e}")
        logger.info(f"  {label_2:20s}: mean = {mean_2:.6e}, std = {std_2:.6e}")
        if mean_1 != 0:
            logger.info(f"  Mean difference:      {abs(mean_2 - mean_1):.6e} ({abs(mean_2-mean_1)/mean_1*100:.2f}%)")
        else:
            logger.info(f"  Mean difference:      {abs(mean_2 - mean_1):.6e} (N/A - mean_1 is zero)")
        if std_1 != 0:
            logger.info(f"  Std difference:       {abs(std_2 - std_1):.6e} ({abs(std_2-std_1)/std_1*100:.2f}%)")
        else:
            logger.info(f"  Std difference:       {abs(std_2 - std_1):.6e} (N/A - std_1 is zero)")
        logger.info(f"  T-test p-value:       {p_value:.4e} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
        logger.info(f"  KS-test p-value:      {ks_p_value:.4e} {'***' if ks_p_value < 0.001 else '**' if ks_p_value < 0.01 else '*' if ks_p_value < 0.05 else ''}")
    
    logger.info("="*70)


