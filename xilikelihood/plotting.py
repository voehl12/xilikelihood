import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import norm
import configparser
from .cl2xi_transforms import pcls2xis, prep_prefactors, compute_kernel
import traceback
import matplotlib.colors as colors
import numpy as np
import scipy.stats as stats
import seaborn as sns  # Add seaborn for better corner plot aesthetics
from .file_handling import read_sims_nd
import pickle  # Add import for loading cache files
from itertools import product
import random
from .data_statistics import bootstrap, bootstrap_statistic_2d
from scipy.interpolate import RegularGridInterpolator, griddata
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Times']})
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

        
def compare_to_sims_2d(axes, bincenters, sim_mean, sim_std, interp, vmax,log=False):

    bincenters_x, bincenters_y = bincenters
    X, Y = np.meshgrid(bincenters_x, bincenters_y)
    exact_grid = interp((Y,X))
    
    if log:
        diff_hist = sim_mean - exact_grid
        rel_res = diff_hist
    else:
        diff_hist = (sim_mean - exact_grid)
        rel_res = diff_hist / sim_std
    mean_dev = np.mean(np.fabs(rel_res))
    print("Mean deviation from simulations: {:.3f} std".format(mean_dev))

    # Display values per pixel using a heatmap
    im = axes.pcolormesh(bincenters_x, bincenters_y,
        rel_res, 
        shading="auto", 
        #extent=(bincenters_x[0], bincenters_x[-1], bincenters_y[0], bincenters_y[-1]),
        cmap="coolwarm", 
        vmin=-vmax, 
        vmax=vmax
    )
    
    # Add text annotation with mean deviation
    axes.text(0.05, 0.95, r'Mean dev: {:.3f}$\sigma$'.format(mean_dev), 
              transform=axes.transAxes, fontsize=10, 
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    #im2 = axes.contour(bincenters_x, bincenters_y,exact_grid,cmap='gray',levels=5)

    return im

def plot_corner(simspath, njobs, lmax, save_path=None, redshift_indices=[0, 1, 2], angular_indices=[0, 1],prefactors=None,theta=None,marginals=None,nbins=100, copula_tag="gauss", cache_dir=None):
    """
    Create a corner plot with 2D marginals and 1D histograms for simulations,
    and overlay PDF contours. Additionally, compare 2D histograms to analytic PDFs.
    """
    # Read simulations and angles
    sims, angles = read_sims_nd(simspath, njobs, lmax,prefactors=prefactors,theta=theta)
    print('loaded sims with shape:',sims.shape)
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
    # Use cache_dir if provided, else default to simspath
    filepath = cache_dir if cache_dir is not None else simspath
    # Create a corner plot
    fig, axes = plt.subplots(len(selected_data[0]), len(selected_data[0]), figsize=(10, 10))
    colorbar_ax = fig.add_axes([0.96, 0.2, 0.02, 0.6])  # Add a new axis for the colorbar
    im = None  # Initialize the colorbar reference
    for i in range(len(selected_data[0])):
        for j in range(len(selected_data[0])):
            redshift_idx_i, angular_idx_i = divmod(i, len(angular_indices))
            redshift_idx_j, angular_idx_j = divmod(j, len(angular_indices))
            ax = axes[j, i]
            if i == 0:  # Label first column
                ax.set_ylabel("corr {:d}, ang {:d}".format(redshift_idx_j, angular_idx_j))
            else:
                ax.set_yticklabels([])
            if j == len(selected_data[0]) - 1:  # Label bottom row
                ax.set_xlabel("corr {:d}, ang {:d}".format(redshift_idx_i, angular_idx_i))
            else:
                ax.set_xticklabels([])
            
            
            if i == j:
                # 1D histogram
                n, bins, _ = ax.hist(selected_data[:, i], bins=nbins, density=True, alpha=0.6, color='C0')
                ax.set_xlim(bins[0], bins[-1])  # Set x-axis to histogram range
                
                # Generate all possible pairs of indices
                
                    
                
                if j == len(selected_data[0]) - 1:
                    redshift_idx_i, angular_idx_i = divmod(i-1, len(angular_indices))
                    axis = 1
                else:
                    redshift_idx_j, angular_idx_j = divmod(j+1, len(angular_indices))
                    axis = 0
                # Load and overlay the 1D marginal from the random pair
                marginal_data = load_2d_pdf(filepath, redshift_idx_i, angular_idx_i, redshift_idx_j,angular_idx_j,integrate_axis=axis, copula_tag=copula_tag)
                
                if marginal_data is not None:
                    x_marginal, marginal = marginal_data
                    ax.plot(x_marginal, marginal, color='C0', linewidth=2, label="1D Marginal")
                if marginals is not None:
                    ax.plot(x[i],pdf[i], color='C3', linewidth=2, label="1D Marginal (Sim)")
                
            elif i < j:
                # 2D marginal
                
                """ hist = ax.hist2d(
                    selected_data[:, i], selected_data[:, j],
                    bins=nbins, cmap="Reds", density=True
                ) """
                plot_2d_from_cache(ax, filepath, redshift_idx_i, angular_idx_i, redshift_idx_j, angular_idx_j, copula_tag=copula_tag)
                

                
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
                pdf_data = load_2d_pdf(filepath, redshift_idx_j, angular_idx_j, redshift_idx_i, angular_idx_i, copula_tag=copula_tag)
                if pdf_data is not None:
                    x_pdf, y_pdf, pdf_grid = pdf_data

                    # Interpolate PDF to match histogram bin centers
                    interp_pdf  = RegularGridInterpolator((y_pdf,x_pdf), pdf_grid,method='cubic')
                    # the y,x order is weird, but necessary due to the indexing convention of the interpolator vs the plotting convention used elsewhere!
                    im = compare_to_sims_2d(ax, (bincenters_x, bincenters_y), sim_mean, std_dev, interp_pdf, vmax=3,log=False)
                    # Compute relative residuals
                ax.set_xlim(hist[1][0],hist[1][-1])
                ax.set_ylim(hist[2][0],hist[2][-1])
                oldax.set_xlim(hist[1][0],hist[1][-1])
                oldax.set_ylim(hist[2][0],hist[2][-1])
                
                    
                    

    if im is not None:
        fig.colorbar(im, cax=colorbar_ax, label="Relative Residuals (std)")  # Add colorbar to the new axis

    # Adjust layout
    #plt.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)  # Adjust padding to reduce space
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05, wspace=0.2, hspace=0.2)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Corner plot saved to {save_path}")
    else:
        plt.show()

def plot_2d_from_cache(ax, filepath, i, j, k, l, copula_tag="gauss"):
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
        data = load_2d_pdf(filepath, i, j, k, l, copula_tag=copula_tag)
        if data is not None:
            x, y, pdf = data
            X, Y = np.meshgrid(x, y)
            #c = ax.contour(X, Y, pdf, levels=5, cmap="Blues")
            c = ax.pcolormesh(X,Y,np.log(pdf), shading="auto",cmap="Blues",vmin=5,vmax=30)
    except Exception as e:
        print(f"Error loading or plotting cache file: {e}")
    



def load_2d_pdf(filepath, i, j, k, l, integrate_axis=None, copula_tag="gauss"):
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
    cache_file = f"{filepath}/likelihood_2d_cache_{copula_tag}_{i}_{j}_{k}_{l}.npz"
    
    try:
        data = np.load(cache_file)
        print(f"Loaded cache file: {cache_file}")
        xs = data["x"]
        x = xs[0]
        y = xs[1]
        logpdf = data["likelihood_2d"]
        pdf = np.exp(logpdf)
        print(f"Fraction of NaN values in PDF: {np.isnan(logpdf).sum() / logpdf.size:.4f}")
        print(f"Fraction of finite values in PDF: {np.isfinite(logpdf).sum() / logpdf.size:.4f}")
        pdf = np.where(np.isfinite(pdf), pdf, np.nan)
        logpdf = np.where(np.isfinite(logpdf), logpdf, np.nan)
        #X, Y = np.meshgrid(x, y, indexing="ij")
        #points = np.array([X[~np.isnan(pdf)], Y[~np.isnan(pdf)]]).T  # Valid points
        #values = pdf[~np.isnan(pdf)]  # Valid values

        # Interpolate missing values
        #pdf = griddata(points, values, (X, Y), method="cubic")

        pdf = np.nan_to_num(pdf, nan=0.0)
        logpdf = np.nan_to_num(logpdf, nan=0.0)
        

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
        
        return x, y, pdf

    except FileNotFoundError:
        print(f"Cache file {cache_file} not found.")
        return None
    except Exception as e:
        print(f"Error loading or processing cache file: {e}")
        return None