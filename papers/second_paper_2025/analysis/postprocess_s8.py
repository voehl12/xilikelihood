import numpy as np
import scipy
from likelihood import XiLikelihood, fiducial_dataspace
import theory_cl
import os
import matplotlib.pyplot as plt
import re
from simulate import xi_sim_nD
from scipy.integrate import cumulative_trapezoid as cumtrapz
import sys
import matplotlib.cm as cm
from matplotlib.ticker import LogLocator, LogFormatter
import matplotlib as mpl
from postprocess_nd_likelihood import exp_norm_mean

numjobs = 200
gauss_posteriors, exact_posteriors, s8, means = [], [], [], []
for i in range(numjobs):
    try:
        posts = np.load('s8posts/s8post_1000sqd_{:d}_fiducial_nonoise.npz'.format(i))
        gauss_post, exact_post = posts['gauss'], posts['exact']
        gauss_posteriors.append(gauss_post)
        exact_posteriors.append(exact_post)
        s8.append(posts['s8'])
        means.append(posts['means'])
    except:
        continue

s8 = np.array(s8)
means = np.array(means)
#s8 = np.linspace(0.6, 0.9, 100)
print(exact_posteriors,gauss_posteriors)





ang_bins = [(0.4541123210836613, 1.010257752338312), (1.010257752338312, 2.247507232845216), (2.247507232845216, 5.000000000000002)]
# Normalize the posteriors
angs = [np.mean(angbin) for angbin in ang_bins]
angs = np.array(angs)
normalized_post, mean_exact = exp_norm_mean(s8,exact_posteriors) 
normalized_gauss_post, mean_gauss = exp_norm_mean(s8,gauss_posteriors)

plt.figure()
plt.plot(s8, normalized_gauss_post, color="red", label="Gaussian Likelihood")
plt.axvline(mean_gauss,color='red')
plt.plot(s8, normalized_post, color="blue", label="Copula Likelihood")
plt.axvline(mean_exact,color='blue')
plt.xlim(0.6,1.0)
plt.xlabel("s8")
plt.ylabel("Posterior")
plt.legend()
plt.savefig("s8_posterior_fiducial_1000sqd_nonoise.png")
exit()
num_redshift_bins = 5

fig, axes = plt.subplots(num_redshift_bins, num_redshift_bins, figsize=(15, 15))
fig.subplots_adjust(hspace=0.3, wspace=0.3)

# Define a color map
colors = cm.viridis(np.linspace(0, 1, len(s8)))
import distributions
# Plot each subplot
fid_index = np.argmin(np.fabs(s8-0.8))
np.savez('fiducial_data.npz',data=means[fid_index])
mpl.style.use('classic')
for i in range(num_redshift_bins):
    for j in range(i + 1):
        ax = axes[i, j]
        n = distributions.get_cov_n((i,j))
        for k, s in enumerate(s8):
            
            color = 'red' if k == fid_index else colors[k] 
            linewidth = 1.5 if k == fid_index else 0.3
            ax.plot(angs*60, means[k, n, :]*angs*60*10**4, color=color,linewidth=linewidth)
            ax.set_xscale('log')
            ax.set_ylim(-2,5)
            ax.set_xlim(7e1,2e2)
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
            ax.xaxis.set_major_formatter(LogFormatter(base=10.0))
        ax.set_title(f'{i+1}{j+1}')
        if i == 4:
            ax.set_xlabel('Angular Separation [arcmin]')
        if j == 0:
            ax.set_ylabel('Correlation')
        #if i == num_redshift_bins - 1 and j == 0:
        #    ax.legend()
        

# Turn off unused subplots
for i in range(num_redshift_bins):
    for j in range(i + 1, num_redshift_bins):
        axes[i, j].axis('off')

fig.savefig('all_correlations.png')