import numpy as np
from config import OUTPUT_DIR
import xilikelihood as xlh
from xilikelihood.file_handling import read_posterior_files
from xilikelihood.distributions import exp_norm_mean
import matplotlib.pyplot as plt
import logging
import sys
import cmasher as cmr
from matplotlib.lines import Line2D
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')

filestring = str(OUTPUT_DIR / 's8post_10000sqd_fiducial_nonoise_nd_*.npz')

log_dir = OUTPUT_DIR / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 's8_posteriors_postprocess.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

regexs = [r'^(?!.*(autos|crocos)).*$',r'autos',r'crocos']
labels = ['All', 'Auto', 'Cross']
regexs_scales = [r'ang{:d}'.format(i) for i in range(3)] #largest angle way too broad
regexs_scales.append(r'allang')
def retrieve_exact_and_gaussian_posteriors(filestring,regex=None):

    posterior = read_posterior_files(filestring, regex=regex)
    gauss_posterior = posterior["gauss_posteriors"]
    angle = posterior.get("angles", None)
    exact_posterior = posterior["exact_posteriors"]
    s8 = posterior["s8"]
    order = np.argsort(s8)
    s8 = s8[order]
    exact_posterior = exact_posterior[order]
    
    gauss_posterior = gauss_posterior[order]
    #print(s8,exact_posterior,gauss_posterior)
    x_exact, exact_posterior, exact_mean, exact_std = exp_norm_mean(s8,exact_posterior)
    x_gauss, gauss_posterior, gauss_mean, gauss_std = exp_norm_mean(s8,gauss_posterior)
    return x_exact, exact_posterior, x_gauss, gauss_posterior, exact_mean, gauss_mean, exact_std, gauss_std, angle

#fig, ax = plt.subplots(figsize=(3.5, 3))
fig,ax = plt.subplots(figsize=(5, 4))
colors = plt.cm.viridis(np.linspace(0, 1, 200))
color = colors[2]
colors = ['blue', 'green', 'orange']  # Colors for different regex patterns
pdf_cm = cmr.torch
colors = cmr.take_cmap_colors(pdf_cm, 4, cmap_range=(0.3, 0.7), return_fmt='hex')
y0 = -14   # vertical baseline for mean-shift arrows
dy = 3  # spacing between bins



    # (optional) Label on the left
    

for i, regex in enumerate(regexs_scales):
    y = y0 + i*dy
    color=colors[i]
    s8_exact, exact_posterior, s8_gauss, gauss_posterior, exact_mean, gauss_mean, exact_std, gauss_std, angle = retrieve_exact_and_gaussian_posteriors(filestring, regex=regex)
    shift = gauss_mean - exact_mean
    #
    print(angle[0])
    ang = r'$\bar{{\theta}}_{:d} = [{:.2f}^{{\circ}},{:.2f}^{{\circ}}]$'.format(i+1, angle[0][0][0], angle[0][0][1])
    if i == len(regexs_scales)-1:
        ang = None
    ax.errorbar(exact_mean, y,
                xerr=exact_std,
                fmt='o', color=color, capsize=4, markersize=6,
                label=ang, alpha=0.5)
    if abs(shift) < 0.001:
        ax.plot(exact_mean,y,'o', color=color, markersize=3)
    else:
        ax.annotate("",
                xy=(gauss_mean, y),
                xytext=(exact_mean, y),
                arrowprops=dict(arrowstyle="<|-", lw=2, color=color, mutation_scale=4))
    #ax.text(0.738, y, b, va='center', ha='left', color=colors[i], fontsize=12)
color=colors[-1]
s8_exact, exact_posterior, s8_gauss, gauss_posterior, exact_mean, gauss_mean, exact_std, gauss_std, angle = retrieve_exact_and_gaussian_posteriors(filestring, regex=regexs_scales[-1])  
ang = r'all $\bar{{\theta}}$'
ax.plot(s8_exact, exact_posterior, label=ang, color=color, linestyle='-')

ax.plot(s8_gauss, gauss_posterior,color=color, linestyle='--')

legend1 = ax.legend(frameon=False, loc='upper right')
ymin = y0 - dy
ax.set_ylim(ymin, gauss_posterior.max()*1.05)

ylim_bottom, ylim_top = ax.get_ylim()
y_norm = (y - ylim_bottom) / (ylim_top - ylim_bottom)
ymax_exact = (exact_posterior.max() - ylim_bottom) / (ylim_top - ylim_bottom)
ymax_gauss = (gauss_posterior.max() - ylim_bottom) / (ylim_top - ylim_bottom)
ax.axvline(exact_mean, ymin=y_norm,ymax=ymax_exact, color=color, linestyle='-',alpha=0.7,linewidth=1)
ax.axvline(gauss_mean, ymin=y_norm, ymax=ymax_gauss, color=color, linestyle='--',alpha=0.7,linewidth=1)
# Second legend for likelihood types
likelihood_lines = [
    Line2D([0], [0], color='black', linestyle='-', label=r'Copula Likelihood'),
    Line2D([0], [0], color='black', linestyle='--', label=r'Gaussian Likelihood')
]
legend2 = ax.legend(handles=likelihood_lines, frameon=False, loc='upper left')

# Add first legend back (matplotlib removes it when creating second legend)
ax.add_artist(legend1)

ax.set_xlim(0.75,0.85)
ax.set_xlabel(r"$S_8$")
ax.set_ylabel(r"Posterior")
#ax.axvline(0.8,color='C3')
#plt.title("Joint posterior from full likelihood, large scales")

fig.savefig("s8posterior_combined_scales_10000sqd.pdf",bbox_inches='tight', dpi=300)
fig.savefig("s8posterior_combined_scales_10000sqd.png",bbox_inches='tight', dpi=300)