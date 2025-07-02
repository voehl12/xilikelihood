import numpy as np
from file_handling import read_posterior_files
from postprocess_nd_likelihood import exp_norm_mean
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{amsmath}')

filestring = "/cluster/home/veoehl/2ptlikelihood/s8posts/s8post_10000sqd_fiducial_nonoise_largescales_*.npz"
regexs = [r'^(?!.*(autos|crocos)).*$',r'autos',r'crocos']
labels = ['All', 'Auto', 'Cross']

def retrieve_exact_and_gaussian_posteriors(filestring,regex=None):

    posterior = read_posterior_files(filestring, regex=regex)
    gauss_posterior = posterior["gauss_posteriors"]

    exact_posterior = posterior["exact_posteriors"]
    s8 = posterior["s8"]
    order = np.argsort(s8)
    s8 = s8[order]
    exact_posterior = exact_posterior[order]
    gauss_posterior = gauss_posterior[order]
    exact_posterior, exact_mean = exp_norm_mean(s8[30:],exact_posterior[30:],reg=200)
    gauss_posterior, gauss_mean = exp_norm_mean(s8[30:],gauss_posterior[30:],reg=200)
    return s8[30:], exact_posterior, gauss_posterior, exact_mean, gauss_mean

fig, ax = plt.subplots(figsize=(3.5, 3))
colors = plt.cm.viridis(np.linspace(0, 1, 200))
color = colors[2]
colors = ['blue', 'green', 'orange']  # Colors for different regex patterns
for i, regex in enumerate(regexs[:1]):
    s8, exact_posterior, gauss_posterior, exact_mean, gauss_mean = retrieve_exact_and_gaussian_posteriors(filestring, regex=regex)
    ax.plot(s8, exact_posterior, label=r"exact", color=color, linestyle='-')
    ax.axvline(exact_mean, color=color, linestyle='-')
    ax.plot(s8, gauss_posterior, label=r"Gaussian", color=color, linestyle='--')
    ax.axvline(gauss_mean, color=color, linestyle='--')
    print(s8[np.argmax(gauss_posterior)])
    #plt.axvline(s8[np.argmax(gauss_posterior)], color=colors[i], linestyle='--', label=f"Max Gaussian ({labels[i]})")
ax.legend(frameon=False, loc='upper right')
ax.set_xlim(0.55,1.05)
ax.set_xlabel(r"$S_8$")
ax.set_ylabel(r"Posterior")
ax.axvline(0.8,color='C3')
#plt.title("Joint posterior from full likelihood, large scales")

fig.savefig("s8posterior_combined_largescales_10000sqd_talk.pdf",bbox_inches='tight', dpi=300)