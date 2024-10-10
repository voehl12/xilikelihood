import numpy as np
from approximations import MultiNormalExpansion, moments_nd, ncmom2cum_nd
import configparser
import postprocess_nd_likelihood
import plotting
import matplotlib.pyplot as plt
#  1) stack/2024-06   3) swig/4.1.1-ipvpwcc   5) python/3.11.6
#  2) gcc/12.2.0      4) cmake/3.27.7

# should eventually become a class
configpath = "/cluster/home/veoehl/2ptlikelihood/config_adjusted.ini"
simspath = "/cluster/work/refregier/veoehl/xi_sims/croco_3x2pt_kids_33_circ1000smoothl30_noisedefault/"
config = postprocess_nd_likelihood.load_config(configpath)

paths = config["Paths"]
print(paths)
covs = np.load(paths["cov"])
cov = covs["matrix"]
marrs = np.load(paths["M"])
mset = marrs["matrix"]

x, pdf_exact = postprocess_nd_likelihood.convert_nd_cf_to_pdf(config)

vmax = 4e12
moments_sims = postprocess_nd_likelihood.compare_to_sims(config, simspath)
print('Calculating analytical moments...')
firsts, seconds, thirds = moments_nd(mset, cov)
print(thirds)
moments = [firsts, seconds, thirds]
print(moments_sims)
thirds_sims = np.array(moments_sims[2])
print(thirds_sims)
plt.figure()
plt.plot(thirds_sims - thirds, label="3rd moments difference")
plt.legend()
plt.savefig("3rd_moments_diff.png")

cumulants = ncmom2cum_nd(moments)
approx = MultiNormalExpansion(cumulants)
pdf_edgeworth = approx.pdf(x.reshape(-1, 2))


lims_low = ((-2, 12), (-1, 8))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6.14))
g = plotting.plot_2D(fig, ax1, x[:, 0, 0], x[0, :, 1], pdf_exact.T, vmax=vmax)
h = plotting.plot_2D(fig, ax2, x[:, 0, 0], x[0, :, 1], pdf_edgeworth.reshape(1023, 1023), vmax=vmax)


fig.savefig("2d_comp_edgeworth.png")
