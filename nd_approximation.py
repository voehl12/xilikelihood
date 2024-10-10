import numpy as np
from approximations import MultiNormalExpansion, moments_nd, ncmom2cum_nd
import configparser
import postprocess_nd_likelihood
import plotting
import matplotlib.pyplot as plt


configpath = "/cluster/home/veoehl/2ptlikelihood/config_adjusted.ini"
simspath = "/cluster/work/refregier/veoehl/2ptlikelihood/sims/"
config = postprocess_nd_likelihood.load_config(configpath)

paths = config["Paths"]
print(paths)
covs = np.load(paths["cov"])
cov = covs["matrix"]
marrs = np.load(paths["M"])
mset = marrs["matrix"]

x, pdf_exact = postprocess_nd_likelihood.convert_nd_cf_to_pdf(config)

vmax = 4e12

moments = moments_nd(mset, cov)

moments_sims = postprocess_nd_likelihood.compare_to_sims(config, simspath)
plt.figure()
plt.plot(moments_sims[2] - moments[2], label="3rd moments difference")
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
