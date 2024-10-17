import numpy as np
from approximations import MultiNormalExpansion, moments_nd, ncmom2cum_nd
import configparser
import postprocess_nd_likelihood
import plotting
import matplotlib.pyplot as plt
import scipy.stats

#  1) stack/2024-06   3) swig/4.1.1-ipvpwcc   5) python/3.11.6
#  2) gcc/12.2.0      4) cmake/3.27.7
# module load stack/2024-06 swig/4.1.1-ipvpwcc python/3.11.6 gcc/12.2.0 cmake/3.27.7
# should eventually become a class
configpath = "/cluster/home/veoehl/2ptlikelihood/config_adjusted.ini"
configpath = "config_adjusted.ini"
# simspath = (
#   "/cluster/work/refregier/veoehl/xi_sims/croco_3x2pt_kids_33_circ1000smoothl30_noisedefault/"
# )
config = postprocess_nd_likelihood.load_config(configpath)

paths = config["Paths"]
print(paths)
covs = np.load(paths["cov"])
cov = covs["matrix"]
marrs = np.load(paths["M"])
mset = marrs["matrix"]

x, pdf_exact = postprocess_nd_likelihood.convert_nd_cf_to_pdf(config)

vmax = 4e12
# moments_sims = postprocess_nd_likelihood.compare_to_sims(config, simspath)
print("Calculating analytical moments...")
firsts, seconds, thirds = moments_nd(mset, cov)
moments = [firsts, seconds, thirds]
print(thirds)
# thirds_sims = np.array(moments_sims[2])
# print(thirds_sims)
# plt.figure()
# plt.plot(thirds_sims - thirds, label="3rd moments difference")
# plt.legend()
# plt.savefig("3rd_moments_diff.png")

cumulants = ncmom2cum_nd(moments)

approx = MultiNormalExpansion(cumulants)
third_cumulant_normalized = approx.normalize_third_cumulant()
print("Normalized third cumulant: {}".format(third_cumulant_normalized))

gauss_comp = scipy.stats.multivariate_normal(mean=cumulants[0], cov=cumulants[1])
test_points = gauss_comp.rvs(1000)
sample_inds_x = [
    np.argmin(np.fabs(x[:, 0, 0] - test_points[i, 0])) for i in range(len(test_points))
]
sample_inds_y = [
    np.argmin(np.fabs(x[0, :, 1] - test_points[i, 1])) for i in range(len(test_points))
]

exact_pdf_values = [pdf_exact[sample_inds_x[i], sample_inds_y[i]] for i in range(len(test_points))]

edgeworth_pdf_values = approx.pdf(test_points)
gauss_pdf_values = gauss_comp.pdf(test_points)

fig, ax = plt.subplots(1, 3, figsize=(10, 6))
ax[0].scatter(test_points[:, 0], test_points[:, 1], c=edgeworth_pdf_values, vmax=vmax)
ax[0].set_title("Edgeworth")
ax[1].scatter(test_points[:, 0], test_points[:, 1], c=gauss_pdf_values, vmax=vmax)
ax[1].set_title("Gaussian")
ax[2].scatter(test_points[:, 0], test_points[:, 1], c=exact_pdf_values, vmax=vmax)
ax[2].set_title("Exact")

fig.savefig("2d_comp_edgeworth.png")
