import numpy as np
from approximations import MultiNormalExpansion, GeneralizedLaplace, moments_nd_jitted, ncmom2cum_nd
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
simspath = (
    "/cluster/work/refregier/veoehl/xi_sims/croco_3x2pt_kids_33_circ1000smoothl30_noisedefault/"
)
config = postprocess_nd_likelihood.load_config(configpath)

paths = config["Paths"]
print(paths)
covs = np.load(paths["cov"])
cov = covs["matrix"]
marrs = np.load(paths["M"])
mset = marrs["matrix"]

x, pdf_exact = postprocess_nd_likelihood.convert_nd_cf_to_pdf(config)

vmax = 4e12
""" sims = postprocess_nd_likelihood.load_sims(config, simspath, njobs=100)
sims = np.array(sims)
orders = [1, 2, 3]
print("Calculating statistics...")
moments_sims = postprocess_nd_likelihood.get_stats_from_sims(sims, axis=1)
bootstrap_func = postprocess_nd_likelihood.get_stats_from_sims
thirds_bootstrap = postprocess_nd_likelihood.bootstrap(
    sims, 1000, axis=1, func=bootstrap_func, func_kwargs={"orders": [3]}
) """

print("Calculating analytical moments...")
firsts, seconds, thirds = moments_nd_jitted(mset, cov, 2)
print("Done.")
moments = [firsts, seconds, thirds]
print(moments)
# thirds_sims = np.array(moments_sims[2])

# print(moments_sims)
# thirds_std = np.std(thirds_bootstrap, axis=0)
# print(thirds_std)
# print((np.array(thirds) - thirds_sims) / thirds_std)

cumulants = ncmom2cum_nd(moments)
gen_laplace_mix = [firsts, cumulants[1], thirds]
approx = MultiNormalExpansion(cumulants)
approx_laplace = GeneralizedLaplace(gen_laplace_mix)
third_cumulant_normalized = approx.normalize_third_cumulant()
print("Normalized third cumulant: {}".format(third_cumulant_normalized))

gauss_comp = scipy.stats.multivariate_normal(mean=cumulants[0], cov=cumulants[1])
test_points = np.random.uniform(0, 2e-6, (5000, 2))
sample_inds_x = [
    np.argmin(np.fabs(x[:, 0, 0] - test_points[i, 0])) for i in range(len(test_points))
]
sample_inds_y = [
    np.argmin(np.fabs(x[0, :, 1] - test_points[i, 1])) for i in range(len(test_points))
]

exact_pdf_values = [pdf_exact[sample_inds_x[i], sample_inds_y[i]] for i in range(len(test_points))]

edgeworth_pdf_values = approx.pdf(test_points)
gauss_pdf_values = gauss_comp.pdf(test_points)
laplace_pdf_values = approx_laplace.pdf(test_points)
print(approx_laplace.param_moments, gen_laplace_mix)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
ax1.scatter(test_points[:, 0], test_points[:, 1], c=edgeworth_pdf_values, vmax=vmax)
ax1.set_title("Edgeworth")
ax2.scatter(test_points[:, 0], test_points[:, 1], c=gauss_pdf_values, vmax=vmax)
ax2.set_title("Gaussian")
ax3.scatter(test_points[:, 0], test_points[:, 1], c=exact_pdf_values, vmax=vmax)
ax3.set_title("Exact")
ax4.scatter(
    test_points[:, 0],
    test_points[:, 1],
    c=laplace_pdf_values,
)
ax4.set_title("Laplace")

fig.savefig("2d_comp_edgeworth.png")
