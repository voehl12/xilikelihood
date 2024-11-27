import numpy as np
from approximations import MultiNormalExpansion, GeneralizedLaplace, moments_nd_jitted, ncmom2cum_nd
import configparser
import postprocess_nd_likelihood
import plotting
import matplotlib.pyplot as plt
import scipy.stats
import setup_nd_likelihood
from scipy.interpolate import griddata

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
cl_55 = "Cl_3x2pt_kids55.txt"
cl_53 = "Cl_3x2pt_kids53.txt"
cl_33 = "Cl_3x2pt_kids33.txt"
cl_paths = (cl_33, cl_55, cl_53)
cl_names = ("3x2pt_kids_33", "3x2pt_kids_55", "3x2pt_kids_53")
noise_contribs = ("default", "default", None)
new_config = setup_nd_likelihood.setup_config()
covs = setup_nd_likelihood.setup_covariances(10000, 256, 30, 10, noise_contribs, cl_names, cl_paths)
setup_nd_likelihood.setup_cls(new_config, cl_paths, cl_names, noise_contribs)


result = "/cluster/work/refregier/veoehl/2Dcf/xip_5535bins_10000/"
setup_nd_likelihood.setup_filenames(
    new_config,
    "covariances/cov_xip_55_53_l30_n256_circ1000.npz",
    "m_xip4_6_xip4_6_l30_n256_circ10000.npz",
    "tsets_10000.npz",
    result,
)
angbins = postprocess_nd_likelihood.get_angbins(config)
setup_nd_likelihood.setup_likelihood(new_config, covs, [(0, 0), (1, 0)], angbins, steps=1024)

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
approx_laplace = GeneralizedLaplace(moments=gen_laplace_mix)
third_cumulant_normalized = approx.normalize_third_cumulant()
print("Normalized third cumulant: {}".format(third_cumulant_normalized))

gauss_comp = scipy.stats.multivariate_normal(mean=cumulants[0], cov=cumulants[1])
grid_size = int(np.sqrt(5000))
x_vals = np.linspace(1e-7, 2e-6, grid_size)
y_vals = np.linspace(1e-7, 2e-6, grid_size)
x_grid, y_grid = np.meshgrid(x_vals, y_vals)
test_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
# Interpolate the data for smooth plotting
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

grid_z_edgeworth = griddata(test_points, edgeworth_pdf_values, (x_grid, y_grid), method="cubic")
grid_z_gauss = griddata(test_points, gauss_pdf_values, (x_grid, y_grid), method="cubic")
grid_z_exact = griddata(test_points, exact_pdf_values, (x_grid, y_grid), method="cubic")
grid_z_laplace = griddata(test_points, laplace_pdf_values, (x_grid, y_grid), method="cubic")

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 10))
c1 = ax1.contourf(x_grid, y_grid, grid_z_edgeworth, levels=100, vmax=vmax)
ax1.set_title("Edgeworth")
ax1.set_xlim(0.1e-6, 2e-6)
ax1.set_ylim(0.1e-6, 2e-6)
fig.colorbar(c1, ax=ax1)

c2 = ax2.contourf(x_grid, y_grid, grid_z_gauss, levels=100, vmax=vmax)
ax2.set_title("Gaussian")
ax2.set_xlim(0.1e-6, 2e-6)
ax2.set_ylim(0.1e-6, 2e-6)
fig.colorbar(c2, ax=ax2)

c3 = ax3.contourf(x_grid, y_grid, grid_z_exact, levels=100, vmax=vmax)
ax3.set_title("Exact")
ax3.set_xlim(0.1e-6, 2e-6)
ax3.set_ylim(0.1e-6, 2e-6)
fig.colorbar(c3, ax=ax3)

c4 = ax4.contourf(x_grid, y_grid, grid_z_laplace, levels=100)
ax4.set_title("Laplace")
ax4.set_xlim(0.1e-6, 2e-6)
ax4.set_ylim(0.1e-6, 2e-6)
fig.colorbar(c4, ax=ax4)

fig.savefig("2d_comp_edgeworth.png")

""" print(approx_laplace.param_moments, gen_laplace_mix)

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
fig.savefig("2d_comp_edgeworth.png") """
