import numpy as np
import approximations as app
import configparser
import postprocess_nd_likelihood
import plotting
import matplotlib.pyplot as plt
import scipy.stats
import setup_nd_likelihood
from scipy.interpolate import griddata
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.stats import norm, multivariate_normal
from scipy.interpolate import griddata
from scipy.interpolate import PchipInterpolator
import cl2xi_transforms
import setup_m

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
covs_allcl = setup_nd_likelihood.setup_covariances(
    1000, 256, 30, 10, noise_contribs, cl_names, cl_paths
)
setup_nd_likelihood.setup_cls(new_config, cl_paths, cl_names, noise_contribs)


result = "/cluster/work/refregier/veoehl/2Dcf/xip_5535bins_1000/"
setup_nd_likelihood.setup_filenames(
    new_config,
    "covariances/cov_xip_55_53_l30_n256_circ1000.npz",
    "m_xip4_6_xip4_6_l30_n256_circ1000.npz",
    "tsets_1000.npz",
    result,
)
angbins = postprocess_nd_likelihood.get_angbins(config)
setup_nd_likelihood.setup_likelihood(new_config, covs_allcl, [(0, 0), (1, 0)], angbins, steps=1024)

exit()
paths = config["Paths"]

covs = np.load(paths["cov"])
cov = covs["matrix"]
marrs = np.load(paths["M"])
mset = marrs["matrix"]


def setup_m_cov_combs(covs, ang_bins):
    m_cov_pairs = []
    for cov in covs:
        cov_mat = cov.cov_alm
        exact_lmax = cov.exact_lmax
        prefactors = cl2xi_transforms.prep_prefactors(ang_bins, cov.wl, cov.lmax, cov.lmax)
        m = setup_m.m_xi_cross((prefactors[0, :, : exact_lmax + 1],), combs=((0, 0),), kind=("p",))[
            0
        ]
        m_cov_pairs.append([m, cov_mat])
    return np.array(m_cov_pairs)


x, pdf_exact = postprocess_nd_likelihood.convert_nd_cf_to_pdf(config)

vmax = 4e12

print("Calculating analytical moments...")
# extend the moments_nd function by also returning the 1d marginals
firsts, seconds = app.moments_nd_jitted(mset, cov, 2, 2)
print("Done.")
moments = [firsts, seconds]
print(moments)

cumulants = app.ncmom2cum_nd(moments)


def get_marginals(ms, cov):
    marginals = []
    cdfs = []
    plt.figure()
    for m in ms:
        # replace this with a function that does moments and exact pdf at once (so the matrix multiplication is only done once)
        exact_x, exact_pdf, _, _ = app.get_exact(m, cov, steps=2048)
        pchip_interp = PchipInterpolator(exact_x[500:-500], exact_pdf[500:-500])

        # Evaluate interpolated PDF
        x_vals = np.linspace(exact_x[500], exact_x[-500], 500)

        pdf_vals = pchip_interp(x_vals)
        plt.plot(x_vals, pdf_vals)
        cdf_X = cumtrapz(pdf_vals, x_vals, initial=0)
        cdf_X /= max(cdf_X)  # Normalize to 1 if needed
        cdfs.append(cdf_X)
        marginals.append([x_vals, pdf_vals])

    return np.array(marginals), np.array(cdfs)


def gaussian_copula_density(u, v, rho):
    # Convert u and v to normal space
    z1 = norm.ppf(u)
    z2 = norm.ppf(v)

    # Bivariate normal PDF with correlation rho
    cov_matrix = [[1, rho], [rho, 1]]
    mvn = multivariate_normal(mean=[0, 0], cov=cov_matrix)
    x_grid, y_grid = np.meshgrid(z1, z2)
    test_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T
    bivariate_pdf = mvn.pdf(test_points)

    # Standard normal PDFs
    pdf_z1 = norm.pdf(z1)
    pdf_z2 = norm.pdf(z2)
    plt.figure()
    plt.plot(z1, pdf_z1)
    plt.plot(z2, pdf_z2)
    plt.yscale("log")
    plt.show()

    pdf_z1_grid, pdf_z2_grid = np.meshgrid(pdf_z1, pdf_z2)
    pdf_points = np.vstack([pdf_z1_grid.ravel(), pdf_z2_grid.ravel()]).T

    # Copula density
    copula_density = bivariate_pdf / (np.prod(pdf_points, axis=1))
    return copula_density


def joint_pdf(cdf_X, cdf_Y, pdf_X, pdf_Y, rho):
    # Compute marginals
    u = cdf_X[1:-1]
    v = cdf_Y[1:-1]
    pdf_x_grid, pdf_y_grid = np.meshgrid(pdf_X[1:-1], pdf_Y[1:-1])
    pdf_points = np.vstack([pdf_x_grid.ravel(), pdf_y_grid.ravel()]).T

    # Compute copula density
    copula_density = gaussian_copula_density(u, v, rho)

    # Joint PDF
    return copula_density * np.prod(pdf_points, axis=1)


# m_cov_pairs = setup_m_cov_combs(covs_allcl, angbins)
# marginals, cdfs = get_marginals(m_cov_pairs[:, 0], m_cov_pairs[:, 1])
marginals, cdfs = get_marginals(mset, cov)


stds = np.diag(np.sqrt(cumulants[1]))
cov = cumulants[1][1, 0]
correlation_coefficient = cov / (np.prod(stds))


# Given correlation coefficient
rho = float(correlation_coefficient)  # Computed from covariance
correlation_matrix = [[1, rho], [rho, 1]]

# Compute joint PDF
joint_pdf_values = joint_pdf(cdfs[0], cdfs[1], marginals[0][1], marginals[1][1], rho)

gauss_comp = scipy.stats.multivariate_normal(mean=cumulants[0], cov=cumulants[1])

grid_size = int(np.sqrt(5000))
x_vals = marginals[0][0][1:-1]
y_vals = marginals[1][0][1:-1]

x_grid, y_grid = np.meshgrid(x_vals, y_vals)
test_points = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

sample_inds_x = [
    np.argmin(np.fabs(x[:, 0, 0] - test_points[i, 0])) for i in range(len(test_points))
]
sample_inds_y = [
    np.argmin(np.fabs(x[0, :, 1] - test_points[i, 1])) for i in range(len(test_points))
]

exact_pdf_values = [pdf_exact[sample_inds_x[i], sample_inds_y[i]] for i in range(len(test_points))]


gauss_pdf_values = gauss_comp.pdf(test_points)

grid_z_copula = griddata(test_points, joint_pdf_values, (x_grid, y_grid), method="cubic")
grid_z_gauss = griddata(test_points, gauss_pdf_values, (x_grid, y_grid), method="cubic")
grid_z_exact = griddata(test_points, exact_pdf_values, (x_grid, y_grid), method="cubic")

fig = plt.figure(figsize=(20, 15))
gs = fig.add_gridspec(4, 4, width_ratios=[4, 1, 4, 1], height_ratios=[1, 4, 1, 4])

# Exact plot
ax1 = fig.add_subplot(gs[1, 0])
c1 = ax1.contourf(x_grid, y_grid, grid_z_exact, levels=100, vmax=vmax)
ax1.set_title("Exact")
ax1.set_xlim(0.1e-6, 2e-6)
ax1.set_ylim(0.1e-6, 2e-6)
fig.colorbar(c1, ax=ax1)

# Exact marginals
ax2 = fig.add_subplot(gs[0, 0], sharex=ax1)
marginal_x_exact = np.trapz(grid_z_exact, y_vals, axis=0)
ax2.plot(x_vals, marginal_x_exact, color="blue")
ax2.set_title("Exact Marginal X")
ax2.set_xlim(0.1e-6, 2e-6)
ax2.set_yticks([])

ax3 = fig.add_subplot(gs[1, 1], sharey=ax1)
marginal_y_exact = np.trapz(grid_z_exact, x_vals, axis=1)
ax3.plot(marginal_y_exact, y_vals, color="red")
ax3.set_title("Exact Marginal Y")
ax3.set_ylim(0.1e-6, 2e-6)
ax3.set_xticks([])

# Copula plot
ax4 = fig.add_subplot(gs[1, 2])
c2 = ax4.contourf(x_grid, y_grid, grid_z_copula, levels=100, vmax=vmax)
ax4.set_title("Copula")
ax4.set_xlim(0.1e-6, 2e-6)
ax4.set_ylim(0.1e-6, 2e-6)
fig.colorbar(c2, ax=ax4)

# Copula marginals
ax5 = fig.add_subplot(gs[0, 2], sharex=ax4)
marginal_x_copula = np.trapz(grid_z_copula, y_vals, axis=0)
ax5.plot(x_vals, marginal_x_copula, color="blue")
ax5.set_title("Copula Marginal X")
ax5.set_xlim(0.1e-6, 2e-6)
ax5.set_yticks([])

ax6 = fig.add_subplot(gs[1, 3], sharey=ax4)
marginal_y_copula = np.trapz(grid_z_copula, x_vals, axis=1)
ax6.plot(marginal_y_copula, y_vals, color="red")
ax6.set_title("Copula Marginal Y")
ax6.set_ylim(0.1e-6, 2e-6)
ax6.set_xticks([])

# Gaussian plot
ax7 = fig.add_subplot(gs[3, 0])
c3 = ax7.contourf(x_grid, y_grid, grid_z_gauss, levels=100, vmax=vmax)
ax7.set_title("Gaussian")
ax7.set_xlim(0.1e-6, 2e-6)
ax7.set_ylim(0.1e-6, 2e-6)
fig.colorbar(c3, ax=ax7)

# Gaussian marginals
ax8 = fig.add_subplot(gs[2, 0], sharex=ax7)
marginal_x_gauss = np.trapz(grid_z_gauss, y_vals, axis=0)
ax8.plot(x_vals, marginal_x_gauss, color="blue")
ax8.set_title("Gaussian Marginal X")
ax8.set_xlim(0.1e-6, 2e-6)
ax8.set_yticks([])

ax9 = fig.add_subplot(gs[3, 1], sharey=ax7)
marginal_y_gauss = np.trapz(grid_z_gauss, x_vals, axis=1)
ax9.plot(marginal_y_gauss, y_vals, color="red")
ax9.set_title("Gaussian Marginal Y")
ax9.set_ylim(0.1e-6, 2e-6)
ax9.set_xticks([])

fig.tight_layout()
fig.savefig("copula.png")

plt.figure()
plt.plot(x_vals, marginals[0][1][1:-1], label="x marginal")
plt.plot(y_vals, marginals[1][1][1:-1], label="y marginal")
plt.plot(x_vals, marginal_x_exact, label="x marginal from 2d")
plt.plot(y_vals, marginal_y_exact, label="y marginal from 2d")
plt.legend()
plt.savefig("marginals.png")
