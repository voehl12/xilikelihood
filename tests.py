# tests for the field class:
import numpy as np
import matplotlib.pyplot as plt

# settings class that sets lmax and noise level and kind of mask, so everything is clean, consistent and gets passed the same parameters
# what to add to simulated fields to make the noise align?


# should calculate Wllmm.
# For no mask, these should be trivial (no mixing).
# Could also add test case for circular mask with 1000 sqd.
# Size should correspond to given ell_max
def test_palm_matching():
    import cov_calc

    palm_kinds = ["ReE", "ImE", "ReB", "ImB"]
    print(cov_calc.match_alm_inds(palm_kinds))
    assert cov_calc.match_alm_inds(palm_kinds) == [0, 1, 2, 3], "Function no work"


def test_cl_class():
    from scipy.special import j0
    import scipy.integrate as integrate
    import grf_classes
    import matplotlib.pyplot as plt

    new_cl = grf_classes.TheoryCl(30, path="Cl_3x2pt_kids55.txt")

    assert np.allclose(new_cl.ee, np.zeros(31)), "Something wrong with zero Cl assignment"


def test_cov_xi():
    import cov_setup
    import matplotlib.pyplot as plt

    covs = np.load("../corrfunc_distr/cov_xip_l10_n256_circ1000.npz")
    cov_xip = covs["cov"]
    circ_cov = cov_setup.Cov(10, [2], clpath="Cl_3x2pt_kids55.txt", circmaskattr=(1000, 256))

    test_cov = circ_cov.cov_alm_xi(pos_m=True)
    assert np.allclose(cov_xip, test_cov), "covariance calculation wrong"
    nomask_cov = cov_setup.Cov(10, [2], clpath="Cl_3x2pt_kids55.txt", circmaskattr=("fullsky", 256))
    nomask_cov_array = nomask_cov.cov_alm_xi(pos_m=True)
    diag = np.diag(nomask_cov_array)
    diag_arr = np.diag(diag)
    assert np.array_equal(nomask_cov_array, diag_arr)
    nomask_cov.maskname = "disguised_fullsky"
    nomask_cov.set_covalmpath()
    nomask_bruteforce_cov = nomask_cov.cov_alm_xi(pos_m=True)
    assert np.allclose(nomask_bruteforce_cov - nomask_cov_array, np.zeros_like(nomask_cov_array)), (
        nomask_bruteforce_cov - nomask_cov_array
    )[
        np.argwhere(
            np.invert(
                (
                    np.isclose(
                        nomask_bruteforce_cov - nomask_cov,
                        np.zeros_like(nomask_cov),
                        atol=1e-10,
                    )
                )
            )
        )
    ]

    # next: constant C_l, pure noise implementation


def noise_test():
    import simulate

    nside = 256
    bins = [(n / 10, (n + 1) / 10) for n in range(1, 100)]
    simulate.xi_sim(
        0,
        None,
        bins,
        mask=None,
        mode="both",
        batchsize=1,
        sigma_n="default",
        nside=nside,
        testing=True,
    )


def noise_corrs():
    import simulate
    import grf_classes
    import matplotlib.pyplot as plt

    nside = 256
    kids55_cl = grf_classes.TheoryCl(3 * nside - 1, path="Cl_3x2pt_kids55.txt")

    bins = [(n / 10, (n + 1) / 10) for n in range(1, 100)]
    maskareas = [1000, 4000, 10000, "fullsky"]
    scalecuts = [30, 100, 3 * nside - 1]
    linestyles = ["dotted", "dashed", "solid", "dashdot"]
    i = 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    for m in maskareas:
        circ_mask = grf_classes.SphereMask([2], circmaskattr=(m, nside))
        color = "C{:d}".format(i)
        i += 1
        for ls, lm in enumerate(scalecuts):
            simulate.xi_sim(
                0,
                None,
                bins,
                mask=circ_mask.mask,
                mode="both",
                batchsize=1,
                sigma_n="default",
                nside=nside,
                noiselmax=lm,
                testing=False,
            )
            sims = np.load("simulations/job0.npz")
            corr_namaster = sims["xip_n"][0, 0]
            corr_treecorr = sims["xip_t"][0]
            bins2 = sims["theta"]
            angsep = [(bins2[k, 0] + bins2[k, 1]) / 2 for k in range(len(bins2))]
            if i == 1:
                ax1.plot(
                    angsep,
                    corr_namaster,
                    color=color,
                    linestyle=linestyles[ls],
                    label=lm,
                )
                ax1.plot(
                    angsep,
                    corr_treecorr,
                    color=color,
                    alpha=0.5,
                    linestyle=linestyles[ls],
                )
                ax1.set_xlim(1, 10)
            if ls == 0:
                ax2.plot(
                    angsep,
                    corr_namaster,
                    color=color,
                    linestyle=linestyles[ls],
                    label="{} sqd".format(str(m)),
                )
                ax2.plot(
                    angsep,
                    corr_treecorr,
                    color=color,
                    alpha=0.5,
                    linestyle=linestyles[ls],
                )
    ax1.set_xlim(1, 10)
    ax1.set_ylim(-2e-6, 2e-6)
    ax1.axhline(y=1e-7, color="black", linestyle="--")
    ax2.axhline(y=1e-7, color="black", linestyle="--")
    ax2.set_xlim(1, 10)
    ax2.set_ylim(-2e-6, 2e-6)
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel(r"$\theta$ [deg]")
    ax2.set_xlabel(r"$\theta$ [deg]")
    ax1.set_ylabel(r"$\xi^+$")
    ax2.set_ylabel(r"$\xi^+$")
    plt.savefig("noise_correlation_nside{:d}.pdf".format(nside))
    plt.show()

    # mask, fullsky, pure noise
    # on the full sky, treecorr underestimates, or namaster too high? check normalization factor
    # assert np.allclose(sims['xip_n'][:,0,0],sims['xip_t'][:,0],atol=1e-07), 'treecorr and namaster do not agree'


def test_xip_pdf():
    import calc_pdf, grf_classes
    import matplotlib.pyplot as plt

    angbin = (1, 2)
    lmax = 30
    mask = grf_classes.SphereMask([2], circmaskattr=(1000, 256), lmax=lmax)
    kids55_cl = grf_classes.TheoryCl(lmax, path="Cl_3x2pt_kids55.txt")
    x, pdf, norm, mean = calc_pdf.pdf_xi_1D(
        angbin,
        c_ell_object=kids55_cl,
        lmax=lmax,
        kind="p",
        mask=mask,
        steps=2048,
        savestuff=True,
    )
    # xnoisel,pdfnoisel,norm = calc_pdf.pdf_xi_1D(angbin,c_ell_object=kids55_cl,sigma_n=1.0,lmax=lmax,kind='p',mask=mask,steps=2048,savestuff=True)
    # xnoises,pdfnoises,norm = calc_pdf.pdf_xi_1D(angbin,c_ell_object=kids55_cl,sigma_n=0.23,lmax=lmax,kind='p',mask=mask,steps=2048,savestuff=True)
    xnoised, pdfnoised, norm, meannoise = calc_pdf.pdf_xi_1D(
        angbin,
        c_ell_object=kids55_cl,
        sigma_n="default",
        lmax=lmax,
        kind="p",
        mask=mask,
        steps=2048,
        savestuff=True,
    )
    return x, xnoised, pdf, pdfnoised, mean, meannoise
    """ plt.figure()
    plt.plot(x,pdf,label=r'no noise')
    plt.plot(xnoisel,pdfnoisel,label=r'$\sigma_{\epsilon} = 1.0$')
    plt.plot(xnoised,pdfnoised,label=r'$\sigma_{\epsilon} = 0.4$')
    plt.plot(xnoises,pdfnoises,label=r'$\sigma_{\epsilon} = 0.23$')
    plt.xlim(-1e-6,3e-6)
    plt.legend()
    plt.savefig('lh_xip_{:d}_{:d}deg_lm30_fullsky.png'.format(*angbin))
    plt.show()
 """


def test_gaussian_cov(theta, binmin_in_arcmin, binmax_in_arcmin, noise_apo=False):
    import grf_classes, setup_cov

    import matplotlib.pyplot as plt

    nside = 256
    lmax = 30
    lmin = 0
    kids55_cl = grf_classes.TheoryCl(lmax, path="Cl_3x2pt_kids55.txt")

    maskarea = 1000
    fsky = maskarea / 41253

    mean, cov, cov_sn = setup_cov.cov_xi_gaussian(
        kids55_cl,
        fsky,
        binmin_in_arcmin,
        binmax_in_arcmin,
        lmin=lmin,
        noise_apo=noise_apo,
    )
    print(cov)
    meann, covn, cov_sn = setup_cov.cov_xi_gaussian(
        kids55_cl,
        fsky,
        binmin_in_arcmin,
        binmax_in_arcmin,
        sigma_e="default",
        lmin=lmin,
        noise_apo=noise_apo,
    )
    print(cov_sn)

    x = np.linspace(-2e-6, 6e-5, 1000)
    return x, mean, cov, meann, covn, cov_sn


def noise_level_scale():
    import matplotlib.pyplot as plt
    import scipy.stats as stats
    from helper_funcs import get_global_sigma_n

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    binedges = np.logspace(np.log10(0.5), np.log10(300), 10)
    binsizes = [(binedges[j + 1] - binedges[j]) for j in range(len(binedges) - 1)]
    angs = [(binedges[j + 1] - binedges[j]) / 2 + binedges[j] for j in range(len(binedges) - 1)]
    print(angs)
    comp_cov = [
        1.9861e-09,
        4.8692e-10,
        1.2242e-10,
        3.2808e-11,
        1.0085e-11,
        3.8592e-12,
        1.7718e-12,
        8.2770e-13,
        3.3261e-13,
    ]
    comp_cov_nonoise = [
        1.7159e-12,
        1.6918e-12,
        1.6244e-12,
        1.5173e-12,
        1.3653e-12,
        1.1437e-12,
        8.3793e-13,
        4.9824e-13,
        2.2372e-13,
    ]
    comp_cov_sn = [
        1.9743e-09,
        4.7649e-10,
        1.1500e-10,
        2.7754e-11,
        6.6982e-12,
        1.6165e-12,
        3.9014e-13,
        9.4157e-14,
        2.2724e-14,
    ]
    diffs, means, diffs_sv, fac_diffs = [], [], [], []
    binstart = 3
    for i, ang in enumerate(angs[binstart:]):
        binsize_in_arcmin = binsizes[i]
        color = "C{:d}".format(i)
        x, mean, cov, meann, covn, cov_sn = test_gaussian_cov(
            ang / 60, binedges[i + binstart], binedges[i + binstart + 1]
        )
        pure_noise_sigma4 = (
            get_global_sigma_n(1000, binedges[i + binstart] / 60, binedges[i + binstart + 1] / 60)
            ** 2
        )
        pdf = stats.norm.pdf(x, mean, np.sqrt(cov))
        pdfnoise = stats.norm.pdf(x, meann, np.sqrt(covn))
        pdfonecov = stats.norm.pdf(x, meann, np.sqrt(comp_cov[i + binstart]))
        pdfonecov_nonoise = stats.norm.pdf(x, meann, np.sqrt(comp_cov_nonoise[i + binstart]))
        pdf_sn = stats.norm.pdf(x, mean, np.sqrt(cov_sn))
        pdfonecov_sn = stats.norm.pdf(x, mean, np.sqrt(comp_cov_sn[i + binstart]))
        means.append(mean)
        diffs.append(comp_cov_sn[i + binstart] - cov_sn)
        fac_diffs.append(comp_cov_sn[i + binstart] / cov_sn)
        diffs_sv.append(comp_cov_nonoise[i + binstart] - cov)
        ax1.plot(
            x,
            stats.norm.pdf(x, mean, np.sqrt(cov)),
            label=r"$\theta = {:3.1f}$ arcmin".format(ang),
            color=color,
        )
        ax1.plot(
            x,
            pdfnoise / np.max(pdfnoise) * np.max(pdf),
            color=color,
            linestyle="dashed",
        )
        ax1.plot(
            x,
            pdfonecov / np.max(pdfonecov) * np.max(pdf),
            color=color,
            linestyle="dotted",
        )
        """ ax2.plot(x,stats.norm.pdf(x,mean,np.sqrt(cov)),label=r'$\theta = {:3.1f}$ arcmin'.format(ang),color=color)
        ax2.plot(x,pdfonecov_nonoise/np.max(pdfonecov_nonoise)*np.max(pdf),color=color,linestyle='dashed')
        ax2.plot(x,pdfonecov_sn/np.max(pdfonecov_sn)*np.max(pdf),color=color,linestyle='dashed')
        ax2.plot(x,pdf_sn/np.max(pdf_sn)*np.max(pdf),color=color) """
    ax2.plot(
        means,
        np.fabs(diffs) / np.array(comp_cov_sn[binstart:]),
        label="relative diff, shot noise",
    )
    # ax2.plot(means,fac_diffs,label='myshotnoise/onecov, shot noise')
    ax2.plot(
        means,
        np.fabs(diffs_sv) / np.array(comp_cov_nonoise[binstart:]),
        label="relative diff, sample variance",
    )
    # ax2.set_yscale('log')
    # ax1.set_xlim(-.2e-7,1.e-7)

    ax1.legend()
    ax2.legend()
    plt.savefig("gaussiancov_noise_l30.png")
    plt.show()


def gauss_exact_comp():
    import matplotlib.pyplot as plt
    import scipy.stats as stats

    x, xnoise, pdf, pdfnoise, meanexact, meannoise = test_xip_pdf()
    gaussx, mean, cov, meann, covn, cov_sn = test_gaussian_cov(1.5, 1 * 60, 2 * 60, noise_apo=False)
    gausspdf = stats.norm.pdf(gaussx, mean, np.sqrt(cov))
    gausspdfnoise = stats.norm.pdf(gaussx, meann, np.sqrt(covn))
    print(np.trapz(gausspdf, x=gaussx))
    plt.figure()
    plt.plot(x, pdf, label=r"no noise", color="C0")
    plt.plot(gaussx, gausspdf, label=r"no noise, Gaussian", color="C0", linestyle="dashed")
    plt.axvline(np.trapz(pdf * x, x=x), color="C0")
    plt.axvline(meanexact, color="C0", linestyle="dashed")
    plt.plot(xnoise, pdfnoise, label=r"$\sigma_{\epsilon} = 0.28$", color="C1")
    plt.axvline(np.trapz(pdfnoise * xnoise, x=xnoise), color="C1")
    plt.plot(
        gaussx,
        gausspdfnoise,
        label=r"$\sigma_{\epsilon} = 0.28$, Gaussian",
        color="C1",
        linestyle="dashed",
    )
    plt.xlim(0, 6e-6)
    plt.legend()
    plt.savefig("gauss_comparison_circ1000_lm30.png")
    plt.show()


def cov_object_test(cov_object, angbin, ax, lims=(0, 2e-5)):
    import calc_pdf, grf_classes

    import scipy.stats as stats

    if cov_object.sigma_e is None:
        alpha = 0.5
    else:
        alpha = 1.0

    x, pdf_lowell, norm, mean_lowell = calc_pdf.pdf_xi_1D(
        angbin, cov_object, steps=4096, savestuff=True, high_ell_extension=False
    )

    ax.plot(x, pdf_lowell, label=r"low $\ell$", color="C3", linestyle="dashed", alpha=alpha)
    ax.axvline(mean_lowell, color="C3", linestyle="dashed", alpha=alpha)
    x, pdf, norm, mean, skewness = calc_pdf.pdf_xi_1D(
        angbin, cov_object, steps=4096, savestuff=True
    )
    ax.plot(
        x,
        stats.norm.pdf(x, cov_object.xi, np.sqrt(cov_object.cov_xi)),
        label=r"high $\ell$",
        color="C3",
        linestyle="dotted",
        alpha=alpha,
    )
    ax.axvline(cov_object.xi, color="C3", linestyle="dotted", alpha=alpha)
    ax.plot(x, pdf, label="exact likelihood", color="C3", alpha=alpha)
    ax.axvline(mean, color="C3", alpha=alpha)
    cov_object.cov_xi_gaussian()
    ax.plot(
        x,
        stats.norm.pdf(x, cov_object.xi, np.sqrt(cov_object.cov_xi)),
        label="Gaussian approximation",
        color="C0",
        alpha=alpha,
    )
    ax.set_title(r"{:.1f} - {:.1f} deg".format(*angbin[0]))
    ax.axvline(cov_object.xi, color="C0", alpha=alpha)
    ax.set_xlim(*lims)
    if cov_object.sigma_e is not None:
        ax.legend()


def plot_exactvsgaussian():
    from cov_setup import Cov

    exact_lmax = 10

    new_cov = Cov(
        exact_lmax,
        [2],
        circmaskattr=(1000, 256),
        clpath="Cl_3x2pt_kids55.txt",
        sigma_e="default",
    )
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(17, 17))
    cov_object_test(new_cov, [(0.5, 1)], ax1, lims=(0, 1.5e-5))
    cov_object_test(new_cov, [(1, 2)], ax2, lims=(-2e-6, 8e-6))
    cov_object_test(new_cov, [(4, 6)], ax3, lims=(-2e-6, 3e-6))
    cov_object_test(new_cov, [(10, 13)], ax4, lims=(-1.3e-6, 1e-6))
    new_cov.sigma_e = None
    cov_object_test(new_cov, [(0.5, 1)], ax1, lims=(0, 1.5e-5))
    cov_object_test(new_cov, [(1, 2)], ax2, lims=(-2e-6, 8e-6))
    cov_object_test(new_cov, [(4, 6)], ax3, lims=(-2e-6, 3e-6))
    cov_object_test(new_cov, [(10, 13)], ax4, lims=(-1.3e-6, 1e-6))
    plt.savefig("plots/exact_vs_gaussian_ellthres{:d}.png".format(exact_lmax))
    plt.show()


def plot_skewness():
    from cov_setup import Cov
    import calc_pdf, grf_classes
    from matplotlib.gridspec import GridSpec
    import scipy.stats as stats

    fig = plt.figure(figsize=(20, 14))

    gs = GridSpec(2, 3)
    new_cov = Cov(
        30,
        [2],
        circmaskattr=(4000, 256),
        clpath="Cl_3x2pt_kids55.txt",
        sigma_e="default",
        l_smooth=20,
        smooth_signal=True,
    )
    # set noise apodization
    # problem when cov is initialized and then set to the same lmax? yes! and only when cov is read from file
    # matters for variance and skewness, mean is not affected
    # cov_xi is different for the first and second exact_lmax = 30
    lmax = [30, 35, 40, 50]
    angbin = [(4, 6)]
    lims = -2e-6, 3e-6
    (
        skews,
        stds,
        means,
        means_lowell_exact,
        means_highell,
        means_lowell_pcl,
        means_lowell_cl,
        means_highell_cl,
    ) = ([], [], [], [], [], [], [], [])
    ax = fig.add_subplot(gs[0, :])
    ax.set_title(r"$\Delta \theta = {:.1f}^{{\circ}} - {:.1f}^{{\circ}}$".format(*angbin[0]))
    color = plt.cm.viridis(np.linspace(0, 1, len(lmax)))
    for i, el in enumerate(lmax):
        new_cov.exact_lmax = el
        new_cov.cl2pseudocl()
        x, pdf, norm, mean, std, skewness, mean_lowell, mean_highell = calc_pdf.pdf_xi_1D(
            angbin, new_cov, steps=4096, savestuff=True
        )
        ax.plot(x, pdf, color=color[i], label=r"$\ell_{{\mathrm{{exact}}}} = {:d}$".format(el))
        ax.set_xlim(lims)
        stds.append(std.real)
        means.append(mean)
        skews.append(skewness.real)
        means_lowell_exact.append(mean_lowell)
        means_highell.append(mean_highell)
        means_highell_cl.append(new_cov.xi_cl)
        new_cov.cov_xi_gaussian(lmax=el)
        means_lowell_pcl.append(new_cov.xi_pcl)
        means_lowell_cl.append(new_cov.xi_cl)
    new_cov.cov_xi_gaussian()
    ax.plot(
        x,
        stats.norm.pdf(x, new_cov.xi_pcl, np.sqrt(new_cov.cov_xi)),
        label="Gaussian approximation, Pseudo Cl",
        color="black",
        linestyle="dotted",
    )
    ax.plot(
        x,
        stats.norm.pdf(x, new_cov.xi_cl, np.sqrt(new_cov.cov_xi)),
        label="Gaussian approximation",
        color="black",
        linestyle="dashed",
    )
    ax.legend()
    ax.set_xlabel(r"$\xi^+ (\Delta \theta)$")
    ax6 = fig.add_subplot(gs[1, 0])
    ax6.set_xlabel(r"$\ell_{{\mathrm{{exact}}}}$")
    ax6.set_ylabel(r"Skewness")
    ax6.plot(lmax, skews)
    ax7 = fig.add_subplot(gs[1, 1])
    ax7.plot(lmax, stds / new_cov.cov_xi)
    ax7.axhline(1, color="black", linestyle="dotted")
    ax7.set_xlabel(r"$\ell_{{\mathrm{{exact}}}}$")
    ax7.set_ylabel(r"$\sigma / \sigma_{\mathrm{Gauss}}$")
    ax8 = fig.add_subplot(gs[1, 2])
    ax8.plot(lmax, means / new_cov.xi_pcl, label="convolution")
    # ax8.plot(lmax, means_lowell / new_cov.xi_pcl, label="low ell")
    # ax8.plot(lmax, means_highell / new_cov.xi_pcl, label="high ell")
    sum_of_means = [means_lowell_exact[i] + means_highell[i] for i in range(len(means))]
    mean_pcl_exact_comp = [means_lowell_pcl[i] / means_lowell_exact[i] for i in range(len(means))]
    mean_pcl_cl_comp = [means_lowell_pcl[i] / means_lowell_cl[i] for i in range(len(means))]
    mean_pcl_cl_highell_comp = [means_highell[i] / means_highell_cl[i] for i in range(len(means))]
    ax8.plot(lmax, mean_pcl_exact_comp, label="low ell comparison, pcl to exact")
    ax8.plot(lmax, mean_pcl_cl_comp, label="low ell comparison, pcl to cl")
    ax8.plot(lmax, mean_pcl_cl_highell_comp, label="high ell comparison, pcl to cl")
    ax8.plot(lmax, sum_of_means / new_cov.xi_pcl, label="sum of means")
    ax8.axhline(1, color="black", linestyle="dotted")
    ax8.set_xlabel(r"$\ell_{{\mathrm{{exact}}}}$")
    ax8.set_ylabel(r"$\mathbb{E}(\xi^+)$ / $\hat{\xi}^+$")
    ax8.legend()

    plt.savefig("skweness{}.png".format(new_cov.set_char_string()))


def sum_testing():
    import wpm_funcs
    from cov_setup import Cov

    exact_lm = 30
    new_cov = Cov(
        exact_lm,
        [2],
        circmaskattr=(4000, 256),
        clpath="Cl_3x2pt_kids55.txt",
        sigma_e="default",
        l_smooth="auto",
    )

    M = 0
    buffer_lmax = exact_lm + 10
    ell, wigners = wpm_funcs.prepare_wigners(2, 3, 5, 2, 2, buffer_lmax)
    wlppM = wpm_funcs.get_wlm_l(new_cov.wlm_lmax, M, new_cov.lmax, ell)
    factors = wpm_funcs.w_factor(ell, 3, 5)

    plt.figure()
    plt.plot(ell, wigners * wlppM * factors)
    plt.show()


def test_files():
    from cov_setup import Cov

    new_cov = Cov(
        20,
        [2],
        circmaskattr=(4000, 256),
        clpath="Cl_3x2pt_kids55.txt",
        sigma_e=None,
        l_smooth=20,
    )
    cov_calc = new_cov.cov_alm_xi()

    cov_from_file = new_cov.cov_alm_xi()
    print(np.max(np.fabs(cov_calc - cov_from_file)))
    plt.figure()
    plt.imshow(cov_calc.real)
    plt.colorbar()

    plt.figure()
    plt.imshow(cov_calc.imag)
    plt.colorbar()

    plt.figure()
    plt.imshow(cov_from_file.real)
    plt.colorbar()

    plt.figure()
    plt.imshow(cov_from_file.imag)
    plt.colorbar()
    plt.show()


def sum_partition():
    from cov_setup import Cov

    new_cov = Cov(
        30,
        [2],
        circmaskattr=(4000, 256),
        clpath="Cl_3x2pt_kids55.txt",
        sigma_e=None,
        l_smooth=20,
    )
    new_cov.cl2pseudocl()
    new_cov.ang_bins_in_deg = [(4, 6)]
    LM = [10, 20, 30, 35, 40, 50]
    low, high, all = [], [], []
    for lmax in LM:
        new_cov.cov_xi_gaussian(lmax=lmax)
        low.append(new_cov.xi_pcl)
        new_cov.cov_xi_gaussian(lmin=lmax)
        high.append(new_cov.xi_pcl)
        new_cov.cov_xi_gaussian()
        all.append(new_cov.xi_pcl)
    sumofmeans = [low[i] + high[i] for i in range(len(low))]
    plt.figure()
    plt.plot(LM, low)
    plt.plot(LM, high)
    plt.plot(LM, all)
    plt.plot(LM, sumofmeans)
    plt.show()
    # could the overlap be the problem?


def sim_test():
    from simulate import TwoPointSimulation

    new_sim = TwoPointSimulation(
        [(4, 6)], circmaskattr=(4000, 256), clpath="Cl_3x2pt_kids55.txt", batchsize=30
    )
    jobnumber = 0
    new_sim.xi_sim(jobnumber, plot=True)


sim_test()
