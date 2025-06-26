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
    import mask_props
    import matplotlib.pyplot as plt

    new_cl = theory_cl.TheoryCl(30, path="Cl_3x2pt_kids55.txt")

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
    import mask_props
    import matplotlib.pyplot as plt

    nside = 256
    kids55_cl = theory_cl.TheoryCl(3 * nside - 1, path="Cl_3x2pt_kids55.txt")

    bins = [(n / 10, (n + 1) / 10) for n in range(1, 100)]
    maskareas = [1000, 4000, 10000, "fullsky"]
    scalecuts = [30, 100, 3 * nside - 1]
    linestyles = ["dotted", "dashed", "solid", "dashdot"]
    i = 0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    for m in maskareas:
        circ_mask = mask_props.SphereMask([2], circmaskattr=(m, nside))
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


def noise_correlation_analytical():
    from mask_props import SphereMask
    from theory_cl import TheoryCl
    from cov_setup import Cov
    from helper_funcs import prep_prefactors, pcl2xi
    exact_lmax = 30
    mask = SphereMask(spins=[2], circmaskattr=(10000, 256), exact_lmax=exact_lmax, l_smooth=30)
    theorycl = TheoryCl(
        mask.lmax,
        sigma_e="default",
        clname="nonoise"
    )

    covariance = Cov(mask,theorycl=theorycl,exact_lmax=exact_lmax)
    covariance.cl2pseudocl()
    prefactors = prep_prefactors([(4,6)],mask.wl,norm_lmax=mask.lmax,out_lmax=mask.lmax)
    noise_xi = pcl2xi((covariance.p_ee,covariance.p_bb,covariance.p_eb),prefactors)
    print(noise_xi)




def test_xip_pdf():
    import calc_pdf, mask_props, theory_cl
    import matplotlib.pyplot as plt

    angbin = (1, 2)
    lmax = 30
    mask = mask_props.SphereMask([2], circmaskattr=(1000, 256), lmax=lmax)
    kids55_cl = theory_cl.TheoryCl(lmax, path="Cl_3x2pt_kids55.txt")
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
    import mask_props, setup_cov

    import matplotlib.pyplot as plt

    nside = 256
    lmax = 30
    lmin = 0
    kids55_cl = theory_cl.TheoryCl(lmax, path="Cl_3x2pt_kids55.txt")

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
    import calc_pdf

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
    import calc_pdf
    from matplotlib.gridspec import GridSpec
    import scipy.stats as stats
    import plotting

    fig = plt.figure(figsize=(22, 14))
    gs = GridSpec(2, 3)
    
    filepath = "/cluster/scratch/veoehl/xi_sims/3x2pt_kids_55_circ1000smoothl30_nonoise"
    sims = plotting.read_sims(filepath, 1000, (4, 6))
    mean_measured, std_measured, skew_measured = np.mean(sims), np.std(sims),stats.skew(sims)
    print(std_measured)

    # set noise apodization
    lmax = [10,20,30]
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
    
    color = plt.cm.viridis(np.linspace(0, 1, len(lmax)))
    for i, el in enumerate(lmax):
        new_cov = Cov(
        el,
        [2],
        circmaskattr=(1000,256),
        clpath="Cl_3x2pt_kids53.txt",
        clname='3x2pt_kids_53',
        sigma_e='default',
        l_smooth_mask=30,
        l_smooth_signal=None,
        cov_ell_buffer=10,
    )
        new_cov.cl2pseudocl()
        x, pdf, norm, mean, std, skewness, mean_lowell = calc_pdf.pdf_xi_1D(
            angbin, new_cov, steps=2048, savestuff=True
        )
        ax.plot(x, pdf, color=color[i], label=r"$\ell_{{\mathrm{{exact}}}} = {:d}$".format(el))
        ax.set_xlim(lims)
        stds.append(std.real)
        means.append(mean)
        skews.append(skewness.real)
        means_lowell_exact.append(mean_lowell)
        means_highell.append(new_cov.xi_pcl)
        means_highell_cl.append(new_cov.xi_cl)
        new_cov.cov_xi_gaussian(lmax=el)
        means_lowell_pcl.append(new_cov.xi_pcl)
        means_lowell_cl.append(new_cov.xi_cl)
    new_cov.cov_xi_gaussian()
    ax.set_title(r"$\Delta \theta = {:.1f}^{{\circ}} - {:.1f}^{{\circ}}$, $A_{{\mathrm{{eff}}}} = {:.1f} \mathrm{{sqd}}$".format(*angbin[0],new_cov.eff_area))
    ax.plot(
        x,
        stats.norm.pdf(x, new_cov.xi_pcl, np.sqrt(new_cov.cov_xi)),
        label="Gaussian approximation, mean from Pseudo Cl",
        color="black",
        linestyle="dotted",
    )
    ax.plot(
        x,
        stats.norm.pdf(x, new_cov.xi_cl, np.sqrt(new_cov.cov_xi)),
        label="Gaussian approximation, mean from Cl",
        color="black",
        linestyle="dashed",
    )
    
    ax = plotting.plot_hist(ax, sims, "test")
    ax.legend()
    ax.set_xlabel(r"$\xi^+ (\Delta \theta)$")
    ax6 = fig.add_subplot(gs[1, 0])
    ax6.set_xlabel(r"$\ell_{{\mathrm{{exact}}}}$")
    ax6.set_ylabel(r"Skewness")
    ax6.plot(lmax, skews,label='predicted')
    ax6.axhline(skew_measured,color='C0',linestyle='dotted',label='measured')
    ax6.legend()
    ax7 = fig.add_subplot(gs[1, 1])
    ax7.plot(lmax, stds / new_cov.cov_xi,label='predicted')
    ax7.axhline(std_measured**2 / new_cov.cov_xi,color='C0',linestyle='dotted',label='measured')
    ax7.axhline(1, color="black", linestyle="dotted")
    ax7.legend()
    ax7.set_xlabel(r"$\ell_{{\mathrm{{exact}}}}$")
    ax7.set_ylabel(r"$\sigma / \sigma_{\mathrm{Gauss}}$")
    ax8 = fig.add_subplot(gs[1, 2])
    
    # ax8.plot(lmax, means_lowell / new_cov.xi_pcl, label="low ell")
    # ax8.plot(lmax, means_highell / new_cov.xi_pcl, label="high ell")
    sum_of_means = [means_lowell_exact[i] + means_highell[i] for i in range(len(means))]
    mean_pcl_exact_comp = [means_lowell_pcl[i] / means_lowell_exact[i] for i in range(len(means))]
    mean_pcl_cl_comp = [means_lowell_pcl[i] / means_lowell_cl[i] for i in range(len(means))]
    mean_pcl_cl_highell_comp = [means_highell[i] / means_highell_cl[i] for i in range(len(means))]
    ax8.plot(lmax, mean_pcl_exact_comp, label="low ell comparison, pcl to exact")
    #ax8.plot(lmax, mean_pcl_cl_comp, label="low ell comparison, pcl to cl")
    #ax8.plot(lmax, mean_pcl_cl_highell_comp, label="high ell comparison, pcl to cl")
    ax8.plot(lmax, means / new_cov.xi_pcl, label="convolution",color='C3')
    ax8.plot(lmax, sum_of_means / new_cov.xi_pcl, label="sum of means",color='C3',linestyle='dashed')
    ax8.axhline(1, color="black", linestyle="dotted")
    ax8.axhline(mean_measured/ new_cov.xi_pcl,color='C0',linestyle='dotted',label='measured')
    ax8.set_xlabel(r"$\ell_{{\mathrm{{exact}}}}$")
    ax8.set_ylabel(r"$\mathbb{E}(\xi^+)$ / $\hat{\xi}^+$")
    #ax8.set_ylim(0.99,1.02)
    ax8.legend()

    plt.savefig("skewness{}.png".format(new_cov.set_char_string()[4:-4]))


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
    low, high, total = [], [], []
    for lmax in LM:
        new_cov.cov_xi_gaussian(lmax=lmax)
        low.append(new_cov.xi_pcl)
        new_cov.cov_xi_gaussian(lmin=lmax)
        high.append(new_cov.xi_pcl)
        new_cov.cov_xi_gaussian()
        total.append(new_cov.xi_pcl)
    sumofmeans = [low[i] + high[i] for i in range(len(low))]
    plt.figure()
    plt.plot(LM, low)
    plt.plot(LM, high)
    plt.plot(LM, total)
    plt.plot(LM, sumofmeans)
    plt.show()
    # could the overlap be the problem?


def sim_test():
    from simulate import TwoPointSimulation
    import numpy as np
    import sys

    jobnumber = int(sys.argv[1])

    new_sim = TwoPointSimulation([(4, 6),(7,10)], maskpath='singlet_lowres.fits',maskname='KiDS1000',l_smooth_mask=30, clpath="Cl_3x2pt_kids55.txt", batchsize=1,simpath="/cluster/scratch/veoehl/xi_sims",sigma_e=None )

    new_sim.xi_sim_1D(jobnumber, plot=True)


def sim_ana_comp():
    from simulate import TwoPointSimulation
    from cov_setup import Cov
    import calc_pdf
    import helper_funcs
    from file_handling import read_pcl_sims
    import healpy as hp
    mask_smooth_l = 30
    lmin = 0
    new_sim = Cov(767,[2],
        maskname='kids_lowres',
        maskpath='singlet_lowres.fits',
        l_smooth_mask=mask_smooth_l,
        clpath="Cl_3x2pt_kids55.txt",
        clname='3x2pt_kids_55',
        sigma_e='default',
         ) 
    """   new_sim = Cov(767,[2],
        circmaskattr=(10000,256),
        l_smooth_mask=mask_smooth_l,
        clpath="Cl_3x2pt_kids55.txt",
        clname='3x2pt_kids_55',
        sigma_e='default',
         ) """
    """new_sim.wl

    pcl_measured = []

    for i in range(20):
        print(i)
        maps_TQU = new_sim.create_maps()
        pcl_22 = new_sim.get_pcl(maps_TQU)
        pcl_measured.append(pcl_22)
    pcl_measured = np.array(pcl_measured) """
    path = '/cluster/scratch/veoehl/xi_sims/3x2pt_kids_55_kids_lowressmoothl30_noisedefault/'
    #path = '/cluster/scratch/veoehl/xi_sims/3x2pt_kids_55_circ10000smoothl30_noisedefault/'
    pcl_measured = read_pcl_sims(path,10)
    new_sim.cl2pseudocl()
    angbins = [(0.23,0.3),(0.3,0.5),(0.6,0.9),(0.75,2),(2,3),(2.5,5),(4,6),(6,9)]
    angs = [np.mean(np.array(tup)) for tup in angbins]
    prefacs = helper_funcs.prep_prefactors(angbins,new_sim.wl,new_sim.lmax,new_sim.lmax)
    corr_func = helper_funcs.pcl2xi((new_sim.p_ee,new_sim.p_bb,new_sim.p_eb),prefacs,new_sim.lmax)
    #prefactors = helper_funcs.prep_prefactors([(4, 6)], new_sim.wl, new_sim.lmax, new_sim.lmax)
    #xi_sim = helper_funcs.pcl2xi(np.mean(pcl_measured, axis=0), prefactors, new_sim.lmax,lmin=lmin)
    #xi_ana_cl = helper_funcs.cl2xi((new_sim.ee,new_sim.bb), (4, 6), new_sim.lmax, lmin=lmin)
    #xi_ana = helper_funcs.pcl2xi((new_sim.p_ee, new_sim.p_bb, new_sim.p_eb), prefactors, new_sim.lmax,lmin=lmin)
    covs,means,pclmeans = [],[],[]
    for angbin in angbins:
        mean,cov = calc_pdf.cov_xi_gaussian_nD((new_sim,),((0,0),),(angbin,))
        covs.append(cov[0,0])
        means.append(mean[0])
    #print(xi_sim, xi_ana,xi_ana_cl)
    #pixweights_t, pixweights_p = hp.pixwin(256,pol=True)
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(8,12))
    assert np.allclose(np.array(means),corr_func[0])
    ax1.plot(np.mean(pcl_measured[:, 0], axis=0), color="C0")
    for i in range(len(prefacs)):
        pclmeans.append(np.mean(prefacs[i,0]*(2 * np.arange(new_sim.lmax+1) + 1)*((np.mean(pcl_measured[:, 0], axis=0)-new_sim.p_ee)+(np.mean(pcl_measured[:, 0], axis=0)-new_sim.p_ee))))
    ax2.plot(angs,pclmeans,'.')
    ax2.set_ylim(-1e-10,1e-10)
    ax1.plot(new_sim.p_ee, color="C0", linestyle="dotted")
    ax1.plot(np.mean(pcl_measured[:, 1], axis=0), color="C1")
    ax1.plot(new_sim.p_bb, color="C1", linestyle="dotted")
    ax3.plot(angs,np.fabs(np.array(corr_func[0]))/np.array(covs)) # also plot covariance here to find good signal to noise level
    fig.savefig('pseudo_cl_comp.png')


def lowell_comp():
    from cov_setup import Cov
    import calc_pdf, mask_props
    from matplotlib.gridspec import GridSpec
    import scipy.stats as stats
    import plotting

    fig1,ax1 = plt.subplots()
    l_exacts = [10,20,30]
    angbin = [(4, 6)]
    l_smooths = [None]
    l_smooths_plot = ['None']
    xi_cl,xi_pcl,mean_exact,close_ells = [],[],[],[]
    ref_cov = new_cov = Cov(
            30,
            [2],
            circmaskattr=(4000, 256),
            clpath="Cl_3x2pt_kids55.txt",
            sigma_e=None,
            l_smooth_mask=None,
            l_smooth_signal=None,
        )
    ref_cov.cl2pseudocl()
    ref_cov.ang_bins_in_deg = angbin
    ref_cov.cov_xi_gaussian(lmax = 30)
    ref_xi = ref_cov.xi_cl
    for l_exact in l_exacts:
        new_cov = Cov(
            l_exact,
            [2],
            circmaskattr=(1000, 256),
            clpath="Cl_3x2pt_kids55.txt",
            sigma_e='default',
            l_smooth_mask=30,
            l_smooth_signal=None,
            cov_ell_buffer=10,
        )
        
        
        
        new_cov.cl2pseudocl()
        x, pdf, norm, mean, std, skewness, mean_lowell = calc_pdf.pdf_xi_1D(
            angbin, new_cov, steps=4096, savestuff=True, high_ell_extension=False
        )
        new_cov.cov_xi_gaussian(lmax=l_exact)
        xi_cl.append(1)

        
        xi_pcl.append(new_cov.xi_pcl/new_cov.xi_pcl)
        mean_exact.append(mean/new_cov.xi_pcl)
    new_cov.cov_xi_gaussian()
    diag_alm = np.diag(new_cov.cov_alm)
    fig3,ax3 = plt.subplots()
    ax3.plot(np.diag(new_cov.cov_alm))
    fig3.savefig('palm_cov.png')
    len_sub = 2*new_cov._exact_lmax+1
    reps = int(len(diag_alm) / (len_sub))
    print(reps)
    print(len(diag_alm[1*len_sub:(1+1)*len_sub]))
    check_pcl = np.array([np.sum(diag_alm[i*len_sub:(i+1)*len_sub]) for i in range(reps)])
    #check_pcl = np.array([np.sum(2*diag_alm[(new_cov._exact_lmax+1) + i*len_sub:(new_cov._exact_lmax+1) +(i*len_sub+1+new_cov._exact_lmax)]) + diag_alm[new_cov._exact_lmax + i*len_sub] for i in range(reps)])
    fig2,(ax21,ax22) = plt.subplots(2)
    # TODO: implement this to work with pos_m=True covariance matrices as already done in assert statement in cov_setup.
    # need to check this again with pos_m=True but add factors 2 that usually come from m matrix. 
    ell_short = 2 * new_cov.ell + 1
    ax21.plot(new_cov.ell,new_cov.p_ee * ell_short,label='pseudo Cl EE')
    ax21.plot(new_cov.ell,new_cov.p_bb * ell_short,label='pseudo Cl BB')
    ax21.plot(np.arange(new_cov._exact_lmax+1),check_pcl[:new_cov._exact_lmax+1]+check_pcl[new_cov._exact_lmax+1:2*(new_cov._exact_lmax+1)],label='partial traces of sigma,EE')
    ax21.plot(np.arange(new_cov._exact_lmax+1),check_pcl[2*(new_cov._exact_lmax+1):3*(new_cov._exact_lmax+1)]+check_pcl[3*(new_cov._exact_lmax+1):4*(new_cov._exact_lmax+1)],label='partial traces of sigma,BB')
    ax22.plot(np.arange(new_cov._exact_lmax+1),(new_cov.p_ee * ell_short)[:new_cov._exact_lmax+1] / (check_pcl[:new_cov._exact_lmax+1]+check_pcl[new_cov._exact_lmax+1:2*(new_cov._exact_lmax+1)]))
    ax22.plot(np.arange(new_cov._exact_lmax+1),(new_cov.p_bb * ell_short)[:new_cov._exact_lmax+1] / (check_pcl[2*(new_cov._exact_lmax+1):3*(new_cov._exact_lmax+1)]+check_pcl[3*(new_cov._exact_lmax+1):4*(new_cov._exact_lmax+1)]))
    ax21.set_yscale('log')
    ax21.set_xlim(0,31)
    ax21.legend()
    #ax22.set_ylim(0.98,1.02)
    fig2.savefig("pcl_vs_cl.png")

    print(ref_xi)
    ax1.plot(l_exacts,xi_pcl,label=r'pseudo $C_{\ell}$')
    #ax.plot(l_smooths_plot,xi_pcl,label=r'$\tilde{C}_{\ell}$')
    ax1.plot(l_exacts,mean_exact,label=r'mean exact likelihood')
    
    ax1.set_xlabel(r'exact $\ell$')
    ax1.set_ylabel(r'$\xi^+$/$\xi^+_{pC_{\ell}}$')
    
    ax1.legend()
    fig1.savefig('lowell_meancomparison_fullm.png')    

def norm_testing():
    from scipy.special import eval_legendre
    from scipy.integrate import quad_vec, quad
    from mask_props import SphereMask
    norm_lmax = 767
    lower,upper = np.radians(4),np.radians(6)
    norm_l = np.arange(norm_lmax + 1)
    new_mask = SphereMask(spins=[2],circmaskattr=(1000,256),l_smooth=30)
    legendres = lambda t_in_rad: eval_legendre(norm_l, np.cos(t_in_rad))
    # TODO: check whether this sum needs to be done after the integration as well (even possible?)
    norm_summed = lambda t_in_rad: 1 / np.sum((2 * norm_l + 1) * legendres(t_in_rad) * new_mask.wl[: norm_lmax + 1]) / (2 * np.pi)
    

    norm_array = lambda t_in_rad: 1 / ((2 * norm_l + 1) * legendres(t_in_rad) * new_mask.wl[: norm_lmax + 1]) / (2 * np.pi)

    norm1 = quad(norm_summed,lower,upper)[0]
    norm2 = np.sum(quad_vec(norm_array,lower,upper)[0])

    print(norm1,norm2)


def plot_full_likelihood():
    from cov_setup import Cov
    import calc_pdf
    from matplotlib.gridspec import GridSpec
    import scipy.stats as stats
    import plotting

    fig = plt.figure(figsize=(8, 6))
    
    
    filepath = "/cluster/scratch/veoehl/xi_sims/3x2pt_kids_55_circ1000smoothl30_nonoise"
    sims = plotting.read_sims(filepath, 1000, (4, 6))
    mean_measured, std_measured, skew_measured = np.mean(sims), np.std(sims),stats.skew(sims)
    
    # set noise apodization
    lmax = 50
    angbin = [(4, 6)]
    lims = -1.5e-6, 2.0e-6
    
    ax = fig.add_subplot()
    
     
    new_cov = Cov(
    lmax,
    [2],
    circmaskattr=(1000,256),
    clpath="Cl_3x2pt_kids55.txt",
    sigma_e=None,
    l_smooth_mask=30,
    l_smooth_signal=None,
    cov_ell_buffer=10,
)
    new_cov.cl2pseudocl()
    x, pdf, norm, mean, std, skewness, mean_lowell = calc_pdf.pdf_xi_1D(
        angbin, new_cov, steps=4096, savestuff=True
    )
    ax.plot(x, pdf, color='C3', label=r"Prediction, $\ell_{{\mathrm{{exact}}}} = {:d}$".format(lmax))
    ax.set_xlim(lims)
    
       
    new_cov.cov_xi_gaussian()
    ax.set_title(r"$\Delta \theta = {:.1f}^{{\circ}} - {:.1f}^{{\circ}}$, $A_{{\mathrm{{eff}}}} = {:.1f} \ \mathrm{{sqd}}$".format(*angbin[0],new_cov.eff_area))
    ax.plot(
        x,
        stats.norm.pdf(x, new_cov.xi_pcl, np.sqrt(new_cov.cov_xi)),
        label=r"Gaussian approximation",
        color="black",
        linestyle="dashed",
    )

    
    ax = plotting.plot_hist(ax, sims, r"Measured",label=True)
    ax.legend()
    ax.set_xlabel(r"$\xi^+ (\Delta \theta)$")
    
    

    plt.savefig("full_likelihood_vs_measured.pdf".format(new_cov.set_char_string()[4:-4]))

def high_low_s8():
    from cov_setup import Cov
    import calc_pdf
    from simulate import TwoPointSimulation   
    import scipy.stats as stats
    
    cl_paths = [("Cl_3x2pt_kids55_lows8.txt",'S8p6'),("Cl_3x2pt_kids55_s8p7.txt",'S8p7'),("Cl_3x2pt_kids55_s8p75.txt",'S8p75'),("Cl_3x2pt_kids55_s8p8.txt","S8p8"),("Cl_3x2pt_kids55_s8p85.txt",'S8p85'),("Cl_3x2pt_kids55_s8p9.txt",'S8p9'),("Cl_3x2pt_kids55_highs8.txt",'S81')]
    s8 = [0.6,0.7,0.75,0.8,0.85,0.9,1.0]
    likelihood = []
    likelihood_gauss = []
    fig = plt.figure(figsize=(10, 6))
    measured_cl = cl_paths[3]
    lims = -0.25e-6, 1e-6
    angbin = [(2,3)]
    measurement = TwoPointSimulation(angbin,circmaskattr=(10000,256),l_smooth_mask=30,clpath=measured_cl[0],clname = measured_cl[1],batchsize=1,simpath="/cluster/home/veoehl/2ptlikelihood",sigma_e=None )
    jobnumber = 1
    measurement.xi_sim_1D(jobnumber)
    xi_measured = np.load(measurement.simpath + "/job{:d}.npz".format(jobnumber))
    xip_measured = xi_measured['xip'][0,0]     
    print(xip_measured)
    ax = fig.add_subplot(121)
    cov_bestfit = Cov(
        30,
        [2],
        circmaskattr=(10000,256),
        clpath=measured_cl[0],
        clname = measured_cl[1],
        sigma_e=None,
        l_smooth_mask=30,
        l_smooth_signal=None,
        cov_ell_buffer=10,
        )
    means_bestfit, covs_bestfit = calc_pdf.cov_xi_gaussian_nD((cov_bestfit,),((0,0),),angbin)
    lag = 10
    colors = plt.cm.viridis(np.linspace(0, 1, len(cl_paths)))
    for i,cl in enumerate(cl_paths):
        color = colors[i]
        cov_s8 = Cov(
        30,
        [2],
        circmaskattr=(10000,256),
        clpath=cl[0],
        clname = cl[1],
        sigma_e=None,
        l_smooth_mask=30,
        l_smooth_signal=None,
        cov_ell_buffer=10,
        )
    
        cov_s8.cl2pseudocl()
        x, pdf, statistics = calc_pdf.pdf_xi_1D(
        angbin, (cov_s8,), steps=4096, savestuff=True)

        meas_ind = np.argmin(np.fabs(x[0]-xip_measured))
        s8_likelihood = pdf[0][meas_ind]
        likelihood.append(s8_likelihood)
        ax.plot(x[0], pdf[0], color=color, label=cl[1])
        ax.axvline(statistics[0][0],color=color)

        means, covs = calc_pdf.cov_xi_gaussian_nD((cov_s8,),((0,0),),angbin)
    
        ax.plot(
            x[0],
            stats.norm.pdf(x[0], means[0], np.sqrt(covs_bestfit[0,0])),
            color=color,
            linestyle="dotted",
        )
        likelihood_gauss.append(stats.norm.pdf(x[0][meas_ind], means[0], np.sqrt(covs_bestfit[0,0])))
        # different bins: check where signal to noise for correlation function is good
        # get p-values for gaussian vs exact (can also be done for pdfs directly)
    
    ax.set_xlim(lims)
    ax.axvline(xip_measured,color='C3',label='measured')
    ax.set_xlabel(r'$\xi^+$')
    ax.legend(frameon=False)
    ax2 = fig.add_subplot(122)
    ax2.plot(s8,likelihood,label='exact likelihood')
    ax2.plot(s8,likelihood_gauss,label='Gaussian likelihood')
    ax2.axvline(s8[3],color='C3')
    ax2.set_xlabel(r'S8')
    ax2.legend(frameon=False)
    plt.savefig('varied_s8.pdf',bbox_inches='tight')
         
              

def test_ndcf():
    import calc_pdf
    from scipy.stats import multivariate_normal
    mu =  np.array([1, 2])
    #mu =  np.array([-8.99339e-07, -6.22804e-07])
    #cov = np.array([[8.06611494e-14, 9.25908846e-14],[9.25908846e-14, 3.96556510e-14]])
    cov = np.array([[3, 1],[0.5, 6]])
    print(np.linalg.eigvals(cov))
    var = multivariate_normal(mean=mu, cov=cov)
    
    xip_max1 = np.max(np.array([np.fabs(mu[0] - 500 * np.sqrt(np.diag(cov)[0])),np.fabs(mu[0] + 500 * np.sqrt(np.diag(cov)[0]))]))
    xip_max2 = np.max(np.array([np.fabs(mu[1] - 500 * np.sqrt(np.diag(cov)[1])),np.fabs(mu[1] + 500 * np.sqrt(np.diag(cov)[1]))]))
    dt_xip1 = 0.45 * 2 * np.pi / xip_max1
    dt_xip2 = 0.45 * 2 * np.pi / xip_max2
    steps = 2048
    t01 = -0.5 * dt_xip1 * (steps - 1)
    t02 = -0.5 * dt_xip2 * (steps - 1)
    t1 = np.linspace(t01, -t01, steps - 1)
    t2 = np.linspace(t02, -t02, steps - 1)
    t_inds = np.arange(len(t1))
    t0_2 = np.array([t01,t02])
    dt_2 = np.array([dt_xip1,dt_xip2])
    t_sets = np.stack(np.meshgrid(t1,t2),-1).reshape(-1,2)
    ind_sets = np.stack(np.meshgrid(t_inds,t_inds),-1).reshape(-1,2)    
    vals = calc_pdf.high_ell_gaussian_cf_nD(t_sets,mu,cov)

    cf_grid = np.full((steps-1,steps-1),np.nan,dtype=complex)
    for j,idx in enumerate(ind_sets):
     
        cf_grid[tuple(idx)] = vals[j]

    x_grid, pdf_grid = calc_pdf.cf_to_pdf_nd(cf_grid, t0_2, dt_2, verbose=True)

    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(14,6))
    x, y = np.meshgrid(x_grid[:,0,0], x_grid[:,1,0])

    pos = np.dstack((x,y))


    k = ax2.pcolormesh(x_grid[:,0,0],x_grid[:,1,0], var.pdf(pos),vmin=0)
    h = ax1.pcolormesh(x_grid[:,0,0],x_grid[:,1,0],pdf_grid,vmin=0)
    l = ax4.pcolormesh(x_grid[:,0,0],x_grid[:,1,0], var.pdf(pos)-pdf_grid)
    ax3.pcolormesh(t1,t2, cf_grid.real)
    fig.colorbar(h, ax=ax1)
    fig.colorbar(k, ax=ax2)
    fig.colorbar(l, ax=ax4)
    lims = (-10,10)
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    #ax3.set_xlim(-2e-6,0)
    #ax3.set_ylim(-1.5e-6,0)
    ax4.set_xlim(lims)
    ax4.set_ylim(lims)
    plt.savefig('2dgaussian')

def plot_pcl_vs_cl():
    from cov_setup import Cov
    def plot_clvpcl(fig, axes,cov,name,linestyle,color):
        ax1 = axes[0]
        ax2 = axes[1]
        if name == '1000 sqd':
            ax1.plot(cov.ell,cov.ee/np.cumsum(cov.ee)[-1],label=r'$C_{\ell}^E$',color='C0')
        ax1.plot(cov.ell,(cov.p_ee+cov.p_bb)/np.cumsum(cov.p_ee+cov.p_bb)[-1],label=r'$\tilde{{C}}_{{\ell}}^E + \tilde{{C}}_{{\ell}}^B$, {}'.format(name),color=color)
        #ax1.plot(new_cov.ell,new_cov.p_bb/np.max(new_cov.p_ee+new_cov.p_bb),label=r'$\tilde{C}_{\ell}^B$')
        ax2.plot(cov.ell,((cov.ee/np.cumsum(cov.ee)[-1]/((cov.p_ee+cov.p_bb)/np.cumsum(cov.p_ee+cov.p_bb)[-1]))-1),label=name,color=color)
        ax2.plot(cov.ell,-((cov.ee/np.cumsum(cov.ee)[-1]/((cov.p_ee+cov.p_bb)/np.cumsum(cov.p_ee+cov.p_bb)[-1]))-1),linestyle='dotted',color=color)
        #ax2.axhline(0,color='black',linestyle='dotted')
        ax1.legend()
        ax2.legend()
        ax1.set_xlabel(r'$\ell$')
        ax2.set_xlabel(r'$\ell$')
        ax2.set_ylabel(r'rel. difference')
        #ax1.set_yscale('log')
        ax2.set_yscale('log')
        ax1.set_xlim(2,80)
        ax2.set_xlim(2,80)
        ax1.set_ylim(2e-3,1.5e-2)
        #ax2.set_ylim(-5e-3,4e-2)
        return fig, (ax1,ax2)

    cov_kids = Cov(
            30,
            [2],
            maskpath='singlet_lowres.fits',
            clpath="Cl_3x2pt_kids55.txt",
            sigma_e=None,
            l_smooth_mask=30,
            l_smooth_signal=None,
            cov_ell_buffer=10,
        )
    cov_1000 = Cov(
            30,
            [2],
            circmaskattr=(1000,256),
            clpath="Cl_3x2pt_kids55.txt",
            sigma_e=None,
            l_smooth_mask=30,
            l_smooth_signal=None,
            cov_ell_buffer=10,
        )
    cov_10000 = Cov(
            30,
            [2],
            circmaskattr=(10000,256),
            clpath="Cl_3x2pt_kids55.txt",
            sigma_e=None,
            l_smooth_mask=30,
            l_smooth_signal=None,
            cov_ell_buffer=10,
        )
    
    
    
    cov_1000.cl2pseudocl()
    cov_10000.cl2pseudocl()
    cov_kids.cl2pseudocl()
    color = plt.cm.inferno(np.linspace(0.5, 0.8, 3))
    fig, (ax1,ax2) = plt.subplots(2,figsize=(10,7))
    
    fig, (ax1,ax2) = plot_clvpcl(fig, (ax1,ax2),cov_1000, '1000 sqd','solid',color[0])
    fig, (ax1,ax2) = plot_clvpcl(fig, (ax1,ax2),cov_10000, '10000 sqd','dashed',color[1])
    fig, (ax1,ax2) = plot_clvpcl(fig, (ax1,ax2),cov_kids, 'KiDS','dotted',color[2])
    plt.savefig('pcl_cl_comparison.pdf')

   
def test_glass():
    from simulate import TwoPointSimulation
    from simulate import get_xi_namaster_nD, get_pcl_nD, add_noise_nD
    import glass.fields
    import healpy as hp
    import matplotlib.pyplot as plt
    import numpy as np
    import time
    import helper_funcs
    import healpy as hp

    cl_55 = "Cl_3x2pt_kids55.txt"
    cl_53 = "Cl_3x2pt_kids53.txt"
    cl_33 = "Cl_3x2pt_kids33.txt"
    cl_paths = (cl_33,cl_55,cl_53)
    cl_names = ('3x2pt_kids_33','3x2pt_kids_55','3x2pt_kids_53')
    batchsize = 50
    angs_in_deg = [(4,6)]
    sim_33 = TwoPointSimulation([(4, 6),(7,10)], circmaskattr=(1000, 256),l_smooth_mask=30, clpath=cl_paths[0], clname=cl_names[0], batchsize=1000,simpath="/cluster/scratch/veoehl/xi_sims",sigma_e='default')
    
    sim_53 = TwoPointSimulation([(4, 6),(7,10)], circmaskattr=(1000, 256),l_smooth_mask=30,clpath=cl_paths[2], clname=cl_names[2], batchsize=1000,simpath="/cluster/scratch/veoehl/xi_sims",sigma_e=None)
    sim_55 = TwoPointSimulation([(4, 6),(7,10)], circmaskattr=(1000, 256),l_smooth_mask=30,clpath=cl_paths[1], clname=cl_names[1], batchsize=1000,simpath="/cluster/scratch/veoehl/xi_sims",sigma_e='default')
    sim_33.cl2pseudocl()
    sim_55.cl2pseudocl()
    sim_53.cl2pseudocl()
    sims = [sim_33,sim_55,sim_53]
    areas = [sim_33.eff_area, sim_55.eff_area]
    lmax = 30
    ell = np.arange(lmax+1)
    print(areas)
    prefactors = helper_funcs.prep_prefactors(angs_in_deg, sim_33.wl, sim_33.lmax, lmax)
    gls = np.array([sim_33.ee,sim_55.ee,sim_53.ee])
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(18,8))
    all_cl = []
    all_xis = []
    pcls, pcls_1D = [],[]
    tic = time.perf_counter()
    for j in range(batchsize):
        print(j)
        tictic = time.perf_counter()
        fields = glass.fields.generate_gaussian(gls, nside=sim_33.nside)


        field_list = []
        while True:
            try:
                mapsTQU = next(fields)
                field_list.append(mapsTQU)
            except StopIteration:
                break
        field_list = np.array(field_list)
        toctoc = time.perf_counter()
        print(toctoc-tictic)
        cl = [hp.anafast(field) for field in field_list]
        cl_mix = hp.anafast(field_list[0],field_list[1])
        cl.append(cl_mix)
        cl = np.array(cl)
        all_cl.append(cl[:,:,:lmax+1])
        
        field_list = add_noise_nD(field_list,256,sigmas=[sim_33.pixelsigma,sim_55.pixelsigma])
        maps_TQU = sim_55.create_maps()
        maps_TQU = sim_55.add_noise(maps_TQU)
        pcl_1D = sim_55.get_pcl(maps_TQU)
        pcl_1D = np.array(pcl_1D)
        pcls_1D.append(pcl_1D[:,:lmax+1])
        pcl = get_pcl_nD(field_list,[sim_33.smooth_mask,sim_55.smooth_mask],fullsky=False)
        pcl = np.array(pcl)
        pcls.append(pcl[:,:,:lmax+1])
        
        xis = get_xi_namaster_nD(field_list,[sim_33.smooth_mask,sim_55.smooth_mask], prefactors, lmax,lmin=0)
        all_xis.append(xis)
    toc = time.perf_counter()
    print(toc-tic)
    all_cl = np.array(all_cl)
    cl = np.mean(all_cl,axis=0)
    pcl = np.mean(np.array(pcls),axis=0)
    pcl_1D = np.mean(np.array(pcls_1D),axis=0)
    for i in range(3):    
        ax1.plot(ell,cl[i,1],color='C{:d}'.format(i),linestyle='dotted',label='Cl measured')
        ax1.plot(ell,gls[i,:lmax+1],color='C{:d}'.format(i),label='input Cl')
        ax2.plot(ell,pcl[i,0],color='C{:d}'.format(i),linestyle='dashed',label='pclE measured')
        #ax.plot(sim_33.ell,pcl33_e,color='black',linestyle='dashed',label='pclE measured 1D')
        ax2.plot(ell,sims[i].p_ee[:lmax+1],color='C{:d}'.format(i),label='pclE predicted')
        ax2.plot(ell,pcl[i,1],color='C{:d}'.format(i+3),linestyle='dashed',label='pclB measured')
        ax2.plot(ell,sims[i].p_bb[:lmax+1],color='C{:d}'.format(i+3),label='pclB predicted')
        ax2.plot(ell,pcl_1D[0],color='black',linestyle='dashed')
    all_xis = np.array(all_xis)
    ax3.hist(all_xis[:,1,0,0],10)
    print(np.mean(all_xis[:,1,0,0]))
    ax2.legend()
    ax1.legend()
    #ax.set_yscale('log')
    plt.savefig('testfield_glass.png')

noise_correlation_analytical()