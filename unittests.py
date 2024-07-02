import numpy as np

testdir = "/Users/voehl/Code/2pt_likelihood/testdata/"


def test_palm_matching():
    import cov_funcs

    palm_kinds = ["ReE", "ImE", "ReB", "ImB"]
    assert cov_funcs.match_alm_inds(palm_kinds) == [0, 1, 2, 3]


def test_cl_class():
    import grf_classes

    new_cl = grf_classes.TheoryCl(30)
    assert np.allclose(new_cl.ee, np.zeros(31))


def test_cov_xi():
    import cov_setup

    covs = np.load(testdir + "/cov_xip_l10_n256_circ1000.npz")
    cov_xip = covs["cov"]
    circ_cov = cov_setup.Cov(10, [2], clpath="Cl_3x2pt_kids55.txt", circmaskattr=(1000, 256))
    test_cov = circ_cov.cov_alm_xi(pos_m=True)
    assert np.allclose(cov_xip, test_cov)

    nomask_cov = cov_setup.Cov(10, [2], clpath="Cl_3x2pt_kids55.txt", circmaskattr=("fullsky", 256))
    nomask_cov_array = nomask_cov.cov_alm_xi(pos_m=True)
    diag = np.diag(nomask_cov_array)
    diag_arr = np.diag(diag)
    assert np.allclose(nomask_cov_array, diag_arr)

    nomask_cov.maskname = "disguised_fullsky"
    nomask_cov.set_covalmpath()
    nomask_bruteforce_cov = nomask_cov.cov_alm_xi(pos_m=True)
    assert np.allclose(nomask_bruteforce_cov - nomask_cov_array, np.zeros_like(nomask_cov_array))


def test_cov_diag():
    # check that the diagonal matches the pseudo-cl to 10%:
    import cov_setup

    exact_lmax = 10
    cov = cov_setup.Cov(
        exact_lmax,
        [2],
        clpath="Cl_3x2pt_kids55.txt",
        circmaskattr=(1000, 256),  # l_smooth_mask=30
    )
    cov_array = cov.cov_alm_xi(pos_m=True)
    diag_alm = np.diag(cov_array)

    len_sub = exact_lmax + 1
    reps = int(len(diag_alm) / (len_sub))
    check_pcl_sub = np.array(
        [
            np.sum(2 * diag_alm[i * len_sub + 1 : (i + 1) * len_sub + 1]) + diag_alm[i * len_sub]
            for i in range(reps)
        ]
    )

    check_pcl = np.zeros((2, exact_lmax + 1))
    check_pcl[0], check_pcl[1] = (
        check_pcl_sub[: exact_lmax + 1] + check_pcl_sub[exact_lmax + 1 : 2 * (exact_lmax + 1)],
        check_pcl_sub[2 * (exact_lmax + 1) : 3 * (exact_lmax + 1)]
        + check_pcl_sub[3 * (exact_lmax + 1) : 4 * (exact_lmax + 1)],
    )
    pcl = np.zeros_like(check_pcl)
    twoell = 2 * cov.ell + 1
    cov.cl2pseudocl()
    pcl[0], pcl[1] = (cov.p_ee * twoell)[: exact_lmax + 1], (cov.p_bb * twoell)[: exact_lmax + 1]
    assert np.allclose(
        pcl, check_pcl, rtol=1e-1
    ), "cov_alm_gen: covariance diagonal does not agree with pseudo C_ell"


def test_analytic_pcl():
    from simulate import TwoPointSimulation
    from cov_setup import Cov
    import helper_funcs

    l_smooth_mask = 30
    lmin = 0
    new_sim = TwoPointSimulation(
        [(4, 6)],
        circmaskattr=(4000, 256),
        l_smooth_mask=l_smooth_mask,
        clpath="Cl_3x2pt_kids55.txt",
        batchsize=10,
        simpath="",
        healpix_datapath=testdir,
    )
    new_sim.wl

    pcl_measured = []

    for i in range(50):
        maps_TQU = new_sim.create_maps()
        pcl_22 = new_sim.get_pcl(maps_TQU)
        pcl_measured.append(pcl_22)
    pcl_measured = np.array(pcl_measured)

    new_sim.cl2pseudocl()

    prefactors = helper_funcs.prep_prefactors([(4, 6)], new_sim.wl, new_sim.lmax, new_sim.lmax)
    xi_sim = helper_funcs.pcl2xi(np.mean(pcl_measured, axis=0), prefactors, new_sim.lmax, lmin=lmin)
    xi_ana_cl = helper_funcs.cl2xi((new_sim.ee, new_sim.bb), (4, 6), new_sim.lmax, lmin=lmin)
    xi_ana = helper_funcs.pcl2xi(
        (new_sim.p_ee, new_sim.p_bb, new_sim.p_eb), prefactors, new_sim.lmax, lmin=lmin
    )
    assert np.allclose(xi_ana_cl, xi_ana, rtol=1e-2)
    assert np.allclose(xi_sim, xi_ana, rtol=1e-2)
    print(xi_sim, xi_ana, xi_ana_cl)


def test_wlmlm():
    # assert np.allclose(wlm[:,:,mid_ind:,:,:],wlm[:,:,:mid_ind+1,:,:])
    # assert np.allclose(wlpmp[:,:,mid_ind:,:,:],wlpmp[:,:,:mid_ind+1,:,:])

    pass


def test_treecorrvsnamaster():
    from simulate import TwoPointSimulation
    from cov_setup import Cov
    from calc_pdf import cov_xi_gaussian_nD

    jobnumber = 0

    new_sim = TwoPointSimulation(
        [(2, 3)],
        circmaskattr=(10000, 256),
        l_smooth_mask=30,
        clname="3x2pt_kids_55",
        clpath="Cl_3x2pt_kids55.txt",
        batchsize=10,
        simpath=None,
        sigma_e="default",
        ximode="comp",
        healpix_datapath=testdir,
    )
    cov = Cov(
        30,
        [2],
        sigma_e="default",
        circmaskattr=(10000, 256),
        l_smooth_mask=30,
        clname="3x2pt_kids_55",
        clpath="Cl_3x2pt_kids55.txt",
    )
    cov_comp = np.sqrt(cov_xi_gaussian_nD((cov,), ((0, 0),), [(2, 3)])[1][0, 0])
    new_sim.xi_sim_1D(jobnumber, save_pcl=False, pixwin=False, plot=False)
    assert np.allclose(new_sim.comp[0], new_sim.comp[1], atol=cov_comp)


def test_means():
    pass


def test_cf2pdf(t, mu, sigma):
    import scipy.stats as stats
    from helper_funcs import gaussian_cf
    from calc_pdf import cf_to_pdf_1d

    cf = gaussian_cf(t, mu, sigma)
    x, pdf_from_cf = cf_to_pdf_1d(t, cf)
    pdf = stats.norm.pdf(x, mu, sigma)
    assert np.allclose(pdf, pdf_from_cf), pdf_from_cf


test_treecorrvsnamaster()
