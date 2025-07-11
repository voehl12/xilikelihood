import numpy as np

def test_treecorr_vs_namaster():
    from simulate import TwoPointSimulation
    from pseudo_alm_cov import Cov
    from distributions import cov_xi_gaussian_nD

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
        healpix_datapath="/Users/voehl/Code/2pt_likelihood/testdata/",
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


def compare_analytic_vs_simulated_pcl():
    from simulate import TwoPointSimulation
    from pseudo_alm_cov import Cov
    import cl2xi_transforms
    from distributions import cov_xi_gaussian_nD

    l_smooth_mask = 30
    lmin = 0
    new_sim = TwoPointSimulation(
        [(4, 6)],
        circmaskattr=(4000, 256),
        l_smooth_mask=l_smooth_mask,
        clpath="cls/Cl_3x2pt_kids55.txt",
        batchsize=10,
        simpath="",
        healpix_datapath=testdir,
    )
    cov = Cov(
        30,
        [2],
        circmaskattr=(4000, 256),
        l_smooth_mask=l_smooth_mask,
        clname="3x2pt_kids_55",
        clpath="Cl_3x2pt_kids55.txt",
    )
    new_sim.wl

    pcl_measured = []

    for i in range(50):
        maps_TQU = new_sim.create_maps()
        pcl_22 = new_sim.get_pcl(maps_TQU)
        pcl_measured.append(pcl_22)
    pcl_measured = np.array(pcl_measured)

    new_sim.cl2pseudocl()

    prefactors = cl2xi_transforms.prep_prefactors([(4, 6)], new_sim.wl, new_sim.lmax, new_sim.lmax)
    xi_sim = cl2xi_transforms.pcl2xi(np.mean(pcl_measured, axis=0), prefactors, new_sim.lmax, lmin=lmin)
    xi_ana_cl = cl2xi_transforms.cl2xi((new_sim.ee, new_sim.bb), (4, 6), new_sim.lmax, lmin=lmin)
    xi_ana = cl2xi_transforms.pcl2xi(
        (new_sim.p_ee, new_sim.p_bb, new_sim.p_eb), prefactors, new_sim.lmax, lmin=lmin
    )
    xi_ana = [xi_ana[0][0], xi_ana[1][0]]
    xi_sim = [xi_sim[0][0], xi_sim[1][0]]
    cov_comp = np.sqrt(cov_xi_gaussian_nD((cov,), ((0, 0),), [(4, 6)])[1][0, 0])
    assert np.allclose(np.array(xi_ana_cl), np.array(xi_ana), atol=cov_comp)
    assert np.allclose(xi_sim, xi_ana, atol=cov_comp)