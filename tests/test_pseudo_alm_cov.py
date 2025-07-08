def test_cov_xi():
    import pseudo_alm_cov
    import matplotlib.pyplot as plt

    covs = np.load("../corrfunc_distr/cov_xip_l10_n256_circ1000.npz")
    cov_xip = covs["cov"]
    circ_cov = pseudo_alm_cov.Cov(10, [2], clpath="Cl_3x2pt_kids55.txt", circmaskattr=(1000, 256))

    test_cov = circ_cov.cov_alm_xi(pos_m=True)
    assert np.allclose(cov_xip, test_cov), "covariance calculation wrong"
    nomask_cov = pseudo_alm_cov.Cov(10, [2], clpath="Cl_3x2pt_kids55.txt", circmaskattr=("fullsky", 256))
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