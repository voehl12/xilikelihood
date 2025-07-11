import os
# Set JAX environment variables BEFORE any other imports
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"

def test_cov_xi(snapshot):
    
    import xilikelihood as xlh
    import numpy as np
    
    from xilikelihood.file_handling import generate_filename
    from xilikelihood.theory_cl import TheoryCl
    from xilikelihood.pseudo_alm_cov import Cov
    

    mask = xlh.SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=10)
    full_mask = xlh.SphereMask(spins=[2], circmaskattr=("fullsky", 256), exact_lmax=10)
    theorycl = TheoryCl(767, clpath="papers/first_paper_2024/data/cls/Cl_3x2pt_kids55.txt")
    # covs = np.load(testdir + "/cov_xip_l10_n256_circ1000.npz")
    # cov_xip = covs["cov"]
    circ_cov = Cov(mask, theorycl, 10)
    test_cov = circ_cov.cov_alm_xi()
    # assert np.allclose(cov_xip, test_cov)
    snapshot.check(test_cov, atol=1e-16)
    nomask_cov = Cov(full_mask, theorycl, 10)
    nomask_cov_array = nomask_cov.cov_alm_xi()
    diag = np.diag(nomask_cov_array)
    diag_arr = np.diag(diag)
    assert np.allclose(nomask_cov_array, diag_arr)

    nomask_cov.mask.name = "disguised_fullsky"
    nomask_cov.covalm_path = generate_filename("cov_xi", {
            "lmax":nomask_cov.exact_lmax,
            "nside":nomask_cov.mask.nside,
            "mask":nomask_cov.mask.name,
            "theory":nomask_cov.theorycl.name,
            "sigma":nomask_cov.theorycl.sigmaname},
            nomask_cov.working_dir
        )

            
        
    nomask_bruteforce_cov = nomask_cov.cov_alm_xi()
    assert np.allclose(nomask_bruteforce_cov - nomask_cov_array, np.zeros_like(nomask_cov_array))



def test_cov_diag():
    # check that the diagonal matches the pseudo-cl to 10%:
    import xilikelihood as xlh
    import numpy as np

    exact_lmax = 10
    mask = xlh.SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=exact_lmax)
    theorycl = xlh.TheoryCl(767, clpath="cls/Cl_3x2pt_kids55.txt")
    cov = xlh.Cov(mask, theorycl, exact_lmax)
    cov_array = cov.cov_alm_xi()
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
