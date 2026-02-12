import pytest


@pytest.mark.slow
def test_cov_xi(snapshot, covariance_test_setup, sample_theory_cl, regtest):
    import xilikelihood as xlh
    import numpy as np

    from xilikelihood.file_handling import generate_filename
    from xilikelihood.theory_cl import TheoryCl, HAS_CCL
    from xilikelihood.pseudo_alm_cov import Cov

    if not HAS_CCL:
        pytest.skip("pyccl not available; skipping theory C_ell based covariance test")

    setup = covariance_test_setup
    exact_lmax = setup["exact_lmax"]

    mask = xlh.SphereMask(spins=[2], circmaskattr=(1000, 32), exact_lmax=exact_lmax)
    full_mask = xlh.SphereMask(spins=[2], circmaskattr=("fullsky", 32), exact_lmax=exact_lmax)
    
    circ_cov = Cov(mask, sample_theory_cl, exact_lmax)
    test_cov = circ_cov.cov_alm_xi()
    snapshot.check(test_cov, atol=1e-12)
    nomask_cov = Cov(full_mask, sample_theory_cl, exact_lmax)
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
    
    # Calculate the difference and analyze it
    diff = nomask_bruteforce_cov - nomask_cov_array
    max_diff = np.max(np.abs(diff))
    
    # Enhanced diagnostics for numerical differences
    tolerance = 1e-15
    large_diff_mask = np.abs(diff) > tolerance
    num_large_diffs = np.sum(large_diff_mask)
    
    regtest.write(f"Matrix shape: {nomask_cov_array.shape}\n")
    regtest.write(f"Maximum absolute difference: {max_diff:.2e}\n")
    regtest.write(f"Mean absolute difference: {np.mean(np.abs(diff)):.2e}\n")
    regtest.write(f"Standard deviation of differences: {np.std(diff):.2e}\n")
    regtest.write(
        "Number of entries exceeding tolerance "
        f"{tolerance:.0e}: {num_large_diffs}\n"
    )
    
    if num_large_diffs > 0:
        large_diff_indices = np.where(large_diff_mask)
        large_diff_values = diff[large_diff_mask]
        
        regtest.write("\nLargest differences (showing up to 10):\n")
        # Sort by absolute value and show the largest ones
        sorted_indices = np.argsort(np.abs(large_diff_values))[::-1][:10]
        for i, idx in enumerate(sorted_indices):
            row, col = large_diff_indices[0][idx], large_diff_indices[1][idx]
            value = large_diff_values[idx]
            orig_val = nomask_cov_array[row, col]
            new_val = nomask_bruteforce_cov[row, col]
            rel_diff = value / orig_val if orig_val != 0 else float('inf')
            regtest.write(
                "  "
                f"[{row:3d}, {col:3d}]: diff={value:+.2e}, "
                f"orig={orig_val:.2e}, new={new_val:.2e}, rel={rel_diff:.2e}\n"
            )
    
    # Check if differences are symmetric (since covariance should be symmetric)
    diff_asymmetry = np.max(np.abs(diff - diff.T))
    regtest.write(f"Maximum asymmetry in differences: {diff_asymmetry:.2e}\n")
    
    # Use a more reasonable tolerance for the actual test
    assert np.allclose(nomask_bruteforce_cov, nomask_cov_array, atol=1e-12), \
        f"Full-sky covariance matrices differ by more than tolerance. Max diff: {max_diff:.2e}"



@pytest.mark.slow
def test_cov_diag(covariance_test_setup, sample_theory_cl):
    # check that the diagonal matches the pseudo-cl to 10%:
    import xilikelihood as xlh
    import numpy as np

    from xilikelihood.theory_cl import HAS_CCL

    if not HAS_CCL:
        pytest.skip("pyccl not available; skipping theory C_ell based covariance test")

    setup = covariance_test_setup
    exact_lmax = setup["exact_lmax"]
    # Use a smaller nside here so the mask coupling matrices and TheoryCl lmax agree.
    # cl2pseudocl uses mask.lmax-sized coupling matrices, so TheoryCl must match mask.lmax.
    mask = setup["mask"]
    theory_lmax = mask.lmax
    
    cov = xlh.pseudo_alm_cov.Cov(mask, sample_theory_cl, exact_lmax)
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
    
    cov.cl2pseudocl(ischain=True)
    twoell = 2 * np.arange(exact_lmax + 1) + 1
    pcl[0] = cov.p_ee[: exact_lmax + 1] * twoell
    pcl[1] = cov.p_bb[: exact_lmax + 1] * twoell
    assert np.allclose(
        pcl, check_pcl, rtol=1e-1
    ), "cov_alm_gen: covariance diagonal does not agree with pseudo C_ell"
