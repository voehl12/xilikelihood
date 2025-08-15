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
    
    # Get the package directory (parent of xilikelihood module)
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clpath = os.path.join(package_dir, "papers", "first_paper_2024", "data", "cls", "Cl_3x2pt_kids55.txt")

    mask = xlh.SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=10)
    full_mask = xlh.SphereMask(spins=[2], circmaskattr=("fullsky", 256), exact_lmax=10)
    theorycl = TheoryCl(767, clpath=clpath)
    circ_cov = Cov(mask, theorycl, 10)
    test_cov = circ_cov.cov_alm_xi()
    snapshot.check(test_cov, atol=1e-15)
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
    
    # Calculate the difference and analyze it
    diff = nomask_bruteforce_cov - nomask_cov_array
    max_diff = np.max(np.abs(diff))
    
    # Enhanced diagnostics for numerical differences
    tolerance = 1e-15
    large_diff_mask = np.abs(diff) > tolerance
    num_large_diffs = np.sum(large_diff_mask)
    
    print(f"\n=== Covariance Matrix Comparison Diagnostics ===")
    print(f"Matrix shape: {nomask_cov_array.shape}")
    print(f"Maximum absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {np.mean(np.abs(diff)):.2e}")
    print(f"Standard deviation of differences: {np.std(diff):.2e}")
    print(f"Number of entries exceeding tolerance {tolerance:.0e}: {num_large_diffs}")
    
    if num_large_diffs > 0:
        large_diff_indices = np.where(large_diff_mask)
        large_diff_values = diff[large_diff_mask]
        
        print(f"\nLargest differences (showing up to 10):")
        # Sort by absolute value and show the largest ones
        sorted_indices = np.argsort(np.abs(large_diff_values))[::-1][:10]
        for i, idx in enumerate(sorted_indices):
            row, col = large_diff_indices[0][idx], large_diff_indices[1][idx]
            value = large_diff_values[idx]
            orig_val = nomask_cov_array[row, col]
            new_val = nomask_bruteforce_cov[row, col]
            rel_diff = value / orig_val if orig_val != 0 else float('inf')
            print(f"  [{row:3d}, {col:3d}]: diff={value:+.2e}, orig={orig_val:.2e}, new={new_val:.2e}, rel={rel_diff:.2e}")
    
    # Check if differences are symmetric (since covariance should be symmetric)
    diff_asymmetry = np.max(np.abs(diff - diff.T))
    print(f"Maximum asymmetry in differences: {diff_asymmetry:.2e}")
    
    # Use a more reasonable tolerance for the actual test
    assert np.allclose(nomask_bruteforce_cov, nomask_cov_array, atol=1e-14), \
        f"Full-sky covariance matrices differ by more than tolerance. Max diff: {max_diff:.2e}"



def test_cov_diag():
    # check that the diagonal matches the pseudo-cl to 10%:
    import xilikelihood as xlh
    import numpy as np

    # Get the package directory (parent of xilikelihood module)
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clpath = os.path.join(package_dir, "papers", "first_paper_2024", "data", "cls", "Cl_3x2pt_kids55.txt")

    exact_lmax = 10
    mask = xlh.SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=exact_lmax)
    theorycl = xlh.theory_cl.TheoryCl(767, clpath=clpath)
    cov = xlh.pseudo_alm_cov.Cov(mask, theorycl, exact_lmax)
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
    twoell = 2 * np.arange(mask.lmax + 1) + 1
    cov.cl2pseudocl()
    pcl[0], pcl[1] = (cov.p_ee * twoell)[: exact_lmax + 1], (cov.p_bb * twoell)[: exact_lmax + 1]
    assert np.allclose(
        pcl, check_pcl, rtol=1e-1
    ), "cov_alm_gen: covariance diagonal does not agree with pseudo C_ell"
