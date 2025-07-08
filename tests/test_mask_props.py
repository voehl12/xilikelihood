def test_wllpmmp(snapshot, tmp_path):
    import mask_props

    mask = mask_props.SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=10)
    path = tmp_path / "wpm.npz"
    w_arr = mask.w_arr(path=path)
    snapshot.check(w_arr, atol=1e-16)