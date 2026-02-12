def test_wllpmmp(snapshot, tmp_path):
    from xilikelihood import mask_props
    import numpy as np

    mask = mask_props.SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=10)
    path = tmp_path / "wpm.npz"
    w_arr = mask.w_arr(path=path)
    with np.printoptions(threshold=40, edgeitems=2, linewidth=120):
        snapshot.check(
            w_arr,
            atol=1e-15,
        )