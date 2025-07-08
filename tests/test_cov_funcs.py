def test_palm_matching():
    import cov_funcs

    palm_kinds = ["ReE", "ImE", "ReB", "ImB"]
    print(cov_funcs.match_alm_inds(palm_kinds))
    assert cov_funcs.match_alm_inds(palm_kinds) == [0, 1, 2, 3], "Function no work"




