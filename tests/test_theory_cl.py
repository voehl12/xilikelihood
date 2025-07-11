def test_cl_class():
    import theory_cl

    new_cl = theory_cl.TheoryCl(30)
    assert np.allclose(new_cl.ee, np.zeros(31))