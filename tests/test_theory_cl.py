def test_cl_class():
    from scipy.special import j0
    import scipy.integrate as integrate
    import theory_cl
    import matplotlib.pyplot as plt

    new_cl = theory_cl.TheoryCl(30, path="Cl_3x2pt_kids55.txt")

    assert np.allclose(new_cl.ee, np.zeros(31)), "Something wrong with zero Cl assignment"