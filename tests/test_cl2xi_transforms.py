import numpy as np
import pytest

from xilikelihood.cl2xi_transforms import HAS_WIGNER, cl2pseudocl, prep_prefactors


def test_prep_prefactors_shape(sample_mask, sample_angular_bins, snapshot):
    if not HAS_WIGNER:
        pytest.skip("wigner not available")
    wl = sample_mask.wl
    norm_lmax = min(10, len(wl) - 1)
    out_lmax = min(5, norm_lmax)
    prefactors = prep_prefactors(sample_angular_bins, wl, norm_lmax, out_lmax)
    assert prefactors.shape == (len(sample_angular_bins), 2, out_lmax + 1)
    snapshot.check(prefactors, rtol=1e-10)


def test_prep_prefactors_invalid(sample_mask):
    if not HAS_WIGNER:
        pytest.skip("wigner not available")
    wl = sample_mask.wl
    with pytest.raises(ValueError):
        prep_prefactors([], wl, 2, 2)
    with pytest.raises(ValueError):
        prep_prefactors([(2.0, 1.0)], wl, 2, 2)


def test_cl2pseudocl_identity():
    lmax = 3
    identity = np.eye(lmax + 1)
    mllp = np.zeros((3, lmax + 1, lmax + 1))
    mllp[0] = identity
    mllp[1] = np.zeros_like(identity)

    class DummyCl:
        def __init__(self, ee, bb):
            self.ee = ee
            self.bb = bb

    cl_e = np.arange(lmax + 1, dtype=float)
    cl_b = np.arange(lmax + 1, dtype=float) + 1.0
    theorycls = [DummyCl(cl_e, cl_b)]

    pseudo = cl2pseudocl(mllp, theorycls)
    assert pseudo.shape == (3, 1, lmax + 1)
    assert np.allclose(np.array(pseudo[0, 0]), cl_e)
    assert np.allclose(np.array(pseudo[1, 0]), cl_b)
    assert np.allclose(np.array(pseudo[2, 0]), cl_b)
