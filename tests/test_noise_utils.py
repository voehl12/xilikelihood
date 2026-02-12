import numpy as np
import pytest

from xilikelihood import noise_utils
from xilikelihood.theory_cl import TheoryCl


def test_get_noise_cl_default_positive():
    noise = noise_utils.get_noise_cl()
    assert np.isfinite(noise)
    assert noise > 0


def test_get_noise_cl_invalid_raises():
    with pytest.raises(ValueError):
        noise_utils.get_noise_cl(sigma_e="bad")


def test_noise_cl_cube_snapshot(snapshot):
    noise_cl = np.array([0.1, 0.2, 0.3])
    cube = noise_utils.noise_cl_cube(noise_cl)
    assert cube.shape == (3, 3, 3)
    snapshot.check(cube, rtol=0.0, atol=0.0)


def test_get_noise_pixelsigma_positive(sample_mask):
    sigma = noise_utils.get_noise_pixelsigma(nside=sample_mask.nside)
    assert np.isfinite(sigma)
    assert sigma > 0


def test_get_noisy_cl_adds_noise(snapshot):
    cl = TheoryCl(lmax=5, sigma_e="default")
    cl_es, cl_bs = noise_utils.get_noisy_cl([cl], lmax=5)
    assert len(cl_es) == 1
    assert len(cl_bs) == 1
    snapshot.check(np.array(cl_es[0]), rtol=0.0, atol=0.0)
    snapshot.check(np.array(cl_bs[0]), rtol=0.0, atol=0.0)
