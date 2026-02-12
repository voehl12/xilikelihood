import numpy as np

from xilikelihood import simulate


def test_create_maps_empty():
    nside = 4
    maps = simulate.create_maps([], nside=nside)
    npix = 12 * nside * nside
    assert maps.shape == (3, npix)
    assert np.allclose(maps, 0.0)


def test_get_noise_sigma_default(sample_theory_cl, sample_mask):
    sample_theory_cl.sigma_e = "default"
    sigma = simulate.get_noise_sigma(sample_theory_cl, sample_mask.nside)
    assert sigma is not None
    assert np.isfinite(sigma)
    assert sigma > 0


def test_add_noise_to_maps_none():
    nside = 4
    maps = simulate.create_maps([], nside=nside)
    noisy_maps = simulate.add_noise_to_maps(maps, nside=nside, noise_sigmas=None)
    assert noisy_maps.shape == maps.shape
    assert np.allclose(noisy_maps, maps)
