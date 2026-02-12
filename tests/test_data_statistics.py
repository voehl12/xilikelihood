import numpy as np

from xilikelihood import data_statistics


def test_bootstrap_shape():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 4))
    samples = data_statistics.bootstrap(data, n=10, axis=0, func=np.mean)
    assert samples.shape == (10, 4)


def test_compute_simulation_moments_mean():
    rng = np.random.default_rng(1)
    sims = rng.normal(size=(8, 5))
    moments = data_statistics.compute_simulation_moments(sims, orders=[1], axis=0)
    assert np.allclose(moments[0], np.mean(sims, axis=0))
