"""Tests for direct Gaussian likelihood mode."""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

import xilikelihood as xlh
from xilikelihood import copula_funcs as cop


@pytest.fixture
def gaussian_test_setup(covariance_test_setup, tmp_path):
    """Create a likelihood instance and shape-consistent deterministic mock data."""
    setup = covariance_test_setup

    likelihood = xlh.XiLikelihood(
        mask=setup["mask"],
        redshift_bins=setup["redshift_bins"],
        ang_bins_in_deg=setup["angular_bins_in_deg"],
        include_ximinus=False,
        exact_lmax=setup["exact_lmax"],
    )

    likelihood.setup_likelihood()

    mock_data_path = tmp_path / "gaussian_mode_mock.npz"
    cov_path = tmp_path / "gaussian_mode_cov.npz"
    mock_data, covariance = xlh.mock_data.create_mock_data(
        likelihood,
        mock_data_path,
        cov_path,
        fiducial_cosmo=setup["cosmo_params"],
        random=None,
    )

    # Stabilize for multivariate_normal with allow_singular=False.
    covariance = np.asarray(covariance)
    covariance = covariance + 1e-8 * np.eye(covariance.shape[0])
    likelihood.gaussian_covariance = covariance

    return likelihood, mock_data, setup["cosmo_params"], covariance


def _make_unconfigured_likelihood(setup):
    """Construct a fresh XiLikelihood instance without setting gaussian_covariance."""
    likelihood = xlh.XiLikelihood(
        mask=setup["mask"],
        redshift_bins=setup["redshift_bins"],
        ang_bins_in_deg=setup["angular_bins_in_deg"],
        include_ximinus=False,
        exact_lmax=setup["exact_lmax"],
    )
    likelihood.setup_likelihood()
    return likelihood


def test_default_likelihood_type_is_copula(gaussian_test_setup):
    likelihood, mock_data, cosmo, _ = gaussian_test_setup
    logl = likelihood.loglikelihood(mock_data, cosmo)
    assert np.isfinite(logl)


def test_unknown_likelihood_type_raises_value_error(gaussian_test_setup):
    likelihood, mock_data, cosmo, _ = gaussian_test_setup
    with pytest.raises(ValueError, match="likelihood_type"):
        likelihood.loglikelihood(mock_data, cosmo, likelihood_type="invalid")


def test_gaussian_mode_returns_finite(gaussian_test_setup):
    likelihood, mock_data, cosmo, _ = gaussian_test_setup
    logl = likelihood.loglikelihood(mock_data, cosmo, likelihood_type="gaussian")
    assert np.isfinite(logl)


def test_both_requires_allow_diagnostic(gaussian_test_setup):
    likelihood, mock_data, cosmo, _ = gaussian_test_setup
    with pytest.raises(ValueError, match="allow_diagnostic"):
        likelihood.loglikelihood(
            mock_data,
            cosmo,
            likelihood_type="both",
            allow_diagnostic=False,
        )


def test_both_returns_two_finite_values(gaussian_test_setup):
    likelihood, mock_data, cosmo, _ = gaussian_test_setup
    logl_copula, logl_gaussian = likelihood.loglikelihood(
        mock_data,
        cosmo,
        likelihood_type="both",
        allow_diagnostic=True,
    )
    assert np.isfinite(logl_copula)
    assert np.isfinite(logl_gaussian)


def test_gaussian_without_fixed_covariance_raises(covariance_test_setup, tmp_path):
    setup = covariance_test_setup
    likelihood = _make_unconfigured_likelihood(setup)

    mock_data_path = tmp_path / "missing_cov_mock.npz"
    cov_path = tmp_path / "missing_cov_cov.npz"
    mock_data, _ = xlh.mock_data.create_mock_data(
        likelihood,
        mock_data_path,
        cov_path,
        fiducial_cosmo=setup["cosmo_params"],
        random=None,
    )

    with pytest.raises(RuntimeError, match="gaussian_covariance"):
        likelihood.loglikelihood(mock_data, setup["cosmo_params"], likelihood_type="gaussian")


def test_both_without_fixed_covariance_raises(covariance_test_setup, tmp_path):
    setup = covariance_test_setup
    likelihood = _make_unconfigured_likelihood(setup)

    mock_data_path = tmp_path / "missing_cov_both_mock.npz"
    cov_path = tmp_path / "missing_cov_both_cov.npz"
    mock_data, _ = xlh.mock_data.create_mock_data(
        likelihood,
        mock_data_path,
        cov_path,
        fiducial_cosmo=setup["cosmo_params"],
        random=None,
    )

    with pytest.raises(RuntimeError, match="gaussian_covariance"):
        likelihood.loglikelihood(
            mock_data,
            setup["cosmo_params"],
            likelihood_type="both",
            allow_diagnostic=True,
        )


def test_copula_mode_does_not_require_fixed_covariance(covariance_test_setup, tmp_path):
    setup = covariance_test_setup
    likelihood = _make_unconfigured_likelihood(setup)

    mock_data_path = tmp_path / "copula_only_mock.npz"
    cov_path = tmp_path / "copula_only_cov.npz"
    mock_data, _ = xlh.mock_data.create_mock_data(
        likelihood,
        mock_data_path,
        cov_path,
        fiducial_cosmo=setup["cosmo_params"],
        random=None,
    )

    logl = likelihood.loglikelihood(mock_data, setup["cosmo_params"], likelihood_type="copula")
    assert np.isfinite(logl)


def test_subset_gaussian_matches_manual_subset_logpdf(gaussian_test_setup):
    likelihood, mock_data, cosmo, _ = gaussian_test_setup
    subset = [(0, 0), (1, 1)]

    logl_subset = likelihood.loglikelihood(
        mock_data,
        cosmo,
        likelihood_type="gaussian",
        data_subset=subset,
    )

    likelihood._prepare_likelihood_components(cosmo, highell=True, prepare_copula=False)
    mean_subset = cop.data_subset(likelihood._mean, subset)
    data_subset = cop.data_subset(mock_data, subset)
    cov_subset = cop.cov_subset(likelihood.gaussian_covariance, subset, likelihood._n_ang_bins)
    logl_manual = multivariate_normal.logpdf(
        np.asarray(data_subset).flatten(),
        mean=np.asarray(mean_subset).flatten(),
        cov=np.asarray(cov_subset),
    )

    np.testing.assert_allclose(logl_subset, logl_manual, rtol=1e-10)


def test_subset_gaussian_in_both_matches_direct_gaussian(gaussian_test_setup):
    likelihood, mock_data, cosmo, _ = gaussian_test_setup
    subset = [(0, 0), (1, 1)]

    _, logl_gaussian_both = likelihood.loglikelihood(
        mock_data,
        cosmo,
        likelihood_type="both",
        allow_diagnostic=True,
        data_subset=subset,
    )
    logl_gaussian = likelihood.loglikelihood(
        mock_data,
        cosmo,
        likelihood_type="gaussian",
        data_subset=subset,
    )
    np.testing.assert_allclose(logl_gaussian_both, logl_gaussian, rtol=1e-12)


def test_default_copula_matches_explicit_copula(gaussian_test_setup):
    likelihood, mock_data, cosmo, _ = gaussian_test_setup
    logl_default = likelihood.loglikelihood(mock_data, cosmo)
    logl_copula = likelihood.loglikelihood(mock_data, cosmo, likelihood_type="copula")
    np.testing.assert_allclose(logl_default, logl_copula, rtol=1e-14)


def test_both_gaussian_component_matches_direct_gaussian(gaussian_test_setup):
    likelihood, mock_data, cosmo, _ = gaussian_test_setup
    _, logl_gaussian_both = likelihood.loglikelihood(
        mock_data,
        cosmo,
        likelihood_type="both",
        allow_diagnostic=True,
    )
    logl_gaussian = likelihood.loglikelihood(mock_data, cosmo, likelihood_type="gaussian")
    np.testing.assert_allclose(logl_gaussian_both, logl_gaussian, rtol=1e-12)


def test_loglikelihood_gaussian_method_matches_dispatch(gaussian_test_setup):
    likelihood, mock_data, cosmo, _ = gaussian_test_setup
    logl_method = likelihood.loglikelihood_gaussian(mock_data, cosmo)
    logl_dispatch = likelihood.loglikelihood(mock_data, cosmo, likelihood_type="gaussian")
    np.testing.assert_allclose(logl_method, logl_dispatch, rtol=1e-14)


def test_gauss_compare_deprecated_wrapper_runs_after_preparation(gaussian_test_setup):
    likelihood, mock_data, cosmo, _ = gaussian_test_setup

    # gauss_compare wrapper uses already-prepared mean; prepare once via gaussian path.
    likelihood.loglikelihood(mock_data, cosmo, likelihood_type="gaussian")

    with pytest.warns(DeprecationWarning):
        logl = likelihood.gauss_compare(mock_data)
    assert np.isfinite(logl)


def test_gausscompare_parameter_deprecated_routing(gaussian_test_setup):
    likelihood, mock_data, cosmo, _ = gaussian_test_setup
    with pytest.warns(DeprecationWarning):
        result = likelihood.loglikelihood(mock_data, cosmo, gausscompare=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
