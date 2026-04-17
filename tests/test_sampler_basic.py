"""Basic regression tests for papers/second_paper_2025/analysis/sampler.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "papers" / "second_paper_2025" / "analysis"
SAMPLER_PATH = ANALYSIS_DIR / "sampler.py"


@pytest.fixture(scope="module")
def sampler_module():
    """Import the analysis sampler module from its file path."""
    sys.path.insert(0, str(ANALYSIS_DIR))
    try:
        spec = importlib.util.spec_from_file_location("analysis_sampler", SAMPLER_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_log_prob_combines_priors_and_likelihood(sampler_module, monkeypatch):
    """The log posterior should add uniform bounds, Gaussian prior, and likelihood."""
    monkeypatch.setattr(sampler_module, "params", ["omega_m", "delta_z", "s8"])
    monkeypatch.setattr(sampler_module, "mvn_param", "delta_z")
    monkeypatch.setattr(sampler_module, "mvn_mean", np.array([0.0, 0.0]))
    monkeypatch.setattr(sampler_module, "mvn_cov", np.eye(2))
    monkeypatch.setattr(sampler_module, "uniform_priors", {"omega_m": (0.1, 0.5), "s8": (0.6, 0.95)})

    def fake_likelihood(theta_dict):
        assert set(theta_dict) == {"omega_m", "delta_z", "s8"}
        return 2.5

    def fake_mvn_prior(theta_dict):
        assert np.allclose(theta_dict["delta_z"], [0.1, -0.2])
        return 1.0

    theta = np.array([0.3, 0.1, -0.2, 0.8])
    logp = sampler_module._log_prob_fn(
        theta,
        params=sampler_module.params,
        mvn_param=sampler_module.mvn_param,
        mvn_mean=sampler_module.mvn_mean,
        uniform_priors=sampler_module.uniform_priors,
        likelihood_fn=fake_likelihood,
        log_prior_mvn_fn=fake_mvn_prior,
    )

    assert logp == pytest.approx(3.5)


def test_log_prob_rejects_out_of_bounds_parameters(sampler_module, monkeypatch):
    """Out-of-bounds parameters should be rejected before the likelihood is evaluated."""
    monkeypatch.setattr(sampler_module, "params", ["omega_m", "delta_z", "s8"])
    monkeypatch.setattr(sampler_module, "mvn_param", "delta_z")
    monkeypatch.setattr(sampler_module, "mvn_mean", np.array([0.0, 0.0]))
    monkeypatch.setattr(sampler_module, "mvn_cov", np.eye(2))
    monkeypatch.setattr(sampler_module, "uniform_priors", {"omega_m": (0.1, 0.5), "s8": (0.6, 0.95)})

    def should_not_run(_theta_dict):
        raise AssertionError("likelihood should not be evaluated for invalid parameters")

    theta = np.array([0.8, 0.0, 0.0, 0.8])
    logp = sampler_module._log_prob_fn(
        theta,
        params=sampler_module.params,
        mvn_param=sampler_module.mvn_param,
        mvn_mean=sampler_module.mvn_mean,
        uniform_priors=sampler_module.uniform_priors,
        likelihood_fn=should_not_run,
        log_prior_mvn_fn=lambda _theta_dict: 0.0,
    )

    assert logp == -np.inf


def test_save_corner_plot_handles_3d_samples(sampler_module, monkeypatch, tmp_path):
    """3D emcee samples should produce both trace and corner outputs."""
    saved_paths = []

    class DummyFig:
        pass

    class DummyCorner:
        @staticmethod
        def corner(*_args, **_kwargs):
            return DummyFig()

    monkeypatch.setattr(sampler_module, "HAS_CORNER", True)
    monkeypatch.setattr(sampler_module, "corner", DummyCorner())
    monkeypatch.setattr(sampler_module, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(sampler_module.plt, "savefig", lambda path, **_kwargs: saved_paths.append(Path(path)))
    monkeypatch.setattr(sampler_module.plt, "close", lambda *_args, **_kwargs: None)

    samples = np.random.randn(4, 3, 2)
    sampler_module.save_corner_plot(samples, labels=["a", "b"], filename="corner_test.png")

    assert saved_paths == [tmp_path / "corner_test_traces.png", tmp_path / "corner_test.png"]
