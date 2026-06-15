# Likelihood Analysis Example

This example shows the core likelihood workflow without running map
simulations. It uses deterministic mock data generated from the fiducial theory
mean, so it does not require the custom GLASS simulation dependency.

## Setup

```python
import numpy as np
import xilikelihood as xlh
from xilikelihood.core_utils import LikelihoodConfig
from xilikelihood.mock_data import create_mock_data

mask = xlh.SphereMask(
    spins=[2],
    circmaskattr=(1000, 256),
    exact_lmax=10,
    l_smooth=30,
)

z = np.linspace(0.01, 3.0, 100)
redshift_bins = [
    xlh.RedshiftBin(nbin=1, z=z, zmean=0.5, zsig=0.1),
    xlh.RedshiftBin(nbin=2, z=z, zmean=1.0, zsig=0.1),
]

angular_bins_in_deg = [(1.0, 2.0), (2.0, 4.0)]
config = LikelihoodConfig(cf_steps=1024, pdf_steps=1024)

likelihood = xlh.XiLikelihood(
    mask=mask,
    redshift_bins=redshift_bins,
    ang_bins_in_deg=angular_bins_in_deg,
    config=config,
)
likelihood.setup_likelihood()
```

## Deterministic Mock Data

```python
fiducial_cosmology = {"omega_m": 0.31, "s8": 0.8}

mock_data, gaussian_covariance = create_mock_data(
    likelihood,
    mock_data_path="mock_data_example.npz",
    gaussian_covariance_path="gaussian_covariance_example.npz",
    fiducial_cosmo=fiducial_cosmology,
    random=None,
)
```

The deterministic mock data are the fiducial theory mean. The returned Gaussian
covariance can be attached to the likelihood for Gaussian comparison modes:

```python
likelihood.gaussian_covariance = gaussian_covariance
```

## Copula Likelihood Evaluation

```python
test_cosmology = {"omega_m": 0.30, "s8": 0.82}
log_likelihood = likelihood.loglikelihood(
    mock_data,
    test_cosmology,
    likelihood_type="copula",
)
print(log_likelihood)
```

## Gaussian Comparison

```python
log_likelihood_gaussian = likelihood.loglikelihood(
    mock_data,
    test_cosmology,
    likelihood_type="gaussian",
)
print(log_likelihood_gaussian)
```

For diagnostics, both likelihood values can be returned together:

```python
log_likelihood_copula, log_likelihood_gaussian = likelihood.loglikelihood(
    mock_data,
    test_cosmology,
    likelihood_type="both",
    allow_diagnostic=True,
)
```

## Notes

- The example is deliberately small. Paper-scale configurations use larger
  masks, more angular bins, and more expensive exact multipole limits.
- The output value is not a paper result; it is a compact workflow example.
- For the official release smoke check, see
  [REPRODUCING.md](../../REPRODUCING.md).
