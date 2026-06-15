# Quick Start Guide

This guide will help you get started with xilikelihood for two-point correlation function likelihood analysis.

## Basic Usage

### 1. Import the Package

```python
import xilikelihood as xlh
import numpy as np
from xilikelihood.core_utils import LikelihoodConfig
from xilikelihood.mock_data import create_mock_data
```

### 2. Set Up Survey Mask and Bins

```python
# Create a survey mask
mask = xlh.SphereMask(
    spins=[2],
    circmaskattr=(1000, 256),
    exact_lmax=10,
    l_smooth=30,
)

# Set up redshift bins (option 1: from file)
redshift_bins = [xlh.RedshiftBin(nbin=1, filepath='path/to/redshift_file.txt')]

# Set up redshift bins (option 2: Gaussian distribution)
z = np.linspace(0.01, 3.0, 100)
redshift_bins = [
    xlh.RedshiftBin(nbin=1, z=z, zmean=0.5, zsig=0.1),
    xlh.RedshiftBin(nbin=2, z=z, zmean=1.0, zsig=0.1)
]

# Set up redshift bins (option 3: custom z, nz arrays)
redshift_bins = [
    xlh.RedshiftBin(nbin=1, z=z_array, nz=nz_array)
]

# Angular bins in degrees
angular_bins_in_deg = [(1.0, 2.0), (2.0, 4.0)]

# Or use fiducial setup
# redshift_bins, angular_bins_in_deg = xlh.fiducial_dataspace()
```

### 3. Set Up Likelihood Analysis
```python
# Create and setup likelihood instance
config = LikelihoodConfig(cf_steps=1024, pdf_steps=1024)
likelihood = xlh.XiLikelihood(
    mask=mask,
    redshift_bins=redshift_bins,
    ang_bins_in_deg=angular_bins_in_deg,
    config=config,
)
likelihood.setup_likelihood()
```

### 4. Evaluate a Likelihood

```python
fiducial_cosmology = {'omega_m': 0.31, 's8': 0.8}
mock_data, gaussian_covariance = create_mock_data(
    likelihood,
    mock_data_path="mock_data_quickstart.npz",
    gaussian_covariance_path="gaussian_covariance_quickstart.npz",
    fiducial_cosmo=fiducial_cosmology,
    random=None,
)

test_cosmology = {'omega_m': 0.30, 's8': 0.82}
log_likelihood = likelihood.loglikelihood(mock_data, test_cosmology)
```

## Example: Parameter Estimation

In a parameter-estimation workflow, the likelihood evaluation is repeated over a
grid or sampler. The following sketch shows the shape of a simple one-parameter
scan; production paper runs use the scripts under
`papers/second_paper_2025/analysis/`.

```python
import numpy as np

# Reuse the likelihood object from above and replace mock_data with a measured
# data vector when analysing real data.
measured_data = mock_data

# Parameter grid
s8_values = np.linspace(0.7, 0.9, 100)
log_likelihoods = []

# Evaluate likelihood over parameter grid
for s8 in s8_values:
    cosmology = {'s8': s8, 'omega_m': 0.3}
    log_likelihood = likelihood.loglikelihood(measured_data, cosmology)
    log_likelihoods.append(log_likelihood)

# Find maximum likelihood estimate
best_s8 = s8_values[np.argmax(log_likelihoods)]
print(f"Best-fit σ₈: {best_s8}")
```

For detailed configuration options including `exact_lmax` and `LikelihoodConfig`, 
see [Configuration](configuration.md).

## Next Steps

- Explore the [API Reference](api/index.md) for detailed function documentation
- Check out the [Likelihood Analysis example](examples/likelihood_analysis.md)
- See [Configuration](configuration.md) for likelihood setup options
