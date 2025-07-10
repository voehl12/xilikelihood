# Quick Start Guide

This guide will help you get started with xilikelihood for two-point correlation function likelihood analysis.

## Basic Usage

### 1. Import the Package

```python
import xilikelihood as xlh
import numpy as np
```

### 2. Set Up Survey Mask and Bins

```python
# Create a survey mask
mask = xlh.SphereMask(spins=[2], circmaskattr=(10000, 256))

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
angular_bins_in_deg = [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]

# Or use fiducial setup
# redshift_bins, angular_bins_in_deg = xlh.fiducial_dataspace()
```

### 3. Generate Theory Power Spectra

```python
# Prepare theory inputs and generate power spectra
numerical_combinations, redshift_bin_combinations, is_cov_cross, shot_noise, mapper = xlh.prepare_theory_cl_inputs(redshift_bins)
theory_cls = xlh.generate_theory_cl(
    mask.lmax,
    redshift_bin_combinations,
    shot_noise,
    cosmo={'omega_m': 0.31, 's8': 0.8}
)
```

### 4. Simulate Correlation Functions (Optional)

```python
# Generate correlation functions from theory
result = xlh.simulate_correlation_functions(
    theory_cls, [mask], angular_bins_in_deg, n_batch=100
)
xi_plus, xi_minus = result['xi_plus'], result['xi_minus']
```

### 5. Set Up Likelihood Analysis
```python
# Create and setup likelihood instance
likelihood = xlh.XiLikelihood(mask, redshift_bins, angular_bins_in_deg)
likelihood.setup_likelihood()
```

## Example: Parameter Estimation

Here's a complete example for estimating σ₈:

```python
import xilikelihood as xlh
import numpy as np

# Set up the analysis
mask = xlh.SphereMask(spins=[2], circmaskattr=(10000, 256))

# Create redshift bins
z = np.linspace(0.01, 3.0, 100)
redshift_bins = [xlh.RedshiftBin(nbin=1, z=z, zmean=0.5, zsig=0.1)]

# Angular bins in degrees
angular_bins_in_deg = [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]

# Set up likelihood
likelihood = xlh.XiLikelihood(mask, redshift_bins, angular_bins_in_deg)
likelihood.setup_likelihood()

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

## Next Steps

- Explore the [API Reference](api/index.md) for detailed function documentation
- Check out the [Examples](examples/index.md) for more advanced use cases
- Learn about [Theory and Methods](theory.md) behind the implementation
