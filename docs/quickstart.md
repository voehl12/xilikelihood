# Quick Start Guide

This guide will help you get started with xilikelihood for two-point correlation function likelihood analysis.

## Basic Usage

### 1. Import the Package

```python
import xilikelihood as xi
import numpy as np
```

### 2. Set Up a Simulation

```python
# Define angular bins (in degrees)
angular_bins = [(2, 3), (3, 4), (4, 5)]

# Create a simulation instance
simulation = xi.TwoPointSimulation(
    angular_bins,
    circmaskattr=(10000, 256),  # circular mask: area and nside
    l_smooth_mask=30,            # smoothing scale
    clpath="Cl_3x2pt_kids55.txt",  # path to power spectrum
    sigma_e=None                 # intrinsic ellipticity noise
)
```

### 3. Generate Correlation Functions

```python
# Generate xi for a specific job/realization
job_number = 1
simulation.xi_sim_1D(job_number)

# Load the results
xi_data = np.load(f"job{job_number}.npz")
xip = xi_data['xip']  # xi_plus
xim = xi_data['xim']  # xi_minus
```

### 4. Set Up a Likelihood

```python
# Load or define your data vector
data_vector = np.array([...])  # your measured xi values

# Create likelihood instance
likelihood = xi.XiLikelihood(
    data_vector=data_vector,
    angular_bins=angular_bins,
    redshift_bins=[(0, 1), (1, 2)],  # redshift bin combinations
    simulation_params={
        'circmaskattr': (10000, 256),
        'l_smooth_mask': 30,
        'clpath': 'Cl_3x2pt_kids55.txt'
    }
)
```

### 5. Evaluate the Likelihood

```python
# Define cosmological parameters
cosmology = {
    's8': 0.8,
    'omega_m': 0.3,
    # ... other parameters
}

# Compute log-likelihood
log_likelihood = likelihood.loglikelihood(data_vector, cosmology)
print(f"Log-likelihood: {log_likelihood}")
```

## Example: Parameter Estimation

Here's a complete example for estimating σ₈:

```python
import xilikelihood as xi
import numpy as np

# Set up the analysis
angular_bins = [(2, 3), (3, 4)]
likelihood = xi.XiLikelihood(
    data_vector=measured_data,
    angular_bins=angular_bins,
    redshift_bins=[(0, 1)],
    simulation_params={
        'circmaskattr': (10000, 256),
        'l_smooth_mask': 30,
        'clpath': 'fiducial_cl.txt'
    }
)

# Parameter grid
s8_values = np.linspace(0.7, 0.9, 100)
log_likelihoods = []

# Evaluate likelihood over parameter grid
for s8 in s8_values:
    cosmology = {'s8': s8, 'omega_m': 0.3}
    log_like = likelihood.loglikelihood(measured_data, cosmology)
    log_likelihoods.append(log_like)

# Find maximum likelihood estimate
best_s8 = s8_values[np.argmax(log_likelihoods)]
print(f"Best-fit σ₈: {best_s8}")
```

## Next Steps

- Explore the [API Reference](api/index.md) for detailed function documentation
- Check out the [Examples](examples/index.md) for more advanced use cases
- Learn about [Theory and Methods](theory.md) behind the implementation
