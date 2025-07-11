# Basic Simulation Example

This example demonstrates how to set up and run a basic two-point correlation function simulation.

## Setup

```python
import xilikelihood as xlh
import numpy as np
import matplotlib.pyplot as plt

# Create a survey mask
mask = xlh.SphereMask(spins=[2], circmaskattr=(10000, 256))

# Set up redshift bins
z = np.linspace(0.01, 3.0, 100)
redshift_bins = [xlh.RedshiftBin(nbin=1, z=z, zmean=0.5, zsig=0.1)]

# Define the angular bins (in degrees)  
angular_bins_in_deg = [(2, 3), (3, 4), (4, 5), (5, 7), (7, 10)]

# Prepare theory inputs
numerical_combinations, redshift_bin_combinations, is_cov_cross, shot_noise, mapper = xlh.prepare_theory_cl_inputs(redshift_bins)

# Generate theory power spectra
theory_cls = xlh.generate_theory_cl(
    mask.lmax,
    redshift_bin_combinations,
    shot_noise,
    cosmo={'omega_m': 0.31, 's8': 0.8}
)
```

## Running the Simulation

```python
# Generate correlation functions
result = xlh.simulate_correlation_functions(
    theory_cls, [mask], angular_bins_in_deg, n_batch=100
)

print(f"Generated xi for {len(angular_bins_in_deg)} angular bins")
print(f"Simulation shape: xi_plus {result['xi_plus'].shape}, xi_minus {result['xi_minus'].shape}")

# Extract results explicitly
xi_plus = result['xi_plus']
xi_minus = result['xi_minus']
theta = np.array([(bins[0] + bins[1])/2 for bins in angular_bins_in_deg])
```

## Visualizing Results

```python
# Plot the correlation functions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Compute statistics
xip_mean = np.mean(xi_plus, axis=0)
xip_std = np.std(xi_plus, axis=0)
xim_mean = np.mean(xi_minus, axis=0)
xim_std = np.std(xi_minus, axis=0)

# Xi plus
ax1.errorbar(theta, xip_mean[0, :], yerr=xip_std[0, :], fmt='o-', label='ξ₊')
ax1.set_xlabel('θ (degrees)')
ax1.set_ylabel('ξ₊(θ)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_title('Xi Plus')

# Xi minus (can be negative!)
ax2.errorbar(theta, xim_mean[0, :], yerr=xim_std[0, :], fmt='s-', label='ξ₋', color='orange')
ax2.set_xlabel('θ (degrees)')
ax2.set_ylabel('ξ₋(θ)')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_title('Xi Minus')

plt.tight_layout()
plt.show()
```

## Multiple Realizations

```python
# Generate multiple batches for statistical analysis
n_batches = 10
batch_size = 50
all_xip = []
all_xim = []

for batch in range(n_batches):
    batch_result = xlh.simulate_correlation_functions(
        theory_cls, [mask], angular_bins_in_deg, n_batch=batch_size
    )
    all_xip.append(batch_result['xi_plus'])
    all_xim.append(batch_result['xi_minus'])

# Combine all realizations
xip_array = np.concatenate(all_xip, axis=0)
xim_array = np.concatenate(all_xim, axis=0)

# Compute statistics
xip_mean = np.mean(xip_array, axis=0)
xip_std = np.std(xip_array, axis=0)

print(f"Mean ξ₊ across {n_batches * batch_size} realizations:")
for i, (angle, mean_val, std_val) in enumerate(zip(theta, xip_mean[0, :], xip_std[0, :])):
    print(f"  Bin {i+1} ({angle:.1f}°): {mean_val:.2e} ± {std_val:.2e}")
```

## Next Steps

- Learn how to use these simulations in [Likelihood Analysis](../api/likelihood.md)
- Explore parameter estimation workflows in the [Quick Start Guide](../quickstart.md)
- See the [API Reference](../api/index.md) for more customization options
