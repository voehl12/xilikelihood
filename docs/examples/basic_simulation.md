# Basic Simulation Example

This example demonstrates how to set up and run a basic two-point correlation function simulation.

## Setup

```python
import xilikelihood as xi
import numpy as np
import matplotlib.pyplot as plt

# Define the angular bins (in degrees)
angular_bins = [(2, 3), (3, 4), (4, 5), (5, 7), (7, 10)]

# Create the simulation object
simulation = xi.TwoPointSimulation(
    angular_bins=angular_bins,
    circmaskattr=(10000, 256),  # 10000 sq deg circular mask, nside=256
    l_smooth_mask=30,           # smoothing scale for mask
    clpath="Cl_3x2pt_kids55.txt",  # power spectrum file
    sigma_e=None                # no intrinsic ellipticity noise
)
```

## Running the Simulation

```python
# Generate correlation functions for job number 1
job_number = 1
simulation.xi_sim_1D(job_number)

# Load the results
results = np.load(f"job{job_number}.npz")
theta = results['theta']  # angular scales
xip = results['xip']      # xi_plus correlation function
xim = results['xim']      # xi_minus correlation function

print(f"Generated xi for {len(angular_bins)} angular bins")
print(f"Angular scales: {theta} degrees")
```

## Visualizing Results

```python
# Plot the correlation functions
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Xi plus
ax1.errorbar(theta, xip[0, 0], fmt='o-', label='ξ₊')
ax1.set_xlabel('θ (degrees)')
ax1.set_ylabel('ξ₊(θ)')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_title('Xi Plus')

# Xi minus  
ax2.errorbar(theta, np.abs(xim[0, 0]), fmt='s-', label='|ξ₋|', color='orange')
ax2.set_xlabel('θ (degrees)')
ax2.set_ylabel('|ξ₋(θ)|')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_title('Xi Minus (absolute value)')

plt.tight_layout()
plt.show()
```

## Multiple Realizations

```python
# Generate multiple realizations for statistical analysis
n_realizations = 10
xip_realizations = []
xim_realizations = []

for job in range(1, n_realizations + 1):
    simulation.xi_sim_1D(job)
    data = np.load(f"job{job}.npz")
    xip_realizations.append(data['xip'][0, 0])
    xim_realizations.append(data['xim'][0, 0])

# Convert to arrays
xip_array = np.array(xip_realizations)
xim_array = np.array(xim_realizations)

# Compute statistics
xip_mean = np.mean(xip_array, axis=0)
xip_std = np.std(xip_array, axis=0)

print(f"Mean ξ₊ across {n_realizations} realizations:")
for i, (angle, mean_val, std_val) in enumerate(zip(theta, xip_mean, xip_std)):
    print(f"  Bin {i+1} ({angle:.1f}°): {mean_val:.2e} ± {std_val:.2e}")
```

## Next Steps

- Learn how to use these simulations in [Likelihood Analysis](likelihood_analysis.md)
- Explore [Parameter Estimation](parameter_estimation.md) workflows
- See [Advanced Usage](advanced_usage.md) for customization options
