# xilikelihood

Two-point correlation function likelihoods for cosmic shear surveys. Exact one-dimensional marginals and a copula approximation to the full n-dimensional likelihood.

## Installation

### Development Installation

For development or if you need the custom GLASS version:

```bash
# Clone the repository
git clone https://github.com/voehl12/xilikelihood.git
cd xilikelihood


# Install the package in development mode
pip install -e .
```

### Custom GLASS Installation

This package requires a custom version of GLASS. If you need it, please reach out. To install it:

```bash
# From the parent directory of xilikelihood
pip install -e glass
```


If you don't have the custom GLASS version, simulation functions will raise an error message.

## Quick Start

```python
import xilikelihood as xlh
import numpy as np

# 1. Create a survey mask
mask = xlh.SphereMask(spins=[2], circmaskattr=(10000, 256))

# 2. Set up redshift bins and angular bins
z = np.linspace(0.01, 3.0, 100)
redshift_bins = [xlh.RedshiftBin(nbin=1, z=z, zmean=0.5, zsig=0.1), 
                 xlh.RedshiftBin(nbin=2, z=z, zmean=1.0, zsig=0.1)]
angular_bins_in_deg = [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]

# Or use fiducial setup
# redshift_bins, angular_bins_in_deg = xlh.fiducial_dataspace()

# 3. Prepare theory inputs and generate power spectra
numerical_combinations, redshift_bin_combinations, is_cov_cross, shot_noise, 
mapper = xlh.prepare_theory_cl_inputs(redshift_bins) # by default with shot noise
theory_cls = xlh.generate_theory_cl(
    mask.lmax,
    redshift_bin_combinations,
    shot_noise,
    cosmo={'omega_m': 0.31, 's8': 0.8}
)

# 4. Simulate correlation functions (optional)
result = xlh.simulate_correlation_functions(
    theory_cls, [mask], angular_bins_in_deg, n_batch=100
)
xi_plus, xi_minus = result['xi_plus'], result['xi_minus']

# 5. Set up likelihood analysis
likelihood = xlh.XiLikelihood(mask, redshift_bins, ang_bins_in_deg)
likelihood.setup_likelihood()

# 6. Evaluate likelihood with data
# log_likelihood = likelihood.loglikelihood(data_vector, cosmology)
# cosmology is a dictionary with cosmological parameters
```

## Key Features

- **Likelihood**: Exact likelihoods for correlation functions (currently only $\xi^+$)
- **Scale-dependent marginals**: Gaussian marginals below configurable angular scale cuts
- **Simulations**: Generate correlation functions from Gaussian random maps
- **Theory**: Compute power spectra and correlation functions
- **Masks**: Handle realistic survey geometries

### Planned Features
- **$\xi^-$ likelihood support**: Extension to include xi_minus correlations


## Dependencies

### Required Dependencies
- `numpy>=1.20.0`
- `scipy>=1.7.0`
- `healpy>=1.14.0`
- `matplotlib>=3.5.0`
- `jax>=0.3.0` (for accelerated computations)

### Optional Dependencies
- **GLASS** (custom version): Required for Gaussian field simulations
  - Install from: `pip install -e ../glass`
  - Without this, simulation functions will raise informative errors
- **TreeCorr**: For alternative correlation function estimation
- **pyccl**: For cosmological computations
- **wigner**: For curved sky correlation function calculations 

The package is designed to work without optional dependencies, providing informative error messages when they're needed.

## Scientific Background

This package implements the methods described in:
- []

## Documentation

Full documentation: https://xilikelihood.readthedocs.io

## Examples

- [Basic simulation example](examples/basic_simulation.py)
- [Likelihood analysis example](examples/likelihood_analysis.py)
- [Mask creation example](examples/mask_creation.py)