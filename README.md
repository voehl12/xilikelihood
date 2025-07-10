# xilikelihood

Two-point correlation function likelihood analysis for cosmic shear surveys.

## Installation

### Standard Installation
```bash
pip install xilikelihood
```

### Development Installation

For development or if you need the custom GLASS version:

```bash
# Clone the repository
git clone [repository_url]
cd xilikelihood

# Install custom GLASS version (required for simulations)
pip install -e ../glass

# Install the package in development mode
pip install -e .
```

### Custom GLASS Installation

This package requires a custom version of GLASS located in the `../glass` directory. To install it:

```bash
# From the parent directory of xilikelihood
pip install -e glass
```

If you don't have the custom GLASS version, simulation functions will raise an informative error message.

## Quick Start

```python
import xilikelihood as tpl

# 1. Create a survey mask
mask = tpl.SphereMask(spins=[2], circmaskattr=(10000, 256))

# 2. Generate theory predictions
theory_cls = tpl.generate_theory_cl(
    lmax=100, 
    cosmo={'omega_m': 0.31, 's8': 0.8}
)

# 3. Simulate correlation functions
angular_bins = [(1.0, 2.0), (2.0, 4.0), (4.0, 8.0)]
xi_plus, xi_minus = tpl.simulate_correlation_functions(
    theory_cls, [mask], angular_bins, n_batch=100
)

# 4. Run likelihood analysis
likelihood = tpl.XiLikelihood(xi_plus, xi_minus, angular_bins)
posterior = likelihood.sample_posterior()
```

## Key Features

- **Simulations**: Generate correlation functions from cosmological models
- **Theory**: Compute power spectra and correlation functions
- **Likelihood**: Bayesian parameter estimation
- **Masks**: Handle realistic survey geometries
- **Statistics**: Bootstrap resampling and moment calculations

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
- **wigner**: For correlation function calculations (some functions need this)

The package is designed to work gracefully without optional dependencies, providing informative error messages when they're needed.

## Scientific Background

This package implements the methods described in:
- [Your paper citation]
- [Related work citations]

## Documentation

Full documentation: https://xilikelihood.readthedocs.io

## Examples

- [Basic simulation example](examples/basic_simulation.py)
- [Likelihood analysis example](examples/likelihood_analysis.py)
- [Mask creation example](examples/mask_creation.py)