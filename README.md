# xilikelihood

Two-point correlation function likelihoods for cosmic shear surveys. Exact one-dimensional marginals and a copula approximation to the full n-dimensional likelihood, with JAX-accelerated numerical kernels that can run on CPU or GPU backends.

## Paper Reproducibility

This repository is being prepared as the reference implementation for:

- **New paper:** https://arxiv.org/abs/2604.07336
- **First paper / earlier method:** https://arxiv.org/abs/2407.08718

The final archived release and DOI will be added once the publication-ready repository tag is created. Citation metadata is provided in [CITATION.cff](CITATION.cff). For reproduction instructions, see [REPRODUCING.md](REPRODUCING.md). For the repository map and maintenance notes, see the [repository reference guide](docs/repository_reference.rst).

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
from xilikelihood.core_utils import LikelihoodConfig

# 1. Create a survey mask
mask = xlh.SphereMask(
    spins=[2],
    circmaskattr=(1000, 256),
    exact_lmax=10,
    l_smooth=30,
)

# 2. Set up redshift bins and angular bins
z = np.linspace(0.01, 3.0, 100)
redshift_bins = [
    xlh.RedshiftBin(nbin=1, z=z, zmean=0.5, zsig=0.1),
    xlh.RedshiftBin(nbin=2, z=z, zmean=1.0, zsig=0.1),
]
angular_bins_in_deg = [(1.0, 2.0), (2.0, 4.0)]

# Or use fiducial setup
# redshift_bins, angular_bins_in_deg = xlh.fiducial_dataspace()

# 3. Configure and set up the likelihood
config = LikelihoodConfig(cf_steps=1024, pdf_steps=1024)
likelihood = xlh.XiLikelihood(
    mask=mask,
    redshift_bins=redshift_bins,
    ang_bins_in_deg=angular_bins_in_deg,
    config=config,
)
likelihood.setup_likelihood()

# 4. Evaluate the likelihood for a xi data vector.
# Replace this toy vector with measured or simulated xi values with the same shape.
observed_xi = np.zeros(likelihood.data_shape_full)
test_cosmology = {"omega_m": 0.30, "s8": 0.82}
log_likelihood = likelihood.loglikelihood(observed_xi, test_cosmology)
print(log_likelihood)
```

For deterministic mock data and Gaussian comparison covariances, use
`xilikelihood.mock_data.create_mock_data(..., random=None)`. This path uses the
fiducial theory mean and does not require GLASS simulations. Simulation-backed
mock data (`random="frommap"`) requires the custom GLASS dependency.

## Key Features

- **Likelihood**: Exact likelihoods for correlation functions (currently only $\xi^+$)
- **JAX acceleration**: Numerical likelihood kernels use JAX and can run on CPU or GPU backends; the official smoke check was recorded on CPU for portability
- **Scale-dependent marginals**: Gaussian marginals below configurable angular scale cuts
- **Simulations**: Generate correlation functions from Gaussian random maps
- **Theory**: Compute power spectra and correlation functions
- **Masks**: Handle realistic survey geometries

### Planned Features
- **$\xi^-$ likelihood support**: Extension to include xi_minus correlations


## Dependencies

### Required Dependencies
- `numpy>=1.20.0,<2.0`
- `scipy>=1.7.0`
- `jax>=0.3.0`
- `jaxlib>=0.3.0`
- `healpy>=1.14.0`
- `treecorr>=4.0.0`
- `pyccl>=2.0.0`
- `matplotlib>=3.5.0`
- `seaborn>=0.11.0`
- `wigner>=1.0.0`

### Custom Simulation Dependency
- **GLASS** (custom version): Required for Gaussian and lognormal field simulations and
  simulation-backed mock data
  - Install from: `pip install -e ../glass`
  - Without this, simulation functions will raise informative errors. Likelihood setup, transformations, and many tests can still be used without running simulations.

The deterministic likelihood and mock-data workflows documented for the
paper-reference smoke checks do not require GLASS simulations.

## Scientific Background

This package implements the methods described in:
- New copula likelihood paper: https://arxiv.org/abs/2604.07336
- First paper / exact low-dimensional likelihood method: https://arxiv.org/abs/2407.08718

## Documentation

Full documentation: https://xilikelihood.readthedocs.io

## Examples

- [Basic simulation example](docs/examples/basic_simulation.md)
- [Quick start guide](docs/quickstart.md)
- [Repository reference](docs/repository_reference.rst)
