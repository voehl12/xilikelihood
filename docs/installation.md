# Installation

## Requirements

xilikelihood requires Python 3.8 or later and has the following dependencies:

- numpy >= 1.20.0
- scipy >= 1.7.0
- jax >= 0.3.0
- jaxlib >= 0.3.0
- healpy >= 1.14.0
- pyccl >= 2.0.0
- wigner >= 1.0.0

for simulations:
- glass (for version see information below)

if simulations with treecorr for comparison are desired:
- treecorr >= 4.0.0

## Special Dependencies

### PyCCL Installation

See https://ccl.readthedocs.io/en/latest/


### GLASS Installation

Generating correlation function simulations with the simulate module requires a custom (old) version of GLASS. Contact the author for further information. Compatibility with the current version of GLASS is planned but not implemented yet.
For lognormal simulations, the current GLASS version is needed, however, other dependencies break, such that at the moment simulations and likelihood computations are recommended to be run with different environments

```bash
# Clone and install the custom GLASS version
git clone https://github.com/glass-dev/glass.git
cd glass
pip install -e .
```


## Installing xilikelihood

### From Source

```bash
# Clone the repository
git clone https://github.com/yourusername/xilikelihood.git
cd xilikelihood

# Install in development mode
pip install -e .[dev,docs]
```

### From PyPI (not available yet)


