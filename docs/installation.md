# Installation

## Requirements

xilikelihood requires Python 3.8 or later and has the following dependencies:

- numpy >= 1.20.0
- scipy >= 1.7.0
- jax >= 0.3.0
- jaxlib >= 0.3.0
- healpy >= 1.14.0
- treecorr >= 4.0.0
- pyccl >= 2.0.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- wigner >= 1.0.0

## Special Dependencies

### GLASS Installation

xilikelihood requires a custom version of GLASS. Install it first:

```bash
# Clone and install the custom GLASS version
git clone https://github.com/glass-dev/glass.git
cd glass
pip install -e .
```

### PyCC Installation

For PyCC, you may need additional system dependencies. On Ubuntu/Debian:

```bash
sudo apt-get install gsl-bin libgsl-dev
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

### From PyPI (when available)

```bash
pip install xilikelihood
```

## Verification

To verify your installation, run the test suite:

```bash
pytest tests/
```
