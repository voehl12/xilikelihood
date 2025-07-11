# Development Notebooks

This directory contains Jupyter notebooks used for development, exploration, 
and parameter tuning during package development.

## Notebooks

- `mask_apo.ipynb` - Mask apodization parameter exploration
  - Tests different smoothing kernels and parameters
  - Compares HEALPix smoothing methods
  - Investigates convergence of spherical harmonic sums
  - Used to determine optimal `l_smooth` parameters for mask processing

- `gaussian_cf.ipynb` - Gaussian characteristic function analysis
  - Explores properties of Gaussian characteristic functions
  - Tests numerical accuracy of CF to PDF transformations
  - Investigates convergence and stability of moment calculations
  - Development work for the distributions module and copula methods



## Usage

```bash
# Activate environment
source ../../venv/bin/activate

# Start Jupyter
jupyter notebook

# Or for JupyterLab
jupyter lab
