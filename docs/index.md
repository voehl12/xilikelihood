# xilikelihood Documentation

Welcome to the documentation for **xilikelihood**, a Python package for computing two-point correlation function likelihoods on masked spin-2 Gaussian random fields. The primary and currently implemented application are cosmic shear correlation function datavectors.

```{toctree}
:maxdepth: 2
:caption: Contents:

installation
quickstart
api/index
examples/index
```

## About xilikelihood

xilikelihood provides tools for:

- Setting up likelihoods for correlation function datavectors
- Evaluating likelihoods with customizable levels of approximation to the exact shape
- Parallel evaluation of the corresponding Gaussian likelihood
- Handling any survey geometries as well as basic shot noise
- Using an MCMC sampler to perform cosmological parameter estimation from a given data vector

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
