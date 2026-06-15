# API Reference

This section documents the main public modules in `xilikelihood`. The package
contains both reusable likelihood machinery and paper-analysis support code; for
new analyses, start with the likelihood workflow below and use the lower-level
transform/simulation utilities only when needed.

```{toctree}
:maxdepth: 2

core
simulation
likelihood
transforms
utilities
```

## Recommended Entry Points

For most likelihood analyses, the central objects are:

- {class}`xilikelihood.XiLikelihood` - sets up and evaluates the xi likelihood.
- {class}`xilikelihood.SphereMask` - creates or loads a survey mask and prepares
  mask-coupling arrays.
- {class}`xilikelihood.RedshiftBin` - stores one tomographic redshift
  distribution.
- {class}`xilikelihood.TheoryCl` - stores or computes the theory angular power
  spectra for a redshift-bin pair.

The usual high-level workflow is:

1. Create a {class}`xilikelihood.SphereMask` with a chosen `exact_lmax`.
2. Build redshift bins, either manually or with
   {func}`xilikelihood.fiducial_dataspace`.
3. Create {class}`xilikelihood.XiLikelihood`.
4. Run {meth}`xilikelihood.XiLikelihood.setup_likelihood` once.
5. Evaluate {meth}`xilikelihood.XiLikelihood.loglikelihood` for data vectors and
   cosmologies.

## Common Supporting Functions

- {func}`xilikelihood.fiducial_dataspace` - KiDS-like redshift/angular setup.
- {func}`xilikelihood.prepare_theory_cl_inputs` - tomographic bin-combination
  bookkeeping for theory spectra.
- {func}`xilikelihood.generate_theory_cl` - build `TheoryCl` objects from a
  cosmology dictionary.
- {func}`xilikelihood.mock_data.create_mock_data` - deterministic or Gaussian
  mock data and Gaussian comparison covariances.

## Lower-Level Utilities

- {func}`xilikelihood.simulate.simulate_correlation_functions` is the main
  simulation interface. It is optional and requires the GLASS simulation
  dependency for map-backed simulations.
- {func}`xilikelihood.cl2xi_transforms.pcl2xi` and related transform functions
  convert pseudo power spectra into xi values. These are important internally
  and for simulation/postprocessing work, but they are not the main likelihood
  API.
- File handling, noise, and plotting helpers support paper workflows and cached
  products; see the module pages for details.
