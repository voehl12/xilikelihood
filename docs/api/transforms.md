# Transform Module

This module contains the lower-level numerical transforms between angular power
spectra and two-point correlation functions.

Most likelihood users do not need to call these functions directly. They are
used internally by `XiLikelihood`, by simulation postprocessing, and by
diagnostic plotting.

Important functions include:

- {func}`xilikelihood.cl2xi_transforms.prep_prefactors` - precompute binned
  angular kernels.
- {func}`xilikelihood.cl2xi_transforms.pcl2xi` and
  {func}`xilikelihood.cl2xi_transforms.pcls2xis` - convert pseudo-Cls into
  xi-plus/xi-minus values.
- {func}`xilikelihood.cl2xi_transforms.cl2pseudocl` - apply mask mode coupling
  to theory spectra.

These routines are sensitive to multipole limits, angular-bin definitions, and
array ordering. Prefer the high-level likelihood/simulation APIs unless you are
working on transform validation or cached simulation products.

```{eval-rst}
.. automodule:: xilikelihood.cl2xi_transforms
   :members:
   :undoc-members:
   :show-inheritance:
```
