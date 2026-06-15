# Core API

The core API is the small set of objects that most users need to set up and
evaluate likelihoods. Lower-level transform, simulation, and file-handling
utilities are documented on their own pages.

```{eval-rst}
.. automodule:: xilikelihood
   :members:
   :undoc-members:
   :show-inheritance:
```

## Main Classes

```{eval-rst}
.. autoclass:: xilikelihood.XiLikelihood
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xilikelihood.SphereMask
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xilikelihood.RedshiftBin
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xilikelihood.TheoryCl
   :members:
   :undoc-members:
   :show-inheritance:
```

## Core Functions

These functions are part of the high-level setup path. Simulation and transform
functions are documented separately because they are optional or lower-level.

```{eval-rst}
.. autofunction:: xilikelihood.fiducial_dataspace

.. autofunction:: xilikelihood.prepare_theory_cl_inputs

.. autofunction:: xilikelihood.generate_theory_cl
```

## Basic File Helpers

```{eval-rst}
.. autofunction:: xilikelihood.save_arrays

.. autofunction:: xilikelihood.load_arrays
```
