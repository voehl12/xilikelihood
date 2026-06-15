# Utilities

This page groups support modules used by the likelihood and paper workflows.
They are useful when inspecting cached products or building reproducibility
scripts, but most analyses can start from `XiLikelihood` and only reach for
these modules as needed.

## File Handling

`xilikelihood.file_handling` standardizes array IO, cache filenames, simulation
readers, and posterior file discovery.

## Mock Data

`xilikelihood.mock_data` creates deterministic mock data, Gaussian mock draws,
and Gaussian comparison covariances for likelihood tests and examples.

## Configuration and Runtime Helpers

`xilikelihood.core_utils` contains `LikelihoodConfig`, JAX backend handling, and
small context managers for memory/logging.

## Masks and Noise

`xilikelihood.mask_props` provides `SphereMask`, and
`xilikelihood.noise_utils` contains KiDS-like shape-noise utilities.

```{eval-rst}
.. automodule:: xilikelihood.file_handling
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xilikelihood.mock_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xilikelihood.core_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xilikelihood.mask_props
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xilikelihood.noise_utils
   :members:
   :undoc-members:
   :show-inheritance:
```
