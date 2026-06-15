# Simulation Module

The simulation module generates correlation-function realizations from theory
power spectra. It is useful for validation and paper comparisons, but it is not
required for deterministic likelihood evaluation.

The main entry point is
{func}`xilikelihood.simulate.simulate_correlation_functions`. The pseudo-Cl
estimator path uses GLASS to generate maps, Healpy to measure pseudo power
spectra, and the same `cl2xi` transform kernels used elsewhere in the package.

## Dependency Note

Map-backed simulations require the custom GLASS version used for the paper
analysis. Deterministic mock data from
{func}`xilikelihood.mock_data.create_mock_data` with `random=None` does not
require GLASS simulations.

## Output Format

Simulation jobs write `jobN.npz` files containing xi-plus/xi-minus data and,
when requested, `pcljobN.npz` files containing pseudo power spectra. The readers
in {mod}`xilikelihood.file_handling` understand this job-array format.

```{eval-rst}
.. automodule:: xilikelihood.simulate
   :members:
   :undoc-members:
   :show-inheritance:
```
