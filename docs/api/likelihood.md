# Likelihood Module

The likelihood module contains the main analysis object,
{class}`xilikelihood.likelihood.XiLikelihood`. This is the entry point for
copula likelihood evaluation and Gaussian comparison likelihoods.

## Main Workflow

1. Instantiate a `SphereMask`.
2. Define redshift bins and angular bins.
3. Create `XiLikelihood(mask, redshift_bins, ang_bins_in_deg, config=...)`.
4. Call `setup_likelihood()` once to prepare mask-dependent arrays and
   projection kernels.
5. Call `loglikelihood(data, cosmology)` for each data vector/cosmology pair.

The default likelihood type is `"copula"`, which evaluates the exact-marginal
Gaussian-copula approximation used by the paper. The `"gaussian"` and `"both"`
modes are diagnostic/comparison paths and require a fixed Gaussian covariance.

## Scale Handling

`LikelihoodConfig.large_angle_threshold` separates angular bins whose marginals
are evaluated with the exact characteristic-function machinery from bins that
use Gaussian marginals. The default threshold is `15/60` degrees.

## xi-minus Status

`include_ximinus=True` exists for development and shape checks, but the code
marks the xi-minus covariance implementation as incomplete. Published analyses
should treat xi-plus as the validated path unless the xi-minus implementation is
revisited.

```{eval-rst}
.. automodule:: xilikelihood.likelihood
   :members:
   :undoc-members:
   :show-inheritance:
```
