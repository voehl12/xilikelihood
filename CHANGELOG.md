# Changelog

## 2.0.0 - 2026-06-16

Paper-reference release for the published copula weak-lensing likelihood
analysis.

- Versioned as ``2.0.0`` to reflect API and workflow evolution since the
  first-paper code state.
- Archived release on Zenodo: https://doi.org/10.5281/zenodo.20717704.
- Keeps the ``v1.1.1`` metadata polish and Read the Docs fixes.
- Clarifies repository-reference maintenance guidance and cache reuse caveats.

## 1.1.1 - 2026-06-16

Post-release polish before Zenodo archival.

- Aligned README dependency wording with package metadata.
- Clarified that GLASS is the custom dependency for simulation-backed paths.
- Emphasized JAX-accelerated likelihood kernels and CPU/GPU backend support.
- Updated package, documentation, and citation metadata version strings.
- Archived release on Zenodo: https://doi.org/10.5281/zenodo.20714933.

## 1.1.0 - 2026-06-16

Paper-reference release preparation for the copula weak-lensing likelihood
analysis.

- Added paper-facing citation metadata and reproduction guidance.
- Documented the official finite copula likelihood smoke check.
- Recorded the compute-node smoke-check result:
  `tests/test_likelihood_gaussian_mode.py::test_copula_mode_does_not_require_fixed_covariance`
  passed on 2026-06-15.
- Clarified bundled data, generated artifacts, and external HPC path policy.
- Added repository and API reference documentation for future maintainers.
- Cleaned publication-facing paper-analysis script notes and stale batch paths.

Known limitations:

- Full paper reproduction remains HPC-oriented and depends on external/generated
  analysis products that are not stored in Git.
- Simulation-backed mock-data paths require the custom GLASS dependency.
- `include_ximinus=True` remains incomplete/deprecated and is not part of the
  validated paper-reference workflow.
