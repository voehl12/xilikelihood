# Reproducing the Paper Results

This repository is the reference implementation for the copula likelihood
analysis described in:

- New paper: https://arxiv.org/abs/2604.07336
- First paper / earlier method: https://arxiv.org/abs/2407.08718

The paper-reference release is archived on Zenodo:
https://doi.org/10.5281/zenodo.20714932.

## Scope

The GitHub release is intended to preserve the source code, tests, examples,
and paper-analysis scripts needed to understand and rerun the likelihood
workflow. It is not intended to contain every generated simulation, posterior,
plot, SLURM log, or cached covariance product from the paper analysis.

The reproducibility target is therefore split into three levels:

1. **Code verification**: install the package and run focused tests.
2. **Likelihood smoke check**: build a small likelihood, generate deterministic
   mock data, and evaluate a log-likelihood.
3. **Full paper analysis**: rerun the paper scripts on an HPC environment with
   the required external/generated data products.

## Environment

Use Python 3.11 if possible; the current development environment used
``likelihood-env`` with Python 3.11.6.

Install the package in editable mode:

```bash
pip install -e .
```

For tests:

```bash
pip install -e ".[dev]"
```

The project depends on scientific Python packages including NumPy, SciPy, JAX,
Healpy, TreeCorr, pyccl, and wigner. The simulation-backed paths additionally
require a custom GLASS version. Deterministic likelihood and mock-data paths do
not require GLASS simulations.

On CPU-only systems or shared clusters, set JAX to CPU before importing the
package:

```bash
export JAX_PLATFORM_NAME=cpu
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""
```

## Official Smoke Check

The official smoke check for the archived release is a finite copula
log-likelihood evaluation:

```bash
likelihood-env/bin/python -m pytest tests/test_likelihood_gaussian_mode.py::test_copula_mode_does_not_require_fixed_covariance -q
```

This target verifies the core likelihood path used by the paper in a small
configuration: likelihood setup, deterministic mock-data generation, exact
marginal construction, copula evaluation, and a finite
``XiLikelihood.loglikelihood(...)`` result without requiring the fixed Gaussian
comparison covariance.

Current result on this branch, run on a Slurm compute node
(``eu-a2p-341``) on 2026-06-15:

```text
1 passed in 14.12s
```

## Broader Verification

The focused mock-data tests are a useful supporting check for deterministic
mock data, Gaussian covariance generation, save/load behavior, and finite
Gaussian mock-data draws:

```bash
likelihood-env/bin/python -m pytest tests/test_mock_data.py -q
```

Current supporting-check result on this branch:

```text
2 passed in 79.11s
```

Run the full test suite when the full dependency stack is available:

```bash
likelihood-env/bin/python -m pytest
```

If using another environment, replace ``likelihood-env/bin/python`` with the
Python executable for that environment.

## Smoke Check Workflow

The test target above is the recommended command to run. The following snippet
shows the same style of workflow in a readable form and exercises the main
likelihood path without map simulations:

```python
import numpy as np
import xilikelihood as xlh
from xilikelihood.core_utils import LikelihoodConfig
from xilikelihood.mock_data import create_mock_data

mask = xlh.SphereMask(
    spins=[2],
    circmaskattr=(1000, 256),
    exact_lmax=10,
    l_smooth=30,
)

z = np.linspace(0.01, 3.0, 100)
redshift_bins = [
    xlh.RedshiftBin(nbin=1, z=z, zmean=0.5, zsig=0.1),
    xlh.RedshiftBin(nbin=2, z=z, zmean=1.0, zsig=0.1),
]
angular_bins_in_deg = [(1.0, 2.0), (2.0, 4.0)]

likelihood = xlh.XiLikelihood(
    mask=mask,
    redshift_bins=redshift_bins,
    ang_bins_in_deg=angular_bins_in_deg,
    config=LikelihoodConfig(cf_steps=1024, pdf_steps=1024),
)
likelihood.setup_likelihood()

mock_data, gaussian_covariance = create_mock_data(
    likelihood,
    mock_data_path="mock_data_smoke.npz",
    gaussian_covariance_path="gaussian_covariance_smoke.npz",
    fiducial_cosmo={"omega_m": 0.31, "s8": 0.8},
    random=None,
)

likelihood.gaussian_covariance = gaussian_covariance
log_likelihood = likelihood.loglikelihood(
    mock_data,
    {"omega_m": 0.30, "s8": 0.82},
)
print(log_likelihood)
```

The output value is not a paper result. The archived-release smoke check is the
pytest command above; this snippet is included to make the workflow easy to
inspect and adapt.

## Paper Analysis Entry Points

The second-paper analysis workspace is:

```text
papers/second_paper_2025/
```

Important files:

- ``papers/second_paper_2025/analysis/config.py``: central constants for masks,
  fiducial cosmology, priors, grid sizes, file names, and cluster paths.
- ``papers/second_paper_2025/analysis/s8_om_posterior.py``: 2D
  ``omega_m``/``S8`` grid posterior jobs.
- ``papers/second_paper_2025/analysis/sampler.py``: MCMC workflow with
  checkpointing and MPI-aware logging.
- ``papers/second_paper_2025/analysis/postprocess_*.py``: posterior and chain
  postprocessing.
- ``papers/second_paper_2025/analysis/plot_sims.py``: comparison plots between
  likelihood contours and simulation products.
- ``papers/second_paper_2025/slurm/``: SLURM job scripts used for cluster runs.

Several of these scripts contain absolute ETH/cluster paths and are intended to
document the production workflow rather than run unchanged on a laptop.

## Data and Generated Artifacts

The GitHub repository should contain source code, tests, documentation, small
test fixtures, and paper scripts. Large or generated products are not expected
to be tracked in GitHub.

### Bundled Inputs and Fixtures

These files are small enough, or important enough for tests/docs, to keep in the
repository:

- Source code in ``xilikelihood/``.
- Tests and regression fixtures in ``tests/``.
- Documentation in ``docs/``.
- Paper-analysis scripts in ``papers/second_paper_2025/analysis/`` and related
  SLURM submission scripts.
- Redshift-bin inputs under ``redshift_bins/`` and ``data/`` when present in
  the GitHub release.
- Small test masks, test PCL products, and other fixtures under ``tests/``.

### Regenerated Caches and Intermediate Products

The following products are generated by setup, likelihood, simulation, or paper
scripts. They are useful for reruns, but are not part of the minimal source
archive:

- Wigner/mask coupling caches in ``wpm_arrays/`` and ``mllp_arrays/``.
- Covariance products in ``covariances/``.
- Pseudo-Cl and xi simulation products in ``pcls/``, ``cfs/``, and
  simulation output directories.
- Mock-data and Gaussian-covariance files produced by
  ``xilikelihood.mock_data.create_mock_data``.

These products can be regenerated from the scripts when the required external
dependencies, compute resources, and input paths are available.

### Paper Outputs and HPC Artifacts

Full paper runs produce products that are intentionally treated as analysis
artifacts rather than source code:

- Posterior grids, chains, corner plots, trace plots, figures, logs, and
  profiler outputs under ``papers/second_paper_2025/`` and scratch paths.
- SLURM ``.out``/``.err`` files and per-rank sampler logs.
- Large ``.npz``, ``.npy``, ``.h5``, ``.fits``, ``.png``, ``.pdf``, and
  profiler files generated during exploratory or production runs.

If a specific final artifact is needed for review or long-term preservation, it
should be attached to the archived release or deposited separately, with the DOI
or download location recorded here.

### External Paths

Some production scripts preserve absolute ETH/cluster paths to document the
actual paper workflow. In particular, second-paper scripts may refer to
``/cluster/home``, ``/cluster/scratch``, or ``/cluster/work`` locations. These
paths are not portable inputs; they identify where the original runs placed
intermediate and output products.

## Optional Simulation Dependency

Simulation-backed mock data and map simulations require the custom GLASS
dependency used by the analysis. The deterministic mock-data path
``create_mock_data(..., random=None)`` uses the fiducial theory mean and does
not require GLASS simulations. The map-backed path
``create_mock_data(..., random="frommap")`` does require GLASS.

## Known Limitations

- ``include_ximinus=True`` is marked as incomplete/deprecated in the code
  because the xi-minus covariance implementation is not fully validated.
- Full paper reproduction is HPC-oriented and may require substantial memory
  and runtime.
- Some paper scripts preserve absolute cluster paths for reproducibility of the
  original workflow.

Current archived release:

- Git tag: ``v2.0.0``
- Zenodo DOI: https://doi.org/10.5281/zenodo.20714932
