Repository Reference
====================

This document is a future-maintainer map of the ``xilikelihood`` repository as
of 2026-06-12. It focuses on where the main scientific logic lives, how the
major workflows connect, and which directories are source code versus cached
research artifacts.

Project Purpose
---------------

``xilikelihood`` computes likelihoods for two-point cosmic shear correlation
function data vectors. The central scientific idea is to use exact
characteristic-function marginals for low multipoles and a Gaussian treatment
for high multipoles or small-scale marginals, then assemble the joint
likelihood with a Gaussian copula. The package also contains simulation,
plotting, paper-analysis, and reproducibility scripts around that core.

The package name is ``xilikelihood``. Some older scripts and docs still say
``2ptlikelihood``; treat those as historical naming drift.

Top-Level Layout
----------------

``xilikelihood/``
    Main Python package. This is the code to modify for reusable behavior.

``tests/``
    Pytest test suite with small covariance fixtures, regression snapshots, and
    focused tests for transforms, masks, distributions, copulas, likelihood
    modes, mock data, simulation helpers, and sampler utility functions.

``docs/``
    Sphinx/MyST documentation source. Built output under ``docs/_build/`` is
    generated and should not be treated as source.

``simulation_scripts/``
    Cluster/job-array scripts for xi simulations. These are operational entry
    points around the package simulation API, not the canonical implementation.

``papers/``
    Paper-specific workflows, generated outputs, plots, SLURM scripts, and
    cached arrays. ``papers/second_paper_2025`` is the most active analysis
    tree.

``development/``
    Experimental methods, notebooks, prototypes, and validation sketches.
    Code here may be exploratory, incomplete, or path-dependent.

``legacy/``
    Deprecated and first-paper-era code preserved for scientific
    reproducibility. New work should start in ``xilikelihood/`` unless it is
    explicitly reproducing an old result.

``data/``, ``redshift_bins/``, ``masks/``, ``wpm_arrays/``, ``mllp_arrays/``,
``covariances/``, ``pcls/``, ``pdfs/``, ``cfs/``, ``s8posts/``, and similar
directories
    Scientific inputs, cached coupling matrices, simulation outputs, posterior
    products, and generated figures. Many are large or derived. Check whether a
    file is an input, cache, or final artifact before editing or deleting.

``likelihood-env/``, ``sim-env/``, ``2ptlikelihood/``
    Checked-in virtual environments. These are not package source.

Installation and Dependencies
-----------------------------

The package metadata is in ``pyproject.toml`` and requires Python 3.8 or newer.
Core dependencies include NumPy, SciPy, JAX/JAXLIB, Healpy, TreeCorr, pyccl,
Matplotlib, Seaborn, and ``wigner``. The simulation path imports
``glass.fields`` directly and expects a custom GLASS version described in the
README.

Development install:

.. code-block:: bash

   pip install -e .

Useful extras:

.. code-block:: bash

   pip install -e ".[dev]"
   pip install -e ".[docs]"

Tests force JAX onto CPU in ``tests/conftest.py`` to avoid GPU initialization
and memory surprises. The package import itself calls
``xilikelihood.core_utils.ensure_jax_device()``, which tries GPU first and
falls back to CPU by setting ``JAX_PLATFORM_NAME=cpu``. On clusters, scripts
often set JAX CPU environment variables explicitly before importing the
package.

Public API
----------

``xilikelihood/__init__.py`` exports the intended high-level interface:

* ``SphereMask`` for survey masks and mask-coupling cache preparation.
* ``RedshiftBin`` and ``TheoryCl`` for redshift distributions and theory
  angular power spectra.
* ``prepare_theory_cl_inputs`` and ``generate_theory_cl`` for tomographic
  theory setup.
* ``XiLikelihood`` and ``fiducial_dataspace`` for likelihood setup and
  evaluation.
* ``save_arrays``, ``load_arrays``, and ``pcl2xi`` as common utilities.
* Advanced submodules such as ``distributions``, ``wpm_funcs``,
  ``theoretical_moments``, ``pseudo_alm_cov``, and ``mock_data``.

The simulation interface is documented as ``xilikelihood.simulate`` rather than
as a top-level export.

Core Package Modules
--------------------

``core_utils.py``
    Shared configuration and runtime helpers. ``LikelihoodConfig`` controls CF
    grid sizes, PDF interpolation size, low/high-ell split behavior, covariance
    buffers, memory cleanup, eigenvalue diagnostics, and the large-angle
    threshold. ``temporary_arrays`` and ``computation_phase`` help with memory
    cleanup and logging.

``theory_cl.py``
    Redshift and theory power-spectrum logic. ``RedshiftBin`` loads
    distributions from files, explicit ``z,nz`` arrays, or Gaussian toy bins.
    ``load_kids_redshift_bins`` reads TOMO files from ``redshift_bins/KiDS``.
    ``TheoryCl`` loads spectra from text files, computes them with pyccl from a
    cosmology dict, optionally applies smoothing and pixel windows, and adds
    shape-noise spectra. ``BinCombinationMapper`` defines the triangular
    tomographic ordering used throughout the package. ``prepare_theory_cl_inputs``
    and ``generate_theory_cl`` are the standard tomographic setup helpers.

``mask_props.py``
    Survey mask handling. ``SphereMask`` can load a FITS mask or create a
    circular/full-sky HEALPix mask. It computes/caches ``wl``, ``wlm``, WPM
    arrays, and ``m_llp`` arrays. ``precompute_for_cov_masked`` is the key
    one-time expensive setup used by the likelihood. Cache filenames are
    generated through ``file_handling.generate_filename`` and usually land in
    ``wpm_arrays/`` and ``mllp_arrays/``.

``wpm_funcs.py`` and ``cov_funcs.py``
    Low-level Wigner and covariance tensor algebra. These are performance- and
    indexing-sensitive modules. ``wpm_funcs`` builds Wigner coupling arrays and
    mask mode-coupling matrices; ``cov_funcs`` contains the JAX/NumPy algebra
    for pseudo-alm covariance pieces.

``pseudo_alm_cov.py``
    ``Cov`` converts a ``SphereMask`` plus ``TheoryCl`` into pseudo-alm
    covariance matrices. ``cov_alm_xi`` is the xi-oriented path used by
    ``XiLikelihood``. Non-chain mode can cache covariances; chain mode skips
    file IO for repeated likelihood evaluations.

``cl2xi_transforms.py``
    Conversion between pseudo/true power spectra and xi-plus/xi-minus. Key
    functions are ``prep_prefactors``, ``pcl2xi``, ``pcls2xis``, ``cl2xi``,
    ``cl2pseudocl``, and Wigner integration/cache helpers. These kernels are
    shared by simulations, mean calculations, and plotting.

``distributions.py``
    Characteristic-function and PDF machinery. It contains JAX-jitted 1D CF
    generation, FFT-based CF-to-PDF conversion, Gaussian high-ell extensions,
    Gaussian xi covariance, xi mean calculations, exact 1D PDF helpers, and
    older moment/covariance utilities. New likelihood work should usually enter
    through ``XiLikelihood`` instead of calling this module directly.

``copula_funcs.py``
    Gaussian copula assembly and point evaluation. It validates marginal PDFs,
    interpolates PDFs/CDFs, converts covariance matrices to correlations,
    regularizes ill-conditioned correlation matrices, extracts data/covariance
    subsets, and evaluates joint log densities.

``likelihood.py``
    Main workflow. ``XiLikelihood`` validates the mask, redshift bins, angular
    bins, and configuration; prepares tomographic shapes; precomputes
    mask-dependent matrices; then evaluates cosmology-dependent likelihoods.
    ``fiducial_dataspace`` creates a KiDS-like redshift-bin and angular-bin
    setup.

``simulate.py``
    Simulation API. ``create_maps`` uses GLASS to generate Gaussian T/Q/U maps.
    ``compute_pseudo_cl`` uses Healpy to measure pseudo-Cls from masked maps.
    ``compute_correlation_functions`` converts pseudo-Cls to xi. The main
    ``simulate_correlation_functions`` wrapper supports ``pcl_estimator`` and
    ``treecorr`` methods and writes job-array-compatible ``jobN.npz`` and
    ``pcljobN.npz`` files.

``file_handling.py``
    Standardized array IO, filename generation, simulation readers, PCL-to-xi
    regeneration, posterior file discovery/loading, and covariance/pseudo-Cl
    save/load helpers. It centralizes directory conventions such as
    ``covariances/``, ``pcls/``, ``wpm_arrays/``, and ``mllp_arrays/``.

``mock_data.py``
    Creates fiducial mock data vectors and Gaussian covariance files for
    downstream posterior analyses. Also loads previously generated mock data
    and covariance products.

``noise_utils.py``
    KiDS-like shape-noise utilities for xi covariance, C_ell noise, per-pixel
    noise, noise cubes, and adding noise to theory spectra.

``data_statistics.py``, ``theoretical_moments.py``, ``diagnostic_tools.py``,
``plotting.py``
    Support modules for bootstrap statistics, moments/cumulants, eigenvalue and
    PDF diagnostics, and paper-style plotting/corner comparisons.

Main Likelihood Workflow
------------------------

The reusable workflow is:

1. Build or load a ``SphereMask`` with an explicit ``exact_lmax`` and optional
   smoothing.
2. Build redshift bins manually or via ``fiducial_dataspace``.
3. Instantiate ``XiLikelihood(mask, redshift_bins, ang_bins_in_deg, config=...)``.
4. Call ``setup_likelihood()`` once. This computes mask-specific arrays,
   prefactors, and covariance index bookkeeping.
5. For each cosmology, call ``loglikelihood(data, cosmology, ...)``.

Internally, each cosmology evaluation:

1. Calls ``generate_theory_cl`` for the tomographic bin combinations.
2. Builds pseudo-alm covariance products through ``Cov``.
3. Computes low-ell means and covariance in xi-space.
4. Optionally adds high-ell Gaussian means/covariance.
5. Builds exact large-angle marginal PDFs from characteristic functions and
   Gaussian small-angle marginals.
6. Converts marginals to CDFs and evaluates the Gaussian copula joint log PDF.

``loglikelihood`` supports ``likelihood_type="copula"`` by default,
``"gaussian"`` for a Gaussian-only diagnostic path, and ``"both"`` only with
``allow_diagnostic=True``. Gaussian modes require ``self.gaussian_covariance``
to be set, typically from ``mock_data.create_mock_data`` or a saved covariance
file.

Data Shapes and Indexing
------------------------

Tomographic redshift-bin combinations use triangular order from
``BinCombinationMapper``: for ``n`` bins there are ``n*(n+1)/2`` combinations,
with auto and cross terms in GLASS-like order. Data arrays are generally shaped
as:

.. code-block:: text

   (n_redshift_bin_combinations, n_correlation_types * n_angular_bins)

With the default ``include_ximinus=False``, ``n_correlation_types`` is 1 and the
columns are xi-plus angular bins. With ``include_ximinus=True``, columns are
xi-plus bins followed by xi-minus bins. The xi-minus path is explicitly marked
deprecated/incomplete in the code because the covariance implementation is not
fully correct for all B/EB terms.

Subsets are lists of ``(redshift_combination_index, angular_index)`` pairs.
Use ``copula_funcs.data_subset`` and ``copula_funcs.cov_subset`` to preserve the
repository's flattening convention.

Simulation Workflow
-------------------

The package-level simulation implementation is in ``xilikelihood/simulate.py``.
``simulation_scripts/xi_sim.py`` and ``xi_sim_nD.py`` are cluster-facing
wrappers. The current README for that directory still contains some historical
naming and scratch-path assumptions; update those scripts carefully if moving
to a different environment.

The pseudo-Cl simulation path:

1. Extracts E-mode spectra from ``TheoryCl`` objects.
2. Generates Gaussian maps with custom GLASS.
3. Adds shape noise to Q/U maps if requested.
4. Masks maps and measures pseudo-Cls with Healpy ``anafast``.
5. Converts pseudo-Cls to xi-plus/xi-minus with ``prep_prefactors`` and
   ``pcl2xi``.
6. Saves ``job{job_id}.npz`` and optionally ``pcljob{job_id}.npz``.

Paper and Analysis Workflows
----------------------------

``papers/second_paper_2025`` is a full analysis workspace. Important source
files:

``analysis/config.py``
    Central constants for exact lmax, fiducial cosmology, masks, angular bins,
    S8/omega_m grids, file naming, and priors. It contains cluster-specific
    absolute paths such as ``PACKAGE_DIR``.

``analysis/mock_data_generation.py``
    Paper-facing wrapper for creating mock data and Gaussian covariance files.

``analysis/s8_om_posterior.py``
    2D grid posterior job-array entry point. It splits an omega_m/S8 grid over
    ``N_JOBS_2D`` jobs, creates fiducial mock data in a temporary directory,
    runs both copula and Gaussian likelihoods, and writes structured
    ``posterior_{job}.npy`` files to scratch.

``analysis/s8_posteriors.py``
    1D and subset posterior helper functions for S8 scans and earlier
    workflows.

``analysis/sampler.py`` and ``analysis/sampler_nested.py``
    MCMC/nested sampling entry points with MPI/logging/checkpoint support.
    ``sampler.py`` currently defaults to a Gaussian likelihood mode and writes
    into ``sampler_output_gaussian`` plus scratch log directories.

``analysis/postprocess_*.py`` and ``analysis/postprocess_utils.py``
    Chain/posterior inspection, stuck-step removal, trace/scatter/corner plots,
    2D posterior assembly, and comparison plotting.

``analysis/plot_sims.py``
    Rebuilds a small likelihood for selected redshift/angular subsets,
    optionally clears cached 2D likelihood files, computes or loads pairwise
    likelihood grids, and calls ``xilikelihood.plotting.plot_corner`` against
    simulation products.

``analysis/scale_cut_sweep.py``
    Runs and plots scale-cut sweeps for posterior sensitivity studies.

Most files under ``papers/second_paper_2025/data``, ``slurm`` output folders,
``analysis/*png/pdf/npz/h5/log/out/err``, and similar paths are generated or
environment-specific artifacts. They are useful for reproducibility, but should
not be mistaken for package logic.

Tests
-----

Run the full suite with:

.. code-block:: bash

   pytest

The pytest configuration lives in ``pyproject.toml``. Tests are verbose by
default and use ``pytest-regtest`` snapshot files in ``tests/_regtest_outputs``.
Important coverage areas:

* ``test_theory_cl.py``: redshift bins, bin-combination mapping, noise setup,
  and theory generation.
* ``test_mask_props.py`` and ``test_wpm_funcs.py``: mask coupling and Wigner
  helpers.
* ``test_cl2xi_transforms.py``: prefactor shapes and pseudo-Cl transforms.
* ``test_pseudo_alm_cov.py`` and ``test_distributions.py``: exact covariance
  and Gaussian xi covariance snapshots/properties.
* ``test_copula_funcs.py``: covariance/correlation conversion, copula density,
  interpolation, subsetting, and edge cases.
* ``test_likelihood.py``: setup and xi-minus shape/mean/covariance behavior.
* ``test_likelihood_gaussian_mode.py``: newer Gaussian-only and diagnostic
  likelihood dispatch.
* ``test_mock_data.py`` and ``test_simulate.py``: mock-data creation and
  simulation utilities.
* ``test_sampler_basic.py``: imports the second-paper sampler and tests prior,
  likelihood composition, and plotting helpers with monkeypatches.

Be aware that tests may create or reuse masks and cached arrays under the
repository root. The fixture uses ``exact_lmax=10`` and ``nside=256`` for speed.

Documentation
-------------

Sphinx configuration is in ``docs/conf.py``. Existing docs are Markdown/MyST
files. Build with:

.. code-block:: bash

   cd docs
   make html

The docs include API stubs and orientation pages for the package modules,
installation, quickstart, configuration, and examples. The quickstart is
likelihood-first; simulation examples import the simulation function from
``xilikelihood.simulate``.

Common Gotchas
--------------

* Importing ``xilikelihood`` can alter JAX backend state or restart a script if
  GPU initialization fails. In scripts/tests that must stay CPU-only, set JAX
  environment variables before importing.
* ``include_ximinus=True`` is not production-ready; code comments warn that the
  xi-minus covariance treatment is incomplete.
* The package mixes source code with many cached/generated artifacts. Use file
  extensions and directory context to decide what is authoritative.
* Many paper scripts contain absolute ETH/cluster paths and scratch output
  paths. Porting them requires path cleanup.
* ``.gitignore`` ignores many generated formats, including ``*.md``, ``*.npz``,
  ``*.npy``, plots, logs, FITS files, and virtualenvs. Existing tracked files
  still remain tracked.
* The high-dimensional ``likelihood_function`` method is deprecated and guarded
  against dimensions greater than four. Prefer point evaluation with
  ``loglikelihood`` or 2D visualization with ``likelihood_function_2d`` after
  preparation.
* ``LikelihoodConfig.large_angle_threshold`` is in degrees. The default is
  ``15/60`` degrees, so exact marginals are used for angular bins whose lower
  edge is at least 15 arcmin.
* Cache filenames encode key parameters such as ``lmax``, ``nside``, mask name,
  theory name, and noise state where relevant. They do not encode every
  possible analysis choice, so a run can intentionally or accidentally reuse an
  existing cache. If a result looks stale, check that the encoded parameters and
  cache directory match the intended analysis before deleting or regenerating
  files.

Maintenance Guidance
--------------------

For reusable behavior, start in ``xilikelihood/`` and add focused tests in
``tests/``. For paper reproduction, change files under ``papers/.../analysis``
and avoid pushing hard-coded paths deeper into the package. For performance
changes, treat ``wpm_funcs.py``, ``cov_funcs.py``, ``pseudo_alm_cov.py``, and
``likelihood.py`` as a coupled system: small indexing changes can affect
regression snapshots and scientific outputs.

When adding a new likelihood feature:

1. Decide whether it changes reusable package behavior or only a paper script.
2. Add or adjust tests for shapes, covariance symmetry/PSD behavior, and one
   small numeric regression.
3. Keep generated outputs out of source paths unless they are intentional
   fixtures.
4. Document any new file naming or cache convention in ``file_handling.py`` or
   this reference.
