# TODO.md or REFACTORING_NOTES.md

## S8 Analysis Refactoring Plan

### Current Status
- `s8_example.py` contains analysis code for paper 2 (copula likelihood S8 posteriors)
- Mixed configuration, data generation, and analysis functions
- Hardcoded paths and parameters

### Proposed Structure
```
papers/paper2_copula_likelihood/
├── analysis/
│   ├── s8_posteriors.py          # Main analysis functions  
│   ├── mock_data_generation.py   # Data creation utilities
│   ├── config.py                 # Configuration parameters
│   └── posterior_analysis.py     # Common utilities
├── data/
│   ├── mock_data_*.npz
│   └── s8posts/                  # Output directory
├── slurm/
│   └── posterior_job.sh          # Job submission
└── README.md
```

### Key Improvements
1. **Separate concerns**: configuration, data generation, analysis
2. **Extract hardcoded values** to config.py
4. **Add proper imports** and path handling
5. **Create reusable utilities** for common patterns
6. **Add documentation** and README

### Functions to Extract
- `create_mock_data()` → `mock_data_generation.py`  
- `setup_likelihood()` → shared utility
- Configuration constants → `config.py`
- S8 grid logic → configurable parameters

### Implementation Status
- [x] Move to papers folder
- [x] Extract configuration  
- [x] Separate data generation
- [x] Clean up analysis functions
- [x] Add documentation
- [x] Create job submission scripts

### REMAINING TASKS
- [x] **Update SLURM scripts** - Still reference old `s8_example.py` path
- [x] **Integrate config.py** - `s8_posteriors.py` has duplicate config 
- [x] **Confirm xi-minus support** - Verified working through existing architecture
- [x] **Scale-dependent marginals** - Implement scale cuts for Gaussian marginals
- [ ] Save all setup/configuration parameters (mask, angular bins, redshift bins, prior ranges, etc.) to a YAML or JSON file at the start of every run.
    - Include: mask parameters, angular bin edges, redshift bin edges, prior ranges, and any other relevant config.
    - Save to a file named e.g. `run_setup_{jobnumber}.yaml` or `run_setup_{jobnumber}.json` in the output/log directory.
    - Log the path to this file in the main log.
- [ ] Add a function to load and print this config for any run, for full reproducibility.
- [ ] Optionally: Save a hash or timestamp of the main code files used for the run.
- [ ] Document this workflow in the README for future users. This will ensure every run is fully reproducible and the setup is always recorded, even if the script changes later.
- [x] **Implement MCMC sampler** - Add an MCMC-based sampling framework in addition to the current nested sampling approach for posterior inference.
- [ ] **Covariance treatment for copula likelihood**
    - Decide whether to keep the covariance matrix fixed (as in standard Gaussian likelihood) or allow it to vary with cosmology (as required for an exact likelihood).
    - Consider the implications for both pure copula and mixed/Gaussian marginal approaches.
    - Document the pros/cons and the impact on inference and computational cost.
    - Implement a switch or clear logic for this choice in the analysis pipeline.
- [ ] **Handle NaN likelihoods due to non-positive definite covariance**
    - Investigate robust strategies for production: 
        - Should we always regularize (set negative eigenvalues to zero)?
        - Should we reject/flag cosmologies that yield non-positive definite covariances?
        - Should we add a small diagonal noise term ("nugget")?
    - Document the pros/cons and user guidance for each approach.
    - Implement a clear, user-configurable policy for handling this in the likelihood code.
- [ ] **Document and handle intermediate (pseudo-alm) covariance regularization**
    - The covariances in question are not the final data vector covariances, but intermediate pseudo-alm (sub-)covariances.
    - Non-positive definiteness at this stage can propagate and cause issues in the final likelihood, but regularizing every intermediate matrix is expensive.
    - Investigate:
        - Whether to regularize/check only at the final, combined covariance level, or also at intermediate steps.
        - Efficient diagnostic strategies (e.g., check eigenvalues only in debug mode, or for a subset of runs).
        - How regularization at the intermediate level affects the final result.
    - Clearly document this subtlety and the chosen strategy in code and user docs.
- [ ] Add a CLI interface to sampler.py for sampler selection (nested/MCMC), scale cut parameterization, and output directory specification.
    - Use argparse or click for robust argument parsing.
    - Ensure all CLI options are logged and saved in the run configuration file.
    - Update documentation to reflect CLI usage and options.
- [ ] Implement systematic posterior shape comparison as a function of scale cut (run samplers for several scale cuts and compare posteriors).
    - Automate running sampler.py for a range of scale cuts.
    - Save and plot posterior summaries for each scale cut.
    - Document findings and add plotting utilities for comparison.
- [ ] Implement Student-t copula version of the systematic scale cut/posterior study.
    - Add option to sampler.py and config to select Student-t copula.
    - Ensure robust handling of edge cases and document limitations.
    - Compare results to Gaussian copula and document differences.
- [ ] **Robust covariance regularization and stabilization strategies**
    - Test regularizing the full covariance matrix by adding a relative nugget: `nugget = epsilon * mean(diag(cov))`, with epsilon ~1e-6 to 1e-3, instead of a fixed value.
    - Compare the effect of different epsilon values and document the impact on likelihood stability and inference.
    - Consider and test other stabilization methods (e.g., adaptive eigenvalue thresholding, dynamic regularization based on condition number, or fallback to nearest positive definite matrix).
    - Log the applied nugget and matrix conditioning for each run for diagnostics.
    - pyccl: spline integration?
    - Summarize best practices and user guidance in the documentation.
- [ ] Add trace plot and autocorrelation diagnostics for all MCMC chains
    - Implement a routine to plot trace plots for each parameter and scale cut, saving to the output directory.
    - Compute and log the integrated autocorrelation time for each parameter (e.g., using emcee or ArviZ utilities).
    - Add warnings or flags if autocorrelation times are large compared to chain length (indicating poor mixing).
    - Document recommended thresholds and interpretation in the README.
- [ ] Run and compare MCMC chains with a fixed covariance matrix
    - Implement an option in the sampler and config to use a fixed (fiducial) covariance for all likelihood evaluations.
    - Run the full scale cut sweep with fixed covariance and compare posteriors to the varying-covariance case.
    - Document the impact on posterior shape, computation time, and stability.
    - Add plotting utilities to overlay fixed vs. varying covariance results.

## ✅ REFACTORING COMPLETE!

All major tasks have been completed successfully:
- ✅ Organized structure in papers/second_paper_2025/
- ✅ Extracted configuration to config.py with centralized parameters
- ✅ Separated data generation utilities  
- ✅ Clean analysis functions with proper imports
- ✅ Comprehensive documentation and README
- ✅ Updated job submission scripts
- ✅ Integrated config.py across all analysis files
- ✅ Confirmed xi-minus support working correctly

The S8 analysis pipeline is now well-organized and maintainable!

## 🚀 NEXT: Scale-Dependent Marginals

Implement scale cuts for Gaussian marginals to enable scale-dependent analysis.