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

// ...existing content...

## Tail Dependence Analysis for Copula Selection

### **Motivation**
Current analysis uses theoretical covariance matrix (same for both Gaussian and Student-t copulas) with fiducial mean data. This doesn't test which copula better captures realistic correlation function dependencies. Need empirical validation using map-based simulations.

### **Implementation Plan**

#### **1. Generate Realistic Correlation Function Ensemble**
```python
def generate_realistic_correlation_data(n_realizations=1000, cosmology=None):
    """
    Generate realistic correlation function measurements from map simulations.
    Captures: shot noise, shape noise, mask effects, non-Gaussian features
    """
    correlation_realizations = []
    
    for i in range(n_realizations):
        # Generate realistic maps with all observational effects
        maps = xlh.generate_realistic_maps(
            cosmology=cosmology or FIDUCIAL_COSMO,
            include_shape_noise=True,
            include_shot_noise=True,
            apply_survey_mask=True,
            add_systematics=True,
            realization_seed=i
        )
        
        # Measure correlation functions from maps
        xi_plus, xi_minus = xlh.measure_correlation_functions(maps, ...)
        correlation_data = flatten_correlation_measurements(xi_plus, xi_minus)
        correlation_realizations.append(correlation_data)
    
    return np.array(correlation_realizations)  # Shape: (n_realizations, n_correlations)
```

#### **2. Transform to Uniform Margins (Isolate Copula Structure)**
```python
def transform_to_uniform_margins(correlation_data):
    """
    Transform each correlation function type to uniform [0,1] margins.
    This isolates the pure dependence structure (copula) from marginal distributions.
    """
    n_realizations, n_correlations = correlation_data.shape
    uniform_data = np.zeros_like(correlation_data)
    
    for i in range(n_correlations):
        # Empirical CDF transformation: ranks → uniform [0,1]
        values = correlation_data[:, i]
        ranks = stats.rankdata(values, method='average')
        uniform_data[:, i] = ranks / (n_realizations + 1)
    
    return uniform_data
```

#### **3. Measure Empirical Tail Dependence**
```python
def measure_tail_dependence(uniform_data, quantile_thresholds=[0.90, 0.95, 0.99]):
    """
    Measure tail dependence coefficients between correlation function pairs.
    
    Key concepts:
    - Upper tail dependence λ_U = lim_{t→1⁻} P(U_j > t | U_i > t)
    - Lower tail dependence λ_L = lim_{t→0⁺} P(U_j ≤ t | U_i ≤ t)
    
    Copula signatures:
    - Gaussian: λ_U = λ_L = 0 (no tail dependence)
    - Student-t: λ_U = λ_L > 0 (symmetric tail dependence)
    """
    n_realizations, n_correlations = uniform_data.shape
    results = {}
    
    for threshold in quantile_thresholds:
        pair_results = {}
        
        for i in range(n_correlations):
            for j in range(i+1, n_correlations):
                u_i = uniform_data[:, i]
                u_j = uniform_data[:, j]
                
                # Upper tail dependence
                upper_mask_i = u_i > threshold
                upper_mask_j = u_j > threshold
                upper_mask_both = upper_mask_i & upper_mask_j
                
                n_upper_i = np.sum(upper_mask_i)
                n_upper_both = np.sum(upper_mask_both)
                lambda_upper = n_upper_both / n_upper_i if n_upper_i > 10 else np.nan
                
                # Lower tail dependence  
                lower_threshold = 1 - threshold
                lower_mask_i = u_i <= lower_threshold
                lower_mask_j = u_j <= lower_threshold
                lower_mask_both = lower_mask_i & lower_mask_j
                
                n_lower_i = np.sum(lower_mask_i)
                n_lower_both = np.sum(lower_mask_both)
                lambda_lower = n_lower_both / n_lower_i if n_lower_i > 10 else np.nan
                
                pair_results[f'corr_{i}_{j}'] = {
                    'lambda_upper': lambda_upper,
                    'lambda_lower': lambda_lower,
                    'correlation_type': classify_correlation_pair(i, j)
                }
        
        results[f'threshold_{threshold}'] = pair_results
    
    return results
```

#### **4. High-Dimensional Analysis Strategies**

**A. Covariance-Guided Analysis** (Prioritize high-correlation pairs):
```python
def covariance_guided_tail_analysis(correlation_data, covariance_matrix, top_k=20):
    """
    Smart approach: Focus tail dependence analysis on pairs with highest linear correlation.
    Rationale: High covariance often correlates with tail dependence.
    """
    # Extract correlation coefficients, sort by strength, analyze top pairs
```

**B. Physical Grouping** (Reduce dimensionality meaningfully):
```python
def block_tail_dependence(correlation_data, block_structure):
    """
    Group measurements by physics:
    - 'small_scales': angular bins 0-2
    - 'large_scales': angular bins 6-8  
    - 'redshift_1': all z-bin 1 measurements
    - 'cross_correlations': ξ+ vs ξ- dependencies
    
    Analyze tail dependence between blocks instead of all pairs.
    """
```

**C. Principal Component Analysis**:
```python
def pc_tail_dependence_analysis(correlation_data):
    """
    1. PCA on correlation data → identify dominant modes
    2. Transform PC scores to uniform margins
    3. Analyze tail dependence between top 5-10 PCs
    Much more tractable than 1,431 pairwise comparisons!
    """
```

#### **5. Copula Model Comparison**
```python
def compare_copulas_on_realistic_data(correlation_realizations):
    """
    Compare how well different copulas model realistic correlation data:
    1. Fit both Gaussian and Student-t copulas to ensemble
    2. Compute likelihood for each realization under each copula
    3. Statistical test (paired t-test) for which fits better
    4. Information criteria (AIC/BIC) comparison
    """
```

#### **6. Integration with Existing Analysis**

**Modify `s8_copula_comparison.py`**:
- Add `--use-simulations` flag to generate realistic data instead of fiducial
- Add `--tail-analysis` flag to perform tail dependence tests
- Compare copula choice recommendation from tail analysis vs. parameter constraint differences

**Expected Workflow**:
```python
# 1. Generate simulation ensemble
ensemble = generate_realistic_correlation_data(n_realizations=1000)

# 2. Analyze tail dependence structure  
tail_results = covariance_guided_tail_analysis(ensemble, empirical_cov)
recommendation = interpret_tail_dependence_for_copulas(tail_results)

# 3. Compare with parameter inference results
if recommendation['copula'] == 'student_t':
    print("Tail dependence analysis supports Student-t copula choice")
    print(f"Confidence: {recommendation['confidence']}")
    
# 4. Validate: does the data-driven choice match the parameter constraint differences?
```

### **Deliverables**

1. **`tail_dependence_analysis.py`** - Complete implementation
2. **Modified `s8_copula_comparison.py`** - Integration with existing workflow  
3. **Validation study** - Compare tail-based recommendations with parameter constraint differences
4. **Documentation** - Clear explanation of methodology and physical interpretation

### **Expected Impact**

- **Data-driven copula selection** instead of ad-hoc choice
- **Physical understanding** of which correlation function dependencies drive copula effects
- **Methodological contribution** - first systematic copula analysis for cosmological 2-point functions
- **Practical guidance** for Stage IV surveys (Euclid, Roman, Rubin)

### **Timeline**
- [ ] Implement basic tail dependence measurement (1 week)
- [ ] Add high-dimensional analysis strategies (1 week)  
- [ ] Integrate with existing S8 analysis (1 week)
- [ ] Validation and documentation (1 week)

### **References**
- **Nelsen (2006)** - "An Introduction to Copulas" (methodology)
- **Sellentin & Heavens (2016)** - Student-t likelihoods in cosmology
- **Your current work** - systematic copula comparison for correlation functions

---