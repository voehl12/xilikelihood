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
3. **Rename functions** for clarity:
   - `posterior_from_1d_firstpaper()` → `posterior_from_measurement()`
   - `posterior_from_1d_autocorr()` → clearer parameter handling
4. **Add proper imports** and path handling
5. **Create reusable utilities** for common patterns
6. **Add documentation** and README

### Functions to Extract
- `create_mock_data()` → `mock_data_generation.py`  
- `setup_likelihood()` → shared utility
- Configuration constants → `config.py`
- S8 grid logic → configurable parameters

### Implementation Priority
- [ ] Move to papers folder
- [ ] Extract configuration  
- [ ] Separate data generation
- [ ] Clean up analysis functions
- [ ] Add documentation
- [ ] Create job submission scripts