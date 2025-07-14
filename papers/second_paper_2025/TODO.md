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
- [ ] **Clean function names** (optional) - `posterior_from_1d_firstpaper()` → `posterior_from_measurement()`

## ✅ REFACTORING COMPLETE!

All major tasks have been completed successfully:
- ✅ Organized structure in papers/second_paper_2025/
- ✅ Extracted configuration to config.py with centralized parameters
- ✅ Separated data generation utilities  
- ✅ Clean analysis functions with proper imports
- ✅ Comprehensive documentation and README
- ✅ Updated job submission scripts
- ✅ Integrated config.py across all analysis files

The S8 analysis pipeline is now well-organized and maintainable!