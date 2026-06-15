# First Paper Figures (2024)

This directory contains code to reproduce all figures from the paper:
"The exact non-Gaussian weak lensing likelihood: A framework to calculate analytic likelihoods for correlation functions on masked Gaussian random fields" by Oehl, V., Tröster, T.

This is a preserved first-paper workspace. It contains historical plotting and
simulation scripts, generated figure products, and paper-specific input spectra.
Some scripts retain the old project name ``2ptlikelihood`` and ETH cluster
paths because they document the original production environment. For new
likelihood analyses, use the package-level ``xilikelihood`` modules and the
second-paper workflow.

## Overview

`paperplots.py` generates all main text and appendix figures using a mix of 
legacy and current package implementations.

## Environment Setup

### Option 1: Use the historical repository virtual environment
```bash
# From the repository root
source 2ptlikelihood/bin/activate
```

### Option 2: Install dependencies manually
```bash
pip install -r requirements.txt # not all are strictly needed probably but it works this way
# Plus the main xilikelihood package (from repository root):
pip install -e ../../
```

## Reproducing Figures

1. Activate environment or install dependencies (see above)

2. Generate all figures:
   ```python
   from paperplots import generate_all_figures
   generate_all_figures()
   ```

3. Individual figures:
   ```python
   from paperplots import fig1, fig2
   fig1()  # Saves to figures/figure1.pdf
   ```

## Data Sources

- `cls/Cl_3x2pt_kids55.txt`: KiDS-450 power spectra
- Large simulation data: Available on request (not included in repo due to size)

## Notes

- Uses legacy file handling and some old module structures
- Bridges to current distributions, mask_props, theory_cl modules  
- Output figures saved to `plots_paperone/` directory
- Some simulation paths are hardcoded to the original cluster locations -
  update as needed for your local setup

## File Structure

```
papers/first_paper_2024/
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── paperplots.py       # Figure generation script
├── data/               # Paper-specific data files
└── plots_paperone/            # Generated figures (not tracked in git)
```
