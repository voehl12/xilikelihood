# simulation_scripts/README.md

# Simulation Scripts

This directory contains scripts for running correlation function simulations using the 2ptlikelihood package.

## Scripts

### `xi_sim.py` - 1D Correlation Function Simulations
- Uses TreeCorr method  
- Single redshift bin combination
- Optimized for speed

```bash
python xi_sim.py <job_number>
```

### `xi_sim_nD.py` - n-dimensional Correlation Function Simulations  
- Uses pseudo_cl method
- Multiple redshift bin combinations

```bash
python xi_sim_nD.py <job_number>
```

## Setup

1. Install the 2ptlikelihood package in development mode:
   ```bash
   cd ../
   pip install -e .
   ```

2. Ensure data files are available:
   - `singlet_lowres.fits` (mask file)
   - `Cl_3x2pt_kids55.txt` (theory power spectra)

## Output

Simulations save results to `/cluster/scratch/veoehl/xi_sims/`:
- `job{N}.npz` - Correlation function results
- `pcljob{N}.npz` - Pseudo-C_l results (if requested)

## Cluster Usage

For SLURM batch jobs:
```bash
sbatch jobarray_xisimnd
```