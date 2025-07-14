# S8 Posterior Analysis

Analysis pipeline for computing S8 posteriors using the copula likelihood method.

## Structure

- `s8_posteriors.py` - Main analysis functions
- `mock_data_generation.py` - Data creation utilities  
- `data/` - Mock data and results
- `slurm/` - Job submission scripts

## Usage

### Main Analysis Scripts

#### `s8_om_posterior.py` - 2D Parameter Grid Analysis
Computes likelihood values across a 2D grid in (Ωₘ, S8) parameter space for posterior mapping:

```bash
# Single job (computes subset of parameter grid)
python s8_om_posterior.py <job_number>

# Submit array job for full grid
sbatch slurm/jobarray_s8om
```

**Key features:**
- Grid sampling: 100×100 points in (Ωₘ ∈ [0.1, 0.5], S8 ∈ [0.5, 1.1])
- Splits work across 500 jobs for parallel processing
- Computes both exact and Gaussian likelihood comparisons
- Outputs: `/cluster/scratch/veoehl/posteriors/posterior_{job}.npy`
- **Postprocessed with:** `postprocess_2dpost.py`

### Generate mock data:
```python
from mock_data_generation import create_mock_data
from s8_posteriors import setup_likelihood, FIDUCIAL_COSMO

likelihood, mask = setup_likelihood()
create_mock_data(likelihood, "data/mock_data.npz", "data/cov.npz", 
                fiducial_cosmo=FIDUCIAL_COSMO)
```

### Run S8 analysis:
```bash
# Single job
python s8_posteriors.py 1

# Submit array job
sbatch slurm/posterior_job.sh
```

## Output

Results saved to `data/s8posts/` with naming convention:
- `s8post_*_auto.npz` - Autocorrelation results
- `s8post_*_croco.npz` - Cross-correlation results  
- `s8post_*_measurement*.npz` - Individual measurement results

## Postprocessing and plotting

The analysis includes four postprocessing scripts for different types of analysis:

### `postprocess_s8.py`
- **Purpose**: Process S8 posteriors from fiducial simulations
- **Input**: `s8posts/s8post_1000sqd_{job}_fiducial_nonoise.npz` files
- **Output**: Comparison plots of Gaussian vs copula likelihoods
- **Features**: Normalizes posteriors, calculates means, creates overlay plots with vertical lines at posterior means

### `postprocess_2dpost.py` 
- **Purpose**: Create 2D posterior maps in (Ωₘ, S8) parameter space
- **Input**: `posteriors/posterior_{job}.npy` files from grid sampling
- **Output**: 2D contour plots and marginal distributions
- **Features**: Assembles grid-based posteriors, normalizes using 2D integration, creates contour plots

### `postprocess_1dposts.py`
- **Purpose**: Analyze 1D posteriors from individual data vector components
- **Input**: Auto-correlation and cross-correlation posterior files
- **Output**: Component-wise posterior comparison plots
- **Features**: Separates auto/cross correlations, validates redshift bin combinations, creates angular bin comparisons

### `postprocess_1dfromndpost.py`
- **Purpose**: Extract 1D S8 posteriors from high-dimensional likelihood analysis
- **Input**: Large-scale analysis files (`s8post_10000sqd_*_largescales_*.npz`)
- **Output**: Publication-quality S8 posterior plots
- **Features**: Supports auto/cross/combined analysis, includes LaTeX formatting, fiducial comparison lines

