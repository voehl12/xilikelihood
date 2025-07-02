# S8 Posterior Analysis

Analysis pipeline for computing S8 posteriors using the copula likelihood method.

## Structure

- `s8_posteriors.py` - Main analysis functions
- `mock_data_generation.py` - Data creation utilities  
- `config.py` - Configuration parameters
- `data/` - Mock data and results
- `slurm/` - Job submission scripts

## Usage

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