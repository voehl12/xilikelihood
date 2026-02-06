# Configuration

## Key Parameters

### exact_lmax

The most important parameter for exact likelihood computation. Controls the maximum 
multipole for exact characteristic function computation and crucially determines computational cost.
See [Oehl & Tröster, 2025](https://arxiv.org/abs/2407.08718) for guidance on this parameter.

Can be specified in:
- `SphereMask(exact_lmax=...)` - mask computation
- `XiLikelihood(exact_lmax=...)` - likelihood computation, takes exact_lmax from mask if not specified.

If set in both, consistency is checked. Higher values = more accurate but more memory/time.


## LikelihoodConfig

Fine-tune computation settings:

```python
from xilikelihood.core_utils import LikelihoodConfig

config = LikelihoodConfig(
    cf_steps=2048,           # Characteristic function grid points
    pdf_steps=2048,          # PDF interpolation points
    ximax_sigma_factor=50.0, # Upper bound for characteristic function grid (in σ)
    large_angle_threshold=1/3, # Threshold for exact vs Gaussian marginals (degrees)
    enable_memory_cleanup=True,
    log_memory_usage=False,
)

likelihood = XiLikelihood(mask, redshift_bins, ang_bins, config=config)
```

config parameters can also be set individually when setting up the XiLikelihood object and will override standard config.