# Configuration

This page summarizes the parameters that most strongly affect reproducibility,
runtime, and memory use. For the full repository map, see the
[repository reference](repository_reference.rst).

## Key Parameters

### exact_lmax

The most important parameter for exact likelihood computation. It controls the
maximum multipole included in the exact characteristic-function calculation and
strongly determines computational cost. See
[Oehl & Tröster, 2025](https://arxiv.org/abs/2407.08718) for the earlier exact
low-dimensional method and
[Oehl & Tröster, 2026](https://arxiv.org/abs/2604.07336) for the copula
extension.

Can be specified in:

- `SphereMask(exact_lmax=...)` - mask computation.
- `XiLikelihood(exact_lmax=...)` - likelihood computation; if omitted, the
  likelihood takes `exact_lmax` from the mask.

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

Config parameters can also be set individually when creating the `XiLikelihood`
object and will override the default config values.

## Large-Angle Threshold

`LikelihoodConfig.large_angle_threshold` is given in degrees. Angular bins whose
lower edge is at least this threshold are treated with exact marginals; smaller
bins use Gaussian marginals. The default is `15/60` degrees.

## Working Directories and Caches

`SphereMask` and `XiLikelihood` may create cache files such as Wigner/mask
coupling arrays and covariance products. Use `working_dir` to make cache
locations explicit in production runs:

```python
mask = SphereMask(..., working_dir="/path/to/cache/root")
config = LikelihoodConfig(working_dir="/path/to/cache/root")
```

For paper reproducibility, record whether a result used regenerated caches or
pre-existing cache files.
