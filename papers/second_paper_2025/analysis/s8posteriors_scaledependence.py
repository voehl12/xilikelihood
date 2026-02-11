"""
Compute S8 posteriors as a function of angular scale.

Runs a batch job for a given `jobnumber` and loops over a subset of
angular-bin indices, calling `posterior_from_nd` from `s8_posteriors`.
Designed for SLURM job arrays; writes logs and output files per job.
"""

import numpy as np
import sys
import logging
from time import time
from pathlib import Path
from s8_posteriors import posterior_from_nd, setup_likelihood_nd_example

from config import OUTPUT_DIR

# Add package root to path
package_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(package_root))

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_dir = OUTPUT_DIR / 'logs'
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 's8_posteriors.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__) 






jobnumber = int(sys.argv[1]) - 1
logger.info(f"Starting S8 posterior computation for job number {jobnumber}")
for i in range(4):
    logger.info(f"Setting up likelihood for angle index {i}")
    likelihood, data_paths = setup_likelihood_nd_example(create_data=True,angle_index=i)
    mock_data_path, gaussian_covariance_path = data_paths
    data = np.load(mock_data_path)["data"]
    covariance = np.load(gaussian_covariance_path)["cov"]
    logger.info(f"Loading data from {mock_data_path}")
    logger.info(f"Loading covariance from {gaussian_covariance_path}")
    posterior_from_nd(jobnumber, likelihood, (data,covariance),s8grid='wide', njobs=100,outputfile='s8post_10000sqd_fiducial_nonoise_nd_ang{}'.format(i))
logger.info("Job completed successfully")