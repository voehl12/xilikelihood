import numpy as np
import logging
import xilikelihood as xlh
from mock_data_generation import create_mock_data
from config import FIDUCIAL_COSMO, EXACT_LMAX, MASK_CONFIG, PACKAGE_DIR

logging.basicConfig(
    level=logging.INFO,  # Uncomment for less verbose output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()  # Console output
    ]
)
logger = logging.getLogger(__name__)

# Enable ALL xilikelihood logging (catch any submodule loggers)
xili_logger = logging.getLogger('xilikelihood')
xili_logger.setLevel(logging.DEBUG)


# definitely run with exact_lmax=20!
# Set root logger to capture everything
logging.getLogger().setLevel(logging.INFO)

redshift_bins, angular_bins = xlh.fiducial_dataspace(min_ang_cutoff=0)
mask = xlh.SphereMask(
    spins=MASK_CONFIG['spins'], 
    circmaskattr=MASK_CONFIG['circmaskattr'], 
    exact_lmax=EXACT_LMAX, 
    l_smooth=MASK_CONFIG['l_smooth'],
    working_dir=PACKAGE_DIR  # Use package root for shared arrays
)
likelihood = xlh.XiLikelihood(
    redshift_bins=redshift_bins,
    ang_bins_in_deg=angular_bins,
    mask=mask,
    include_ximinus=False,  # Set to False for small scales
    exact_lmax=EXACT_LMAX,
)
# Generate mock data and covariance
mock_data_path = "mock_data_smallscales.npz"
gaussian_covariance_path = "gaussian_covariance_smallscales.npz"
#create_mock_data(likelihood, mock_data_path, gaussian_covariance_path)
mock_data = np.load(mock_data_path)["data"]
gaussian_covariance = np.load(gaussian_covariance_path)["cov"]

likelihood.gaussian_covariance = gaussian_covariance
likelihood.setup_likelihood()
logL = likelihood.loglikelihood(mock_data,FIDUCIAL_COSMO)