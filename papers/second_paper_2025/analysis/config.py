# config.py in the analysis folder
"""Configuration for S8 posterior analysis."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "s8posts"
PACKAGE_DIR = '/cluster/home/veoehl/xilikelihood/'  # Go up from analysis to package root

REDSHIFT_BINS_PATH = BASE_DIR.parent.parent.parent / "redshift_bins/KiDS"

# Analysis parameters
EXACT_LMAX = 20
FIDUCIAL_COSMO = {"omega_m": 0.31, "s8": 0.8}

# Mask configuration
MASK_CONFIG = {
    "spins": [2], 
    "circmaskattr": (10000, 256), 
    "l_smooth": 30
}

# Angular bins
ANG_BINS = [(2, 3)]  # Single bin for 1D analysis
FIDUCIAL_ANG_BINS = "from_fiducial_dataspace"  # Use xlh.fiducial_dataspace()

# S8 grids for different analyses
S8_GRIDS = {
    "narrow": (0.7, 0.9, 100),    # (min, max, n_points)
    "medium": (0.6, 1.0, 200), 
    "wide": (0.4, 1.2, 200),
    "test": (0.7,0.9,20)
}

# Parameter grids for 2D analysis
PARAM_GRIDS = {
    "omega_m": (0.2, 0.45, 200),
    "s8": (0.7, 0.9, 200)
}

# Job configuration
N_JOBS_2D = 500  # For s8_om_posterior.py

# Scale cuts for marginal analysis
SCALE_CUTS = {
    "large_scales": {"min_arcmin": 15, "max_arcmin": 300},    # Conservative linear scales
    "medium_scales": {"min_arcmin": 5, "max_arcmin": 50},    # Intermediate scales  
    "small_scales": {"min_arcmin": 0.5, "max_arcmin": 15},   # Non-linear scales
    "all_scales": {"min_arcmin": 0.5, "max_arcmin": 300},    # Full range
    "custom": {"min_arcmin": None, "max_arcmin": None}       # User-defined
}

# File naming patterns
DATA_FILES = {
    "1d_firstpaper": {
        "mock_data": "mock_data_10000sqd_nonoise_firstpaper.npz",
        "covariance": "gaussian_covariance_10000sqd_nonoise_firstpaper.npz"
    },
    "nd_analysis": {
        "mock_data": "mock_data_10000sqd_nonoise.npz", 
        "covariance": "gaussian_covariance_10000sqd_nonoise.npz"
    },
    "1000sqd": {
        "mock_data": "fiducial_data_1000sqd.npz",
        "covariance": "gaussian_covariance_1000sqd.npz"
    },
    "10000sqd": {
        "mock_data": "mock_data_10000sqd.npz",
        "covariance": "gaussian_covariance_10000sqd.npz"
    },
    "10000sqd_random": {
        "mock_data": "mock_data_10000sqd_random.npz",
        "covariance": "gaussian_covariance_10000sqd_random.npz"
    },
    "10000sqd_kidslike": {
        "mock_data": "mock_data_10000sqd_kidslike.npz",
        "covariance": "gaussian_covariance_10000sqd_kidslike.npz"
    },
    "10000sqd_kidsplus": {
        "mock_data": "mock_data_10000sqd_kidsplus.npz",
        "covariance": "gaussian_covariance_10000sqd_kidsplus.npz" # including one extra angular bin, 5-10deg
    },
    
}