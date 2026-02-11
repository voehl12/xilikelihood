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
MASK_CONFIG_STAGE4 = {
    "spins": [2], 
    "circmaskattr": (10000, 256), 
    "l_smooth": 30
}
MASK_CONFIG_STAGE3 = {
    "spins": [2], 
    "circmaskattr": (1000, 256), 
    "l_smooth": 30
}

MASK_CONFIG_HIGHRES_STAGE4 = {
    "spins": [2], 
    "circmaskattr": (10000, 512), 
    "l_smooth": 30
}

MASK_CONFIG_MEDRES_STAGE4 = {
    "spins": [2], 
    "circmaskattr": (10000, 1024), 
    "l_smooth": 30
}

MASK_CONFIG_EXTREMERES_STAGE4 = {
    "spins": [2], 
    "circmaskattr": (10000, 2048), 
    "l_smooth": 30
}

MASK_CONFIG_HIGHRES_STAGE3 = {
    "spins": [2], 
    "circmaskattr": (1000, 512), 
    "l_smooth": 30
}

MASK_CONFIG_MEDRES_STAGE3 = {
    "spins": [2], 
    "circmaskattr": (1000, 1024), 
    "l_smooth": 30
}

# Angular bins
ANG_BINS = [(2, 3)]  # Single bin for 1D analysis
FIDUCIAL_ANG_BINS = "from_fiducial_dataspace"  # Use xlh.fiducial_dataspace()

# S8 grids for different analyses
S8_GRIDS = {
    "narrow": (0.7, 0.9, 200),    # (min, max, n_points)
    "medium": (0.6, 1.0, 400), 
    "wide": (0.4, 1.2, 800)
}

# Parameter grids for 2D analysis
PARAM_GRIDS_NARROW = {
    "omega_m": (0.2, 0.45, 100),
    "s8": (0.7, 0.9, 100)
}
PARAM_GRIDS_MEDIUM = {
    "omega_m": (0.1, 0.5, 100),
    "s8": (0.7, 0.9, 100)
}
PARAM_GRIDS_WIDE = {
    "omega_m": (0.1, 0.5, 100),
    "s8": (0.5, 1.1, 100)
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
    "1000sqd_kidsplus": {
        "mock_data": "mock_data_1000sqd_kidsplus.npz",
        "covariance": "gaussian_covariance_1000sqd_kidsplus.npz"
    },
    "10000sqd_kidsplus_lownoise": {
        "mock_data": "mock_data_10000sqd_kidsplus_lownoise.npz",
        "covariance": "gaussian_covariance_10000sqd_kidsplus_lownoise.npz"
    },
    "10000sqd_kidsplus_inclsmall": {
        "mock_data": "mock_data_10000sqd_kidsplus_inclsmall.npz",
        "covariance": "gaussian_covariance_10000sqd_kidsplus_inclsmall.npz"
    },
    "1000sqd_kidsplus_inclsmall": {
        "mock_data": "mock_data_1000sqd_kidsplus_inclsmall.npz",
        "covariance": "gaussian_covariance_1000sqd_kidsplus_inclsmall.npz"
    },
    "1000sqd_kidsplus_highres": {
        "mock_data": "mock_data_1000sqd_kidsplus_highres.npz",
        "covariance": "gaussian_covariance_1000sqd_kidsplus_highres.npz"
    },
    "10000sqd_kidsplus_nonoise": {
        "mock_data": "mock_data_10000sqd_kidsplus_nonoise.npz",
        "covariance": "gaussian_covariance_10000sqd_kidsplus_nonoise.npz"
    }

    
}

# Prior configuration
# https://www.aanda.org/articles/aa/pdf/2021/01/aa39070-20.pdf
# https://arxiv.org/pdf/2007.01844
PRIOR_CONFIG = {
    # List of parameter names in order
    "parameters": ["s8","w_c","w_b", "h", "n_s", "A_IA", "delta_z"],
    
    # Uniform priors: (min, max)
    "uniform": {
        "s8": (0.1, 1.3),
        "w_c": (0.051,0.255),
        "w_b": (0.019, 0.026),
        "h": (0.64, 0.82),
        "n_s": (0.84, 1.1),
        "A_IA": (-6.0, 6.0),
            
    },
    
    # Multivariate Gaussian prior: {"mean": [...], "cov": [[...]]}
    # Set to None if not using
    "multivariate_gaussian": {
        "parameter": "delta_z",  # Which parameter has the multivariate Gaussian prior
        "mean": [0.0,0.002,0.013,0.011,-0.006],  # Mean vector
        "cov": [[0.0106,0.15,0.10,0.00,0.05],
                [0.15,0.0113,0.31,-0.17,-0.02],
                [0.10,0.31,0.0118,-0.10,-0.02],
                [0.00,-0.17,-0.10,0.0087,0.34],
                [0.05,-0.02,-0.02,0.34,0.0097]]  # Covariance matrix (1x1 for single param, NxN for N params)
    }
    
    # To disable multivariate Gaussian, set to None:
    # "multivariate_gaussian": None
}