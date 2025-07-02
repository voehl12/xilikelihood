# config.py in the analysis folder
"""Configuration for S8 posterior analysis."""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = DATA_DIR / "s8posts"
REDSHIFT_BINS_PATH = BASE_DIR.parent.parent.parent / "redshift_bins/KiDS/..."

# Analysis parameters
EXACT_LMAX = 30
FIDUCIAL_COSMO = {"omega_m": 0.31, "s8": 0.8}
MASK_CONFIG = {"spins": [2], "circmaskattr": (10000, 256), "l_smooth": 30}
ANG_BINS = [(2, 3)]

# S8 grids for different analyses
S8_GRIDS = {
    "narrow": (0.7, 0.9, 200),
    "medium": (0.6, 1.0, 200), 
    "wide": (0.4, 1.2, 200)
}