# xilikelihood/__init__.py
"""
xilikelihood: Two-point correlation function likelihood analysis.
"""

__version__ = "0.1.0"

# Core user-facing functions
from .simulate import simulate_correlation_functions, TwoPointSimulation
from .theory_cl import generate_theory_cl, prepare_theory_cl_inputs  
from .likelihood import XiLikelihood, fiducial_dataspace
from .mask_props import SphereMask
from .file_handling import save_arrays, load_arrays, generate_filename
from .cl2xi_transforms import pcl2xi, prep_prefactors
from .data_statistics import bootstrap, compute_simulation_moments

# Advanced users can access submodules
from . import distributions
from . import wpm_funcs
from . import theoretical_moments
from . import pseudo_alm_cov

__all__ = [
    # Main workflow functions
    'simulate_correlation_functions',
    'TwoPointSimulation',
    'generate_theory_cl', 
    'prepare_theory_cl_inputs',
    'XiLikelihood',
    'fiducial_dataspace',
    
    # Essential objects
    'SphereMask',
    
    # Key utilities
    'save_arrays',
    'load_arrays',
    'generate_filename',
    'pcl2xi',
    'prep_prefactors',
    'bootstrap',
    'compute_simulation_moments',
    
    # Advanced submodules
    'distributions',
    'wpm_funcs', 
    'theoretical_moments',
    
    # Package info
    '__version__',
]