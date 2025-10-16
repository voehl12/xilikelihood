"""
Core utilities for the likelihood module.

Configuration and context managers specifically for likelihood computations.
"""

import gc
import time
import logging

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration for Likelihood
# ============================================================================

@dataclass
class LikelihoodConfig:
    """Configuration parameters for likelihood computation."""
    
    # CF computation settings
    cf_steps: int = 2048
    pdf_steps: int = 2048
    ximax_sigma_factor: float = 200.0
    ximin_sigma_factor: float = 70.0
    
    # Covariance settings
    cov_ell_buffer: int = 10
    
    # Memory and performance
    enable_memory_cleanup: bool = True
    log_memory_usage: bool = True
    max_memory_gb: float = 8.0
    
    # Validation and debugging
    validate_means: bool = False  # Expensive validation check for mean calculations
    analyze_eigenvalues: bool = False
    use_fixed_covariance: bool = False
    large_angle_threshold: float = 1  # Threshold for large angle bins in degrees
    
    # Student-t coupling settings
    use_student_t: bool = False
    student_t_dof: float = 4.0  # degrees of freedom
    student_t_stability_check: bool = True  # enable stability validation
    
    # File paths
    working_dir: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.cf_steps <= 0:
            raise ValueError("cf_steps must be positive")
        if self.ximax_sigma_factor <= 0:
            raise ValueError("ximax_sigma_factor must be positive")
        if self.ximin_sigma_factor <= 0:
            raise ValueError("ximin_sigma_factor must be positive")
        if self.student_t_dof <= 2.0:
            raise ValueError("student_t_dof must be > 2.0 for finite variance")
        if self.student_t_dof > 100.0:
            logger.warning(f"student_t_dof={self.student_t_dof} is very large, consider using Gaussian mode instead")
        


# ============================================================================
# JAX availability
# ============================================================================
# export XLA_FLAGS=--xla_gpu_cuda_data_dir=/cluster/software/stacks/2024-06/spack/opt/spack/linux-ubuntu22.04-x86_64_v3/gcc-12.2.0/cuda-12.1.1-5znnrjb5x5xr26nojxp3yhh6v77il7ie

import os
import sys

def ensure_jax_device():
    """
    Ensure JAX uses a working backend (GPU if available, otherwise CPU).
    If GPU is unavailable or initialization fails, force CPU and restart the script.
    Call this BEFORE importing jax anywhere else.
    """
    if os.environ.get("JAX_PLATFORM_NAME", "").lower() == "cpu":
        print("JAX_PLATFORM_NAME is already set to CPU. Skipping device check.")
        return

    try:
        import jax
        import jax.numpy as jnp
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform in ['gpu', 'cuda', 'rocm']]
        if gpu_devices:
            try:
                with jax.default_device(gpu_devices[0]):
                    x = jnp.array([1.0, 2.0, 3.0])
                    y = jnp.sum(x)
                    _ = y.devices()
                    # Additional test: try a more complex JAX op (sqrt)
                    try:
                        z = jnp.sqrt(x)
                        _ = z.devices()
                    except Exception as sqrt_error:
                        print("JAX GPU test failed for jnp.sqrt(x):", sqrt_error)
                        print("You may need to set 'export XLA_FLAGS=--xla_gpu_cuda_data_dir=...' for complex ops.")
                        raise
                print(f"JAX GPU backend is available: {gpu_devices}")
                return
            except Exception as gpu_error:
                print(f"JAX GPU test failed: {gpu_error}")
        else:
            print("No JAX GPU devices found.")
    except Exception as e:
        print(f"JAX import/device check failed: {e}")

    print("Forcing JAX to CPU backend and restarting script...")
    os.environ["JAX_PLATFORM_NAME"] = "cpu"
    sys.modules.pop("jax", None)
    sys.modules.pop("jax.numpy", None)
    os.execv(sys.executable, [sys.executable] + sys.argv)


# ============================================================================
# Context Managers for Memory Management
# ============================================================================

@contextmanager
def temporary_arrays():
    """Context manager for automatic cleanup of large temporary arrays."""
    temp_arrays = []
    
    def track_array(arr, name="temp_array"):
        """Track an array for automatic cleanup."""
        temp_arrays.append((arr, name))
        return arr
    
    try:
        yield track_array
    finally:
        total_memory = 0
        for arr, name in temp_arrays:
            if hasattr(arr, 'nbytes'):
                total_memory += arr.nbytes
            del arr
        
        if temp_arrays:
            logger.debug(f"Cleaned up {len(temp_arrays)} arrays ({total_memory/1024**2:.1f} MB)")
        gc.collect()

@contextmanager  
def computation_phase(phase_name: str, log_memory: bool = True):
    """Context manager for logging and memory management of computation phases."""
    logger.info(f"Starting {phase_name}...")
    start_time = time.time()
    
    initial_memory = None
    if log_memory:
        try:
            import psutil
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024**3  # GB
            logger.debug(f"{phase_name} initial memory: {initial_memory:.2f} GB")
        except ImportError:
            logger.debug("psutil not available, skipping memory monitoring")
    
    try:
        yield
    except Exception as e:
        logger.error(f"Error in {phase_name}: {e}")
        raise
    finally:
        elapsed = time.time() - start_time
        
        if log_memory and initial_memory is not None:
            try:
                final_memory = process.memory_info().rss / 1024**3
                logger.debug(f"{phase_name} final memory: {final_memory:.2f} GB")
            except:
                pass
            
        logger.info(f"Completed {phase_name} in {elapsed:.2f}s")
        gc.collect()

@contextmanager
def logging_context(log_file=None, level="INFO", console_output=True):
    """
    Context manager for setting up logging configuration.
    
    Parameters:
    -----------
    log_file : str or Path, optional
        Path to log file. If None, no file logging.
    level : str
        Logging level ("DEBUG", "INFO", "WARNING", "ERROR")
    console_output : bool
        Whether to also output to console
        
    Yields:
    -------
    logger : logging.Logger
        Configured logger instance
    """
    # Store original logging configuration
    original_handlers = logging.root.handlers[:]
    original_level = logging.root.level
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Set up new configuration
    log_level = getattr(logging, level.upper())
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = []
    
    # Add file handler if requested
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(console_handler)
    
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # Configure xilikelihood package logging
    xlh_logger = logging.getLogger('xilikelihood')
    xlh_logger.setLevel(log_level)
    
    # Get the main logger
    logger = logging.getLogger('s8_copula_comparison')
    logger.setLevel(log_level)
    
    try:
        logger.info(f"Logging initialized (level: {level})")
        if log_file:
            logger.info(f"Log file: {log_file}")
        yield logger
    finally:
        # Restore original logging configuration
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        logging.root.handlers = original_handlers
        logging.root.level = original_level

# ============================================================================
# Utility Functions
# ============================================================================

def check_property_equal(instances, property_name):
    """
    Check if a specific property of all instances is equal.

    Parameters:
        instances (list): A list of instances to check.
        property_name (str): The name of the property to check.

    Returns:
        bool: True if the property is equal for all instances, False otherwise.
    """
    if not instances:
        return True  # If the list is empty, return True

    first_value = getattr(instances[0], property_name)
    return all(getattr(instance, property_name) == first_value for instance in instances)



# ============================================================================
# Fiducial Cosmology
# ============================================================================

def fiducial_cosmo():
    """
    Return standard fiducial cosmology parameters used throughout xilikelihood.
    
    This provides a consistent set of cosmological parameters for testing,
    examples, and default analysis configurations. Based on Planck 2018 
    results and commonly used in weak lensing surveys.
    
    Returns
    -------
    dict
        Dictionary containing cosmological parameters:
        - 'omega_m': Matter density parameter (0.31)
        - 's8': Amplitude of matter fluctuations (0.8)
        
    Examples
    --------
    >>> import xilikelihood as xlh
    >>> cosmo = xlh.fiducial_cosmo()
    >>> print(cosmo)
    {'omega_m': 0.31, 's8': 0.8}
    
    >>> # Use in likelihood evaluation
    >>> likelihood = xlh.XiLikelihood(...)
    >>> logL = likelihood.loglikelihood(data, xlh.fiducial_cosmo())
    """
    return {
        'omega_m': 0.31,  # Matter density parameter (Planck 2018-like)
        's8': 0.8         # Amplitude of matter fluctuations
    }


__all__ = [
    'LikelihoodConfig',
    'fiducial_cosmo',
    'temporary_arrays', 
    'computation_phase',
    'check_property_equal',
    'logging_context',
]
