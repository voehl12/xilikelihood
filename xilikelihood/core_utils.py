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
    cf_steps: int = 4096
    ximax_sigma_factor: float = 40.0
    ximin_sigma_factor: float = 5.0
    
    # Covariance settings
    cov_ell_buffer: int = 10
    
    # Memory and performance
    enable_memory_cleanup: bool = True
    log_memory_usage: bool = True
    max_memory_gb: float = 8.0
    
    # Validation and debugging
    validate_means: bool = False  # Expensive validation check for mean calculations
    
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
        


# ============================================================================
# JAX availability
# ============================================================================

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



__all__ = [
    'LikelihoodConfig',
    'temporary_arrays', 
    'computation_phase',
    'check_property_equal',
]
