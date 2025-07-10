"""
File I/O utilities for cosmological data analysis.

This module provides standardized file handling operations for:
- Covariance matrices and pseudo power spectra
- Simulation data (xi+/-, power spectra)
- PDF and correlation function data
- Batch job result processing

All functions use proper logging and error handling for robust operation.
"""

import os
import numpy as np

import glob
import re  # Import regular expressions
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

from .cl2xi_transforms import pcls2xis

logger = logging.getLogger(__name__)

__all__ = [
    # Core file operations
    'ensure_directory_exists',
    'check_for_file',
    'save_arrays',
    'load_arrays',
    'create_array_directory',
    'generate_filename',
    
    # Simulation data readers
    'read_sims_nd',
    'read_pcl_sims',
    'xi_sims_from_pcl',
    
    # Batch processing
    'read_posterior_files',
    
    # Utility functions
    'save_matrix',
    
    
]


# ============================================================================
# Core File Operations
# ============================================================================


def ensure_directory_exists(directory: str) -> None:
    """
    Ensure a directory exists, creating it if necessary.
    
    Parameters:
    -----------
    directory : str
        Path to directory that should exist
        
    Notes:
    ------
    Uses os.makedirs with exist_ok=True for safe directory creation.
    Replaces unsafe os.system("mkdir") calls.
    """
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Created directory: {directory}")

def check_for_file(filepath: str, kind: str = "file") -> bool:
    """
    Check if a file exists with proper logging.
    
    Parameters:
    -----------
    filepath : str
        Path to file to check
    kind : str, default="file"
        Type description for logging
        
    Returns:
    --------
    bool
        True if file exists, False otherwise
    """
    logger.debug(f"Checking for {kind}: {filepath}")
    if os.path.isfile(filepath):
        logger.debug(f"Found {kind}")
        return True
    else:
        logger.debug(f"No {kind} found")
        return False

def create_array_directory(base_dir: str, subdir_name: str) -> Path:
    """Create directory for array storage if it doesn't exist."""
    array_dir = Path(base_dir) / subdir_name
    array_dir.mkdir(exist_ok=True)
    return array_dir

def save_arrays(
    data: Union[np.ndarray, Dict[str, np.ndarray]],
    filepath: str,
    compress: bool = True,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    General purpose array saving with optional metadata.
    
    Parameters
    ----------
    data : ndarray or dict
        Single array or dictionary of arrays to save
    filepath : str
        Output file path
    compress : bool
        Whether to use compression (default: True)
    metadata : dict, optional
        Additional metadata to store
    """
    ensure_directory_exists(os.path.dirname(filepath))
    
    if isinstance(data, np.ndarray):
        data = {"data": data}
    
    if metadata:
        data.update(metadata)
    
    save_func = np.savez_compressed if compress else np.savez
    save_func(filepath, **data)
    
    file_size_mb = os.path.getsize(filepath) / 1024**2
    logger.info(f"Saved arrays to {filepath}: {file_size_mb:.1f} MB")





def load_arrays(filepath: str, keys: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """
    General purpose array loading.
    
    Parameters
    ----------
    filepath : str
        File to load from
    keys : list of str, optional
        Specific keys to load (loads all if None)
        
    Returns
    -------
    dict
        Dictionary of loaded arrays
    """
    if not check_for_file(filepath):
        raise FileNotFoundError(f"Array file not found: {filepath}")
    
    logger.debug(f"Loading arrays from: {filepath}")
    try:
        with np.load(filepath) as data_file:
            if keys is None:
                result = {key: data_file[key].copy() for key in data_file.files}
            else:
                result = {key: data_file[key].copy() for key in keys if key in data_file.files}
            
            logger.debug(f"Loaded arrays with keys: {list(result.keys())}")
            return result
    except Exception as e:
        raise RuntimeError(f"Failed to load array from {filepath}: {e}")
    

def generate_filename(
    data_type: str,
    parameters: Dict[str, Union[int, str]],
    base_dir: Optional[str] = None,
    extension: str = ".npz"
) -> str:
    """
    Generate standardized filenames for analysis data.
    
    Parameters
    ----------
    data_type : str
        Type of data ("cov", "pcl", "sims", etc.)
    parameters : dict
        Dictionary of parameters to include in filename
    base_dir : str, optional
        Base directory (creates subdirectory based on data_type)
    extension : str
        File extension (default: ".npz")
        
    Returns
    -------
    str
        Generated filename
        
    Examples
    --------
    >>> # Covariance matrix
    >>> filename = generate_filename(
    ...     "cov", 
    ...     {"lmax": 100, "nside": 256, "mask": "kids", "theory": "fiducial"}
    ... )
    >>> print(filename)  # "cov_l100_n256_kids_fiducial.npz"
    
    >>> # Pseudo-Cl
    >>> filename = generate_filename(
    ...     "pcl",
    ...     {"nside": 256, "mask": "circular", "sigma": "default"},
    ...     base_dir="/data"
    ... )
    >>> print(filename)  # "/data/pcls/pcl_n256_circular_default.npz"
    """
    # Build filename components
    components = [data_type]
    
    # Add parameters in a standardized order
    param_order = ["lmax", "nside", "mask", "theory", "sigma", "job", "batch"]
    
    for param in param_order:
        if param in parameters:
            value = parameters[param]
            if param in ["lmax", "nside"]:
                components.append(f"{param[0]}{value}")  # l100, n256
            else:
                components.append(str(value))
    
    # Add any remaining parameters
    for key, value in parameters.items():
        if key not in param_order:
            components.append(f"{key}_{value}")
    
    filename = "_".join(components) + extension
    
    if base_dir:
        # Create subdirectory based on data type
        subdir_map = {
            "cov_xi": "covariances",
            "cov": "covariances",
            "pcl": "pcls", 
            "sims": "simulations",
            "pdf": "pdfs",
            "post": "posteriors",
            "wpm": "wpm_arrays",
            "mllp": "mllp_arrays",
        }
        subdir = subdir_map.get(data_type, data_type + "s")
        full_dir = os.path.join(base_dir, subdir)
        ensure_directory_exists(full_dir)
        return os.path.join(full_dir, filename)
    
    return filename



def check_array_file_size(filepath: str) -> str:
    """Check and return human-readable file size."""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024**2)
        return f"{size_mb:.1f} MB"
    return "File not found"




# ============================================================================
# Simulation Data Readers
# ============================================================================

def xi_sims_from_pcl(job_id: int, prefactors: np.ndarray, filepath: str, lmax: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert pseudo power spectra to correlation functions for a simulation job.
    
    Parameters:
    -----------
    job_id : int
        Job number identifier
    prefactors : ndarray
        Conversion prefactors for xi calculation
    filepath : str
        Base path to simulation files
    lmax : int, optional
        Maximum multipole to use
        
    Returns:
    --------
    tuple
        (xi_plus, xi_minus) correlation functions
    """
    pcl_file_path = f"{filepath}/pcljob{job_id:d}.npz"
    
    if not check_for_file(pcl_file_path, f"PCL job {job_id}"):
        raise FileNotFoundError(f"PCL file not found: {pcl_file_path}")
    
    try:
        with np.load(pcl_file_path) as pclfile:
            if lmax is not None:
                file_lmax = int(pclfile["lmax"])
                if lmax > file_lmax:
                    raise ValueError(
                        f"Requested lmax ({lmax}) too high for simulated pcljob {job_id}. "
                        f"File has lmax={file_lmax}. Need new simulations."
                    )
            
            pcl_s = np.array([pclfile['pcl_e'], pclfile['pcl_b'], pclfile['pcl_eb']])
        
        xips, xims = pcls2xis(pcl_s, prefactors, out_lmax=lmax)
        return xips, xims
        
    except Exception as e:
        raise RuntimeError(f"Failed to process PCL job {job_id}: {e}")

def read_sims_nd(filepath: str, njobs: int, lmax: int, kind: str = "xip", 
                 prefactors: Optional[np.ndarray] = None, theta: Optional[List] = None) -> Tuple[np.ndarray, List]:
    """
    Read n-dimensional simulation data from multiple job files.
    
    Parameters:
    -----------
    filepath : str
        Base path to simulation files
    njobs : int
        Number of job files to read
    lmax : int
        Maximum multipole for calculations
    kind : str, default="xip"
        Type of correlation function ("xip" or "xim")
    prefactors : ndarray, optional
        Conversion prefactors (required for older simulations)
    theta : list, optional
        Angular bins to include (uses all if None)
        
    Returns:
    --------
    tuple
        (all_xi, theta_angles) - concatenated xi data and angle list
    """
    logger.info(f"Reading {kind} simulations from {filepath}")
    logger.info(f"Processing {njobs} jobs with lmax={lmax}")
    all_xi = []
    missing = []
    angles = None  # To store angles from the first file

    for i in range(1, njobs + 1):
        job_file = f"{filepath}/job{i:d}.npz"
        
        if not check_for_file(job_file, f"job {i}"):
            logger.warning(f"Missing job number {i}")
            missing.append(i)
            continue
        try:
            with np.load(job_file) as xifile:
                file_lmax = int(xifile["lmax"])
                
                # Check if lmax matches - if not, need to recompute from PCL
                if lmax != file_lmax:
                    logger.warning(f"lmax mismatch in job {i}: file has {file_lmax}, requested {lmax}. "
                                 f"Regenerating xi from PCL data.")
                    
                    # Get angles and prefactors
                    if angles is None:
                        angles = [tuple(angle) for angle in xifile["theta"]]
                        logger.debug(f"Found {len(angles)} angular bins from file")
                    
                    # Check for prefactors in the file or use provided ones
                    if "prefactors" in xifile.files:
                        file_prefactors = xifile["prefactors"]
                        logger.debug("Using prefactors from xi file")
                    elif prefactors is not None:
                        file_prefactors = prefactors
                        logger.debug("Using provided prefactors")
                    else:
                        raise ValueError(f"Prefactors must be provided for job {i} with lmax mismatch")
                    
                    if len(file_prefactors) != len(angles):
                        raise ValueError(f"Prefactors length ({len(file_prefactors)}) doesn't match "
                                       f"theta length ({len(angles)}) for job {i}")
                    
                    # Regenerate xi from PCL data
                    xips, xims = xi_sims_from_pcl(i, file_prefactors, filepath, lmax=lmax)
                    xi = xips if kind == "xip" else xims
                    
                    # Save regenerated data to new location
                    _save_regenerated_xi_data(filepath, i, xips, xims, lmax, angles, file_lmax)
                    
                else:
                    # lmax matches - use existing xi data
                    try:
                        xi = xifile[kind]  # Shape (batchsize, n_corr, n_theta)
                        logger.debug(f"Using existing {kind} data from job {i}")
                    except KeyError:
                        raise KeyError(f"Key '{kind}' not found in job {i} file")
                    
                    if angles is None:
                        angles = [tuple(angle) for angle in xifile["theta"]]
                        logger.debug(f"Found {len(angles)} angular bins from file")
                
                all_xi.append(xi)
                
        except Exception as e:
            logger.error(f"Error processing job {i}: {e}")
            missing.append(i)

    # Handle missing files
    if missing:
        missing_str = ",".join(map(str, missing))
        logger.warning(f"Missing jobs: {missing_str}")

    # Validate theta parameter against available angles
    if theta is not None:
        if not set(theta).issubset(set(angles)):
            raise ValueError("Provided theta contains angles not in the dataset")
    else:
        theta = angles

    # Concatenate all simulation data
    if all_xi:
        all_xi = np.concatenate(all_xi, axis=0)
        logger.info(f"Loaded {all_xi.shape[0]} simulations with shape {all_xi.shape}")
    else:
        logger.error("No simulation data loaded!")
        all_xi = np.array([])
    
    return all_xi, theta

def _save_regenerated_xi_data(filepath: str, job_id: int, xips: np.ndarray, xims: np.ndarray, 
                             lmax: int, angles: List, old_lmax: int) -> None:
    """Save regenerated xi data to new file structure with updated lmax."""
    
    # Create new folder path with updated lmax
    new_folder = filepath.replace(f"llim_{old_lmax}", f"llim_{lmax}")
    ensure_directory_exists(new_folder)
    
    new_filepath = os.path.join(new_folder, f"job{job_id:d}.npz")
    
    # Convert angles back to numpy array format expected by the file
    theta_array = np.array(angles)
    
    np.savez(
        new_filepath,
        xip=xips,
        xim=xims,
        lmax=lmax,
        theta=theta_array,
    )
    
    logger.info(f"Saved regenerated xi data to: {new_filepath}")
    file_size_mb = os.path.getsize(new_filepath) / 1024**2
    logger.debug(f"Regenerated file size: {file_size_mb:.2f} MB")


def read_pcl_sims(filepath: str, njobs: int) -> np.ndarray:
    """
    Read pseudo power spectra from simulation job files.
    
    Parameters:
    -----------
    filepath : str
        Base path to PCL simulation files
    njobs : int
        Number of job files to read
        
    Returns:
    --------
    ndarray
        Concatenated PCL data array
    """
    logger.info(f"Reading PCL simulations from {filepath}")
    
    allpcl = []
    missing = []
    
    for i in range(1, njobs + 1):
        pcl_file = f"{filepath}/pcljob{i:d}.npz"
        
        if check_for_file(pcl_file, f"PCL job {i}"):
            try:
                with np.load(pcl_file) as pclfile:
                    pcl_s = np.array([pclfile['pcl_e'], pclfile['pcl_b'], pclfile['pcl_eb']])
                    pcl_s = np.swapaxes(pcl_s, 0, 1)
                    allpcl.extend(pcl_s)
            except Exception as e:
                logger.error(f"Error reading PCL job {i}: {e}")
                missing.append(i)
        else:
            logger.warning(f"Missing PCL job number {i}")
            missing.append(i)
    
    if missing:
        missing_str = ",".join(map(str, missing))
        logger.warning(f"Missing PCL jobs: {missing_str}")
    
    allpcl = np.array(allpcl)
    logger.info(f"Loaded PCL data with shape: {allpcl.shape}")
    
    return allpcl


# ============================================================================
# Utility functions
# ============================================================================

def save_matrix(m, filename, kind="M"):
    print("Saving {} matrix.".format(kind))
    np.savez(filename, matrix=m)





# ============================================================================
# Batch Processing Functions
# ============================================================================

def read_posterior_files(pattern: str, regex: Optional[str] = None, 
                        flatten_posteriors: Optional[bool] = None) -> Dict[str, np.ndarray]:
    """
    Read files matching a pattern and extract posterior data.
    
    Automatically detects available fields and handles different file structures:
    - Single posterior split across many files -> flattens automatically
    - Multiple complete posteriors in separate files -> keeps as 2D array
    
    Parameters:
    -----------
    pattern : str
        Glob pattern to match files (e.g., 's8posts/s8post_*.npz')
    regex : str, optional
        Regular expression to further filter matched files
    flatten_posteriors : bool, optional
        If True, flatten all posteriors into 1D arrays
        If False, keep as 2D array (one posterior per row)
        If None (default), auto-detect based on file structure
        
    Returns:
    --------
    dict
        Dictionary containing arrays of extracted posterior data.
        Available keys depend on what's found in files:
        - 'gauss_posteriors', 'exact_posteriors' (always present if data exists)
        - 's8' (if available)
        - 'means', 'combs' (if available)
        - 'available' (boolean mask of successfully read files)
        - 'file_info' (metadata about file structure)
    """
    logger.info(f"Reading posterior files with pattern: {pattern}")
    
    files = glob.glob(pattern)
    if not files:
        logger.warning(f"No files found matching pattern: {pattern}")
        return {"available": [], "file_info": {"total_files": 0}}
    
    if regex:
        files = [f for f in files if re.search(regex, f)]
        logger.debug(f"Filtered to {len(files)} files using regex: {regex}")
    
    # Data containers
    gauss_posteriors, exact_posteriors = [], []
    s8_values, means_values, combs_values = [], [], []
    available = []
    
    # Track what fields are available
    available_fields = set()
    posterior_shapes = []
    
    for file_path in files:
        try:
            with np.load(file_path) as posts:
                # Check what fields are available
                file_fields = set(posts.files)
                available_fields.update(file_fields)
                
                # Always try to load gauss and exact
                if 'gauss' in posts and 'exact' in posts:
                    gauss_post = posts['gauss']
                    exact_post = posts['exact']
                    
                    # Track shapes for auto-detection
                    posterior_shapes.append(gauss_post.shape)
                    
                    gauss_posteriors.append(gauss_post)
                    exact_posteriors.append(exact_post)
                    
                    # Load optional fields if available
                    if 's8' in posts:
                        s8_values.append(posts['s8'])
                    
                    if 'means' in posts:
                        means_values.append(posts['means'])
                        
                    if 'comb' in posts:
                        combs_values.append(posts['comb'])
                    elif 'combs' in posts:  # Handle both naming conventions
                        combs_values.append(posts['combs'])
                    
                    available.append(True)
                    logger.debug(f"Successfully loaded: {file_path}")
                    
                else:
                    logger.warning(f"Missing required fields (gauss/exact) in {file_path}")
                    available.append(False)
                    
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            available.append(False)
    
    # Auto-detect file structure if not specified
    if flatten_posteriors is None:
        flatten_posteriors = _should_flatten_posteriors(posterior_shapes)
    
    logger.info(f"Successfully loaded {sum(available)}/{len(files)} files")
    logger.debug(f"Available fields: {sorted(available_fields)}")
    logger.debug(f"Flattening posteriors: {flatten_posteriors}")
    
    # Process the data based on flattening decision
    result = {
        "available": available,
        "file_info": {
            "total_files": len(files),
            "successful_files": sum(available),
            "available_fields": sorted(available_fields),
            "flattened": flatten_posteriors,
            "posterior_shapes": posterior_shapes
        }
    }
    
    if gauss_posteriors:
        if flatten_posteriors:
            result["gauss_posteriors"] = np.concatenate([p.flatten() for p in gauss_posteriors])
            result["exact_posteriors"] = np.concatenate([p.flatten() for p in exact_posteriors])
        else:
            result["gauss_posteriors"] = np.array(gauss_posteriors)
            result["exact_posteriors"] = np.array(exact_posteriors)
    
    # Handle optional fields
    if s8_values:
        if flatten_posteriors:
            result["s8"] = np.concatenate([s.flatten() for s in s8_values])
        else:
            result["s8"] = np.array(s8_values)
    
    if means_values:
        if flatten_posteriors:
            result["means"] = np.concatenate([m.flatten() for m in means_values])
        else:
            result["means"] = np.array(means_values)
    
    if combs_values:
        if flatten_posteriors:
            result["combs"] = np.concatenate([c.flatten() for c in combs_values])
        else:
            result["combs"] = np.array(combs_values)
    
    return result

def _should_flatten_posteriors(shapes: List[Tuple]) -> bool:
    """
    Auto-detect whether posteriors should be flattened based on their shapes.
    
    Heuristic:
    - If all arrays are short (< 5 elements), likely parts of single posterior -> flatten
    - If arrays are long and similar length, likely complete posteriors -> don't flatten
    """
    if not shapes:
        return True
    
    # Get array lengths (assuming 1D arrays)
    lengths = [np.prod(shape) for shape in shapes]
    
    # If all arrays are very short, probably parts of a single posterior
    if all(length < 5 for length in lengths):
        logger.debug(f"Detected posterior fragments (lengths: {lengths}) -> flattening")
        return True
    
    # If arrays are reasonably long and similar length, probably complete posteriors
    if len(set(lengths)) == 1 and lengths[0] > 100:  # All same length and long
        logger.debug(f"Detected complete posteriors (length {lengths[0]}) -> keeping as 2D")
        return False
    
    # If mixed lengths but all reasonably long, probably complete posteriors of different types
    if all(length > 50 for length in lengths):
        logger.debug(f"Detected complete posteriors (various lengths: {set(lengths)}) -> keeping as 2D")
        return False
    
    # Default: flatten for safety
    logger.debug(f"Uncertain structure (lengths: {lengths[:5]}{'...' if len(lengths) > 5 else ''}) -> flattening")
    return True

# Helper function for backward compatibility
def read_posterior_files_simple(pattern: str, regex: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Simplified version that always flattens for backward compatibility.
    
    Returns the old format with flattened arrays.
    """
    result = read_posterior_files(pattern, regex, flatten_posteriors=True)
    
    # Return in old format
    return {
        "gauss_posteriors": result.get("gauss_posteriors", np.array([])),
        "exact_posteriors": result.get("exact_posteriors", np.array([])),
        "s8": result.get("s8", np.array([])),
        "available": result["available"]
    }








# ============================================================================
# Deprecated Functions
# ============================================================================

def load_pdfs(*args, **kwargs):
    """DEPRECATED: Moved to legacy.file_handling_v1."""
    import warnings
    warnings.warn(
        "load_pdfs has been moved to the legacy folder. "
        "Use 'from legacy.file_handling_v1 import load_pdfs' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from legacy.file_handling_v1 import load_pdfs as legacy_load_pdfs
    return legacy_load_pdfs(*args, **kwargs)

def load_cfs(*args, **kwargs):
    """DEPRECATED: Moved to legacy.file_handling_v1."""
    import warnings
    warnings.warn(
        "load_cfs has been moved to the legacy folder. "
        "Use 'from legacy.file_handling_v1 import load_cfs' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from legacy.file_handling_v1 import load_cfs as legacy_load_cfs
    return legacy_load_cfs(*args, **kwargs)

def read_2D_cf(*args, **kwargs):
    """DEPRECATED: Moved to legacy.file_handling_v1."""
    import warnings
    warnings.warn(
        "read_2D_cf has been moved to the legacy folder. "
        "Use 'from legacy.file_handling_v1 import read_2D_cf' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    from legacy.file_handling_v1 import read_2D_cf as legacy_read_2D_cf
    return legacy_read_2D_cf(*args, **kwargs)

def read_xi_sims(*args, **kwargs):
    """DEPRECATED: Use read_sims_nd instead."""
    import warnings
    warnings.warn(
        "read_xi_sims is deprecated. Use read_sims_nd() for new analyses.",
        DeprecationWarning,
        stacklevel=2
    )
    raise NotImplementedError("Use read_sims_nd() instead - provides better functionality.")


# ============================================================================
# Covariance Matrix I/O
# ============================================================================

def save_covariance_matrix(cov_matrix: np.ndarray, filepath: str) -> None:
    """Save covariance matrix to compressed numpy file."""
    import warnings
    warnings.warn(
        "save_covariance_matrix is deprecated. Use save_arrays({'cov': matrix}, filepath) instead.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.info(f"Saving covariance matrix to: {filepath}")
    
    # Ensure directory exists
    ensure_directory_exists(os.path.dirname(filepath))
    
    np.savez_compressed(filepath, cov=cov_matrix)
    
    # Log file size
    file_size_mb = os.path.getsize(filepath) / 1024**2
    logger.info(f"Saved covariance matrix: {file_size_mb:.1f} MB")


def load_covariance_matrix(filepath: str) -> np.ndarray:
    """Load covariance matrix from numpy file."""
    import warnings
    warnings.warn(
        "load_covariance_matrix is deprecated. Use load_arrays(filepath)['cov'] instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if not check_for_file(filepath):
        raise FileNotFoundError(f"Covariance file not found: {filepath}")
    
    logger.info(f"Loading covariance matrix from: {filepath}")
    
    try:
        with np.load(filepath) as cov_file:
            cov_matrix = cov_file["cov"]
            logger.debug(f"Loaded covariance matrix shape: {cov_matrix.shape}")
            return cov_matrix.copy()  # Return copy to avoid file handle issues
    except Exception as e:
        raise RuntimeError(f"Failed to load covariance matrix from {filepath}: {e}")




# ============================================================================
# Pseudo Power Spectra I/O
# ============================================================================

def generate_pseudo_cl_filename(
    nside: int, 
    mask_name: str, 
    theory_name: str, 
    sigma_name: str, 
    base_dir: Optional[str] = None
) -> str:
    """Generate standardized filename for pseudo-Cl."""
    import warnings
    warnings.warn(
        "generate_pseudo_cl_filename is deprecated. Use generate_filename with pcl key instead.",
        DeprecationWarning,
        stacklevel=2
    )
    filename = f"pcl_n{nside:d}_{mask_name}_{theory_name}_{sigma_name}.npz"
    
    if base_dir:
        pcl_dir = os.path.join(base_dir, "pcls")
        ensure_directory_exists(pcl_dir)
        return os.path.join(pcl_dir, filename)
    
    return filename

def save_pseudo_cl(pseudo_cl_dict: Dict[str, np.ndarray], filepath: str) -> None:
    """Save pseudo power spectra to file."""
    import warnings
    warnings.warn(
        "save_pseudo_cl is deprecated. Use save_arrays with pcl dictionary instead.",
        DeprecationWarning,
        stacklevel=2
    )
    logger.info(f"Saving pseudo-Cl to: {filepath}")
    
    ensure_directory_exists(os.path.dirname(filepath))
    np.savez_compressed(filepath, **pseudo_cl_dict)
    
    file_size_mb = os.path.getsize(filepath) / 1024**2
    logger.debug(f"Saved pseudo-Cl file: {file_size_mb:.2f} MB")

def load_pseudo_cl(filepath: str) -> Optional[Dict[str, np.ndarray]]:
    """Load pseudo power spectra from file."""
    import warnings
    warnings.warn(
        "load_pseudo_cl is deprecated. Use load_arrays instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if not check_for_file(filepath):
        return None
    
    logger.info(f"Loading cached pseudo-Cl from: {filepath}")
    
    try:
        with np.load(filepath) as pcl_file:
            result = {key: pcl_file[key].copy() for key in pcl_file.files}
            logger.debug(f"Successfully loaded pseudo-Cl with keys: {list(result.keys())}")
            return result
    except Exception as e:
        logger.warning(f"Failed to load pseudo-Cl from {filepath}: {e}")
        return None


def generate_array_filename(prefix: str, lmax: int, nside: int, name: str, 
                           buffer: int = 0) -> str:
    """Generate standardized filename for array storage."""
    import warnings
    warnings.warn(
        "generate_array_filename is deprecated. Use generate_filename instead.",
        DeprecationWarning,
        stacklevel=2
    )
    if buffer > 0:
        return f"{prefix}_l{lmax + buffer}_n{nside}_{name}.npz"
    else:
        return f"{prefix}_l{lmax}_n{nside}_{name}.npz"
