import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

from xilikelihood import generate_theory_cl, prepare_theory_cl_inputs, fiducial_dataspace, SphereMask
from xilikelihood.simulate import simulate_correlation_functions
from random import randint
from time import time, sleep
import sys
import logging

# Configure logging
def setup_logging(job_id, log_dir="logs"):
    """Set up logging to file and console."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"xi_sim_job_{job_id}.log")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler - important messages only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized for job {job_id}")
    logger.info(f"Log file: {log_file}")
    return logger


sleep(randint(1,10))

jobnumber = int(sys.argv[1])

# Set up logging for this job
logger = setup_logging(jobnumber)
logger.info(f"Starting simulation job {jobnumber}")

rootfolder = '/cluster/home/veoehl/xilikelihood/'
""" rootfolder = '/cluster/home/veoehl/xilikelihood/'
cl_55 = rootfolder+"Cl_3x2pt_kids55.txt"
cl_53 = rootfolder+"Cl_3x2pt_kids53.txt"
cl_33 = rootfolder+"Cl_3x2pt_kids33.txt"
cl_paths = (cl_33,cl_55,cl_53)
cl_names = ('3x2pt_kids_33','3x2pt_kids_55','3x2pt_kids_53')
batchsize = 1000
noise_contribs = ('default','default',None)
seps_in_deg = [(1,2),(4, 6),(7,10)] """
redshift_bins, angular_separation_bins = fiducial_dataspace()
n_redshift_bins = len(redshift_bins)
cosmo = {'omega_m': 0.31, 's8': 0.8}

logger.info(f"Using {n_redshift_bins} redshift bins")
logger.info(f"Cosmology: {cosmo}")

mask = SphereMask(spins=[2], circmaskattr=(10000, 256), l_smooth=30, exact_lmax=20)
logger.info(f"Created mask with effective area: {mask.eff_area:.1f} sq deg")

sim_lmax = mask.lmax
numerical_combinations, redshift_bin_combinations, is_cov_cross, shot_noise, mapper = prepare_theory_cl_inputs(redshift_bins, noise=None)

logger.info("Generating theory power spectra...")
theory_cls = generate_theory_cl(lmax=sim_lmax, redshift_bin_combinations=redshift_bin_combinations,shot_noise=shot_noise,cosmo=cosmo)
logger.info(f"Generated {len(theory_cls)} theory spectra")

xi_lmax = 20
batchsize = 1000
logger.info(f"Simulation parameters: lmax={xi_lmax}, batch_size={batchsize}")

if jobnumber == 1:
    plot = True
    logger.info("Diagnostic plots will be generated for this job")
else:
    plot = False

logger.info("Starting correlation function simulation...")
simulate_correlation_functions(theory_cls,[mask], angular_separation_bins,job_id=jobnumber,lmin=0,lmax=xi_lmax,plot_diagnostics=plot,save_pcl=True,field_type='gaussian',method='pcl_estimator',n_batch=batchsize,run_name="KiDS_setup",save_path="/cluster/scratch/veoehl/xi_sims")

logger.info(f"Job {jobnumber} completed successfully")