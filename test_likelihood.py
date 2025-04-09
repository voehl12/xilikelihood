import pytest
import numpy as np
from likelihood import XiLikelihood
from grf_classes import SphereMask, TheoryCl, RedshiftBin
from cov_setup import Cov

@pytest.fixture
def likelihood_instance():
    # Set up a minimal instance of XiLikelihood
    mask = SphereMask(spins=[2], circmaskattr=(1000, 256), exact_lmax=30, l_smooth=30)
    redshift_bin = RedshiftBin(5, filepath='redshift_bins/KiDS/K1000_NS_V1.0.0A_ugriZYJHKs_photoz_SG_mask_LF_svn_309c_2Dbins_v2_DIRcols_Fid_blindC_TOMO5_Nz.txt')
    theorycl = TheoryCl(mask.lmax, cosmo={'omega_m': 0.31, 's8': 0.8}, z_bins=(redshift_bin, redshift_bin))
    ang_bins_in_deg = [(0.5, 1.0), (4, 6)]

    likelihood = XiLikelihood(
        mask=mask,
        redshift_bins=[redshift_bin],
        ang_bins_in_deg=ang_bins_in_deg,
        exact_lmax=30,
        noise='default',
    )
    likelihood.initiate_mask_specific()
    likelihood.precompute_combination_matrices()
    likelihood.initiate_theory_cl({'omega_m': 0.31, 's8': 0.8})
    likelihood._prepare_matrix_products()
    return likelihood

def test_get_cfs_1d_lowell(likelihood_instance, regtest):
    # Call the method
    likelihood_instance._get_cfs_1d_lowell()

    # Capture outputs
    variances = likelihood_instance._variances
    eigvals = likelihood_instance._eigvals
    t_lowell = likelihood_instance._t_lowell
    cfs_lowell = likelihood_instance._cfs_lowell
    ximax = likelihood_instance._ximax
    ximin = likelihood_instance._ximin

    # Write outputs to the regression test snapshot
    regtest.write(f"Variances:\n{variances}\n")
    regtest.write(f"Eigenvalues:\n{eigvals}\n")
    regtest.write(f"t_lowell:\n{t_lowell}\n")
    regtest.write(f"cfs_lowell:\n{cfs_lowell}\n")
    regtest.write(f"ximax:\n{ximax}\n")
    regtest.write(f"ximin:\n{ximin}\n")