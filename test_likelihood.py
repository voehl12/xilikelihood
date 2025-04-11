import pytest
import numpy as np
from likelihood import XiLikelihood
from grf_classes import SphereMask,RedshiftBin
from theory_cl import TheoryCl
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

def test_initiate_theory_cl(likelihood_instance, snapshot, regtest):
    # Define cosmological parameters
    cosmo_params = {'omega_m': 0.31, 's8': 0.8}

    # Call the method
    likelihood_instance.initiate_theory_cl(cosmo_params)

    # Capture outputs
    theory_cl = likelihood_instance._theory_cl

    # Assert that theory_cl is a list of objects with expected attributes
    assert isinstance(theory_cl, list), "theory_cl should be a list"
    assert len(theory_cl) > 0, "theory_cl should not be empty"
    for cl in theory_cl:
        assert hasattr(cl, 'lmax'), "Each theory_cl object should have an 'lmax' attribute"
        assert hasattr(cl, 'cosmo'), "Each theory_cl object should have a 'cosmo' attribute"
        assert cl.cosmo == cosmo_params, "Cosmological parameters should match the input"
        assert hasattr(cl, 'ee'), "Each theory_cl object should have an 'ee' attribute"
        assert hasattr(cl, 'sigma_e'), "Each theory_cl object should have a 'sigma_e' attribute"
        assert cl.sigma_e == likelihood_instance.noise, "sigma_e should match the noise value"

        # Write sigma_e to the regression test snapshot
        regtest.write(f"sigma_e for TheoryCl instance:\n{cl.sigma_e}\n")

        # Snapshot test for the 'ee' array
        snapshot.check(cl.ee)


