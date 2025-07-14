import numpy as np
import xilikelihood as xlh

FIDUCIAL_COSMO = {"omega_m": 0.31, "s8": 0.8}


def create_mock_data(likelihood,mock_data_path,gaussian_covariance_path,random=None):
    """Create mock data for the likelihood analysis."""
    
    theory_cls = likelihood.initiate_theory_cl(FIDUCIAL_COSMO)
    likelihood.initiate_mask_specific()
    likelihood.precompute_combination_matrices()
    likelihood._prepare_matrix_products()

    likelihood.get_covariance_matrix_lowell()
    likelihood.get_covariance_matrix_highell()
    likelihood._get_means_highell()

    gaussian_covariance = likelihood._cov_highell + likelihood._cov_lowell
    fiducial_mean = likelihood._means_highell + likelihood._means_lowell
    if random is None:
        mock_data = fiducial_mean
    elif random == 'gaussian':
        mock_data = np.random.multivariate_normal(
            mean=likelihood._means_highell + likelihood._means_lowell,
            cov=likelihood._cov_highell + likelihood._cov_lowell,
            size=1,
        )
    elif random == 'frommap':
        sim = xlh.simulate_correlation_functions(
        theory_cls,
        [likelihood.mask],
        likelihood.ang_bins_in_deg,
        job_id=42,
        lmin=0,
        lmax=likelihood.mask.lmax,
        save_pcl=True,
    )
        mock_data = sim[0, :, 0, :]

    else:
        raise ValueError("Invalid random option. Choose 'gaussian' or 'frommap'.")
    # Save the covariance matrix and mock data to files
    np.savez(gaussian_covariance_path,cov=gaussian_covariance,s8=FIDUCIAL_COSMO['s8'])
    np.savez(mock_data_path,data=mock_data,s8=FIDUCIAL_COSMO['s8'])        
    print("Mock data and covariance matrix saved to {} and {}.".format(mock_data_path,gaussian_covariance_path))


if __name__ == "__main__":
    # Example usage
    likelihood = xlh.XiLikelihood(
        mask=xlh.SphereMask(spins=[2], circmaskattr=(1000, 256)),
        redshift_bins=[xlh.RedshiftBin(nbin=1, z=np.linspace(0.01, 3.0, 100), zmean=0.5, zsig=0.1)],
        angular_bins_in_deg=[(1.0, 2.0), (2.0, 4.0)],
        include_ximinus=True,
    )
    
    mock_data_path = 'mock_data.npz'
    gaussian_covariance_path = 'gaussian_covariance.npz'
    
    create_mock_data(likelihood, mock_data_path, gaussian_covariance_path, random='gaussian')