import numpy as np

def validate_moments_calculation():
    import distributions
    import theoretical_moments as moments

    steps = 2048
    cov = np.random.random((10, 10))
    m = np.random.random(10)

    mean_trace = np.trace(m[:, None] * cov)
    var_trace = 2 * np.trace(m[:, None] * cov @ m[:, None] * cov)

    ximax = mean_trace + 10 * np.sqrt(var_trace)
    t, cf = distributions.calc_quadcf_1D(ximax, steps, cov, m, is_diag=True)
    x_low, pdf_low = distributions.cf_to_pdf_1d(t, cf)
    mean_lowell_pdf = np.trapz(x_low * pdf_low, x=x_low)
    var_lowell_pdf = np.trapz(x_low**2 * pdf_low, x=x_low)
    mean_lowell_cf, var_lowell_cf = moments.nth_moment(2, t, cf)

    assert np.allclose(mean_trace, mean_lowell_cf, rtol=1e-2), (mean_trace, mean_lowell_cf)
    assert np.allclose(var_trace, var_lowell_cf, rtol=1e-2), (var_trace, var_lowell_cf)
    assert np.allclose(mean_lowell_cf, mean_lowell_pdf, rtol=1e-2), (
        mean_lowell_cf,
        mean_lowell_pdf,
    )
    assert np.allclose(var_lowell_pdf, var_lowell_cf, rtol=1e-2)

def validate_hermite_polynomials():
    import scipy.stats as stats
    from edgeworth_experiments import MultiNormalExpansion
    import itertools

    point_3d = np.random.random(3)
    mu = np.zeros(3)
    cov = np.eye(3)
    expansion = MultiNormalExpansion([mu, cov])
    points_3d = [point_3d]
    first_order = expansion.hermite_nd(points_3d, 1)
    second_order = expansion.hermite_nd(points_3d, 2)
    third_order = expansion.hermite_nd(points_3d, 3)
    pdf = stats.multivariate_normal.pdf(point_3d, mean=mu, cov=cov)

    def multivariate_gaussian_derivatives(x, mu, sigma):
        """
        Compute the gradient and Hessian of the multivariate Gaussian PDF.

        Parameters:
        x (np.ndarray): Point at which to evaluate the derivatives.
        mu (np.ndarray): Mean vector of the Gaussian distribution.
        sigma (np.ndarray): Covariance matrix of the Gaussian distribution.

        Returns:
        gradient (np.ndarray): Gradient of the Gaussian PDF at x.
        hessian (np.ndarray): Hessian of the Gaussian PDF at x.
        """
        k = len(mu)
        sigma_inv = np.linalg.inv(sigma)
        diff = x - mu
        pdf = stats.multivariate_normal.pdf(x, mean=mu, cov=sigma)

        # Gradient
        gradient = -pdf * sigma_inv @ diff

        # Hessian
        hessian = pdf * (sigma_inv @ np.outer(diff, diff) @ sigma_inv - sigma_inv)

        return gradient, hessian

    gradient_3d, hessian_3d, third_3d = multivariate_gaussian_derivatives(point_3d, mu, cov)
    gradient_hermite, hessian_hermite, third_hermite = (
        -first_order[0] * pdf,
        second_order[0] * pdf,
        -third_order[0] * pdf,
    )
    assert np.allclose(gradient_3d, gradient_hermite)
    assert np.allclose(hessian_3d, hessian_hermite)
    assert np.allclose(third_3d, third_hermite)


def validate_edgeworth_expansion():
    import scipy.stats as stats
    import postprocess_nd_likelihood
    from theoretical_moments import convert_moments_to_cumulants_nd
    from edgeworth_experiments import MultiNormalExpansion
    import matplotlib.pyplot as plt

    def compare_edgeworth_t_distribution(
        mu, Sigma, nu_values, n_samples=1000000, n_test_points=5000
    ):
        """
        Compare the Edgeworth expansion and the multivariate t-distribution for increasing values of nu.

        Parameters
        ----------
        mu : array-like
            Mean vector of the distributions.
        Sigma : array-like
            Covariance matrix of the distributions.
        nu_values : list
            List of degrees of freedom to compare.
        n_samples : int, optional
            Number of samples to generate from the multivariate t-distribution.
        n_test_points : int, optional
            Number of test points to evaluate the PDFs.

        Returns
        -------
        None
        """
        fig, axes = plt.subplots(len(nu_values), 2, figsize=(10, 6 * len(nu_values)))

        for idx, nu in enumerate(nu_values):
            # Generate samples from the multivariate t-distribution
            samples = stats.multivariate_t.rvs(loc=mu, shape=Sigma, df=nu, size=n_samples).T
            moments = postprocess_nd_likelihood.get_stats_from_sims(samples, [1, 2, 3])

            cumulants = convert_moments_to_cumulants_nd(moments)
            analytical_mean = mu
            analytical_cov = (nu / (nu - 2)) * Sigma

            edgeworth_expansion = MultiNormalExpansion(cumulants)
            third_cumulant_normalized = edgeworth_expansion.normalize_third_cumulant()
            print(f"Normalized third cumulant for nu={nu}: {third_cumulant_normalized}")
            gaussian = stats.multivariate_normal(mean=mu, cov=Sigma)
            test_points = stats.uniform.rvs(size=(n_test_points, len(mu)), loc=-5, scale=10)
            edgeworth_pdf_values = edgeworth_expansion.pdf(test_points)
            t_dist_pdf_values = stats.multivariate_t.pdf(test_points, loc=mu, shape=Sigma, df=nu)
            gaussian_pdf_values = gaussian.pdf(test_points)
            empirical_mean = np.mean(samples.T, axis=0)
            empirical_cov = np.cov(samples.T, rowvar=False)

            analytical_mean = mu
            analytical_cov = (nu / (nu - 2)) * Sigma

            print(f"Empirical Mean for nu={nu}:")
            print(empirical_mean)
            print(f"Analytical Mean for nu={nu}:")
            print(analytical_mean)
            print(f"Empirical Covariance for nu={nu}:")
            print(empirical_cov)
            print(f"Analytical Covariance for nu={nu}:")
            print(analytical_cov)

            # Optionally, you can assert that the values are close (within some tolerance)
            assert np.allclose(
                empirical_mean, analytical_mean, atol=1e-1
            ), f"Mean values do not match within tolerance for nu={nu}"
            assert np.allclose(
                empirical_cov, analytical_cov, atol=1e-1
            ), f"Covariance values do not match within tolerance for nu={nu}"

            ax1 = axes[idx, 0] if len(nu_values) > 1 else axes[0]
            scatter1 = ax1.scatter(
                test_points[:, 0],
                test_points[:, 1],
                c=(edgeworth_pdf_values - t_dist_pdf_values) / t_dist_pdf_values,
                cmap="viridis",
                marker="o",
                vmin=-0.4,
                vmax=0.4,
            )
            ax1.set_xlabel("X1")
            ax1.set_ylabel("X2")
            ax1.set_title(f"Edgeworth vs t-Distribution for nu={nu}")
            fig.colorbar(scatter1, ax=ax1, label="Fractional PDF Value Difference")

            ax2 = axes[idx, 1] if len(nu_values) > 1 else axes[1]
            scatter2 = ax2.scatter(
                test_points[:, 0],
                test_points[:, 1],
                c=(gaussian_pdf_values - t_dist_pdf_values) / t_dist_pdf_values,
                cmap="viridis",
                marker="o",
                vmin=-0.4,
                vmax=0.4,
            )
            ax2.set_xlabel("X1")
            ax2.set_ylabel("X2")
            ax2.set_title(f"Gaussian vs t-Distribution for nu={nu}")
            fig.colorbar(scatter2, ax=ax2, label="Fractional PDF Value Difference")

            textstr = f"Normalized third cumulant:\n{third_cumulant_normalized}"
            props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
            ax1.text(
                0.05,
                0.95,
                textstr,
                transform=ax1.transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=props,
            )

            error_edgeworth = np.mean(np.abs(edgeworth_pdf_values - t_dist_pdf_values))
            error_gaussian = np.mean(np.abs(gaussian_pdf_values - t_dist_pdf_values))
            print(f"Mean absolute error for Edgeworth expansion: {error_edgeworth}")
            print(f"Mean absolute error for Gaussian distribution: {error_gaussian}")

        plt.tight_layout()
        plt.show()

    # Example usage
    mu = np.array([0, 0])
    Sigma = np.array([[1, 0.5], [0.5, 1]])
    nu_values = [5, 10, 20, 50, 100]  # List of degrees of freedom to compare
    compare_edgeworth_t_distribution(mu, Sigma, nu_values)