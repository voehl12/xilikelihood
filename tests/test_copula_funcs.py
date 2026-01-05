"""
Tests for copula_funcs module.

Tests cover:
- Gaussian and Student-t copula density functions
- PDF/CDF utilities and interpolation
- Covariance/correlation matrix conversions
- Joint PDF computation
"""

import numpy as np
import pytest
from scipy.stats import norm, t, multivariate_normal
import xilikelihood as xlh
from xilikelihood import copula_funcs


@pytest.fixture
def simple_covariance():
    """2x2 covariance matrix with correlation 0.5"""
    return np.array([[1.0, 0.5], [0.5, 1.0]])


@pytest.fixture
def simple_cdfs():
    """Simple CDF values for testing"""
    return np.array([0.3, 0.7])


class TestCovarianceCorrelation:
    """Test covariance/correlation matrix conversions."""
    
    def test_covariance_to_correlation_identity(self):
        """Correlation matrix from correlation matrix should be unchanged."""
        corr = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = copula_funcs.covariance_to_correlation(corr)
        assert np.allclose(result, corr)
    
    def test_covariance_to_correlation_diagonal_ones(self):
        """Diagonal should always be 1.0."""
        cov = np.array([[4.0, 2.0], [2.0, 9.0]])
        result = copula_funcs.covariance_to_correlation(cov)
        assert np.allclose(np.diag(result), 1.0)
    
    def test_covariance_to_correlation_symmetric(self):
        """Result should be symmetric."""
        cov = np.array([[2.0, 1.0], [1.0, 3.0]])
        result = copula_funcs.covariance_to_correlation(cov)
        assert np.allclose(result, result.T)
    
    def test_covariance_to_correlation_3x3(self):
        """Test with 3x3 matrix."""
        cov = np.array([
            [1.0, 0.5, 0.3],
            [0.5, 2.0, 0.8],
            [0.3, 0.8, 1.5]
        ])
        result = copula_funcs.covariance_to_correlation(cov)
        assert result.shape == (3, 3)
        assert np.allclose(np.diag(result), 1.0)
        assert np.allclose(result, result.T)


class TestGaussianCopula:
    """Test Gaussian copula functions."""
    
    def test_gaussian_copula_point_density_independence(self):
        """With zero correlation, copula density should be zero (in log space)."""
        cov = np.eye(2)
        cdfs = np.array([0.5, 0.5])
        
        log_density = copula_funcs.gaussian_copula_point_density(cdfs, cov)
        
        # For independent variables, copula density = 1, log density = 0
        assert np.isclose(log_density, 0.0, atol=1e-10)
    
    def test_gaussian_copula_point_density_shape(self, simple_covariance, simple_cdfs):
        """Output should be scalar."""
        result = copula_funcs.gaussian_copula_point_density(simple_cdfs, simple_covariance)
        assert np.isscalar(result)
    
    def test_gaussian_copula_point_density_finite(self, simple_covariance):
        """Result should be finite for valid inputs."""
        cdfs = np.array([0.2, 0.8])
        result = copula_funcs.gaussian_copula_point_density(cdfs, simple_covariance)
        assert np.isfinite(result)
    
    def test_gaussian_copula_point_density_symmetric(self):
        """Swapping dimensions with same correlation should give same result."""
        cov = np.array([[1.0, 0.6], [0.6, 1.0]])
        cdfs1 = np.array([0.3, 0.7])
        cdfs2 = np.array([0.7, 0.3])
        
        result1 = copula_funcs.gaussian_copula_point_density(cdfs1, cov)
        result2 = copula_funcs.gaussian_copula_point_density(cdfs2, cov)
        
        assert np.isclose(result1, result2, rtol=1e-10)
    
    def test_gaussian_copula_density_batch(self):
        """Test batch processing with multiple points."""
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        # Shape: (2 bins, 5 points)
        cdfs = np.array([
            [0.1, 0.3, 0.5, 0.7, 0.9],
            [0.2, 0.4, 0.5, 0.6, 0.8]
        ])
        
        result = copula_funcs.gaussian_copula_density(cdfs, cov)
        
        # Creates meshgrid: 5x5 = 25 points
        assert result.shape == (25,)
        assert np.all(np.isfinite(result))


class TestStudentTCopula:
    """Test Student-t copula functions."""
    
    def test_student_t_copula_density_shape(self):
        """Output shape should match number of grid points."""
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        cdfs = np.array([
            [0.1, 0.5, 0.9],
            [0.2, 0.5, 0.8]
        ])
        df = 10.0
        
        result = copula_funcs.student_t_copula_density(cdfs, cov, df)
        
        # Creates meshgrid: 3x3 = 9 points
        assert result.shape == (9,)
        assert np.all(np.isfinite(result))
    
    def test_student_t_copula_density_independence(self):
        """With zero correlation, should behave like independent variables."""
        cov = np.eye(2)
        cdfs = np.array([[0.5], [0.5]])
        df = 10.0
        
        result = copula_funcs.student_t_copula_density(cdfs, cov, df)
        
        # For independent variables, copula log-density contribution should be near zero
        # Note: Student-t has heavier tails, so not exactly zero
        assert np.isfinite(result[0])
        assert np.abs(result[0]) < 1.0  # Reasonable bound
    
    def test_student_t_copula_density_finite(self):
        """Result should be finite for valid inputs."""
        cov = np.array([[1.0, 0.6], [0.6, 1.0]])
        cdfs = np.array([[0.3, 0.7], [0.4, 0.6]])
        df = 5.0
        
        result = copula_funcs.student_t_copula_density(cdfs, cov, df)
        
        assert np.all(np.isfinite(result))
    
    def test_student_t_vs_gaussian_high_df(self):
        """With high df, Student-t should approach Gaussian."""
        # Use low correlation to avoid numerical issues with Student-t
        cov = np.array([[1.0, 0.1], [0.1, 1.0]])
        # Use CDFs away from boundaries to avoid numerical issues
        cdfs = np.array([[0.3, 0.7], [0.4, 0.6]])
        
        # High df (not too high to avoid gamma function issues)
        result_student = copula_funcs.student_t_copula_density(cdfs, cov, df=500.0)
        result_gaussian = copula_funcs.gaussian_copula_density(cdfs, cov)
        
        # Should be similar with low correlation
        # Filter out any NaN values if they occur
        mask = np.isfinite(result_student) & np.isfinite(result_gaussian)
        assert np.any(mask), "All values are NaN"
        # Check that they're similar (within 10% with low correlation)
        assert np.allclose(result_student[mask], result_gaussian[mask], rtol=0.1)


class TestPDFCDFUtilities:
    """Test PDF/CDF interpolation and evaluation."""
    
    def test_pdf_to_cdf_normalization(self):
        """CDF should end at 1.0."""
        # Create simple Gaussian PDF
        x = np.linspace(-5, 5, 100)
        pdf = norm.pdf(x, 0, 1)
        
        xs = np.tile(x, (2, 1, 1))  # Shape (2, 1, 100)
        pdfs = np.tile(pdf, (2, 1, 1))
        
        cdfs, pdfs_interp, xs_interp = copula_funcs.pdf_to_cdf(xs, pdfs, num_points=200)
        
        # Final CDF value should be close to 1
        assert np.allclose(cdfs[:, :, -1], 1.0, atol=1e-3)
    
    def test_pdf_to_cdf_monotonic(self):
        """CDF should be monotonically increasing."""
        x = np.linspace(-3, 3, 50)
        pdf = norm.pdf(x, 0, 1)
        
        # Correct: xs should be x-values, pdfs should be PDF values
        xs = x.reshape(1, 1, -1)
        pdfs = pdf.reshape(1, 1, -1)
        
        cdfs, _, _ = copula_funcs.pdf_to_cdf(xs, pdfs, num_points=100)
        
        # Check monotonicity
        assert np.all(np.diff(cdfs[0, 0, :]) >= -1e-10)  # Allow tiny numerical errors
    
    def test_interpolate_and_evaluate_at_data(self):
        """Interpolation at data points should match original."""
        x = np.linspace(0, 10, 20)
        pdf = np.exp(-x)  # Simple exponential
        
        # Shape: (n_corr, n_ang, n_points) = (1, 1, 20)
        xs = x.reshape(1, 1, -1)
        pdfs = pdf.reshape(1, 1, -1)
        # x_data shape should be (n_corr, n_ang) = (1, 1)
        # Evaluating at 3 separate points requires 3 separate calls or different function
        # For this test, evaluate at a single point
        x_data = x[10].reshape(1, 1)  # Single point, shape (1, 1)
        
        result = copula_funcs.interpolate_and_evaluate(x_data, xs, pdfs)
        expected = pdf[10].reshape(1, 1)
        
        assert np.allclose(result, expected, rtol=1e-2)  # Interpolation has some error


class TestJointPDF:
    """Test joint PDF computation."""
    
    def test_joint_logpdf_2d_shape(self):
        """2D joint PDF should have correct shape."""
        # Create simple 2D PDFs
        x1 = np.linspace(-3, 3, 50)
        x2 = np.linspace(-3, 3, 50)
        pdf1 = norm.pdf(x1, 0, 1)
        pdf2 = norm.pdf(x2, 0, 1)
        
        # Shape: (1 redshift combo, 2 angular bins, 50 points)
        pdfs = np.array([[pdf1, pdf2]])
        
        # CDFs
        cdf1 = norm.cdf(x1, 0, 1)
        cdf2 = norm.cdf(x2, 0, 1)
        cdfs = np.array([[cdf1, cdf2]])
        
        # Covariance (2x2 for 2 dimensions)
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        
        result = copula_funcs.joint_logpdf(cdfs, pdfs, cov, copula_type="gaussian")
        
        # Should be 50x50 grid
        assert result.shape == (50, 50)
        assert np.all(np.isfinite(result))
    
    def test_joint_logpdf_independence(self):
        """With zero correlation, joint should be sum of marginals."""
        x = np.linspace(-3, 3, 30)
        pdf = norm.pdf(x, 0, 1)
        cdf = norm.cdf(x, 0, 1)
        
        pdfs = np.array([[pdf, pdf]])
        cdfs = np.array([[cdf, cdf]])
        cov = np.eye(2)  # Independent
        
        result = copula_funcs.joint_logpdf(cdfs, pdfs, cov, copula_type="gaussian")
        
        # For independent Gaussians, joint log PDF at (x_i, x_j) should be 
        # log(pdf(x_i)) + log(pdf(x_j))
        log_pdf = np.log(pdf)
        expected = log_pdf[:, None] + log_pdf[None, :]
        
        assert np.allclose(result, expected, rtol=1e-5)
    
    def test_joint_logpdf_student_t(self):
        """Test Student-t copula version."""
        x = np.linspace(-3, 3, 20)
        pdf = norm.pdf(x, 0, 1)
        cdf = norm.cdf(x, 0, 1)
        
        pdfs = np.array([[pdf, pdf]])
        cdfs = np.array([[cdf, cdf]])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        result = copula_funcs.joint_logpdf(cdfs, pdfs, cov, 
                                           copula_type="student-t", df=10.0)
        
        assert result.shape == (20, 20)
        assert np.all(np.isfinite(result))


class TestMatrixUtilities:
    """Test matrix conditioning and utilities."""
    
    def test_get_well_conditioned_matrix_eigenvalues(self):
        """All eigenvalues should be >= min_eigenvalue."""
        # Create poorly conditioned matrix
        corr = np.array([[1.0, 0.999], [0.999, 1.0]])
        min_eig = 0.01
        
        result = copula_funcs.get_well_conditioned_matrix(corr, min_eigenvalue=min_eig)
        
        eigenvalues = np.linalg.eigvalsh(result)
        assert np.all(eigenvalues >= min_eig - 1e-10)
    
    def test_get_well_conditioned_matrix_preserves_good_matrix(self):
        """Well-conditioned matrix should be mostly unchanged."""
        corr = np.array([[1.0, 0.3], [0.3, 1.0]])
        
        result = copula_funcs.get_well_conditioned_matrix(corr, min_eigenvalue=1e-4)
        
        assert np.allclose(result, corr, rtol=1e-3)
    
    def test_meshgrid_and_recast_2d(self):
        """Test meshgrid utility for 2D case."""
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        
        result = copula_funcs.meshgrid_and_recast([x, y])
        
        # Should be (3*2, 2) = (6, 2)
        assert result.shape == (6, 2)
        
        # Check a few values
        assert np.allclose(result[0], [1, 4])
        assert np.allclose(result[-1], [3, 5])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_gaussian_copula_extreme_correlation(self):
        """Test with very high correlation (near singular)."""
        cov = np.array([[1.0, 0.95], [0.95, 1.0]])
        cdfs = np.array([0.5, 0.5])
        
        result = copula_funcs.gaussian_copula_point_density(cdfs, cov)
        
        # Should handle near-singular matrix
        assert np.isfinite(result)
    
    def test_student_t_low_df(self):
        """Test Student-t with low degrees of freedom."""
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        cdfs = np.array([[0.3, 0.7], [0.4, 0.6]])
        
        result = copula_funcs.student_t_copula_density(cdfs, cov, df=3.0)
        
        assert np.all(np.isfinite(result))
    
    def test_cdf_boundary_values(self):
        """Test copula with CDF values near 0 or 1."""
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        
        # Clip to avoid exactly 0 or 1 (which would give infinite quantiles)
        cdfs = np.array([0.01, 0.99])
        
        result = copula_funcs.gaussian_copula_point_density(cdfs, cov)
        
        assert np.isfinite(result)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
