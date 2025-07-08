def test_normal_distribution():
    """Test with standard normal distribution (should give known results)."""
    t = np.linspace(-5, 5, 1000)
    cf = np.exp(-0.5 * t**2)  # Standard normal characteristic function
    
    moments = nth_moment(3, t, cf)
    mean = moments[0].real
    variance = moments[1].real - mean**2
    skew = skewness(t, cf)
    
    # Test known values for standard normal
    assert abs(mean) < 1e-10, f"Mean should be ~0, got {mean}"
    assert abs(variance - 1) < 1e-6, f"Variance should be ~1, got {variance}"
    assert abs(skew) < 1e-6, f"Skewness should be ~0, got {skew}"