import numpy as np

def test_cf2pdf():
    import scipy.stats as stats
    from distributions import gaussian_cf, cf_to_pdf_1d
    

    mu = 0
    sigma = 1
    val_max = mu + 10 * sigma
    dt = 0.45 * 2 * np.pi / val_max
    steps = 2048
    t0 = -0.5 * dt * (steps - 1)
    t = np.linspace(t0, -t0, steps - 1)
    cf = gaussian_cf(t, mu, sigma)
    x, pdf_from_cf = cf_to_pdf_1d(t, cf)
    pdf = stats.norm.pdf(x, mu, sigma)
    assert np.allclose(pdf, pdf_from_cf), pdf_from_cf