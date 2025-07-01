import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


a = np.logspace(0,2,10)
scale = np.logspace(-5,-3,10)



marginals = [stats.gamma(a=a, scale=scale) for a, scale in zip(a, scale)]
xs = [np.linspace(marginal.ppf(0.0000001), marginal.ppf(0.9999999), 1000) for marginal in marginals]
pdfs = [marginal.pdf(x) for marginal, x in zip(marginals, xs)]


for i, (marginal, x, pdf, a , scale) in enumerate(zip(marginals, xs, pdfs, a, scale)):
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(x, pdf, label=f'Marginal {i+1}: a={a}, scale={scale}')
    ax.set_title('Gamma Marginals')
    ax.set_xlabel('x')
    ax.set_ylabel('Probability Density Function (PDF)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('gamma_test_{:d}.png'.format(i),dpi=300)
    plt.close(fig)  # Close the figure to free memory


