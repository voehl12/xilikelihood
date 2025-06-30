import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from coupling_utils import simple_model, generate_covariance, copula_likelihood
from postprocess_nd_likelihood import exp_norm_mean
from make_interactive_plot_corr import interactive_plot

""" def simple_model(parameter, ndim=10):
    
    A simple model that generates an n-dimensional data vector
    based on a single parameter.

    Parameters:
    - parameter: float
        The single parameter controlling the model.
    - ndim: int
        The dimensionality of the data vector.

    Returns:
    - datavector: ndarray
        An n-dimensional data vector.
    
    # Base vector (e.g., a linear sequence)
    base_vector = 1e-3*np.linspace(1, ndim, ndim)
    
    # Apply the parameter as a scaling factor
    datavector = parameter**2 * base_vector
    
    return datavector """


# Example usage
fiducial_parameter = 5.0
ndim = 10
fiducial_datavector = simple_model(fiducial_parameter, ndim)
prior = np.linspace(0.5, 8.5, 1000)
lh = 'gamma'

correlation = 1.0
cov = generate_covariance(fiducial_datavector, correlation)





# Generate the data vector
prior_model = simple_model(prior, ndim)
# Get the marginals

all_post = copula_likelihood(fiducial_datavector, cov, prior_model, type=lh)
gaussian_direct = multivariate_normal.logpdf(prior_model, mean=fiducial_datavector, cov=cov)

# Plotting the results
fig, ax = plt.subplots(1, 1, figsize=(10, 10), sharex=True)

post_normalized_gaussian, mean_gaussian = exp_norm_mean(prior, gaussian_direct,reg=0)
post_normalized, mean = exp_norm_mean(prior, all_post,reg=0)
# Upper panel: Log-posterior comparison
ax.plot(prior, post_normalized, label='Copula Likelihood', color='blue')
ax.plot(prior, post_normalized_gaussian, label='Gaussian Likelihood', color='orange')
#ax.plot(prior, np.prod(evs_1d, axis=1), label='Product of 1D', color='green')
ax.axvline(x=fiducial_parameter, color='red', linestyle='-', label='Fiducial')
ax.axvline(x=prior[np.argmax(gaussian_direct)], color='orange', linestyle='--', label='Max Gaussian')
ax.axvline(x=prior[np.argmax(all_post)], color='blue', linestyle='-', label='Max Copula')
ax.axvline(x=mean,color='blue',linestyle=':', label='Mean Copula')
ax.axvline(x=mean_gaussian,color='orange',linestyle=':', label='Mean Gaussian')
#ax.axvline(x=prior[np.argmax(np.prod(evs_1d, axis=1))], color='green', linestyle='--', label='Max Product of 1D')
ax.set_xlim(prior[0], prior[-1])
#ax.set_ylim(0,4)
ax.set_ylabel('Posterior')
ax.legend()

# Lower panel: Difference plot
""" difference = np.array(all_post) - np.array(gaussian_direct)
ax[1].plot(prior, difference, label='Difference (Copula - Gaussian)', color='red')
ax[1].axhline(0, color='black', linestyle='--', linewidth=0.8)
ax[1].set_xlabel('Parameter')
ax[1].set_ylabel('Posterior Difference')
ax[1].set_xlim(2.8,3.5)
ax[1].set_ylim(np.min(difference), np.min(difference)*-3)
ax[1].legend() """
plt.savefig('log_posterior_{}_{:d}d_test.png'.format(lh,ndim),dpi=300)



""" plt.figure(figsize=(10, 6))
plt.axvline(x=fiducial_parameter, color='red', linestyle='-', label='Fiducial Parameter')
for i in range(evs_1d.shape[1]):
    plt.plot(prior, evs_1d[:, i], label=f'dim {i+1}')
    plt.axvline(x=prior[np.argmax(evs_1d[:, i])], color='green', linestyle='--')
plt.xlabel('Parameter')

plt.legend()
plt.savefig('evs_1d_{}_{:d}d.png'.format(lh,ndim), dpi=300) """

correlation_values = np.linspace(0, 1, 50)
n_dim_values = np.arange(1, 11)
interactive_plot(correlation_values,prior, model=simple_model,fiducial=fiducial_parameter)