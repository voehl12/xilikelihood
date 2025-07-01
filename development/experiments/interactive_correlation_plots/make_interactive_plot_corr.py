import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.stats import multivariate_normal, gamma
# Import for numerical integration
import plotly.graph_objects as go
from coupling_utils import generate_covariance, copula_likelihood
from postprocess_nd_likelihood import exp_norm_mean


def interactive_plot(correlation_values, prior, model, fiducial=5.0):
    # Initial parameters

    # Create the initial plot
    fig = go.Figure()

    # Generate initial covariance and likelihood
    correlation = correlation_values[0]
    datavector = model(fiducial)
    prior_model = model(prior)
    cov = generate_covariance(datavector, correlation)
    posterior_vals = copula_likelihood(datavector, cov, prior_model)
    posterior_vals, mean_val = exp_norm_mean(prior, posterior_vals, reg=0)  # Normalize using integral
    gaussian_posterior = multivariate_normal.logpdf(prior_model, mean=datavector, cov=cov)
    gaussian_posterior, mean_gaussian = exp_norm_mean(prior, gaussian_posterior, reg=0)  # Normalize using integral
    max_gaussian = prior[np.argmax(gaussian_posterior)]
    # Add the Gaussian posterior curve
    fig.add_trace(go.Scatter(x=[fiducial, fiducial], y=[0, max(gaussian_posterior)], mode="lines", name="Fiducial", line=dict(color="red")))
    fig.add_trace(go.Scatter(x=prior, y=gaussian_posterior, mode="lines", name="Posterior from Gaussian likelihood", line=dict(color="orange")))
    fig.add_trace(
        go.Scatter(
            x=[mean_gaussian, mean_gaussian],
            y=[0, max(gaussian_posterior)],
            mode="lines",
            line=dict(color="orange", dash="dash"),
            name="Gaussian Mean",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[max_gaussian, max_gaussian],
            y=[0, max(gaussian_posterior)],
            mode="lines",
            line=dict(color="orange", dash="dot"),
            name="Gaussian Max",
        )
    )
    # Compute mean and max values
    max_val = prior[np.argmax(posterior_vals)]

    # Add the posterior curve
    fig.add_trace(go.Scatter(x=prior, y=posterior_vals, mode="lines", name="Posterior from Gamma likelihood", line=dict(color="blue")))

    # Add vertical lines for mean and maximum
    fig.add_trace(
        go.Scatter(
            x=[mean_val, mean_val],
            y=[0, max(posterior_vals)],
            mode="lines",
            line=dict(color="blue", dash="dash"),
            name="Mean",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[max_val, max_val],
            y=[0, max(posterior_vals)],
            mode="lines",
            line=dict(color="blue", dash="dot"),
            name="Max",
        )
    )

    # Create frames for the slider
    frames = []
    max_y = 0  # Track the maximum y-value for adjusting the y-axis
    for corr in correlation_values:
        cov = generate_covariance(datavector, corr)
        posterior_vals = copula_likelihood(datavector, cov, prior_model)
        posterior_vals, mean_val = exp_norm_mean(prior, posterior_vals, reg=0)  # Normalize using integral
        max_val = prior[np.argmax(posterior_vals)]
        gaussian_posterior = multivariate_normal.logpdf(prior_model, mean=datavector, cov=cov)
        gaussian_posterior, mean_gaussian = exp_norm_mean(prior, gaussian_posterior, reg=0)  # Normalize using integral
        max_gaussian = prior[np.argmax(gaussian_posterior)]
        # Update the maximum y-value for dynamic y-axis adjustment
        max_y = max(max(max_y, max(posterior_vals)), max(gaussian_posterior))
        # Create a frame for the current correlation

        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=[fiducial, fiducial],
                        y=[0, max_y * 1.1],
                        mode="lines",
                        name="Fiducial",
                        line=dict(color="red"),
                    ),
                    go.Scatter(
                        x=prior, y=gaussian_posterior, mode="lines", name="Posterior from Gaussian likelihood", line=dict(color="orange")
                    ),
                    go.Scatter(
                        x=[mean_gaussian, mean_gaussian],
                        y=[0, max(gaussian_posterior)],
                        mode="lines",
                        line=dict(color="orange", dash="dash"),
                        name=f"Mean: {mean_gaussian:.2f}",
                    ),
                    go.Scatter(
                        x=[max_gaussian, max_gaussian],
                        y=[0, max(gaussian_posterior)],
                        mode="lines",
                        line=dict(color="orange", dash="dot"),
                        name=f"Max: {max_gaussian:.2f}",
                    ),
                    go.Scatter(
                        x=prior, y=posterior_vals, mode="lines", name="Posterior from Gamma likelihood", line=dict(color="blue")
                    ),
                    go.Scatter(
                        x=[mean_val, mean_val],
                        y=[0, max(posterior_vals)],
                        mode="lines",
                        line=dict(color="blue", dash="dash"),
                        name=f"Mean: {mean_val:.2f}",
                    ),
                    go.Scatter(
                        x=[max_val, max_val],
                        y=[0, max(posterior_vals)],
                        mode="lines",
                        line=dict(color="blue", dash="dot"),
                        name=f"Max: {max_val:.2f}",
                    ),
                ],
                name=f"corr={corr:.2f}",
            )
        )

    # Add frames to the figure
    fig.frames = frames

    # Add slider for correlation
    sliders = [
        {
            "steps": [
                {
                    "args": [
                        [f"corr={corr:.2f}"],
                        {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"},
                    ],
                    "label": f"{corr:.2f}",
                    "method": "animate",
                }
                for corr in correlation_values
            ],
            "currentvalue": {"prefix": "Correlation: ", "font": {"size": 16}},
            "x": 0.1,
            "xanchor": "left",
            "y": -0.2,  # Position the slider higher
            "yanchor": "top",
        },
    ]

    # Update layout with slider and dynamic y-axis
    fig.update_layout(
        title="Effect of Covariance Structure on Posterior Distribution",
        xaxis_title="Parameter",
        yaxis_title="Posterior",
        yaxis=dict(range=[0, max_y * 1.1]),  # Adjust y-axis to fit the maximum posterior
        sliders=sliders,
        updatemenus=[
            {
                "buttons": [
                    {
                        "args": [
                            None,
                            {"frame": {"duration": 300, "redraw": True}, "mode": "immediate"},
                        ],
                        "label": "Play",
                        "method": "animate",
                    },
                    {
                        "args": [
                            [None],
                            {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                        ],
                        "label": "Pause",
                        "method": "animate",
                    },
                ],
                "direction": "left",
                "pad": {"r": 10, "t": 87},
                "showactive": False,
                "type": "buttons",
                "x": 0.1,
                "xanchor": "right",
                "y": -0.3,  # Position the button below the slider
                "yanchor": "top",
            }
        ],
    )

    fig.write_html("interactive_plot_corr.html")



