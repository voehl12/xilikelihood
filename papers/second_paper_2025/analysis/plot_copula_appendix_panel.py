#!/usr/bin/env python3
"""
Focused appendix plot script for copula vs marginal changes.

Runs only the specific cases described in plan_plots.md and reruns
likelihoods for transparency (no cached data).
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from copula_analysis import analyze_single_copula
from copula_core import simple_linear_model, generate_covariance
from copula_plotting import COLORS



def run_case(param_grid, fiducial_param, n_data, corr, marginal_choice, df, vary_covariance_with_param=False):
    fiducial_data = simple_linear_model(fiducial_param, n_data_points=n_data)
    cov = generate_covariance(fiducial_data, corr)

    baseline = analyze_single_copula(
        param_grid,
        fiducial_data,
        cov,
        'gaussian',
        'normal',
        n_data,
        df=None,
        corr=corr,
        vary_covariance_with_param=vary_covariance_with_param,
    )
    coupling = analyze_single_copula(
        param_grid,
        fiducial_data,
        cov,
        'student_t',
        'normal',
        n_data,
        df=df,
        corr=corr,
        vary_covariance_with_param=vary_covariance_with_param,
    )
    marginal = analyze_single_copula(
        param_grid,
        fiducial_data,
        cov,
        'gaussian',
        marginal_choice,
        n_data,
        df=None,
        corr=corr,
        vary_covariance_with_param=vary_covariance_with_param,
    )
    both = analyze_single_copula(
        param_grid,
        fiducial_data,
        cov,
        'student_t',
        marginal_choice,
        n_data,
        df=df,
        corr=corr,
        vary_covariance_with_param=vary_covariance_with_param,
    )

    results = {
        'baseline': baseline,
        'coupling': coupling,
        'marginal': marginal,
        'both': both,
    }

    missing = [key for key, value in results.items() if value is None]
    if missing:
        raise RuntimeError(
            f"Missing results for n_data={n_data}, corr={corr}: {', '.join(missing)}"
        )

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Create 2x2 appendix plot for copula/marginal effects."
    )
    parser.add_argument('--output', default='.', help='Output directory')
    parser.add_argument('--filename', default='copula_appendix_2x2.png')
    parser.add_argument('--fiducial', type=float, default=5.0)
    parser.add_argument('--n-grid', type=int, default=1000)
    parser.add_argument('--n-data-low', type=int, default=3)
    parser.add_argument('--n-data-high', type=int, default=15)
    parser.add_argument('--corr-low', type=float, default=0.3)
    parser.add_argument('--corr-high', type=float, default=0.8)
    parser.add_argument('--marginal', choices=['lognormal', 'student_t', 'gamma'], default='gamma')
    parser.add_argument('--df', type=int, default=50)
    parser.add_argument(
        '--vary-covariance-with-param',
        action='store_true',
        help='Recompute covariance from each parameter prediction instead of using fiducial covariance.',
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    param_grid = np.linspace(-3, 15, args.n_grid)

    n_data_values = [args.n_data_low, args.n_data_high]
    corr_values = [args.corr_low, args.corr_high]

    all_cases = {}
    for n_data in n_data_values:
        for corr in corr_values:
            all_cases[(n_data, corr)] = run_case(
                param_grid,
                args.fiducial,
                n_data,
                corr,
                args.marginal,
                args.df,
                vary_covariance_with_param=args.vary_covariance_with_param,
            )

    global_max = 0.0
    for case in all_cases.values():
        for result in case.values():
            global_max = max(global_max, np.max(result['posterior']))

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True, sharey=True)

    for row, n_data in enumerate(n_data_values):
        for col, corr in enumerate(corr_values):
            ax = axes[row, col]
            case = all_cases[(n_data, corr)]

            ax.plot(
                param_grid,
                case['baseline']['posterior'],
                color='black',
                linewidth=1,
                label='Baseline (Gauss + normal)',
            )
            ax.plot(
                param_grid,
                case['coupling']['posterior'],
                color=COLORS['student_t'],
                linewidth=1,
                label='Coupling only (t + normal)',
            )
            ax.plot(
                param_grid,
                case['marginal']['posterior'],
                color=COLORS['gaussian'],
                linewidth=1,
                label=f"Marginal only (Gauss + {args.marginal})",
            )

            ax.plot(
                param_grid,
                case['both']['posterior'],
                color=COLORS['student_t'],
                linewidth=1,
                linestyle=(0, (5, 5)),
                label=f"Both (t + {args.marginal})",
            )
            ax.plot(
                param_grid,
                case['both']['posterior'],
                color=COLORS['gaussian'],
                linewidth=1,
                linestyle=(5, (5, 5)),
                label='_nolegend_',
            )

            ax.axvline(
                args.fiducial,
                color='black',
                linestyle=':',
                linewidth=1.5,
                label='_nolegend_',
            )

            for key, color, linestyle in [
                ('baseline', 'black', '-'),
                ('coupling', COLORS['student_t'], '-'),
                ('marginal', COLORS['gaussian'], '-'),
                ('both', COLORS['student_t'], (0, (5, 5))),
            ]:
                ax.axvline(
                    case[key]['maximum'],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1,
                    alpha=0.8,
                )
            ax.axvline(
                case['both']['maximum'],
                color=COLORS['gaussian'],
                linestyle=(5, (5, 5)),
                linewidth=1,
                alpha=0.8,
            )

            ax.text(
                0.98,
                0.98,
                rf"$n={n_data},\ \rho={corr:.1f}$",
                transform=ax.transAxes,
                ha='right',
                va='top',
                fontsize=10,
            )

            ax.set_xlim(param_grid[0], param_grid[-1])
            ax.set_ylim(0.0, global_max * 1.05)

    axes[1, 0].set_xlabel(r'Parameter $\theta$')
    axes[1, 1].set_xlabel(r'Parameter $\theta$')
    axes[0, 0].set_ylabel(r'Posterior $p(\theta|d)$')
    axes[1, 0].set_ylabel(r'Posterior $p(\theta|d)$')

    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 1].legend(handles, labels, frameon=False,loc='upper left')

    plt.tight_layout()
    output_path = os.path.join(args.output, args.filename)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved plot: {output_path}")


if __name__ == '__main__':
    main()
