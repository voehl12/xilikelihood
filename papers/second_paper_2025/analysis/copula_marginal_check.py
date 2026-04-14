#!/usr/bin/env python3
"""
Small marginal recovery check using point-evaluated copula densities.

This mirrors the spirit of marginal_recovery.py but uses the simple copula
analysis setup (simple_linear_model + generate_covariance) and evaluates a
2D grid for selected dimension pairs.
"""

import argparse
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t, lognorm, gamma, beta
from scipy.integrate import simpson

import xilikelihood as xlh
from copula_core import simple_linear_model, generate_covariance


logger = logging.getLogger(__name__)


def setup_logging(log_path=None, log_level='INFO'):
    level = getattr(logging, log_level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Reset handlers to avoid duplicated logs when rerunning in interactive sessions.
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def _marginal_pdf_cdf(x_vals, pred, std, marginal_type):
    if marginal_type == 'normal':
        pdf = norm.pdf(x_vals, loc=pred, scale=std)
        cdf = norm.cdf(x_vals, loc=pred, scale=std)
        return pdf, cdf

    if marginal_type == 'student_t':
        nu = 5
        scale = std * np.sqrt((nu - 2) / nu)
        pdf = t.pdf(x_vals, df=nu, loc=pred, scale=scale)
        cdf = t.cdf(x_vals, df=nu, loc=pred, scale=scale)
        return pdf, cdf

    if marginal_type == 'lognormal':
        sigma = 0.7
        mu = np.log(max(pred, 1e-10)) - 0.5 * sigma**2
        pdf = lognorm.pdf(x_vals, s=sigma, scale=np.exp(mu))
        cdf = lognorm.cdf(x_vals, s=sigma, scale=np.exp(mu))
        return pdf, cdf

    if marginal_type == 'gamma':
        shape = (pred / std) ** 2
        scale = (std ** 2) / pred
        pdf = gamma.pdf(x_vals, a=shape, scale=scale)
        cdf = gamma.cdf(x_vals, a=shape, scale=scale)
        return pdf, cdf
    
    if marginal_type == 'uniform':
        pdf = np.ones_like(x_vals)
        cdf = x_vals
        return pdf, cdf

    raise ValueError(f"Unknown marginal type: {marginal_type}")


def _make_grid(pred, std, n_grid, marginal_type, span=10.0, eps=1e-3,grid_spacing='uniform'):
    if marginal_type == 'uniform':
        return np.linspace(eps, 1-eps,n_grid)
    low = pred - span * std
    high = pred + span * std
    if marginal_type == 'normal':
        if grid_spacing == 'uniform':
            return np.linspace(low, high, n_grid)
        elif grid_spacing == 'beta':
            p = beta.ppf(np.linspace(eps,1-eps, n_grid), 0.3, 0.3)  # clustered in (0,1)
            return norm.ppf(p, loc=pred, scale=std)
    if marginal_type in {'lognormal', 'gamma'}:
        low = max(1e-6, low)
        high = max(low * 1.01, high)
        if grid_spacing == 'uniform':
            return np.linspace(low, high, n_grid)
        elif grid_spacing == 'beta':
            p = beta.ppf(np.linspace(eps,1-eps, n_grid), 0.3, 0.3)  # clustered in (0,1)
            return norm.ppf(p, loc=pred, scale=std)  


def _copula_log_density_grid(cdf_grid, cov, copula_type, df):
    if copula_type == 'gaussian':
        return xlh.copula_funcs.gaussian_copula_density(cdf_grid, cov)
    if copula_type == 'student_t':
        return xlh.copula_funcs.student_t_copula_density(cdf_grid, cov, df)
    raise ValueError(f"Unknown copula type: {copula_type}")


def _integrate_axis(values, x_vals, axis, method):
    if method == 'trapz':
        return np.trapz(values, x=x_vals, axis=axis)
    if method == 'simpson':
        return simpson(values, x=x_vals, axis=axis)
    raise ValueError(f"Unknown integration method: {method}")


def check_pair(fiducial_data, cov, pair, marginal_type, copula_type, df, n_grid, integration_method='trapz'):
    i, j = pair
    pred = np.array([fiducial_data[i], fiducial_data[j]])
    cov_2d = cov[np.ix_([i, j], [i, j])]
    std = np.sqrt(np.diag(cov_2d))

    x1 = _make_grid(pred[0], std[0], n_grid, marginal_type)
    x2 = _make_grid(pred[1], std[1], n_grid, marginal_type)

    pdf1_true, cdf1 = _marginal_pdf_cdf(x1, pred[0], std[0], marginal_type)
    pdf2_true, cdf2 = _marginal_pdf_cdf(x2, pred[1], std[1], marginal_type)

    cdf1 = np.clip(cdf1, 1e-12, 1 - 1e-12)
    cdf2 = np.clip(cdf2, 1e-12, 1 - 1e-12)
    cdfs = np.array([cdf1, cdf2])
    
    log_copula = _copula_log_density_grid(cdfs, cov_2d, copula_type, df)
    log_copula = log_copula.reshape(n_grid, n_grid)

    log_joint = log_copula + np.log(pdf1_true)[:, None] + np.log(pdf2_true)[None, :]
    joint = np.exp(log_joint)

    marg1_rec = _integrate_axis(joint, x2, axis=1, method=integration_method)
    marg2_rec = _integrate_axis(joint, x1, axis=0, method=integration_method)

    eps = 1e-15
    pdf1_safe = np.maximum(pdf1_true, eps)
    pdf2_safe = np.maximum(pdf2_true, eps)
    marg1_safe = np.maximum(marg1_rec, eps)
    marg2_safe = np.maximum(marg2_rec, eps)

    dx1 = x1[1] - x1[0]
    dx2 = x2[1] - x2[0]
    kl1 = np.sum(pdf1_safe * np.log(pdf1_safe / marg1_safe)) * dx1
    kl2 = np.sum(pdf2_safe * np.log(pdf2_safe / marg2_safe)) * dx2
    abs_diff_1 = np.abs(pdf1_true - marg1_rec)
    abs_diff_2 = np.abs(pdf2_true - marg2_rec)
    l1_1 = np.sum(abs_diff_1) * dx1
    l1_2 = np.sum(abs_diff_2) * dx2
    mad_1 = np.mean(abs_diff_1)
    mad_2 = np.mean(abs_diff_2)

    return {
        'pair': pair,
        'kl_mean': 0.5 * (kl1 + kl2),
        'l1_mean': 0.5 * (l1_1 + l1_2),
        'kl_1': kl1,
        'kl_2': kl2,
        'l1_1': l1_1,
        'l1_2': l1_2,
        'mad_mean': 0.5 * (mad_1 + mad_2),
        'mad_1': mad_1,
        'mad_2': mad_2,
        'x1': x1,
        'x2': x2,
        'pdf1_true': pdf1_true,
        'pdf2_true': pdf2_true,
        'marg1_rec': marg1_rec,
        'marg2_rec': marg2_rec,
    }


def check_nd(fiducial_data, cov, marginal_type, copula_type, df, n_grid, integration_method='trapz'):
    n_dims = len(fiducial_data)
    std = np.sqrt(np.diag(cov))

    x_grids = []
    pdf_true = []
    cdf_vals = []

    for i in range(n_dims):
        x_i = _make_grid(fiducial_data[i], std[i], n_grid, marginal_type,grid_spacing='uniform')
        pdf_i, cdf_i = _marginal_pdf_cdf(x_i, fiducial_data[i], std[i], marginal_type)
        x_grids.append(x_i)
        pdf_true.append(pdf_i)
        cdf_vals.append(np.clip(cdf_i, 1e-12, 1 - 1e-12))

    cdf_grid = np.array(cdf_vals)
    log_copula = _copula_log_density_grid(cdf_grid, cov, copula_type, df)
    expected_shape = (n_grid,) * n_dims
    try:
        log_copula = np.asarray(log_copula).reshape(expected_shape)
    except ValueError as exc:
        raise ValueError(
            f"Unexpected copula grid shape: got {np.shape(log_copula)}, expected product {n_grid}^{n_dims}"
        ) from exc

    finite_mask = np.isfinite(log_copula)
    nonfinite_fraction = 1.0 - np.mean(finite_mask)

    log_joint = np.array(log_copula, copy=True)
    for axis in range(n_dims):
        reshape = [1] * n_dims
        reshape[axis] = n_grid
        log_joint += np.log(pdf_true[axis]).reshape(reshape)

    with np.errstate(over='ignore', invalid='ignore'):
        joint = np.exp(log_joint)
    joint = np.nan_to_num(joint, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-15
    kl_vals = []
    l1_vals = []
    mad_vals = []
    recovered = []

    joint_mass = joint
    for axis in range(n_dims - 1, -1, -1):
        joint_mass = _integrate_axis(joint_mass, x_grids[axis], axis=axis, method=integration_method)

    for target_axis in range(n_dims):
        marg = joint
        for axis in range(n_dims - 1, -1, -1):
            if axis == target_axis:
                continue
            marg = _integrate_axis(marg, x_grids[axis], axis=axis, method=integration_method)

        true_pdf = pdf_true[target_axis]
        true_safe = np.maximum(true_pdf, eps)
        rec_safe = np.maximum(marg, eps)
        abs_diff = np.abs(true_pdf - marg)
        dx = x_grids[target_axis][1] - x_grids[target_axis][0]

        kl_vals.append(np.sum(true_safe * np.log(true_safe / rec_safe)) * dx)
        l1_vals.append(np.sum(abs_diff) * dx)
        mad_vals.append(np.mean(abs_diff))
        recovered.append(marg)

    return {
        'n_dims': n_dims,
        'kl_mean': float(np.mean(kl_vals)),
        'l1_mean': float(np.mean(l1_vals)),
        'mad_mean': float(np.mean(mad_vals)),
        'kl_max': float(np.max(kl_vals)),
        'l1_max': float(np.max(l1_vals)),
        'mad_max': float(np.max(mad_vals)),
        'joint_mass': float(joint_mass),
        'nonfinite_log_copula_fraction': float(nonfinite_fraction),
        'x_grids': x_grids,
        'pdf_true': pdf_true,
        'pdf_rec': recovered,
    }


def _parse_grid_list(text):
    values = []
    for item in text.split(','):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    if not values:
        raise ValueError('Grid list is empty. Provide comma-separated integers.')
    return values


def run_convergence_study(fiducial_data, cov, marginal_type, copula_type, df, n_grids, integration_method):
    rows = []
    for n_grid in n_grids:
        result = check_nd(
            fiducial_data,
            cov,
            marginal_type,
            copula_type,
            df,
            n_grid,
            integration_method=integration_method,
        )
        rows.append((
            n_grid,
            result['mad_mean'],
            result['mad_max'],
            result['joint_mass'],
            result['nonfinite_log_copula_fraction'],
        ))
    return rows


def plot_pair(result, output_dir, label):
    pair = result['pair']
    x1 = result['x1']
    x2 = result['x2']
    pdf1_true = result['pdf1_true']
    pdf2_true = result['pdf2_true']
    marg1_rec = result['marg1_rec']
    marg2_rec = result['marg2_rec']

    fig, axes = plt.subplots(2, 2, figsize=(8, 6))

    axes[0, 0].plot(x1, pdf1_true, color='black', linewidth=1.5, label='Target')
    axes[0, 0].plot(x1, marg1_rec, color='tab:blue', linewidth=1.2, label='Recovered')
    axes[0, 0].set_title(f"Marginal 1 (pair {pair})")
    axes[0, 0].set_ylabel('PDF')
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(x2, pdf2_true, color='black', linewidth=1.5, label='Target')
    axes[0, 1].plot(x2, marg2_rec, color='tab:orange', linewidth=1.2, label='Recovered')
    axes[0, 1].set_title(f"Marginal 2 (pair {pair})")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(x1, np.abs(pdf1_true - marg1_rec), color='tab:blue', linewidth=1.2)
    axes[1, 0].set_title('Absolute difference (marginal 1)')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('|PDF - recovered|')

    axes[1, 1].plot(x2, np.abs(pdf2_true - marg2_rec), color='tab:orange', linewidth=1.2)
    axes[1, 1].set_title('Absolute difference (marginal 2)')
    axes[1, 1].set_xlabel('x')

    fig.suptitle(label, fontsize=11)
    plt.tight_layout()

    filename = f"marginal_recovery_pair_{pair[0]}_{pair[1]}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path


def grid_search_metrics(fiducial, marginal_type, copula_type, corr, n_grid, output_dir,
                        n_data_min=2, n_data_max=12, n_data_steps=6,
                        df_min=3, df_max=50, df_steps=10,
                        max_joint_cells=2_000_000,
                        integration_method='trapz'):
    n_data_values = np.linspace(n_data_min, n_data_max, n_data_steps).round().astype(int)
    n_data_values = np.unique(np.clip(n_data_values, 2, None))
    df_values = np.linspace(df_min, df_max, df_steps)

    mad_grid = np.full((len(df_values), len(n_data_values)), np.nan)

    for j, n_data in enumerate(n_data_values):
        fiducial_data = simple_linear_model(fiducial, n_data_points=int(n_data))
        cov = generate_covariance(fiducial_data, corr)

        n_grid_eff = int(np.floor(max_joint_cells ** (1.0 / n_data)))
        n_grid_eff = max(8, min(n_grid, n_grid_eff))
        logger.info(
            'grid-search setup: n_data=%d, effective_n_grid=%d, corr=%.3f',
            n_data,
            n_grid_eff,
            corr,
        )

        for i, df in enumerate(df_values):
            if copula_type == 'student_t' and df <= 2:
                continue
            result = check_nd(
                fiducial_data,
                cov,
                marginal_type,
                copula_type,
                float(df),
                n_grid_eff,
                integration_method=integration_method,
            )
            mad_grid[i, j] = result['mad_mean']

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    im = ax.imshow(
        mad_grid,
        origin='lower',
        aspect='auto',
        extent=[n_data_values[0], n_data_values[-1], df_min, df_max],
    )
    ax.set_title('Mean absolute difference')
    ax.set_xlabel('n_data')
    ax.set_ylabel('df')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"nD marginal recovery grid ({marginal_type}, copula={copula_type}, corr={corr:.2f})",
        fontsize=10,
    )
    plt.tight_layout()

    output_path = os.path.join(output_dir, 'marginal_recovery_grid_mad.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Copula marginal recovery check')
    parser.add_argument('--fiducial', type=float, default=5.0)
    parser.add_argument('--n-data', type=int, default=4)
    parser.add_argument('--corr', type=float, default=0.3)
    parser.add_argument('--marginal', choices=['normal', 'lognormal', 'student_t', 'gamma','uniform'], default='normal')
    parser.add_argument('--copula', choices=['gaussian', 'student_t'], default='student_t')
    parser.add_argument('--df', type=int, default=1000)
    parser.add_argument('--n-grid', type=int, default=100)
    parser.add_argument('--pairs', type=int, default=10)
    parser.add_argument('--plot', action='store_true', help='Save marginal comparison plots')
    parser.add_argument('--output', default='.', help='Output directory for plots')
    parser.add_argument('--grid-search', action='store_true',
                        help='Run n_data/df nD grid search and plot MAD heatmap')
    parser.add_argument('--grid-n-grid', type=int, default=100,
                        help='Grid resolution for the marginal recovery grid search')
    parser.add_argument('--grid-n-data-min', type=int, default=2)
    parser.add_argument('--grid-n-data-max', type=int, default=4)
    parser.add_argument('--grid-n-data-steps', type=int, default=3)
    parser.add_argument('--grid-df-min', type=float, default=3.0)
    parser.add_argument('--grid-df-max', type=float, default=500.0)
    parser.add_argument('--grid-df-steps', type=int, default=10)
    parser.add_argument('--max-joint-cells', type=int, default=100_000_000,
                        help='Upper bound for n_grid**n_data during nD integration')
    parser.add_argument('--integration-method', choices=['trapz', 'simpson'], default='trapz')
    parser.add_argument('--convergence-study', action='store_true',
                        help='Evaluate sensitivity to n_grid for current n_data/corr/df setup')
    parser.add_argument('--convergence-grids', default='20,30,40,60,80,100,200',
                        help='Comma-separated n_grid values for convergence study')
    parser.add_argument('--log-file', default='copula_marginal_check.log',
                        help='Log file path for this script output and diagnostics')
    parser.add_argument('--log-level', default='DEBUG',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    args = parser.parse_args()

    setup_logging(args.log_file, args.log_level)
    logger.info('Starting run with args: %s', vars(args))

    if args.plot:
        os.makedirs(args.output, exist_ok=True)

    fiducial_data = simple_linear_model(args.fiducial, n_data_points=args.n_data)
    cov = generate_covariance(fiducial_data, args.corr)

    nd_result = check_nd(
        fiducial_data,
        cov,
        args.marginal,
        args.copula,
        args.df,
        args.n_grid,
        integration_method=args.integration_method,
    )

    max_pairs = min(args.pairs, args.n_data // 2)
    pairs = [(i, i + 1) for i in range(max_pairs)]
    if args.plot and pairs:
        for pair in pairs:
            pair_result = check_pair(
                fiducial_data,
                cov,
                pair,
                args.marginal,
                args.copula,
                args.df,
                args.n_grid,
                integration_method=args.integration_method,
            )
            label = (
                f"copula={args.copula}, marginal={args.marginal}, "
                f"df={args.df}, n_data={args.n_data}, corr={args.corr}"
            )
            plot_pair(pair_result, args.output, label)

    logger.info('Marginal recovery summary')
    logger.info('  copula=%s, marginal=%s, df=%s', args.copula, args.marginal, args.df)
    logger.info(
        '  n_data=%d, corr=%.3f, n_grid=%d, integration=%s',
        args.n_data,
        args.corr,
        args.n_grid,
        args.integration_method,
    )
    logger.info('  KL mean: %.4e (max %.4e)', nd_result['kl_mean'], nd_result['kl_max'])
    logger.info('  L1 mean: %.4e (max %.4e)', nd_result['l1_mean'], nd_result['l1_max'])
    logger.info('  MAD mean: %.4e (max %.4e)', nd_result['mad_mean'], nd_result['mad_max'])
    logger.info('  Joint mass: %.6f', nd_result['joint_mass'])
    logger.info('  Non-finite log-copula fraction: %.3e', nd_result['nonfinite_log_copula_fraction'])
    if args.plot and pairs:
        logger.info('  pair plots generated for: %s', pairs)

    if args.convergence_study:
        n_grids = _parse_grid_list(args.convergence_grids)
        rows = run_convergence_study(
            fiducial_data,
            cov,
            args.marginal,
            args.copula,
            args.df,
            n_grids,
            args.integration_method,
        )
        logger.info('Convergence study (n_grid, MAD_mean, MAD_max, joint_mass, nonfinite_frac)')
        for row in rows:
            logger.info('  %4d  %.4e  %.4e  %.6f  %.3e', row[0], row[1], row[2], row[3], row[4])

    if args.grid_search:
        grid_path = grid_search_metrics(
            args.fiducial,
            args.marginal,
            args.copula,
            args.corr,
            args.grid_n_grid,
            args.output,
            n_data_min=args.grid_n_data_min,
            n_data_max=args.grid_n_data_max,
            n_data_steps=args.grid_n_data_steps,
            df_min=args.grid_df_min,
            df_max=args.grid_df_max,
            df_steps=args.grid_df_steps,
            max_joint_cells=args.max_joint_cells,
            integration_method=args.integration_method,
        )
        logger.info('Grid search plot saved: %s', grid_path)

    logger.info('Run completed successfully')


if __name__ == '__main__':
    main()
