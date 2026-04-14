"""Compare Gaussian and Copula emcee chains with shared postprocessing logic.

Usage:
    python postprocess_emcee_comparison.py --config postprocess_emcee_comparison.toml
"""

import argparse
from pathlib import Path

from plot_utils_style import configure_paper_plot_style
import postprocess_utils as putils
import chain_utils_io as cio


FIDUCIALS = {
    "s8": 0.8,
    "w_c": 0.1309,
    "w_b": 0.021,
    "h": 0.7,
    "n_s": 0.97,
    "A_IA": 0.0,
    "delta_z_0": 0.0,
    "delta_z_1": 0.0,
    "delta_z_2": 0.0,
    "delta_z_3": 0.0,
    "delta_z_4": 0.0,
    "omega_c": 0.26714,
    "omega_b": 0.04286,
    "omega_m": 0.31,
}

DEFAULT_PARAM_LABELS = {
    "s8": r"$S_8$",
    "w_c": r"$\omega_c$",
    "w_b": r"$\omega_b$",
    "h": r"$h$",
    "n_s": r"$n_s$",
    "A_IA": r"$A_{\mathrm{IA}}$",
    "omega_c": r"$\Omega_c$",
    "omega_b": r"$\Omega_b$",
    "omega_m": r"$\Omega_m$",
    "delta_z_0": r"$\delta_{z1}$",
    "delta_z_1": r"$\delta_{z2}$",
    "delta_z_2": r"$\delta_{z3}$",
    "delta_z_3": r"$\delta_{z4}$",
    "delta_z_4": r"$\delta_{z5}$",
}

DEFAULT_CONFIG = {
    "analysis": {
        "burn": 100,
        "thin": 1,
        "truths": None,
        "title": None,
        "params": None,
        "param_labels": None,
        "prior_ranges": None,
        "omega_transform": False,
        "smooth_scale_1D": 1.5,
        "smooth_scale_2D": 1.5,
        "corner_fig_size": None,
    },
    "stuck_filter": {
        "enabled": False,
        "atol": 0.0,
        "rtol": 1e-12,
        "min_run": 5,
        "pre_padding": 1,
        "post_padding": 0,
        "remove_all_zero_steps": True,
        "zero_tol": 0.0,
        "remove_isolated_global_drops": True,
        "global_drop_rel": 0.02,
        "params": None,
    },
    "comparison": {
        "base_dir": ".",
        "copula_base_dir": None,
        "gaussian_base_dir": None,
        "run_name": None,
        "jobnumber": None,
        "copula_likelihood_type": "copula",
        "gaussian_likelihood_type": "gaussian",
        "output_dir": None,
        "output_prefix": "emcee_comparison",
    },
}


def get_plot_labels(params, param_labels=None):
    labels = dict(DEFAULT_PARAM_LABELS)
    if isinstance(param_labels, dict):
        labels.update(param_labels)
    return [labels.get(p, p) for p in params]


def _process_chain(samples, params, analysis_cfg, stuck_cfg):
    """Apply shared processing logic to one chain set."""
    if stuck_cfg["enabled"]:
        (
            samples,
            _kept_mask,
            _removed_runs,
            _detection_params,
            _kept_indices,
            _n_zero_removed,
            _n_global_drop_removed,
        ) = putils.remove_stuck_steps(
            samples,
            params,
            atol=float(stuck_cfg["atol"]),
            rtol=float(stuck_cfg["rtol"]),
            min_run=int(stuck_cfg["min_run"]),
            pre_padding=int(stuck_cfg["pre_padding"]),
            post_padding=int(stuck_cfg["post_padding"]),
            remove_all_zero_steps=bool(stuck_cfg["remove_all_zero_steps"]),
            zero_tol=float(stuck_cfg["zero_tol"]),
            remove_isolated_global_drops=bool(stuck_cfg["remove_isolated_global_drops"]),
            global_drop_rel=float(stuck_cfg["global_drop_rel"]),
            selected_params=stuck_cfg["params"],
        )

    omega_transform = bool(analysis_cfg.get("omega_transform", False))
    params_to_plot = analysis_cfg.get("params")
    if params_to_plot:
        omega_transform = omega_transform or any(p.startswith("omega") for p in params_to_plot)

    truths = [FIDUCIALS.get(p, None) for p in params]
    if omega_transform:
        samples, params, truths = putils.apply_omega_transform(samples, params, truths)

    return samples, params, truths


def main():
    configure_paper_plot_style()

    parser = argparse.ArgumentParser(description="Compare Gaussian and Copula emcee chains")
    parser.add_argument(
        "--config",
        type=str,
        default="postprocess_emcee_comparison.toml",
        help="TOML config file",
    )
    args = parser.parse_args()

    config = putils.load_postprocess_config(
        args.config,
        DEFAULT_CONFIG,
        relative_to=Path(__file__).resolve().parent,
    )

    analysis_cfg = config["analysis"]
    stuck_cfg = config["stuck_filter"]
    cmp_cfg = config["comparison"]

    prior_ranges = analysis_cfg.get("prior_ranges")
    if prior_ranges is not None:
        if not isinstance(prior_ranges, dict):
            raise ValueError("[analysis].prior_ranges in config must be a table of parameter -> [min, max]")
        for p, bounds in prior_ranges.items():
            if not isinstance(bounds, list) or len(bounds) != 2:
                raise ValueError(
                    f"[analysis].prior_ranges.{p} must be [min, max] (use null for open bounds)"
                )
            lo, hi = bounds
            if lo is not None and not isinstance(lo, (int, float)):
                raise ValueError(
                    f"Lower bound for [analysis].prior_ranges.{p} must be a number or null"
                )
            if hi is not None and not isinstance(hi, (int, float)):
                raise ValueError(
                    f"Upper bound for [analysis].prior_ranges.{p} must be a number or null"
                )
            if lo is not None and hi is not None and float(lo) >= float(hi):
                raise ValueError(
                    f"[analysis].prior_ranges.{p} has invalid bounds [{lo}, {hi}] (need min < max)"
                )

    base_dir = Path(cmp_cfg["base_dir"]).expanduser().resolve()
    copula_base_dir = cmp_cfg.get("copula_base_dir")
    gaussian_base_dir = cmp_cfg.get("gaussian_base_dir")
    if copula_base_dir is None:
        copula_base_dir = base_dir
    else:
        copula_base_dir = Path(copula_base_dir).expanduser().resolve()
    if gaussian_base_dir is None:
        gaussian_base_dir = base_dir
    else:
        gaussian_base_dir = Path(gaussian_base_dir).expanduser().resolve()
    run_name = cmp_cfg.get("run_name")
    jobnumber = cmp_cfg.get("jobnumber")

    copula_files = cio.discover_chain_files(
        copula_base_dir,
        run_name=run_name,
        likelihood_type=cmp_cfg.get("copula_likelihood_type", "copula"),
        jobnumber=jobnumber,
        include_checkpoint=False,
    )
    gaussian_files = cio.discover_chain_files(
        gaussian_base_dir,
        run_name=run_name,
        likelihood_type=cmp_cfg.get("gaussian_likelihood_type", "gaussian"),
        jobnumber=jobnumber,
        include_checkpoint=False,
    )

    copula_bundle = cio.load_and_merge_chains(copula_files, strict_param_match=True)
    gaussian_bundle = cio.load_and_merge_chains(gaussian_files, strict_param_match=True)

    copula_samples = copula_bundle["samples"]
    gaussian_samples = gaussian_bundle["samples"]
    params_c = copula_bundle["params"]
    params_g = gaussian_bundle["params"]

    if params_c != params_g:
        raise ValueError(
            "Parameter mismatch between copula and gaussian chains. "
            f"copula={params_c}, gaussian={params_g}"
        )

    params = params_c

    copula_samples, params_c_proc, truths_c = _process_chain(copula_samples, params, analysis_cfg, stuck_cfg)
    gaussian_samples, params_g_proc, truths_g = _process_chain(gaussian_samples, params, analysis_cfg, stuck_cfg)

    if params_c_proc != params_g_proc:
        raise ValueError(
            "Processed parameter mismatch after transforms. "
            f"copula={params_c_proc}, gaussian={params_g_proc}"
        )

    params = params_c_proc
    truths = analysis_cfg.get("truths") or truths_c or truths_g

    burn_in = int(analysis_cfg["burn"])
    thin = int(analysis_cfg["thin"])
    if burn_in < 0:
        raise ValueError(f"burn must be >= 0, got {burn_in}")
    if thin < 1:
        raise ValueError(f"thin must be >= 1, got {thin}")

    if burn_in >= copula_samples.shape[0] or burn_in >= gaussian_samples.shape[0]:
        raise ValueError(
            "burn exceeds available steps in one of the chain sets. "
            f"burn={burn_in}, copula_steps={copula_samples.shape[0]}, gaussian_steps={gaussian_samples.shape[0]}"
        )

   
    output_dir = cmp_cfg.get("output_dir")
    if output_dir is None:
        output_dir = base_dir
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = cmp_cfg.get("output_prefix", "emcee_comparison")
    if run_name:
        prefix = f"{prefix}_{run_name}"
    if jobnumber is not None:
        prefix = f"{prefix}_job{jobnumber}"

    param_labels = analysis_cfg.get("param_labels")
    params_to_plot = analysis_cfg.get("params")
    title = analysis_cfg.get("title")

    corner_path = output_dir / f"{prefix}_corner_overlay.png"
    putils.plot_corner_overlay(
        copula_samples,
        gaussian_samples,
        params,
        corner_path,
        burn_in=burn_in,
        thin=thin,
        selected_params=params_to_plot,
        truths=truths,
        title=title,
        get_plot_labels_fn=get_plot_labels,
        param_labels=param_labels,
        smooth_scale_1D=float(analysis_cfg["smooth_scale_1D"]),
        smooth_scale_2D=float(analysis_cfg["smooth_scale_2D"]),
        corner_fig_size=analysis_cfg.get("corner_fig_size"),
        filled_gaussian=True,
        filled_copula=True,
        prior_ranges=prior_ranges,
    )

    print(f"\nComparison outputs saved in: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
