"""Minimal chain IO helpers for emcee postprocessing and comparisons."""

from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

import numpy as np


# Matches names like:
# emcee_samples_12_runname_gaussian.npz
# emcee_samples_12_runname_gaussian_chain3.npz
# emcee_checkpoint_12_runname_gaussian.npz
_FILENAME_RE = re.compile(
    r"^emcee_(?P<kind>samples|checkpoint)_(?P<jobnumber>\d+)_"
    r"(?P<run_name>.+?)_(?P<likelihood_type>copula|gaussian)"
    r"(?:_chain(?P<chain_index>\d+))?\.npz$"
)


def parse_chain_filename(path: Path) -> Dict[str, Optional[object]]:
    """Parse sampler metadata from a chain/checkpoint filename."""
    p = Path(path)
    match = _FILENAME_RE.match(p.name)
    if match is None:
        raise ValueError(
            "Filename does not match expected sampler pattern: "
            f"{p.name}"
        )

    chain_index = match.group("chain_index")
    return {
        "path": p,
        "kind": match.group("kind"),
        "jobnumber": int(match.group("jobnumber")),
        "run_name": match.group("run_name"),
        "likelihood_type": match.group("likelihood_type"),
        "chain_index": int(chain_index) if chain_index is not None else None,
    }


def discover_chain_files(
    base_dir: Path,
    run_name: Optional[str] = None,
    likelihood_type: Optional[str] = None,
    jobnumber: Optional[int] = None,
    include_checkpoint: bool = False,
) -> List[Path]:
    """Discover sampler chain files with optional metadata filters.

    Returns sorted paths. For samples files, ordering is deterministic:
    base file (no _chainX) first, then _chain1, _chain2, ...
    """
    base_dir = Path(base_dir)
    if not base_dir.exists() or not base_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    if likelihood_type is not None and likelihood_type not in {"copula", "gaussian"}:
        raise ValueError(
            f"likelihood_type must be 'copula' or 'gaussian', got: {likelihood_type}"
        )

    paths = sorted(base_dir.glob("emcee_*.npz"))
    records: List[Dict[str, Optional[object]]] = []

    for path in paths:
        try:
            record = parse_chain_filename(path)
        except ValueError:
            continue

        if not include_checkpoint and record["kind"] != "samples":
            continue
        if run_name is not None and record["run_name"] != run_name:
            continue
        if likelihood_type is not None and record["likelihood_type"] != likelihood_type:
            continue
        if jobnumber is not None and record["jobnumber"] != int(jobnumber):
            continue

        records.append(record)

    if not records:
        raise FileNotFoundError(
            "No matching chain files found in "
            f"{base_dir} for filters run_name={run_name}, "
            f"likelihood_type={likelihood_type}, jobnumber={jobnumber}."
        )

    def sort_key(rec: Dict[str, Optional[object]]) -> Tuple[int, str, int, int]:
        # Group by job and run for stable ordering.
        job = int(rec["jobnumber"])
        run = str(rec["run_name"])
        # Base chain first, then _chainN.
        idx = rec["chain_index"]
        chain_rank = -1 if idx is None else int(idx)
        return (job, run, 0 if rec["kind"] == "samples" else 1, chain_rank)

    records.sort(key=sort_key)
    return [Path(rec["path"]) for rec in records]


def load_chain_npz(path: Path) -> Dict[str, object]:
    """Load one emcee sampler .npz file and normalize metadata."""
    p = Path(path)
    if p.suffix != ".npz":
        raise ValueError(f"Unsupported file type for chain loading: {p}")

    data = np.load(p, allow_pickle=True)
    samples = data["samples"]
    params = list(data["params"])

    metadata: Dict[str, object] = {
        "filepath": str(p),
        "n_walkers": int(data.get("n_walkers", samples.shape[1])),
        "n_steps": int(data.get("n_steps", samples.shape[0])),
    }

    if "steps_completed" in data:
        completed = int(data["steps_completed"])
        metadata["steps_completed"] = completed
        if completed < 0:
            raise ValueError(f"Invalid steps_completed={completed} in {p}")
        if completed < samples.shape[0]:
            samples = samples[:completed]
            metadata["n_steps"] = completed

    # Expand legacy delta_z representation to explicit per-bin params.
    ndim = samples.shape[-1]
    if len(params) != ndim and "delta_z" in params:
        idx = params.index("delta_z")
        params = params[:idx] + [f"delta_z_{i}" for i in range(5)] + params[idx + 1 :]

    return {
        "samples": samples,
        "params": params,
        "metadata": metadata,
    }


def load_and_merge_chains(
    paths: List[Path],
    strict_param_match: bool = True,
) -> Dict[str, object]:
    """Load and concatenate chains along step axis with compatibility checks."""
    if not paths:
        raise ValueError("paths cannot be empty")

    loaded = [load_chain_npz(Path(p)) for p in paths]

    ref_params = loaded[0]["params"]
    ref_shape = loaded[0]["samples"].shape[1:]

    merged_samples = []
    for item in loaded:
        samples = item["samples"]
        params = item["params"]

        if strict_param_match and params != ref_params:
            raise ValueError(
                "Parameter mismatch across chain files. "
                f"Reference params={ref_params}, got={params} in {item['metadata']['filepath']}"
            )

        if samples.shape[1:] != ref_shape:
            raise ValueError(
                "Walker/dimension shape mismatch across chains. "
                f"Reference {ref_shape}, got {samples.shape[1:]} in {item['metadata']['filepath']}"
            )

        merged_samples.append(samples)

    combined = np.concatenate(merged_samples, axis=0)

    return {
        "samples": combined,
        "params": ref_params,
        "metadata": {
            "source_files": [str(Path(p)) for p in paths],
            "n_files": len(paths),
            "n_steps_total": int(combined.shape[0]),
            "n_walkers": int(combined.shape[1]),
            "ndim": int(combined.shape[2]),
        },
    }
