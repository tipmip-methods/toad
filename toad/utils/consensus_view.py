"""Grid-agnostic helpers for consensus label fields (maps, summaries, HealPix)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd
import xarray as xr


def infer_consensus_time_dim(da: xr.DataArray) -> str | None:
    """Return the time-like dimension name on a consensus DataArray, if any."""
    if da.ndim < 2:
        return None
    for dim in da.dims:
        if dim in ("time", "GMST") or "time" in str(dim).lower():
            return str(dim)
    if "hp_pixel" in da.dims:
        return str(next(d for d in da.dims if d != "hp_pixel"))
    return str(da.dims[0])


def collapse_consensus_for_map(
    clusters: xr.DataArray,
    time_dim: str | None = None,
) -> np.ndarray:
    """Time-collapse spacetime consensus labels for map display.

    At each spatial location, the mode of finite non-negative cluster ids is used.
    Locations that are only noise (-1) across time remain -1.
    """
    if clusters.ndim < 2:
        return np.asarray(clusters.values)

    resolved_time_dim = time_dim or infer_consensus_time_dim(clusters)
    if resolved_time_dim is None or resolved_time_dim not in clusters.dims:
        return np.asarray(clusters.values)

    spatial_dims = [d for d in clusters.dims if d != resolved_time_dim]
    arr = clusters.transpose(resolved_time_dim, *spatial_dims).values
    flat = arr.reshape(arr.shape[0], -1)
    out = np.full(flat.shape[1], np.nan, dtype=np.float64)
    for i in range(flat.shape[1]):
        vals = flat[:, i]
        pos = vals[(vals >= 0) & np.isfinite(vals)]
        if pos.size:
            out[i] = np.bincount(pos.astype(int)).argmax()
        elif np.any(vals == -1):
            out[i] = -1.0
    return out.reshape([clusters.sizes[d] for d in spatial_dims])


def consensus_cluster_ids(clusters_map: np.ndarray) -> list[int]:
    """Sorted unique non-negative cluster ids from a collapsed map."""
    ids = np.unique(clusters_map[(clusters_map >= 0) & np.isfinite(clusters_map)])
    return [int(x) for x in ids]


def nside_from_npix(npix: int) -> int:
    """Infer HEALPix nside from pixel count (npix = 12 * nside²)."""
    if npix <= 0 or npix % 12 != 0:
        raise ValueError(f"Invalid HEALPix npix={npix}.")
    order = 0.5 * np.log2(npix / 12.0)
    if not np.isclose(order, round(order)):
        raise ValueError(f"npix={npix} is not a valid HEALPix pixel count.")
    return 1 << int(round(order))


def resolve_healpix_nside(
    npix: int,
    nside: int | None = None,
    attrs: Mapping[str, Any] | None = None,
) -> int:
    """Resolve HEALPix nside from an explicit value or dataset attributes."""
    if nside is not None:
        resolved = int(nside)
    elif attrs is not None:
        resolved = None
        for key in ("nside", "NSIDE"):
            candidate = attrs.get(key)
            if candidate is not None:
                resolved = int(candidate)
                break
        if resolved is None:
            resolved = nside_from_npix(npix)
    else:
        resolved = nside_from_npix(npix)

    if 12 * resolved**2 != npix:
        raise ValueError(
            f"HEALPix nside={resolved} implies npix={12 * resolved**2}, "
            f"but data has npix={npix}."
        )
    return resolved


def build_simple_consensus_summary_df(
    clusters_map: np.ndarray,
    rate_map: np.ndarray,
    shift_times_by_cluster: Mapping[int, np.ndarray],
    *,
    numeric: bool = True,
) -> pd.DataFrame:
    """Build a per-cluster summary table from collapsed consensus fields.

    This is the MMA-oriented summary (spatial footprint + pooled shift times).
    TOAD's richer :meth:`Aggregation.consensus_summary` uses strict spacetime
    agreement with in-memory cluster variables.
    """
    del numeric  # reserved for API compatibility with MMA.get_consensus_summary
    rows: list[dict[str, Any]] = []
    for cid in consensus_cluster_ids(clusters_map):
        mask = (clusters_map == cid) & np.isfinite(clusters_map)
        size = int(np.sum(mask))
        rate_vals = rate_map[mask]
        mean_consensus_rate = float(np.nanmean(rate_vals)) if rate_vals.size else np.nan
        times = shift_times_by_cluster.get(cid, np.array([]))
        mean_shift = float(np.mean(times)) if len(times) > 0 else np.nan
        std_shift = float(np.std(times)) if len(times) > 1 else np.nan
        rows.append(
            {
                "cluster_id": cid,
                "size": size,
                "mean_consensus_rate": mean_consensus_rate,
                "mean_mean_shift_time": mean_shift,
                "std_mean_shift_time": std_shift,
            }
        )
    return pd.DataFrame(rows)
