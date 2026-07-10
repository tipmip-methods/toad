"""Per-voxel member-support spacetime consensus on a HEALPix grid."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from toad.clustering import sorted_cluster_labels
from toad.healpix import build_ring1_spatial_edges
from toad.postprocessing.member_support_consensus import min_consensus_members


@dataclass(frozen=True)
class HealpixConsensusSupport:
    """Precomputed HEALPix member-support votes for repeated thresholding.

    Returned by :meth:`~toad.mma.MMA.build_consensus` on HEALPix exports and
    consumed by :meth:`~toad.mma.MMA.apply_consensus_threshold`.
    """

    cluster_vars: tuple[str, ...]
    native_union: np.ndarray
    member_vote_count: np.ndarray
    context: HealpixSpacetimeContext
    temporal_tolerance: int
    spatial_tolerance: int
    nside: int

    @property
    def n_members(self) -> int:
        return len(self.cluster_vars)


@dataclass(frozen=True)
class HealpixSpacetimeContext:
    """Fixed HEALPix spacetime layout for one member-support consensus run."""

    time_dim: str
    pixel_dim: str
    T: int
    npix: int
    nside: int
    time_coord: xr.DataArray
    spatial_rows: np.ndarray
    spatial_cols: np.ndarray


def build_healpix_spatial_edges(
    nside: int,
    *,
    k_neighbors: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Undirected ring-1 edges on the full HEALPix pixel set."""
    if k_neighbors != 8:
        warnings.warn(
            "k_neighbors is deprecated and ignored; HEALPix ring-1 neighbours are used.",
            DeprecationWarning,
            stacklevel=2,
        )
    return build_ring1_spatial_edges(nside)


def _spatial_neighbourhoods_for_tolerance(
    *,
    spatial_indices: np.ndarray,
    npix: int,
    spatial_rows: np.ndarray,
    spatial_cols: np.ndarray,
    spatial_tolerance: int,
) -> dict[int, np.ndarray]:
    """All HEALPix pixels within ``spatial_tolerance`` hops on the ring-1 graph."""
    unique_s = np.unique(np.asarray(spatial_indices, dtype=np.int64))
    adjacency: list[list[int]] = [[] for _ in range(npix)]
    for u, v in zip(spatial_rows.tolist(), spatial_cols.tolist()):
        adjacency[int(u)].append(int(v))
        adjacency[int(v)].append(int(u))

    neighbourhoods: dict[int, np.ndarray] = {}
    for s in unique_s.tolist():
        root = int(s)
        seen = {root}
        frontier = {root}
        for _ in range(spatial_tolerance):
            next_frontier: set[int] = set()
            for node in frontier:
                next_frontier.update(adjacency[node])
            next_frontier -= seen
            if not next_frontier:
                break
            seen.update(next_frontier)
            frontier = next_frontier
        neighbourhoods[root] = np.asarray(sorted(seen), dtype=np.int64)
    return neighbourhoods


def _dilate_healpix_support_mask(
    mask_tpix: np.ndarray,
    *,
    temporal_tolerance: int,
    spatial_tolerance: int,
    spatial_rows: np.ndarray,
    spatial_cols: np.ndarray,
) -> np.ndarray:
    """Dilate a native-event mask on (time, hp_pixel) for support counting."""
    k_t = max(0, int(temporal_tolerance))
    k_s = max(0, int(spatial_tolerance))
    mask = np.asarray(mask_tpix, dtype=bool)
    if k_t == 0 and k_s == 0:
        return mask.copy()

    T, npix = mask.shape
    spatial_nbrs = _spatial_neighbourhoods_for_tolerance(
        spatial_indices=np.arange(npix, dtype=np.int64),
        npix=npix,
        spatial_rows=spatial_rows,
        spatial_cols=spatial_cols,
        spatial_tolerance=k_s,
    )

    out = np.zeros_like(mask)
    for t in range(T):
        t_lo = max(0, t - k_t)
        t_hi = min(T - 1, t + k_t)
        for tt in range(t_lo, t_hi + 1):
            active_s = np.flatnonzero(mask[tt])
            if active_s.size == 0:
                continue
            for s in range(npix):
                if np.any(mask[tt, spatial_nbrs[s]]):
                    out[t, s] = True
    return out


def _labels_tpix(
    td: Any,
    cvar_name: str,
    time_dim: str,
    pixel_dim: str,
) -> np.ndarray:
    da = td.data[cvar_name]
    return da.transpose(time_dim, pixel_dim).values


def _accumulate_member_support_healpix(
    td: Any,
    *,
    cluster_vars: list[str],
    temporal_tolerance: int,
    spatial_tolerance: int,
    show_progress: bool,
    context: HealpixSpacetimeContext,
) -> tuple[np.ndarray, np.ndarray]:
    n_st = context.T * context.npix
    native_union = np.zeros(n_st, dtype=bool)
    member_vote_count = np.zeros(n_st, dtype=np.int16)

    for cvar in tqdm(
        cluster_vars,
        total=len(cluster_vars),
        disable=not show_progress,
        desc="healpix member-support consensus",
    ):
        labels = _labels_tpix(td, cvar, context.time_dim, context.pixel_dim)
        orig_flat = np.asarray(labels).reshape(-1)
        native_mask = np.isfinite(orig_flat) & (orig_flat >= 0)
        if not np.any(native_mask):
            continue
        native_union |= native_mask
        dilated = _dilate_healpix_support_mask(
            native_mask.reshape(context.T, context.npix),
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
            spatial_rows=context.spatial_rows,
            spatial_cols=context.spatial_cols,
        )
        member_vote_count[dilated.reshape(-1)] += 1

    return native_union, member_vote_count


def _component_graph_edges_for_kept_voxels(
    *,
    keep: np.ndarray,
    context: HealpixSpacetimeContext,
    temporal_tolerance: int,
    spatial_tolerance: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kept_nodes = np.flatnonzero(keep).astype(np.int64, copy=False)
    if kept_nodes.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    connect_t = max(1, int(temporal_tolerance))
    connect_s = max(1, int(spatial_tolerance))
    kept_t = kept_nodes // context.npix
    kept_s = kept_nodes % context.npix
    node_to_pos = {int(node): i for i, node in enumerate(kept_nodes.tolist())}
    neighbourhoods = _spatial_neighbourhoods_for_tolerance(
        spatial_indices=kept_s,
        npix=context.npix,
        spatial_rows=context.spatial_rows,
        spatial_cols=context.spatial_cols,
        spatial_tolerance=connect_s,
    )
    keep_ts = keep.reshape(context.T, context.npix)
    rows: list[int] = []
    cols: list[int] = []
    for i, (t, s) in enumerate(zip(kept_t.tolist(), kept_s.tolist())):
        lo_t = max(0, int(t) - connect_t)
        hi_t = min(context.T - 1, int(t) + connect_t)
        for tt in range(lo_t, hi_t + 1):
            for ss in neighbourhoods[int(s)].tolist():
                if not keep_ts[tt, ss]:
                    continue
                cand = int(tt) * context.npix + int(ss)
                j = node_to_pos.get(cand)
                if j is None or j <= i:
                    continue
                rows.append(i)
                cols.append(j)

    return (
        kept_nodes,
        np.asarray(rows, dtype=np.int64),
        np.asarray(cols, dtype=np.int64),
    )


def _label_retained_voxels(
    keep: np.ndarray,
    *,
    context: HealpixSpacetimeContext,
    temporal_tolerance: int,
    spatial_tolerance: int,
) -> np.ndarray:
    n_st = keep.size
    kept_nodes, rows, cols = _component_graph_edges_for_kept_voxels(
        keep=keep,
        context=context,
        temporal_tolerance=temporal_tolerance,
        spatial_tolerance=spatial_tolerance,
    )
    if kept_nodes.size == 0:
        return np.full(n_st, -1, dtype=np.int64)
    graph = coo_matrix(
        (np.ones(rows.shape[0], dtype=np.float32), (rows, cols)),
        shape=(kept_nodes.size, kept_nodes.size),
    )
    _, labels_kept = connected_components(graph, directed=False, return_labels=True)
    labels_flat = np.full(n_st, -1, dtype=np.int64)
    labels_flat[kept_nodes] = labels_kept.astype(np.int64, copy=False)
    return labels_flat


def _all_inputs_no_shift_mask_flat(
    td: Any,
    cluster_vars: list[str],
    context: HealpixSpacetimeContext,
) -> np.ndarray:
    all_nan = np.ones((context.T, context.npix), dtype=bool)
    for cvar in cluster_vars:
        lab = _labels_tpix(td, cvar, context.time_dim, context.pixel_dim)
        if lab.shape != (context.T, context.npix):
            raise ValueError(
                f"Label field {cvar!r} has shape {lab.shape}, expected "
                f"({context.T}, {context.npix})."
            )
        all_nan &= np.isnan(np.asarray(lab, dtype=np.float64))
    return all_nan.ravel()


def _mark_no_shift_nan(
    ds: xr.Dataset,
    td: Any,
    cluster_vars: list[str],
    context: HealpixSpacetimeContext,
) -> xr.Dataset:
    all_none = _all_inputs_no_shift_mask_flat(td, cluster_vars, context)
    da_c = ds["clusters"]
    flat = np.asarray(da_c.data, dtype=np.float64).ravel().copy()
    flat[(flat == -1) & all_none] = np.nan
    new_lab = flat.reshape(context.T, context.npix)
    rate = np.asarray(ds["rate"].data, dtype=np.float32).copy().reshape(-1)
    rate[~np.isfinite(new_lab.ravel())] = np.nan
    rate = rate.reshape(context.T, context.npix)
    return xr.Dataset(
        {
            "clusters": xr.DataArray(
                new_lab,
                coords=da_c.coords,
                dims=da_c.dims,
                attrs=da_c.attrs,
                name=da_c.name,
            ),
            "rate": xr.DataArray(
                rate,
                coords=ds["rate"].coords,
                dims=ds["rate"].dims,
                attrs=ds["rate"].attrs,
                name=ds["rate"].name,
            ),
        }
    )


def _filter_min_cluster_area(
    da_labels: xr.DataArray,
    da_rate: xr.DataArray,
    min_cluster_area: int,
    *,
    time_dim: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    if min_cluster_area <= 0:
        return da_labels, da_rate
    lab = np.asarray(da_labels.data, dtype=np.float64)
    time_axis = da_labels.dims.index(time_dim)
    lab_ts = np.moveaxis(lab, time_axis, 0).reshape(lab.shape[time_axis], -1)
    valid = np.isfinite(lab_ts) & (lab_ts >= 0)
    if not np.any(valid):
        return da_labels, da_rate
    label_ids = lab_ts[valid].astype(np.int64, copy=False)
    spatial_ids = np.broadcast_to(
        np.arange(lab_ts.shape[1], dtype=np.int64).reshape(1, -1),
        lab_ts.shape,
    )[valid]
    pairs = np.unique(np.column_stack((label_ids, spatial_ids)), axis=0)
    unique_ids, areas = np.unique(pairs[:, 0], return_counts=True)
    remove = unique_ids[areas < int(min_cluster_area)]
    if remove.size == 0:
        return da_labels, da_rate
    flat = lab.ravel().copy()
    fin = np.isfinite(flat)
    flat[fin & np.isin(flat, remove.astype(np.float64))] = -1.0
    flat = sorted_cluster_labels(flat)
    return (
        xr.DataArray(
            flat.reshape(lab.shape),
            coords=da_labels.coords,
            dims=da_labels.dims,
            attrs=da_labels.attrs,
            name=da_labels.name,
        ),
        da_rate,
    )


def build_healpix_consensus_support(
    td: Any,
    *,
    cluster_vars: list[str],
    temporal_tolerance: int,
    spatial_tolerance: int,
    nside: int,
    k_neighbors: int = 8,
    show_progress: bool = True,
) -> HealpixConsensusSupport:
    """Precompute dilated member-support votes on a HEALPix cluster grid."""
    if temporal_tolerance < 0:
        raise ValueError(
            f"`temporal_tolerance` must be >= 0, got {temporal_tolerance}."
        )
    if spatial_tolerance < 0:
        raise ValueError(f"`spatial_tolerance` must be >= 0, got {spatial_tolerance}.")
    if not cluster_vars:
        raise ValueError("cluster_vars must not be empty.")

    pixel_dim = "hp_pixel"
    time_dim = td.time_dim
    sample = td.data[cluster_vars[0]]
    if pixel_dim not in sample.dims:
        raise ValueError(
            f"Expected cluster labels on dimension {pixel_dim!r}, got dims={sample.dims}."
        )

    T = int(sample.sizes[time_dim])
    npix = int(sample.sizes[pixel_dim])
    if npix != 12 * nside**2:
        raise ValueError(
            f"hp_pixel size {npix} does not match nside={nside} (expected {12 * nside**2})."
        )

    spatial_rows, spatial_cols = build_healpix_spatial_edges(
        nside, k_neighbors=k_neighbors
    )
    context = HealpixSpacetimeContext(
        time_dim=time_dim,
        pixel_dim=pixel_dim,
        T=T,
        npix=npix,
        nside=nside,
        time_coord=sample[time_dim],
        spatial_rows=spatial_rows,
        spatial_cols=spatial_cols,
    )
    native_union, member_vote_count = _accumulate_member_support_healpix(
        td,
        cluster_vars=cluster_vars,
        temporal_tolerance=temporal_tolerance,
        spatial_tolerance=spatial_tolerance,
        show_progress=show_progress,
        context=context,
    )
    return HealpixConsensusSupport(
        cluster_vars=tuple(cluster_vars),
        native_union=native_union,
        member_vote_count=member_vote_count,
        context=context,
        temporal_tolerance=temporal_tolerance,
        spatial_tolerance=spatial_tolerance,
        nside=nside,
    )


def consensus_dataset_from_healpix_support(
    td: Any,
    support: HealpixConsensusSupport,
    *,
    min_consensus: float,
    min_cluster_area: int | None = 2,
) -> xr.Dataset:
    """Build interim HEALPix consensus labels and rate from precomputed votes."""
    cluster_vars = list(support.cluster_vars)
    context = support.context
    time_dim = context.time_dim
    pixel_dim = context.pixel_dim
    T, npix = context.T, context.npix
    n_members = len(cluster_vars)
    n_st = T * npix

    if not np.any(support.native_union):
        da_clusters = xr.DataArray(
            np.full((T, npix), -1, dtype=np.int32),
            coords={time_dim: context.time_coord, pixel_dim: np.arange(npix)},
            dims=[time_dim, pixel_dim],
            name="clusters",
        )
        da_rate = xr.DataArray(
            np.zeros((T, npix), dtype=np.float32),
            coords={time_dim: context.time_coord, pixel_dim: np.arange(npix)},
            dims=[time_dim, pixel_dim],
            name="rate",
        )
        return _mark_no_shift_nan(
            xr.Dataset({"clusters": da_clusters, "rate": da_rate}),
            td,
            cluster_vars,
            context,
        )

    min_votes = min_consensus_members(n_members, min_consensus)
    keep = support.native_union & (support.member_vote_count >= min_votes)
    rate_flat = np.zeros(n_st, dtype=np.float32)
    if np.any(support.native_union):
        rate_flat[support.native_union] = (
            support.member_vote_count[support.native_union].astype(np.float32)
            / n_members
        )

    if np.any(keep):
        labels_flat = _label_retained_voxels(
            keep,
            context=context,
            temporal_tolerance=support.temporal_tolerance,
            spatial_tolerance=support.spatial_tolerance,
        )
        labels_flat = sorted_cluster_labels(labels_flat)
    else:
        labels_flat = np.full(n_st, -1, dtype=np.int64)

    da_clusters = xr.DataArray(
        np.asarray(labels_flat, dtype=np.int32).reshape(T, npix),
        coords={time_dim: context.time_coord, pixel_dim: np.arange(npix)},
        dims=[time_dim, pixel_dim],
        name="clusters",
    )
    da_rate = xr.DataArray(
        np.asarray(rate_flat, dtype=np.float32).reshape(T, npix),
        coords={time_dim: context.time_coord, pixel_dim: np.arange(npix)},
        dims=[time_dim, pixel_dim],
        name="rate",
    )
    ds = _mark_no_shift_nan(
        xr.Dataset({"clusters": da_clusters, "rate": da_rate}),
        td,
        cluster_vars,
        context,
    )
    if min_cluster_area is not None:
        da_c, da_r = _filter_min_cluster_area(
            ds["clusters"],
            ds["rate"],
            min_cluster_area,
            time_dim=time_dim,
        )
        ds = xr.Dataset({"clusters": da_c, "rate": da_r})
    return ds


def run_healpix_member_support_consensus(
    td: Any,
    *,
    cluster_vars: list[str],
    min_consensus: float,
    temporal_tolerance: int,
    spatial_tolerance: int,
    nside: int,
    k_neighbors: int = 8,
    min_cluster_area: int | None = 2,
    show_progress: bool = True,
) -> xr.Dataset:
    """Run member-support consensus on HEALPix cluster label fields."""
    if not (0.0 <= min_consensus <= 1.0):
        raise ValueError(f"`min_consensus` must be in [0, 1], got {min_consensus}.")

    support = build_healpix_consensus_support(
        td,
        cluster_vars=cluster_vars,
        temporal_tolerance=temporal_tolerance,
        spatial_tolerance=spatial_tolerance,
        nside=nside,
        k_neighbors=k_neighbors,
        show_progress=show_progress,
    )
    return consensus_dataset_from_healpix_support(
        td,
        support,
        min_consensus=min_consensus,
        min_cluster_area=min_cluster_area,
    )
