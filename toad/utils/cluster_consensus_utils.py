from typing import Any, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import coo_matrix

from toad.regridding.healpix import HealPixRegridder


def _empty_transition_time_df() -> pd.DataFrame:
    """Return an empty long-form transition-time dataframe with stable dtypes."""
    return pd.DataFrame(
        {
            "consensus_cluster_id": pd.Series(dtype=np.int64),
            "cluster_var": pd.Series(dtype=str),
            "transition_time": pd.Series(dtype=np.float64),
        }
    )


def _empty_label_field_shift_time_df() -> pd.DataFrame:
    """Empty table for :func:`label_field_shift_time_samples` (per-label-field events)."""
    return pd.DataFrame(
        {
            "cluster_id": pd.Series(dtype=np.int64),
            "transition_time": pd.Series(dtype=np.float64),
        }
    )


def _consensus_cluster_vars(td: Any, da_clusters: xr.DataArray) -> list[str]:
    """Resolve which input cluster variables contributed to a consensus result."""
    cluster_vars_attr = da_clusters.attrs.get("cluster_vars")
    if cluster_vars_attr is None:
        return list(td.cluster_vars)
    return list(cluster_vars_attr)


def _largest_cluster_ids(
    td: Any,
    cluster_var: str,
    top_n_clusters: int | None = None,
) -> np.ndarray:
    """Return cluster ids, optionally restricted to the largest N by actual size."""
    cluster_ids = np.asarray(td.get_cluster_ids(cluster_var), dtype=np.int64)
    cluster_ids = cluster_ids[cluster_ids >= 0]
    if top_n_clusters is None or int(top_n_clusters) <= 0 or cluster_ids.size <= 1:
        return cluster_ids

    cluster_counts = td.get_cluster_counts(cluster_var, exclude_noise=True)
    if len(cluster_counts) == 0:
        return np.array([], dtype=np.int64)

    sorted_ids = np.fromiter(
        cluster_counts.keys(), dtype=np.int64, count=len(cluster_counts)
    )
    return sorted_ids[: int(top_n_clusters)]


def _consensus_input_support_mask(
    td: Any,
    da_clusters: xr.DataArray,
    cluster_var: str,
    *,
    spatial_dims: Tuple[str, str] | None = None,
    time_dim: str | None = None,
) -> xr.DataArray:
    """Mask where one input clustering actually supports the final consensus labels.

    The returned boolean mask has the same dimensionality as ``da_clusters`` and
    marks exact 3D support on ``(time, y, x)`` voxels.
    """
    if spatial_dims is None:
        spatial_dims = tuple(td.space_dims)
    if time_dim is None and td.time_dim in da_clusters.dims:
        time_dim = td.time_dim

    top_n_clusters = da_clusters.attrs.get("top_n_clusters")
    allowed = _largest_cluster_ids(td, cluster_var, top_n_clusters)

    labels = td.data[cluster_var].transpose(
        td.time_dim, spatial_dims[0], spatial_dims[1]
    )
    support_3d = labels.isin(allowed)
    if time_dim is not None and time_dim in da_clusters.dims:
        return support_3d.transpose(time_dim, spatial_dims[0], spatial_dims[1])
    return support_3d.any(dim=td.time_dim)


def _add_adjacent_true_pairs(
    mask2d: np.ndarray,
    edge_set: set[tuple[int, int]],
    flat_idx_2d: np.ndarray,
) -> None:
    """Add undirected 8-neighbour edges for True cells in a 2D mask.

    Modifies edge_set in-place by adding edges between adjacent True cells.

    Args:
        mask2d: 2D boolean array indicating valid cells.
        edge_set: Set to which edges will be added (modified in-place).
        flat_idx_2d: 2D array of flattened indices for each grid cell.
    """
    # Horizontal neighbors
    common = mask2d[:, :-1] & mask2d[:, 1:]
    if common.any():
        a = flat_idx_2d[:, :-1][common].ravel()
        b = flat_idx_2d[:, 1:][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))
    # Vertical neighbors
    common = mask2d[:-1, :] & mask2d[1:, :]
    if common.any():
        a = flat_idx_2d[:-1, :][common].ravel()
        b = flat_idx_2d[1:, :][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))
    # Diagonal neighbors: top-left to bottom-right
    common = mask2d[:-1, :-1] & mask2d[1:, 1:]
    if common.any():
        a = flat_idx_2d[:-1, :-1][common].ravel()
        b = flat_idx_2d[1:, 1:][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))
    # Diagonal neighbors: top-right to bottom-left
    common = mask2d[:-1, 1:] & mask2d[1:, :-1]
    if common.any():
        a = flat_idx_2d[:-1, 1:][common].ravel()
        b = flat_idx_2d[1:, :-1][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))


def _add_wrapped_longitude_pairs(
    mask2d: np.ndarray,
    edge_set: set[tuple[int, int]],
    flat_idx_2d: np.ndarray,
) -> None:
    """Add 8-neighbour seam edges between first/last grid columns."""
    if mask2d.shape[1] < 2:
        return

    common = mask2d[:, 0] & mask2d[:, -1]
    if common.any():
        a = flat_idx_2d[:, 0][common].ravel()
        b = flat_idx_2d[:, -1][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))

    common = mask2d[:-1, 0] & mask2d[1:, -1]
    if common.any():
        a = flat_idx_2d[:-1, 0][common].ravel()
        b = flat_idx_2d[1:, -1][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))

    common = mask2d[1:, 0] & mask2d[:-1, -1]
    if common.any():
        a = flat_idx_2d[1:, 0][common].ravel()
        b = flat_idx_2d[:-1, -1][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))


def _build_healpix_edges_from_regridder(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    regridder: HealPixRegridder | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build undirected native-neighbour edges on the used HealPix subset."""
    if lat2d.size == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    if regridder is None:
        regridder = HealPixRegridder()
    if not isinstance(regridder, HealPixRegridder):
        raise ValueError(
            "Only HealPixRegridder is currently supported for consensus clustering. "
            f"Got {type(regridder).__name__}."
        )

    try:
        from astropy_healpix import neighbours
    except ImportError as exc:
        raise ImportError(
            "HealPix consensus adjacency requires `astropy-healpix` for native neighbour lookup."
        ) from exc

    coords_latlon_flat = np.column_stack([lat2d.ravel(), lon2d.ravel()])
    hp_index_global = regridder.map_orig_to_regrid(coords_latlon_flat)
    unique_hp_pixels = np.unique(hp_index_global)
    if unique_hp_pixels.size == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            hp_index_global.astype(np.int64, copy=False),
        )

    hp_index_flat = np.searchsorted(unique_hp_pixels, hp_index_global).astype(
        np.int64, copy=False
    )
    with np.errstate(invalid="ignore"):
        neighbour_global = np.asarray(
            neighbours(unique_hp_pixels, regridder.nside, order="ring"),
            dtype=np.int64,
        )

    source_local = np.broadcast_to(
        np.arange(unique_hp_pixels.size, dtype=np.int64),
        neighbour_global.shape,
    )
    valid = neighbour_global >= 0
    if not np.any(valid):
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            hp_index_flat,
        )

    neighbour_valid = neighbour_global[valid]
    source_valid = source_local[valid]

    target_local = np.searchsorted(unique_hp_pixels, neighbour_valid)
    in_subset = (target_local < unique_hp_pixels.size) & (
        unique_hp_pixels[target_local] == neighbour_valid
    )
    if not np.any(in_subset):
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            hp_index_flat,
        )

    rows = source_valid[in_subset]
    cols = target_local[in_subset]
    keep = rows != cols
    if not np.any(keep):
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            hp_index_flat,
        )

    rows = rows[keep]
    cols = cols[keep]
    edge_pairs = np.column_stack([np.minimum(rows, cols), np.maximum(rows, cols)])
    edge_pairs = np.unique(edge_pairs, axis=0)

    return (
        edge_pairs[:, 0].astype(np.int64, copy=False),
        edge_pairs[:, 1].astype(np.int64, copy=False),
        hp_index_flat,
    )


def _build_consensus_summary_df_spacetime(
    td: Any,
    labels3d: xr.DataArray,
    consistency3d: xr.DataArray,
    spatial_dims: Tuple[str, str],
    time_dim: str,
) -> pd.DataFrame:
    """Build summary statistics over all ``(time × space)`` consensus labels.

    One row is returned per cluster id that appears anywhere in the spacetime field. Column
    ``area`` is the number of unique spatial cells in the cluster footprint. Transition-time
    fields are derived from the
    actual event-time voxels returned by :func:`consensus_shift_time_distribution`.
    Pooled shift columns use all transition-time samples in the long table for that
    cluster (across all ``cluster_var``), not the median-of-medians.

    Args:
        td: TOAD object containing clustering results.
        labels3d: 3D DataArray of consensus cluster labels (``-1`` = shift but not
            in consensus; ``NaN`` = no abrupt shift in any input, matching ``compute_clusters``).
        consistency3d: 3D consistency scores, same dims as ``labels3d``.
        spatial_dims: Tuple of spatial dimension names.
        time_dim: Name of the time dimension (must be a dim of ``labels3d``).

    Returns:
        DataFrame with one row per consensus cluster id present in ``labels3d``.
    """
    sd0, sd1 = spatial_dims
    dim = labels3d.name if labels3d.name else "cluster"
    # Exclude noise (-1) and no-shift (NaN); ``x == x`` is False for NaN.
    cluster_map = labels3d.where((labels3d >= 0) & (labels3d == labels3d))

    empty_cols = [
        "cluster_id",
        "mean_consistency",
        "area",
        f"mean_{sd0}",
        f"mean_{sd1}",
        "median_median_shift_time",
        "std_median_shift_time",
        "median_std_shift_time",
        "std_std_shift_time",
        "pooled_median_shift_time",
        "pooled_std_shift_time",
    ]
    v = np.asarray(labels3d.values, dtype=np.float64)
    if not np.any(np.isfinite(v) & (v >= 0)):
        return pd.DataFrame({c: [] for c in empty_cols})

    mean_consistency = consistency3d.groupby(cluster_map).mean(skipna=True)
    cluster_ids = mean_consistency[dim].values.astype(int)
    area_vals: list[int] = []
    mean_sd0_vals: list[float] = []
    mean_sd1_vals: list[float] = []
    coord0 = td.data[sd0]
    coord1 = td.data[sd1]
    for cid in cluster_ids:
        footprint = (labels3d == cid).any(dim=time_dim)
        area_vals.append(int(footprint.sum(skipna=True).item()))
        mean_sd0_vals.append(float(coord0.where(footprint).mean(skipna=True).item()))
        mean_sd1_vals.append(float(coord1.where(footprint).mean(skipna=True).item()))

    df = pd.DataFrame(
        {
            "cluster_id": cluster_ids,
            "mean_consistency": mean_consistency.values.astype(np.float32),
            "area": np.asarray(area_vals, dtype=np.int32),
            f"mean_{sd0}": np.asarray(mean_sd0_vals, dtype=np.float32),
            f"mean_{sd1}": np.asarray(mean_sd1_vals, dtype=np.float32),
        }
    )

    dist_ds, df_cell = consensus_shift_time_distribution(
        td,
        labels3d,
        spatial_dims=spatial_dims,
        time_dim=time_dim,
        shift_threshold=0.0,
    )
    if len(dist_ds.data_vars) == 0:
        df_transitions = pd.DataFrame(
            {
                "cluster_id": df["cluster_id"].values.astype(int),
                "median_median_shift_time": np.nan,
                "std_median_shift_time": np.nan,
                "median_std_shift_time": np.nan,
                "std_std_shift_time": np.nan,
            }
        )
    else:
        # Per-input-map spatial median / std over voxels; outer medians are over cluster_var.
        per_cluster_per_model_median = dist_ds["spatial_median_transition_time"]
        per_cluster_per_model_std = dist_ds["spatial_std_transition_time"]

        finite_median = per_cluster_per_model_median.where(
            np.isfinite(per_cluster_per_model_median)
        )
        median_median_shift_time = finite_median.median(dim="cluster_var", skipna=True)

        std_median_shift_time_by = finite_median.std(
            dim="cluster_var", skipna=True
        ).fillna(0.0)
        finite_std = per_cluster_per_model_std.where(
            np.isfinite(per_cluster_per_model_std)
        )
        median_std_shift_time = finite_std.median(
            dim="cluster_var", skipna=True
        ).fillna(0.0)
        std_std_shift_time = finite_std.std(dim="cluster_var", skipna=True).fillna(0.0)

        group_dim = "consensus_cluster_id"
        df_transitions = pd.DataFrame(
            {
                "cluster_id": median_median_shift_time[group_dim].values.astype(int),
                "median_median_shift_time": median_median_shift_time.values.astype(
                    np.float32
                ),
                "std_median_shift_time": std_median_shift_time_by.values.astype(
                    np.float32
                ),
                "median_std_shift_time": median_std_shift_time.values.astype(
                    np.float32
                ),
                "std_std_shift_time": std_std_shift_time.values.astype(np.float32),
            }
        )

    df = df.merge(df_transitions, on="cluster_id", how="left")
    pooled_m, pooled_s = _pooled_median_std_from_df_cell(
        df["cluster_id"].to_numpy(np.int64), df_cell
    )
    df["pooled_median_shift_time"] = pooled_m
    df["pooled_std_shift_time"] = pooled_s
    return df


def _pooled_median_std_from_df_cell(
    cluster_ids: np.ndarray, df_cell: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Median and sample std of pooled transition_time samples per consensus cluster id.

    Pooled = all voxels in the long table from :func:`consensus_shift_time_distribution`,
    with duplicate ``(cluster_var, …)`` rows kept (each input can contribute many events).
    """
    n = int(cluster_ids.size)
    med = np.full(n, np.nan, dtype=np.float32)
    std = np.full(n, np.nan, dtype=np.float32)
    if (
        df_cell is None
        or df_cell.empty
        or "consensus_cluster_id" not in df_cell.columns
    ):
        return med, std
    for i, raw in enumerate(np.asarray(cluster_ids, dtype=np.int64).ravel()):
        cid = int(raw)
        sub = df_cell.loc[df_cell["consensus_cluster_id"] == cid, "transition_time"]
        v = sub.to_numpy(dtype=np.float64, copy=False)
        v = v[np.isfinite(v)]
        if v.size:
            med[i] = np.float32(np.median(v))
            std[i] = np.float32(np.std(v, ddof=1)) if v.size > 1 else np.float32(np.nan)
    return med, std


def label_field_shift_time_samples(
    td: Any,
    da_labels: xr.DataArray,
    spatial_dims: Tuple[str, str] | None = None,
    time_dim: str | None = None,
) -> pd.DataFrame:
    """Event-time of each spacetime cell with a non-noise label (one clustering, any kind).

    For each point where ``da_labels >= 0``, records the numeric :attr:`~toad.TOAD.numeric_time_value`
    at that timestep. Same time-coordinate convention as
    :func:`consensus_shift_time_distribution` for the consensus case (event times at labelled
    voxels), but **without** intersecting with other cluster maps—use a single
    3D label field (e.g. a normal ``*cluster`` variable on ``td`` or consensus labels
    with one logical map if passed manually).

    Args:
        td: TOAD instance (time coordinate, ``space_dims``).
        da_labels: 3D cluster labels, ``(time, y, x)`` (or with ``time_dim`` as below).
        spatial_dims: Defaults to ``tuple(td.space_dims)``.
        time_dim: Defaults to ``td.time_dim`` when that dimension is present on ``da_labels``.

    Returns:
        Long-form ``DataFrame`` with columns ``cluster_id``, ``transition_time``.
    """
    if spatial_dims is None:
        spatial_dims = tuple(td.space_dims)

    if time_dim is None and td.time_dim in da_labels.dims:
        time_dim = td.time_dim
    if time_dim is None or time_dim not in da_labels.dims:
        raise ValueError(
            "`label_field_shift_time_samples` requires 3D time-resolved label field "
            f"(time_dim={time_dim!r} not in dims={tuple(da_labels.dims)})."
        )

    if not np.any(np.isfinite(np.asarray(da_labels.values, dtype=np.float64))):
        return _empty_label_field_shift_time_df()

    time_values = np.asarray(td.numeric_time_values, dtype=np.float64)
    t_len = int(da_labels.sizes[time_dim])
    if time_values.size != t_len:
        raise ValueError(
            f"TOAD time coordinate length {time_values.size} != label field "
            f"length {t_len!r} along {time_dim!r}."
        )

    lab = np.asarray(da_labels.values, order="C")
    tt_b = np.broadcast_to(
        time_values.reshape((-1, 1, 1)),
        lab.shape,
    )
    m = lab >= 0
    cid = lab[m].astype(np.int64, copy=False)
    ttv = tt_b[m]
    fin = np.isfinite(ttv)
    if not np.any(fin):
        return _empty_label_field_shift_time_df()
    return pd.DataFrame(
        {
            "cluster_id": cid[fin].astype(np.int64, copy=False),
            "transition_time": ttv[fin].astype(np.float64, copy=False),
        }
    )


def label_field_shift_time_distributions(
    td: Any,
    da_labels: xr.DataArray,
    spatial_dims: Tuple[str, str] | None = None,
    time_dim: str | None = None,
) -> dict[int, np.ndarray]:
    """Plotting-friendly transition-time samples grouped by cluster id in one label field.

    Pooled the same way as :func:`consensus_shift_time_distributions` (one array per id).
    In spacetime, the same cell can appear at multiple times if the cluster footprint
    spans several timesteps.
    """
    df = label_field_shift_time_samples(
        td, da_labels, spatial_dims=spatial_dims, time_dim=time_dim
    )
    if df.empty:
        return {}
    out: dict[int, np.ndarray] = {}
    for cid, grp in df.groupby("cluster_id", sort=True):
        vals = grp["transition_time"].to_numpy(dtype=np.float64, copy=True)
        out[int(np.asarray(cid).item())] = vals[np.isfinite(vals)]
    return out


def consensus_shift_time_distribution(
    td: Any,
    da_clusters: xr.DataArray,
    spatial_dims: Tuple[str, str] | None = None,
    time_dim: str | None = None,
    shift_threshold: float = 0.0,
) -> tuple[xr.Dataset, pd.DataFrame]:
    """Per-consensus-cluster event-time samples used to build summary shift columns.

    This function uses the actual numeric time coordinate of each supported labelled
    voxel. In other words, the exported samples come from the exact peak-event voxels
    that belong to the consensus cluster, not from a derived 2D transition-time map.
    This matches the intended interpretation of a spacetime consensus cluster as a set
    of event voxels at specific times.

    **Dataset (always returned when clusterings exist):**

    - ``spatial_mean_transition_time``: for each ``(consensus_cluster_id, cluster_var)``,
      mean event time over the labelled ``(time, y, x)`` voxels themselves.
    - ``spatial_median_transition_time``: median event time over those voxels (same support).
    - ``spatial_std_transition_time``: standard deviation within that consensus set, per
      ``cluster_var``.

    Summary table columns are derived from these arrays:

    - ``median_median_shift_time`` = median over ``cluster_var`` of ``spatial_median``
      (finite only) — i.e. median across input clusterings of the per-map spatial median time.
    - ``std_median_shift_time`` = std over ``cluster_var`` of ``spatial_median`` (spread of
      per-map medians across input clusterings).
    - ``median_std_shift_time`` = median over ``cluster_var`` of ``spatial_std``.
    - ``std_std_shift_time`` = std over ``cluster_var`` of ``spatial_std``.
    - In :func:`_build_consensus_summary_df_spacetime`, also ``pooled_median_shift_time``
      and ``pooled_std_shift_time``: median and (sample) std of all ``transition_time`` rows
      for that ``consensus_cluster_id`` in the long dataframe (pooled over inputs).

    **Long DataFrame** (second return value): one row per labelled voxel and input
    clustering with columns ``consensus_cluster_id``, ``cluster_var``, ``transition_time``.
    For spacetime consensus the same physical ``(y, x)`` can therefore appear multiple
    times if the same consensus cluster is present there at multiple timesteps.

    Args:
        td: TOAD instance with ``cluster_vars`` and shifts.
        da_clusters: Time-resolved consensus ``clusters`` from
            :meth:`Aggregation.compute_consensus`.
        spatial_dims: Grid dimension names; default ``tuple(td.space_dims)``.
        time_dim: Time dimension of ``da_clusters`` if 3D; default ``td.time_dim`` when
            that dimension is present.
        shift_threshold: Unused for spacetime consensus. Kept only for API compatibility.

    Returns:
        ``(dataset, dataframe)``. If there are no cluster variables or no positive
        consensus labels, returns an empty Dataset and an empty DataFrame with the
        expected columns.
    """
    if spatial_dims is None:
        spatial_dims = tuple(td.space_dims)
    sd0, sd1 = spatial_dims

    if time_dim is None and td.time_dim in da_clusters.dims:
        time_dim = td.time_dim
    if time_dim is None or time_dim not in da_clusters.dims:
        raise ValueError(
            "`consensus_shift_time_distribution` requires a time-resolved consensus field."
        )

    labels = da_clusters
    lv = np.asarray(labels.values, dtype=np.float64)
    if not np.any(np.isfinite(lv) & (lv >= 0)):
        empty_ds = xr.Dataset(attrs={"note": "no consensus clusters (all noise)"})
        return empty_ds, _empty_transition_time_df()

    cluster_vars = _consensus_cluster_vars(td, da_clusters)
    if not cluster_vars:
        empty_ds = xr.Dataset(attrs={"note": "no cluster_vars on td"})
        return empty_ds, _empty_transition_time_df()

    parts_pc: list[pd.DataFrame] = []
    time_values = np.asarray(td.numeric_time_values, dtype=np.float64)
    for cvar in cluster_vars:
        support_mask = _consensus_input_support_mask(
            td,
            da_clusters,
            cvar,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
        )
        support_labels = labels.where(support_mask, other=-1)
        lab = np.asarray(support_labels.values)
        tt_b = np.broadcast_to(
            time_values.reshape((-1, 1, 1)),
            lab.shape,
        )
        m = lab >= 0
        cid = lab[m]
        ttv = tt_b[m]
        fin = np.isfinite(ttv)
        if not np.any(fin):
            continue
        parts_pc.append(
            pd.DataFrame(
                {
                    "consensus_cluster_id": cid[fin].astype(np.int64, copy=False),
                    "cluster_var": cvar,
                    "transition_time": ttv[fin].astype(np.float64, copy=False),
                }
            )
        )
    if not parts_pc:
        empty_ds = xr.Dataset(attrs={"note": "no supported transition-time samples"})
        return empty_ds, _empty_transition_time_df()

    df_cell = pd.concat(parts_pc, ignore_index=True)

    cluster_ids = np.sort(df_cell["consensus_cluster_id"].unique().astype(np.int64))
    mean_arr = np.full((cluster_ids.size, len(cluster_vars)), np.nan, dtype=np.float32)
    median_arr = np.full(
        (cluster_ids.size, len(cluster_vars)), np.nan, dtype=np.float32
    )
    std_arr = np.full((cluster_ids.size, len(cluster_vars)), np.nan, dtype=np.float32)
    cluster_id_to_idx = {int(cid): i for i, cid in enumerate(cluster_ids.tolist())}
    cluster_var_to_idx = {cvar: i for i, cvar in enumerate(cluster_vars)}
    grouped = (
        df_cell.groupby(["consensus_cluster_id", "cluster_var"])["transition_time"]
        .agg(["mean", "std", "median"])
        .reset_index()
    )
    for cid, cvar, mean_val, std_val, median_val in grouped.itertuples(
        index=False, name=None
    ):
        i = cluster_id_to_idx[int(cid)]
        j = cluster_var_to_idx[str(cvar)]
        mean_arr[i, j] = np.float32(mean_val)
        std_arr[i, j] = np.float32(0.0 if pd.isna(std_val) else std_val)
        median_arr[i, j] = np.float32(median_val)

    ds_out = xr.Dataset(
        {
            "spatial_mean_transition_time": xr.DataArray(
                mean_arr,
                dims=["consensus_cluster_id", "cluster_var"],
                coords={
                    "consensus_cluster_id": cluster_ids.astype(np.int64),
                    "cluster_var": pd.Index(cluster_vars, name="cluster_var"),
                },
            ),
            "spatial_median_transition_time": xr.DataArray(
                median_arr,
                dims=["consensus_cluster_id", "cluster_var"],
                coords={
                    "consensus_cluster_id": cluster_ids.astype(np.int64),
                    "cluster_var": pd.Index(cluster_vars, name="cluster_var"),
                },
            ),
            "spatial_std_transition_time": xr.DataArray(
                std_arr,
                dims=["consensus_cluster_id", "cluster_var"],
                coords={
                    "consensus_cluster_id": cluster_ids.astype(np.int64),
                    "cluster_var": pd.Index(cluster_vars, name="cluster_var"),
                },
            ),
        },
        attrs={
            "shift_threshold": str(shift_threshold),
            "spatial_dims": f"{sd0}, {sd1}",
            "support_rule": "input clustering must overlap the consensus cluster",
        },
    )

    return ds_out, df_cell


def consensus_shift_time_distributions(
    td: Any,
    da_clusters: xr.DataArray,
    spatial_dims: Tuple[str, str] | None = None,
    time_dim: str | None = None,
    shift_threshold: float = 0.0,
    *,
    distribution_result: tuple[xr.Dataset, pd.DataFrame] | None = None,
    source_input_cluster_var: str | None = None,
) -> dict[int, np.ndarray]:
    """Plotting-friendly transition-time samples grouped by consensus cluster id.

    This is a convenience wrapper around :func:`consensus_shift_time_distribution`.
    It aggregates the returned long-form dataframe across all input ``cluster_var``
    values and returns one 1D ``numpy`` array per consensus cluster, suitable for
    violin plots or histograms.

    The samples match the values underlying the summary shift columns. In spacetime
    mode this means a spatial cell can contribute multiple times if it appears in the
    same consensus component at multiple timesteps.

    Args:
        distribution_result: If provided, must be a value already returned from
            :func:`consensus_shift_time_distribution` for the same ``td``, ``da_clusters``,
            and options; the inner call is skipped (avoids duplicate work when both the
            dataset and grouped arrays are needed).
        source_input_cluster_var: If set, keep only rows whose ``cluster_var`` column
            equals this name (one input clustering’s events for each consensus id).
    """
    if distribution_result is not None:
        _, df_cell = distribution_result
    else:
        _, df_cell = consensus_shift_time_distribution(
            td,
            da_clusters,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
            shift_threshold=shift_threshold,
        )
    if source_input_cluster_var is not None:
        df_cell = df_cell[df_cell["cluster_var"] == source_input_cluster_var]
    if df_cell.empty:
        return {}

    out: dict[int, np.ndarray] = {}
    for cid, grp in df_cell.groupby("consensus_cluster_id", sort=True):
        vals = grp["transition_time"].to_numpy(dtype=np.float64, copy=True)
        out[int(np.asarray(cid).item())] = vals[np.isfinite(vals)]
    return out


def _native_edges_from_mask(
    mask2d: np.ndarray,
    flat_idx_2d: np.ndarray,
    stitch_longitude: bool = False,
) -> tuple[list[int], list[int]]:
    """Return undirected native 8-neighbour edges where ``mask2d`` is True.

    Args:
        mask2d: 2D boolean array indicating valid cells.
        flat_idx_2d: 2D array of flattened indices for each grid cell.
        stitch_longitude: If True, connect first/last columns as a wrapped meridian seam,
            including diagonal seam neighbours.

    Returns:
        Tuple of two lists (rows, cols) representing undirected adjacency edges
        between True cells in the mask (i < j for all edges).
    """
    edges: set[tuple[int, int]] = set()
    _add_adjacent_true_pairs(mask2d, edges, flat_idx_2d)
    if stitch_longitude:
        _add_wrapped_longitude_pairs(mask2d, edges, flat_idx_2d)
    if not edges:
        return [], []
    r, c = zip(*edges)
    return list(r), list(c)


def _compute_weighted_consensus(
    rows_V: list[int] | np.ndarray,
    cols_V: list[int] | np.ndarray,
    rows_A: list[int] | np.ndarray,
    cols_A: list[int] | np.ndarray,
    shape: tuple[int, int],
    min_consensus: float,
    data_A: np.ndarray | None = None,
):
    """Build V, A CSR matrices, compute W=V/A on V support, threshold by min_consensus.

    Args:
        rows_V: Row indices for vote edges (1-D int array or sequence; no Python ``tolist()`` needed).
        cols_V: Column indices for vote edges.
        rows_A: Row indices for availability edges.
        cols_A: Column indices for availability edges.
        shape: Shape tuple (n_nodes, n_nodes) for the sparse matrices.
        min_consensus: Minimum consensus threshold (in [0,1]). Edges with weight >= min_consensus are kept.
        data_A: Optional weights for availability edges. If omitted, each availability edge
            contributes 1. This is useful when many clusterings share the same availability
            edge set, so the denominator can be represented once with a larger weight instead
            of by duplicating identical edges.

    Returns:
        Sparse CSR matrix W containing weighted consensus scores, thresholded by min_consensus.
        W[i,j] = V[i,j] / A[i,j] for edges present in V, zero otherwise if below threshold.
    """
    rv = np.asarray(rows_V, dtype=np.int64)
    cv = np.asarray(cols_V, dtype=np.int64)
    ra = np.asarray(rows_A, dtype=np.int64)
    ca = np.asarray(cols_A, dtype=np.int64)
    da = (
        np.ones(ra.shape[0], dtype=np.float32)
        if data_A is None
        else np.asarray(data_A, dtype=np.float32)
    )
    V = coo_matrix(
        (np.ones(rv.shape[0], dtype=np.float32), (rv, cv)),
        shape=shape,
    ).tocsr()
    A = coo_matrix(
        (da, (ra, ca)),
        shape=shape,
    ).tocsr()
    # Note: tocsr() already sums duplicates, so sum_duplicates() is not needed
    V = V.maximum(V.T)
    A = A.maximum(A.T)
    V_idx = V.nonzero()
    A_on_V = A[V_idx].A1
    with np.errstate(divide="ignore", invalid="ignore"):
        W = V.copy()
        W.data = np.divide(V.data, A_on_V, out=np.zeros_like(V.data), where=A_on_V > 0)
    mask_keep = W.data >= float(min_consensus)
    W.data = np.where(mask_keep, W.data, 0).astype(W.data.dtype, copy=False)
    W.eliminate_zeros()
    return W


def _aggregate_labels_to_healpix(
    labels_2d: np.ndarray,
    hp_index_flat: np.ndarray,
    n_hp: int,
) -> np.ndarray:
    """Map per-cell cluster labels onto HealPix nodes; conflicts become noise (-1).

    Multiple original cells can map to the same HealPix pixel. If they disagree on
    positive cluster id at this time, that pixel is treated as noise for edge voting.
    """
    flat = np.asarray(labels_2d).ravel()
    out = np.full(n_hp, -1, dtype=flat.dtype)
    if flat.size == 0 or n_hp == 0:
        return out

    hp = np.asarray(hp_index_flat, dtype=np.int64)
    valid = np.isfinite(flat) & (flat >= 0)
    if not np.any(valid):
        return out

    hp_valid = hp[valid]
    labels_valid = flat[valid]

    order = np.argsort(hp_valid, kind="stable")
    hp_sorted = hp_valid[order]
    labels_sorted = labels_valid[order]

    unique_hp, start_idx = np.unique(hp_sorted, return_index=True)
    min_labels = np.minimum.reduceat(labels_sorted, start_idx)
    max_labels = np.maximum.reduceat(labels_sorted, start_idx)
    consistent = min_labels == max_labels
    out[unique_hp[consistent]] = min_labels[consistent]
    return out


def _dilate_cluster_labels_spacetime(
    labels_ts: np.ndarray,
    allowed_labels: np.ndarray,
    *,
    temporal_tolerance: int,
    spatial_tolerance: int = 0,
    spatial_rows: np.ndarray | None = None,
    spatial_cols: np.ndarray | None = None,
) -> np.ndarray:
    """Dilate sparse peak-event labels on a ``(time, space)`` lattice.

    For each allowed cluster id ``cid`` at node ``(t, s)``, the dilated output marks
    the same cluster id on all nodes ``(t', s')`` that are within ``temporal_tolerance``
    timesteps and ``spatial_tolerance`` spatial graph hops. Spatial graph hops are
    defined by the undirected edge list ``spatial_rows`` / ``spatial_cols``.

    If dilations from different cluster ids overlap at the same lattice node, that node
    is marked as ``-1`` (conflict / ambiguous) so it cannot contribute positive votes.
    """
    labels = np.asarray(labels_ts)
    k_t = max(0, int(temporal_tolerance))
    k_s = max(0, int(spatial_tolerance))
    if labels.ndim != 2:
        raise ValueError(
            f"`labels_ts` must have shape (time, space), got ndim={labels.ndim}."
        )
    if labels.size == 0 or allowed_labels.size == 0:
        return labels.copy()
    if k_s > 0 and (spatial_rows is None or spatial_cols is None):
        raise ValueError(
            "`spatial_rows` and `spatial_cols` are required when spatial_tolerance > 0."
        )
    if k_t == 0 and k_s == 0:
        return labels.copy()

    if np.issubdtype(labels.dtype, np.floating):
        out = np.full(labels.shape, np.nan, dtype=labels.dtype)
    else:
        out = np.full(labels.shape, -1, dtype=labels.dtype)
    assigned = np.zeros(labels.shape, dtype=bool)

    occ_mask = np.isfinite(labels) & np.isin(labels, allowed_labels)
    occ_t, occ_s = np.nonzero(occ_mask)
    if occ_t.size == 0:
        return out
    occ_vals = labels[occ_t, occ_s]

    neighborhoods: dict[int, np.ndarray] = {}
    if k_s == 0:
        for s in np.unique(occ_s):
            neighborhoods[int(s)] = np.array([int(s)], dtype=np.int64)
    else:
        assert spatial_rows is not None and spatial_cols is not None
        n_space = labels.shape[1]
        adjacency: list[list[int]] = [[] for _ in range(n_space)]
        for u, v in zip(spatial_rows.tolist(), spatial_cols.tolist()):
            adjacency[int(u)].append(int(v))
            adjacency[int(v)].append(int(u))

        for s in np.unique(occ_s):
            root = int(s)
            seen = {root}
            frontier = {root}
            for _ in range(k_s):
                next_frontier: set[int] = set()
                for node in frontier:
                    next_frontier.update(adjacency[node])
                next_frontier -= seen
                if not next_frontier:
                    break
                seen.update(next_frontier)
                frontier = next_frontier
            neighborhoods[root] = np.array(sorted(seen), dtype=np.int64)

    # Process only actual peak-event voxels. This is much cheaper than scanning the full
    # lattice once per cluster id when labels are sparse.
    for t, s, cid in zip(occ_t.tolist(), occ_s.tolist(), occ_vals.tolist()):
        lo = max(0, t - k_t)
        hi = min(labels.shape[0], t + k_t + 1)
        for s_dst in neighborhoods[int(s)].tolist():
            region_assigned = assigned[lo:hi, s_dst]
            region_out = out[lo:hi, s_dst]

            same_cid = region_out == cid
            conflicts = region_assigned & ~same_cid
            region_out[conflicts] = -1

            # Write conflicts first, then fill only truly fresh cells. Once a node has been
            # assigned, later different ids can only keep it at -1 rather than reclaim it.
            fresh = ~region_assigned
            region_out[fresh] = cid
            assigned[lo:hi, s_dst] = True

    return out


def _dilate_cluster_labels_in_time(
    labels_tyx: np.ndarray,
    allowed_labels: np.ndarray,
    temporal_tolerance: int,
) -> np.ndarray:
    """Dilate peak-event labels along time at fixed spatial cells."""
    labels = np.asarray(labels_tyx)
    out = _dilate_cluster_labels_spacetime(
        labels.reshape(labels.shape[0], -1),
        allowed_labels,
        temporal_tolerance=temporal_tolerance,
        spatial_tolerance=0,
    )
    return out.reshape(labels.shape)


def _build_spacetime_graph_edges(
    T: int,
    n_space: int,
    spatial_rows: np.ndarray,
    spatial_cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Undirected edges for a (time × space) lattice: spatial neighbours at each t + time chains.

    Nodes are indexed ``flat = t * n_space + s`` with ``t`` in ``[0, T)`` and ``s`` in
    ``[0, n_space)`` (same flattening order as ``labels.reshape(T, -1).ravel()``).

    * At each time slice, replicate ``spatial_rows`` / ``spatial_cols`` from the chosen
      spatial graph (native 8-neighbour grid or HealPix adjacency).
    * Between consecutive times, connect ``(t, s)`` to ``(t+1, s)`` for every spatial node ``s``.
    """
    sr = np.asarray(spatial_rows, dtype=np.int64)
    sc = np.asarray(spatial_cols, dtype=np.int64)
    if T <= 0 or n_space <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    t_off = (np.arange(T, dtype=np.int64) * n_space)[:, None]
    er_sp = (sr[None, :] + t_off).ravel() if sr.size else np.array([], dtype=np.int64)
    ec_sp = (sc[None, :] + t_off).ravel() if sc.size else np.array([], dtype=np.int64)
    if T <= 1:
        return er_sp, ec_sp
    s = np.arange(n_space, dtype=np.int64)
    t_lo = np.arange(T - 1, dtype=np.int64)
    u_t = (t_lo[:, None] * n_space + s[None, :]).ravel()
    v_t = ((t_lo + 1)[:, None] * n_space + s[None, :]).ravel()
    return np.concatenate([er_sp, u_t]), np.concatenate([ec_sp, v_t])


def _trim_spacetime_consensus_to_original_support(
    labels_flat: np.ndarray,
    consistency_flat: np.ndarray,
    original_support_flat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim consensus outputs back to undilated event support.

    Matching may use temporally dilated labels, but the public spacetime consensus mask should
    only contain voxels that were present in at least one original undilated input clustering.
    """
    keep = (
        np.asarray(original_support_flat, dtype=bool)
        & np.isfinite(labels_flat)
        & (labels_flat >= 0)
    )
    labels_trim = np.asarray(labels_flat).copy()
    cons_trim = np.asarray(consistency_flat).copy()
    labels_trim[~keep] = -1
    cons_trim[~keep] = 0
    return labels_trim, cons_trim


def _build_empty_consensus_time_resolved(
    T: int,
    y_len: int,
    x_len: int,
    coords_spatial: dict,
    spatial_dims: Tuple[str, str],
    time_coord: xr.DataArray,
    time_dim: str,
) -> xr.Dataset:
    """Empty time-resolved consensus (all noise, zero consistency)."""
    sd0, sd1 = spatial_dims
    da_clusters = xr.DataArray(
        np.full((T, y_len, x_len), -1, dtype=np.int32),
        coords={time_dim: time_coord, **coords_spatial},
        dims=[time_dim, sd0, sd1],
        name="clusters",
    )
    da_consistency = xr.DataArray(
        np.zeros((T, y_len, x_len), dtype=np.float32),
        coords={time_dim: time_coord, **coords_spatial},
        dims=[time_dim, sd0, sd1],
        name="consistency",
    )
    return xr.Dataset({"clusters": da_clusters, "consistency": da_consistency})
