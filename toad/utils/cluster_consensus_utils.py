from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.neighbors import NearestNeighbors

from toad.regridding.base import BaseRegridder
from toad.regridding.healpix import HealPixRegridder


def _add_adjacent_true_pairs(
    mask2d: np.ndarray,
    edge_set: set[tuple[int, int]],
    flat_idx_2d: np.ndarray,
    use_eight: bool,
) -> None:
    """Adds undirected neighbor edges for True cells in a 2D mask."""
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
    if use_eight:
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


def _latlon_to_unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert (lat, lon) in degrees to unit sphere Cartesian coords."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def build_knn_edges_from_latlon(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    k: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Build undirected edges using K-nearest neighbors on a sphere."""
    N = lat2d.size
    if N == 0:
        return np.array([], np.int64), np.array([], np.int64)

    flat_idx = np.arange(N, dtype=np.int64)
    xyz = _latlon_to_unit_xyz(lat2d.ravel(), lon2d.ravel())

    nn = NearestNeighbors(n_neighbors=min(k + 1, N))
    nn.fit(xyz)
    _, nbrs = nn.kneighbors(xyz)

    rows = np.repeat(flat_idx, nbrs.shape[1] - 1)
    cols = nbrs[:, 1:].ravel()

    mask = rows < cols
    return rows[mask], cols[mask]


def build_knn_edges_from_regridder(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    k: int = 8,
    regridder: BaseRegridder | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build undirected edges using KNN after mapping to a regularized grid (e.g., HealPix)."""
    N = lat2d.size
    if N == 0:
        return (
            np.array([], np.int64),
            np.array([], np.int64),
            np.array([], np.int64),
        )

    coords_latlon_flat = np.column_stack([lat2d.ravel(), lon2d.ravel()])

    if regridder is None:
        regridder = HealPixRegridder()

    hp_index_flat = regridder.map_orig_to_regrid(coords_latlon_flat)

    xyz = _latlon_to_unit_xyz(coords_latlon_flat[:, 0], coords_latlon_flat[:, 1])
    nn = NearestNeighbors(n_neighbors=min(k + 1, N)).fit(xyz)
    _, nbrs = nn.kneighbors(xyz)

    knn_rows = np.repeat(np.arange(len(xyz)), nbrs.shape[1] - 1)
    knn_cols = nbrs[:, 1:].ravel()

    keep = knn_rows < knn_cols
    knn_rows = hp_index_flat[knn_rows[keep]]
    knn_cols = hp_index_flat[knn_cols[keep]]

    return knn_rows, knn_cols, hp_index_flat


def _build_empty_consensus(
    td,
    y_len: int,
    x_len: int,
    coords_spatial: dict,
    spatial_dims: Tuple[str, str],
) -> Tuple[xr.Dataset, pd.DataFrame]:
    """Construct empty consensus outputs (all noise, zero consistency).

    Used for early returns when no edges or surviving edges exist.
    """
    da_consensus_labels = xr.DataArray(
        np.full((y_len, x_len), -1, dtype=np.int32),
        coords=coords_spatial,
        dims=spatial_dims,
        name="consensus_clusters",
    )
    da_consistency = xr.DataArray(
        np.full((y_len, x_len), 0, dtype=np.float32),
        coords=coords_spatial,
        dims=spatial_dims,
        name="consensus_consistency",
    )
    ds_out = xr.Dataset(
        {
            "consensus_clusters": da_consensus_labels,
            "consensus_consistency": da_consistency,
        }
    )
    summary_df = build_consensus_summary_df(
        da_consensus_labels, da_consistency, td, spatial_dims
    )
    return ds_out, summary_df


def build_consensus_summary_df(
    td,
    labels2d: xr.DataArray,
    consistency2d: xr.DataArray,
    spatial_dims: Tuple[str, str],
) -> pd.DataFrame:
    """Build a summary DataFrame of cluster statistics from 2D label and consistency arrays."""
    sd0, sd1 = spatial_dims
    dim = labels2d.name if labels2d.name else "cluster"
    cluster_map = labels2d.where(labels2d != -1)

    if np.all(labels2d.values == -1):
        cols = [
            "mean_consistency",
            "size",
            f"mean_{sd0}",
            f"mean_{sd1}",
            "mean_transition_time",
            "std_transition_time",
            "mean_within_model_spread",
            "std_within_model_spread",
        ]
        return pd.DataFrame({c: [] for c in cols})

    mean_consistency = consistency2d.groupby(cluster_map).mean(skipna=True)
    cluster_sizes = (
        xr.ones_like(cluster_map)
        .where(cluster_map.notnull())
        .groupby(cluster_map)
        .sum(skipna=True)
    )
    space_dim0_mean = (
        td.data[sd0].where(cluster_map >= 0).groupby(cluster_map).mean(skipna=True)
    )
    space_dim1_mean = (
        td.data[sd1].where(cluster_map >= 0).groupby(cluster_map).mean(skipna=True)
    )

    df = pd.DataFrame(
        {
            "cluster_id": mean_consistency[dim].values.astype(int),
            "mean_consistency": mean_consistency.values.astype(np.float32),
            "size": cluster_sizes.values.astype(np.int32),
            f"mean_{sd0}": space_dim0_mean.values.astype(np.float32),
            f"mean_{sd1}": space_dim1_mean.values.astype(np.float32),
        }
    )

    transition_time_maps = []
    for cluster_var in td.cluster_vars:
        shift_var = td.data[cluster_var].shifts_variable
        transition_time_maps.append(
            td.cluster_stats(shift_var).time.compute_transition_time(
                shift_threshold=0.0
            )
        )

    if len(transition_time_maps) == 0:
        df_transitions = pd.DataFrame(
            {
                "cluster_id": df["cluster_id"].values.astype(int),
                "mean_transition_time": np.nan,
                "std_transition_time": np.nan,
                "mean_within_model_spread": np.nan,
                "std_within_model_spread": np.nan,
            }
        )
    else:
        cluster_var_index = pd.Index(td.cluster_vars, name="cluster_var")
        transition_time_stack = xr.concat(transition_time_maps, dim=cluster_var_index)

        per_cluster_per_model_mean = transition_time_stack.groupby(cluster_map).mean(
            skipna=True
        )
        per_cluster_per_model_std = transition_time_stack.groupby(cluster_map).std(
            skipna=True
        )

        mean_transition_time = per_cluster_per_model_mean.mean(
            dim="cluster_var", skipna=True
        )
        std_transition_time_by = per_cluster_per_model_mean.std(
            dim="cluster_var", skipna=True
        )
        mean_within_model_spread = per_cluster_per_model_std.mean(
            dim="cluster_var", skipna=True
        )
        std_within_model_spread = per_cluster_per_model_std.std(
            dim="cluster_var", skipna=True
        )

        group_dim = mean_consistency.dims[0]
        df_transitions = pd.DataFrame(
            {
                "cluster_id": mean_transition_time[group_dim].values.astype(int),
                "mean_transition_time": mean_transition_time.values.astype(np.float32),
                "std_transition_time": std_transition_time_by.values.astype(np.float32),
                "mean_within_model_spread": mean_within_model_spread.values.astype(
                    np.float32
                ),
                "std_within_model_spread": std_within_model_spread.values.astype(
                    np.float32
                ),
            }
        )

    df = df.merge(df_transitions, on="cluster_id", how="left")
    return df
