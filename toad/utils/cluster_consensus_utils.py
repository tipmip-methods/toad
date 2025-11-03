from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors

from toad.regridding.base import BaseRegridder
from toad.regridding.healpix import HealPixRegridder


def _add_adjacent_true_pairs(
    mask2d: np.ndarray,
    edge_set: set[tuple[int, int]],
    flat_idx_2d: np.ndarray,
    use_eight: bool,
) -> None:
    """Adds undirected neighbor edges for True cells in a 2D mask.

    Modifies edge_set in-place by adding edges between adjacent True cells.
    Uses 4-connectivity (Von Neumann) by default, or 8-connectivity (Moore) if use_eight=True.

    Args:
        mask2d: 2D boolean array indicating valid cells.
        edge_set: Set to which edges will be added (modified in-place).
        flat_idx_2d: 2D array of flattened indices for each grid cell.
        use_eight: If True, include diagonal neighbors (8-connectivity); else only horizontal/vertical (4-connectivity).
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


def _build_knn_edges_from_latlon(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    k: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Build undirected edges using K-nearest neighbors on a sphere.

    Args:
        lat2d: 2D array of latitude values.
        lon2d: 2D array of longitude values.
        k: Number of nearest neighbors to consider (default: 8).

    Returns:
        Tuple of two arrays (rows, cols) representing undirected edges, where
        rows[i] and cols[i] are the indices of connected grid cells (i < j for all edges).
    """
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


def _build_knn_edges_from_regridder(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    k: int = 8,
    regridder: BaseRegridder | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build undirected edges using KNN after mapping to a regularized grid (e.g., HealPix).

    Args:
        lat2d: 2D array of latitude values.
        lon2d: 2D array of longitude values.
        k: Number of nearest neighbors to consider (default: 8).
        regridder: Optional regridder instance. If None, uses HealPixRegridder.

    Returns:
        Tuple of three arrays:
        - knn_rows: HealPix pixel indices for edge source nodes.
        - knn_cols: HealPix pixel indices for edge target nodes.
        - hp_index_flat: Mapping from original grid cells to HealPix pixel indices.
    """
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


def _build_empty_consensus_summary_df(
    td,
    y_len: int,
    x_len: int,
    coords_spatial: dict,
    spatial_dims: Tuple[str, str],
) -> Tuple[xr.Dataset, pd.DataFrame]:
    """Construct empty consensus outputs (all noise, zero consistency).

    Used for early returns when no edges or surviving edges exist.

    Args:
        td: TOAD object containing clustering results.
        y_len: Length of first spatial dimension.
        x_len: Length of second spatial dimension.
        coords_spatial: Dictionary of spatial coordinates.
        spatial_dims: Tuple of spatial dimension names.

    Returns:
        Tuple of (Dataset, DataFrame) with empty consensus results (all pixels marked as noise).
    """
    da_consensus_labels = xr.DataArray(
        np.full((y_len, x_len), -1, dtype=np.int32),
        coords=coords_spatial,
        dims=spatial_dims,
        name="clusters",
    )
    da_consistency = xr.DataArray(
        np.full((y_len, x_len), 0, dtype=np.float32),
        coords=coords_spatial,
        dims=spatial_dims,
        name="consistency",
    )
    ds_out = xr.Dataset(
        {
            "clusters": da_consensus_labels,
            "consistency": da_consistency,
        }
    )
    summary_df = _build_consensus_summary_df(
        td, da_consensus_labels, da_consistency, spatial_dims
    )
    return ds_out, summary_df


def _build_consensus_summary_df(
    td,
    labels2d: xr.DataArray,
    consistency2d: xr.DataArray,
    spatial_dims: Tuple[str, str],
) -> pd.DataFrame:
    """Build a summary DataFrame of cluster statistics from 2D label and consistency arrays.

    Args:
        td: TOAD object containing clustering results.
        labels2d: 2D DataArray of consensus cluster labels (-1 for noise).
        consistency2d: 2D DataArray of consensus consistency scores.
        spatial_dims: Tuple of spatial dimension names.

    Returns:
        DataFrame with one row per consensus cluster, containing statistics like
        cluster_id, mean_consistency, size, spatial means, and transition time metrics.
    """
    sd0, sd1 = spatial_dims
    dim = labels2d.name if labels2d.name else "cluster"
    cluster_map = labels2d.where(labels2d != -1)

    if np.all(labels2d.values == -1):
        cols = [
            "mean_consistency",
            "size",
            f"mean_{sd0}",
            f"mean_{sd1}",
            "mean_mean_shift_time",
            "std_mean_shift_time",
            "mean_std_shift_time",
            "std_std_shift_time",
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
                "mean_mean_shift_time": np.nan,
                "std_mean_shift_time": np.nan,
                "mean_std_shift_time": np.nan,
                "std_std_shift_time": np.nan,
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

        mean_mean_shift_time = per_cluster_per_model_mean.mean(
            dim="cluster_var", skipna=True
        )
        std_mean_shift_time_by = per_cluster_per_model_mean.std(
            dim="cluster_var", skipna=True
        )
        mean_std_shift_time = per_cluster_per_model_std.mean(
            dim="cluster_var", skipna=True
        )
        std_std_shift_time = per_cluster_per_model_std.std(
            dim="cluster_var", skipna=True
        )

        group_dim = mean_consistency.dims[0]
        df_transitions = pd.DataFrame(
            {
                "cluster_id": mean_mean_shift_time[group_dim].values.astype(int),
                "mean_mean_shift_time": mean_mean_shift_time.values.astype(np.float32),
                "std_mean_shift_time": std_mean_shift_time_by.values.astype(np.float32),
                "mean_std_shift_time": mean_std_shift_time.values.astype(np.float32),
                "std_std_shift_time": std_std_shift_time.values.astype(np.float32),
            }
        )

    df = df.merge(df_transitions, on="cluster_id", how="left")
    return df


def _knn_edges_from_mask(
    mask_bool_flat: np.ndarray, knn_rows: np.ndarray, knn_cols: np.ndarray
) -> tuple[list[int], list[int]]:
    """Return undirected KNN edges where both endpoints are True in mask_bool_flat.

    Args:
        mask_bool_flat: Boolean array indicating valid nodes.
        knn_rows: Array of edge source node indices.
        knn_cols: Array of edge target node indices.

    Returns:
        Tuple of two lists (rows, cols) representing undirected edges where both
        endpoints are True in the mask (i < j for all edges).
    """
    both = mask_bool_flat[knn_rows] & mask_bool_flat[knn_cols]
    if not np.any(both):
        return [], []
    r = knn_rows[both]
    c = knn_cols[both]
    # ensure i<j for undirected
    m = r < c
    return r[m].tolist(), c[m].tolist()


def _native_edges_from_mask(
    mask2d: np.ndarray, flat_idx_2d: np.ndarray, use_eight: bool
) -> tuple[list[int], list[int]]:
    """Return undirected native adjacency edges (4/8) where mask2d is True.

    Args:
        mask2d: 2D boolean array indicating valid cells.
        flat_idx_2d: 2D array of flattened indices for each grid cell.
        use_eight: If True, use 8-connectivity (Moore neighborhood); else 4-connectivity (Von Neumann).

    Returns:
        Tuple of two lists (rows, cols) representing undirected adjacency edges
        between True cells in the mask (i < j for all edges).
    """
    edges: set[tuple[int, int]] = set()
    _add_adjacent_true_pairs(mask2d, edges, flat_idx_2d, use_eight)
    if not edges:
        return [], []
    r, c = zip(*edges)
    return list(r), list(c)


def _compute_weighted_consensus(
    rows_V: list[int],
    cols_V: list[int],
    rows_A: list[int],
    cols_A: list[int],
    shape: tuple[int, int],
    min_consensus: float,
):
    """Build V, A CSR matrices, compute W=V/A on V support, threshold by min_consensus.

    Args:
        rows_V: Row indices for vote edges.
        cols_V: Column indices for vote edges.
        rows_A: Row indices for availability edges.
        cols_A: Column indices for availability edges.
        shape: Shape tuple (n_nodes, n_nodes) for the sparse matrices.
        min_consensus: Minimum consensus threshold (in [0,1]). Edges with weight >= min_consensus are kept.

    Returns:
        Sparse CSR matrix W containing weighted consensus scores, thresholded by min_consensus.
        W[i,j] = V[i,j] / A[i,j] for edges present in V, zero otherwise if below threshold.
    """
    V = coo_matrix(
        (
            np.ones(len(rows_V), dtype=np.float32),
            (np.array(rows_V, dtype=np.int64), np.array(cols_V, dtype=np.int64)),
        ),
        shape=shape,
    ).tocsr()
    A = coo_matrix(
        (
            np.ones(len(rows_A), dtype=np.float32),
            (np.array(rows_A, dtype=np.int64), np.array(cols_A, dtype=np.int64)),
        ),
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
