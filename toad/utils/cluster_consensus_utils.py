from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from tqdm import tqdm

from toad.clustering import sorted_cluster_labels
from toad.regridding.base import BaseRegridder
from toad.regridding.healpix import HealPixRegridder


@dataclass
class _EdgeCollectionContext:
    """Context for consensus edge collection. Built once, passed to _collect_consensus_edges."""

    spatial_dims: Tuple[str, str]
    y_len: int
    x_len: int
    flat_idx_2d: np.ndarray
    regrid_enabled: bool
    use_knn: bool
    neighbor_connectivity: int
    top_n_clusters: int | None
    show_progress: bool
    # For regrid case
    hp_index_flat: np.ndarray | None = None
    N_hp: int = 0
    mask_hp: np.ndarray | None = None
    valid_hp: np.ndarray | None = None
    knn_rows: np.ndarray | None = None
    knn_cols: np.ndarray | None = None
    # For non-regrid case
    present_mask2d: np.ndarray | None = None


def _collect_consensus_edges(
    td,
    cluster_vars: list[str],
    ctx: _EdgeCollectionContext,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """Collect vote and availability edges from all cluster variables.

    Returns:
        Tuple of (rows_V, cols_V, rows_A, cols_A) for weighted consensus.
    """
    rows_V, cols_V = [], []
    rows_A, cols_A = [], []

    for cvar in tqdm(cluster_vars, disable=not ctx.show_progress):
        unique_ids = td.get_cluster_ids(cvar)
        if unique_ids.size == 0:
            continue
        if ctx.top_n_clusters is not None and ctx.top_n_clusters > 0:
            unique_ids = unique_ids[: ctx.top_n_clusters]

        if ctx.regrid_enabled:
            assert ctx.mask_hp is not None and ctx.valid_hp is not None
            assert (
                ctx.hp_index_flat is not None
                and ctx.knn_rows is not None
                and ctx.knn_cols is not None
            )
            labels3d = td.data[cvar].values
            labels_2d = np.logical_or.reduce(
                [(labels3d == cid).any(axis=0) for cid in unique_ids]
            )
            ctx.mask_hp.fill(False)
            ctx.mask_hp[np.unique(ctx.hp_index_flat[labels_2d.ravel()])] = True
            rA, cA = _knn_edges_from_mask(ctx.valid_hp, ctx.knn_rows, ctx.knn_cols)
            rows_A.extend(rA)
            cols_A.extend(cA)
            rV, cV = _knn_edges_from_mask(ctx.mask_hp, ctx.knn_rows, ctx.knn_cols)
            rows_V.extend(rV)
            cols_V.extend(cV)
        else:
            labels = td.data[cvar].values
            map_edges_V: set[tuple[int, int]] = set()
            for cid in unique_ids:
                mask2d = (labels == cid).any(axis=0)
                if ctx.use_knn:
                    assert ctx.knn_rows is not None and ctx.knn_cols is not None
                    mask_flat = mask2d.ravel()
                    both_true = mask_flat[ctx.knn_rows] & mask_flat[ctx.knn_cols]
                    for i, j in zip(ctx.knn_rows[both_true], ctx.knn_cols[both_true]):
                        map_edges_V.add((int(i), int(j)))
                else:
                    _add_adjacent_true_pairs(
                        mask2d,
                        map_edges_V,
                        ctx.flat_idx_2d,
                        ctx.neighbor_connectivity == 8,
                    )
            if map_edges_V:
                r, c = zip(*map_edges_V)
                rows_V.extend(r)
                cols_V.extend(c)
            assert ctx.present_mask2d is not None
            rA, cA = (
                _knn_edges_from_mask(
                    ctx.present_mask2d.ravel(), ctx.knn_rows, ctx.knn_cols
                )
                if ctx.use_knn
                else _native_edges_from_mask(
                    ctx.present_mask2d, ctx.flat_idx_2d, ctx.neighbor_connectivity == 8
                )
            )
            rows_A.extend(rA)
            cols_A.extend(cA)

    return rows_V, cols_V, rows_A, cols_A


def _graph_to_labels_and_consistency(
    W: csr_matrix,
    ever_clustered: np.ndarray,
    y_len: int,
    x_len: int,
    min_cluster_size: int,
    regrid_enabled: bool,
    hp_index_flat: np.ndarray | None,
    lat_shape: tuple[int, ...] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert weighted consensus graph to labels_2d and consistency arrays."""
    node_sum = np.array(W.sum(axis=1)).ravel()
    node_deg = np.array(W.count_nonzero(axis=1)).ravel().astype(np.float32)
    consistency_hp = np.divide(
        node_sum, node_deg, out=np.zeros_like(node_sum), where=node_deg > 0
    )

    if regrid_enabled and hp_index_flat is not None and lat_shape is not None:
        consistency = consistency_hp[hp_index_flat].reshape(lat_shape)
    else:
        consistency = consistency_hp.reshape((y_len, x_len))

    bin_adj = W.copy()
    bin_adj.data[:] = 1.0
    bin_adj = bin_adj.maximum(bin_adj.T)
    _, labels_flat = connected_components(bin_adj, directed=False, return_labels=True)

    if regrid_enabled and hp_index_flat is not None and lat_shape is not None:
        labels_flat_orig = labels_flat[hp_index_flat]
        labels_2d = labels_flat_orig.reshape(lat_shape)
        deg_hp = np.array(bin_adj.getnnz(axis=1))
        deg_orig = deg_hp[hp_index_flat]
        deg_2d = deg_orig.reshape(lat_shape)
        labels_2d[deg_2d == 0] = -1
    else:
        labels_2d = labels_flat.reshape((y_len, x_len))
        deg = np.array(bin_adj.getnnz(axis=1)).reshape((y_len, x_len))
        labels_2d[deg == 0] = -1

    labels_2d = sorted_cluster_labels(labels_2d.flatten()).reshape(labels_2d.shape)

    if min_cluster_size > 1:
        flat = labels_2d.flatten()
        valid_mask = (flat >= 0) & np.isfinite(flat)
        for cid in np.unique(flat[valid_mask]):
            if np.sum(flat == cid) < min_cluster_size:
                flat[flat == cid] = -1
        labels_2d = flat.reshape(labels_2d.shape)
        labels_2d = sorted_cluster_labels(labels_2d.flatten()).reshape(labels_2d.shape)

    labels_2d = labels_2d.astype(np.float32)
    labels_2d[~ever_clustered] = np.nan
    consistency[~ever_clustered] = np.nan

    return labels_2d, consistency


def _create_consensus_output_arrays(
    labels_2d: np.ndarray,
    consistency: np.ndarray,
    coords_spatial: dict,
    spatial_dims: Tuple[str, str],
    shared_attrs: dict,
    cluster_name: str = "consensus_clusters",
    consistency_name: str = "consensus_consistency",
) -> tuple[xr.DataArray, xr.DataArray]:
    """Create consensus cluster and consistency DataArrays with proper attributes."""
    from toad.utils import _attrs

    da_consensus_labels = xr.DataArray(
        labels_2d,
        coords=coords_spatial,
        dims=list(spatial_dims),
        name=cluster_name,
    )
    da_consistency = xr.DataArray(
        consistency,
        coords=coords_spatial,
        dims=list(spatial_dims),
        name=consistency_name,
    )
    da_consensus_labels.attrs.update(
        {
            **shared_attrs,
            "description": "Spatial consensus clusters (time-collapsed).",
            _attrs.VARIABLE_TYPE: _attrs.TYPE_CONSENSUS_CLUSTER,
        }
    )
    da_consensus_labels.attrs[_attrs.CONSENSUS_CONSISTENCY_VARIABLE] = consistency_name
    da_consistency.attrs.update(
        {
            **shared_attrs,
            "description": "Consitency scores for each grid cell.",
            _attrs.VARIABLE_TYPE: _attrs.TYPE_CONSENSUS_CONSISTENCY,
        }
    )
    return da_consensus_labels, da_consistency


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


def _build_knn_edges_from_coords_2d(
    coords_2d: np.ndarray,
    k: int = 8,
    use_sphere: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build KNN edges from 2D coordinates (x,y) or (lat,lon).

    Args:
        coords_2d: (N, 2) array. For use_sphere: (lat, lon) in degrees.
        k: Number of nearest neighbors.
        use_sphere: If True, treat coords as lat/lon and use spherical distance.

    Returns:
        Tuple (knn_rows, knn_cols) of undirected edges between flat indices.
    """
    N = coords_2d.shape[0]
    if N == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    if use_sphere:
        xyz = _latlon_to_unit_xyz(coords_2d[:, 0], coords_2d[:, 1])
    else:
        xyz = coords_2d
    nn = NearestNeighbors(n_neighbors=min(k + 1, N))
    nn.fit(xyz)
    _, nbrs = nn.kneighbors(xyz)
    flat_idx = np.arange(N, dtype=np.int64)
    rows = np.repeat(flat_idx, nbrs.shape[1] - 1)
    cols = nbrs[:, 1:].ravel()
    mask = rows < cols
    return rows[mask], cols[mask]


def _build_knn_edges_healpix_full(
    nside: int, k: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Build KNN edges for the full HealPix grid (all npix pixels).

    Args:
        nside: HealPix nside parameter.
        k: Number of nearest neighbors.

    Returns:
        Tuple (knn_rows, knn_cols) of undirected edges between HealPix pixel indices.
    """
    regridder = HealPixRegridder(nside=nside)
    npix = 12 * nside**2
    lats = np.zeros(npix)
    lons = np.zeros(npix)
    for i in range(npix):
        lats[i], lons[i] = regridder.healpix_to_latlon(i)
    xyz = _latlon_to_unit_xyz(lats, lons)
    nn = NearestNeighbors(n_neighbors=min(k + 1, npix))
    nn.fit(xyz)
    _, nbrs = nn.kneighbors(xyz)
    flat_idx = np.arange(npix, dtype=np.int64)
    rows = np.repeat(flat_idx, nbrs.shape[1] - 1)
    cols = nbrs[:, 1:].ravel()
    mask = rows < cols
    return rows[mask], cols[mask]


def run_healpix_consensus(
    hp_cluster_masks: list[list[tuple[int, np.ndarray]]],
    nside: int,
    min_consensus: float = 0.5,
    min_cluster_size: int = 1,
    k_neighbors: int = 8,
    top_n_clusters: int | None = None,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run consensus clustering on HealPix-native cluster masks (ever-in semantics).

    Args:
        hp_cluster_masks: Per-model list of (cluster_id, mask_bool). Each mask is bool (npix,).
            A pixel can participate in multiple clusters across time.
        nside: HealPix nside (must match the masks).
        min_consensus: Minimum consensus threshold in [0, 1].
        min_cluster_size: Minimum cluster size; smaller clusters demoted to -1.
        k_neighbors: K for KNN graph on HealPix.
        top_n_clusters: If set, only consider top N clusters by size per model.
        show_progress: Whether to show tqdm progress.

    Returns:
        Tuple (labels, consistency) each of shape (npix,).
    """
    if nside <= 0:
        raise ValueError("nside must be positive.")
    npix = 12 * nside**2
    for model_masks in hp_cluster_masks:
        for cid, mask in model_masks:
            if mask.shape != (npix,):
                raise ValueError(
                    f"All masks must have shape ({npix},). Got {mask.shape}."
                )

    ever_clustered = np.zeros(npix, dtype=bool)
    for model_masks in hp_cluster_masks:
        for _cid, mask in model_masks:
            ever_clustered |= mask

    knn_rows, knn_cols = _build_knn_edges_healpix_full(nside, k=k_neighbors)

    rows_V, cols_V = [], []
    rows_A, cols_A = [], []

    for model_masks in tqdm(hp_cluster_masks, disable=not show_progress):
        to_process = list(model_masks)
        if top_n_clusters is not None and top_n_clusters > 0:
            counts = [(i, np.sum(mask)) for i, (cid, mask) in enumerate(to_process)]
            counts.sort(key=lambda x: x[1], reverse=True)
            idx = [c[0] for c in counts[:top_n_clusters]]
            to_process = [to_process[i] for i in idx]
        for cid, mask in to_process:
            rV, cV = _knn_edges_from_mask(mask, knn_rows, knn_cols)
            rows_V.extend(rV)
            cols_V.extend(cV)
        rA, cA = _knn_edges_from_mask(ever_clustered, knn_rows, knn_cols)
        rows_A.extend(rA)
        cols_A.extend(cA)

    if len(rows_V) == 0:
        labels = np.full(npix, np.nan, dtype=np.float32)
        labels[ever_clustered] = -1
        consistency = np.full(npix, np.nan, dtype=np.float32)
        return labels, consistency

    W = _compute_weighted_consensus(
        rows_V, cols_V, rows_A, cols_A, (npix, npix), min_consensus
    )
    if W.nnz == 0:
        labels = np.full(npix, np.nan, dtype=np.float32)
        labels[ever_clustered] = -1
        consistency = np.full(npix, np.nan, dtype=np.float32)
        return labels, consistency

    node_sum = np.array(W.sum(axis=1)).ravel()
    node_deg = np.array(W.count_nonzero(axis=1)).ravel().astype(np.float32)
    consistency = np.divide(
        node_sum, node_deg, out=np.zeros_like(node_sum), where=node_deg > 0
    ).astype(np.float32)

    bin_adj = W.copy()
    bin_adj.data[:] = 1.0
    bin_adj = bin_adj.maximum(bin_adj.T)
    _, labels_flat = connected_components(bin_adj, directed=False, return_labels=True)

    deg = np.array(bin_adj.getnnz(axis=1))
    labels_flat = labels_flat.astype(np.float32)
    labels_flat[deg == 0] = -1

    labels_flat = sorted_cluster_labels(labels_flat)

    if min_cluster_size > 1:
        valid_mask = (labels_flat >= 0) & np.isfinite(labels_flat)
        for cid in np.unique(labels_flat[valid_mask]):
            if np.sum(labels_flat == cid) < min_cluster_size:
                labels_flat[labels_flat == cid] = -1
        labels_flat = sorted_cluster_labels(labels_flat)

    labels_flat = labels_flat.astype(np.float32)  # allow NaN for ~ever_clustered
    labels_flat[~ever_clustered] = np.nan
    consistency[~ever_clustered] = np.nan

    return labels_flat, consistency


def run_native_consensus(
    cluster_masks: list[list[tuple[int, np.ndarray]]],
    n_spatial: int,
    knn_rows: np.ndarray,
    knn_cols: np.ndarray,
    min_consensus: float = 0.5,
    min_cluster_size: int = 1,
    top_n_clusters: int | None = None,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run consensus clustering on native-format cluster masks (ever-in semantics).

    Same algorithm as run_healpix_consensus but for arbitrary spatial grids.
    KNN edges must be precomputed from the native grid coordinates (e.g. x,y or lat,lon).

    Args:
        cluster_masks: Per-model list of (cluster_id, mask_bool). Each mask is bool (n_spatial,).
        n_spatial: Number of spatial points (all masks must have this size).
        knn_rows, knn_cols: Precomputed KNN edges (undirected, i < j).
        min_consensus: Minimum consensus threshold in [0, 1].
        min_cluster_size: Minimum cluster size; smaller demoted to -1.
        top_n_clusters: If set, only consider top N clusters by size per model.
        show_progress: Whether to show tqdm progress.

    Returns:
        Tuple (labels, consistency) each of shape (n_spatial,).
    """
    for model_masks in cluster_masks:
        for cid, mask in model_masks:
            if mask.shape != (n_spatial,):
                raise ValueError(
                    f"All masks must have shape ({n_spatial},). Got {mask.shape}."
                )

    ever_clustered = np.zeros(n_spatial, dtype=bool)
    for model_masks in cluster_masks:
        for _cid, mask in model_masks:
            ever_clustered |= mask

    rows_V, cols_V = [], []
    rows_A, cols_A = [], []

    for model_masks in tqdm(cluster_masks, disable=not show_progress):
        to_process = list(model_masks)
        if top_n_clusters is not None and top_n_clusters > 0:
            counts = [(i, np.sum(mask)) for i, (cid, mask) in enumerate(to_process)]
            counts.sort(key=lambda x: x[1], reverse=True)
            idx = [c[0] for c in counts[:top_n_clusters]]
            to_process = [to_process[i] for i in idx]
        for cid, mask in to_process:
            rV, cV = _knn_edges_from_mask(mask, knn_rows, knn_cols)
            rows_V.extend(rV)
            cols_V.extend(cV)
        rA, cA = _knn_edges_from_mask(ever_clustered, knn_rows, knn_cols)
        rows_A.extend(rA)
        cols_A.extend(cA)

    if len(rows_V) == 0:
        labels = np.full(n_spatial, np.nan, dtype=np.float32)
        labels[ever_clustered] = -1
        consistency = np.full(n_spatial, np.nan, dtype=np.float32)
        return labels, consistency

    W = _compute_weighted_consensus(
        rows_V, cols_V, rows_A, cols_A, (n_spatial, n_spatial), min_consensus
    )
    if W.nnz == 0:
        labels = np.full(n_spatial, np.nan, dtype=np.float32)
        labels[ever_clustered] = -1
        consistency = np.full(n_spatial, np.nan, dtype=np.float32)
        return labels, consistency

    node_sum = np.array(W.sum(axis=1)).ravel()
    node_deg = np.array(W.count_nonzero(axis=1)).ravel().astype(np.float32)
    consistency = np.divide(
        node_sum, node_deg, out=np.zeros_like(node_sum), where=node_deg > 0
    ).astype(np.float32)

    bin_adj = W.copy()
    bin_adj.data[:] = 1.0
    bin_adj = bin_adj.maximum(bin_adj.T)
    _, labels_flat = connected_components(bin_adj, directed=False, return_labels=True)

    deg = np.array(bin_adj.getnnz(axis=1))
    labels_flat = labels_flat.astype(np.float32)
    labels_flat[deg == 0] = -1

    labels_flat = sorted_cluster_labels(labels_flat)

    if min_cluster_size > 1:
        valid_mask = (labels_flat >= 0) & np.isfinite(labels_flat)
        for cid in np.unique(labels_flat[valid_mask]):
            if np.sum(labels_flat == cid) < min_cluster_size:
                labels_flat[labels_flat == cid] = -1
        labels_flat = sorted_cluster_labels(labels_flat)

    labels_flat = labels_flat.astype(np.float32)
    labels_flat[~ever_clustered] = np.nan
    consistency[~ever_clustered] = np.nan

    return labels_flat, consistency


def _build_knn_edges_from_regridder(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    k: int = 8,
    regridder: BaseRegridder | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build undirected edges using KNN after mapping to a regularized grid (e.g., HealPix).

    This function correctly computes KNN on HealPix pixel centers rather than original
    grid coordinates, avoiding circular cluster artifacts around the poles.

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

    # Currently only HealPixRegridder is supported for consensus clustering
    # because we need to convert regridded indices back to lat/lon centers
    if not isinstance(regridder, HealPixRegridder):
        raise ValueError(
            f"Only HealPixRegridder is currently supported for consensus clustering. "
            f"Got {type(regridder).__name__}. "
            f"This restriction may be lifted in the future if a generic interface is added."
        )

    # Map original grid cells to HealPix pixel indices
    hp_index_flat = regridder.map_orig_to_regrid(coords_latlon_flat)

    # Get unique HealPix pixel indices that are actually used
    unique_hp_pixels = np.unique(hp_index_flat)
    N_hp = len(unique_hp_pixels)

    if N_hp == 0:
        return (
            np.array([], np.int64),
            np.array([], np.int64),
            hp_index_flat,
        )

    # Get center coordinates of each unique HealPix pixel
    # This ensures we compute KNN on the actual HealPix grid, not the original grid
    hp_centers_lat = np.zeros(N_hp)
    hp_centers_lon = np.zeros(N_hp)
    for i, hp_pix in enumerate(unique_hp_pixels):
        lat, lon = regridder.healpix_to_latlon(int(hp_pix))
        hp_centers_lat[i] = lat
        hp_centers_lon[i] = lon

    # Convert HealPix pixel centers to 3D Cartesian coordinates
    xyz_hp = _latlon_to_unit_xyz(hp_centers_lat, hp_centers_lon)

    # Compute KNN on HealPix pixel centers
    nn = NearestNeighbors(n_neighbors=min(k + 1, N_hp))
    nn.fit(xyz_hp)
    _, nbrs = nn.kneighbors(xyz_hp)

    # Build edges between HealPix pixel indices
    knn_rows = np.repeat(unique_hp_pixels, nbrs.shape[1] - 1)
    knn_cols = unique_hp_pixels[nbrs[:, 1:].ravel()]

    # Keep only undirected edges (i < j)
    keep = knn_rows < knn_cols
    knn_rows = knn_rows[keep]
    knn_cols = knn_cols[keep]

    return knn_rows, knn_cols, hp_index_flat


def _build_consensus_summary_df(
    td,
    labels2d: xr.DataArray,
    consistency2d: xr.DataArray,
    spatial_dims: Tuple[str, str],
) -> pd.DataFrame:
    """Build a summary DataFrame of cluster statistics from 2D label and consistency arrays.

    Args:
        td: TOAD object containing clustering results.
        labels2d: 2D DataArray of consensus cluster labels. NaN = no abrupt shifts
            detected in any input; -1 = shifts detected but not in consensus cluster;
            values >= 0 = cluster membership.
        consistency2d: 2D DataArray of consensus consistency scores.
        spatial_dims: Tuple of spatial dimension names.

    Returns:
        DataFrame with one row per consensus cluster, containing statistics like
        cluster_id, mean_consistency, size, spatial means, and transition time metrics.
    """
    sd0, sd1 = spatial_dims
    dim = labels2d.name if labels2d.name else "cluster"
    cluster_map = labels2d.where((labels2d >= 0) & labels2d.notnull())

    valid_labels = labels2d.values
    has_any_cluster = np.any((valid_labels >= 0) & ~np.isnan(valid_labels))
    if not has_any_cluster:
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

    if len(td.cluster_vars) == 0:
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
        consensus_cluster_ids = df["cluster_id"].values.astype(int)
        mean_mean_list = []
        std_mean_list = []
        mean_std_list = []
        std_std_list = []

        for cid in consensus_cluster_ids:
            region_mask = labels2d == cid
            means_per_var = []
            stds_per_var = []
            for cluster_var in td.cluster_vars:
                times = td.get_cluster_times_in_region(
                    region_mask, cluster_var=cluster_var, numeric=True
                )
                if len(times) > 0:
                    means_per_var.append(np.nanmean(times))
                    stds_per_var.append(np.nanstd(times) if len(times) > 1 else 0.0)

            valid_means = [m for m in means_per_var if np.isfinite(m)]
            valid_stds = [s for s in stds_per_var if np.isfinite(s)]
            mean_mean_list.append(
                np.float32(np.nanmean(valid_means)) if valid_means else np.nan
            )
            std_mean_list.append(
                np.float32(np.nanstd(valid_means)) if len(valid_means) > 1 else 0.0
            )
            mean_std_list.append(
                np.float32(np.nanmean(valid_stds)) if valid_stds else 0.0
            )
            std_std_list.append(
                np.float32(np.nanstd(valid_stds)) if len(valid_stds) > 1 else 0.0
            )

        df_transitions = pd.DataFrame(
            {
                "cluster_id": consensus_cluster_ids,
                "mean_mean_shift_time": np.array(mean_mean_list, dtype=np.float32),
                "std_mean_shift_time": np.array(std_mean_list, dtype=np.float32),
                "mean_std_shift_time": np.array(mean_std_list, dtype=np.float32),
                "std_std_shift_time": np.array(std_std_list, dtype=np.float32),
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
