"""Consensus clustering utilities for MMA (multi-model aggregation)."""

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from toad.clustering import sorted_cluster_labels
from toad.regridding.healpix import HealPixRegridder


def _latlon_to_unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """Convert (lat, lon) in degrees to unit sphere Cartesian coords."""
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)
    return np.stack([x, y, z], axis=-1)


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
    """Build KNN edges for the full HealPix grid (all npix pixels)."""
    regridder = HealPixRegridder(nside=nside)
    npix = 12 * nside**2
    lats, lons = regridder.pixels_to_latlon(np.arange(npix))
    coords = np.column_stack([lats, lons])
    return _build_knn_edges_from_coords_2d(coords, k, use_sphere=True)


def _knn_edges_from_mask(
    mask_bool_flat: np.ndarray, knn_rows: np.ndarray, knn_cols: np.ndarray
) -> tuple[list[int], list[int]]:
    """Return undirected KNN edges where both endpoints are True in mask_bool_flat."""
    both = mask_bool_flat[knn_rows] & mask_bool_flat[knn_cols]
    if not np.any(both):
        return [], []
    r = knn_rows[both]
    c = knn_cols[both]
    m = r < c
    return r[m].tolist(), c[m].tolist()


def _compute_weighted_consensus(
    rows_V: list[int],
    cols_V: list[int],
    rows_A: list[int],
    cols_A: list[int],
    shape: tuple[int, int],
    min_consensus: float,
) -> csr_matrix:
    """Build V, A CSR matrices, compute W=V/A on V support, threshold by min_consensus."""
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


def _run_consensus_core(
    cluster_masks: list[list[tuple[int, np.ndarray]]],
    n_spatial: int,
    knn_rows: np.ndarray,
    knn_cols: np.ndarray,
    min_consensus: float = 0.5,
    min_cluster_size: int = 1,
    top_n_clusters: int | None = None,
    show_progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared consensus logic for HealPix and native grids.

    Returns:
        Tuple (labels, consistency) each of shape (n_spatial,).
    """
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

    knn_rows, knn_cols = _build_knn_edges_healpix_full(nside, k=k_neighbors)
    return _run_consensus_core(
        hp_cluster_masks,
        n_spatial=npix,
        knn_rows=knn_rows,
        knn_cols=knn_cols,
        min_consensus=min_consensus,
        min_cluster_size=min_cluster_size,
        top_n_clusters=top_n_clusters,
        show_progress=show_progress,
    )


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

    return _run_consensus_core(
        cluster_masks,
        n_spatial=n_spatial,
        knn_rows=knn_rows,
        knn_cols=knn_cols,
        min_consensus=min_consensus,
        min_cluster_size=min_cluster_size,
        top_n_clusters=top_n_clusters,
        show_progress=show_progress,
    )
