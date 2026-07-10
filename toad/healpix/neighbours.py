"""HEALPix spatial neighbourhood helpers."""

from __future__ import annotations

import healpix_geo.auto as hp_auto
import numpy as np

from toad.healpix.grid import HealpixGrid


def k_ring_neighbourhood(
    ipix: np.ndarray,
    grid: HealpixGrid,
    *,
    ring: int,
) -> np.ndarray:
    """Return all cells within ``ring`` HEALPix hops (includes each input cell)."""
    if ring < 0:
        raise ValueError(f"ring must be >= 0, got {ring}.")
    ipix_in = np.asarray(ipix, dtype=np.uint64)
    return hp_auto.kth_neighbourhood(ipix_in, grid.to_geo_grid(), ring=ring)


def build_ring1_spatial_edges(nside: int) -> tuple[np.ndarray, np.ndarray]:
    """Undirected ring-1 adjacency edges for the full HEALPix pixel set."""
    grid = HealpixGrid(nside=nside)
    npix = grid.npix
    pixels = np.arange(npix, dtype=np.uint64)
    nbrs = k_ring_neighbourhood(pixels, grid, ring=1)

    rows: list[int] = []
    cols: list[int] = []
    for pixel, neighbours in enumerate(nbrs):
        for neighbour in neighbours:
            n = int(neighbour)
            if n != pixel and pixel < n:
                rows.append(pixel)
                cols.append(n)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)
