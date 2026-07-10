"""HEALPix cell polygon geometry for map rendering."""

from __future__ import annotations

import healpix_geo.ring as hp_ring
import numpy as np

from toad.healpix.convert import normalize_lon_180
from toad.healpix.grid import HealpixGrid


def ipix_vertices(
    ipix: np.ndarray,
    grid: HealpixGrid,
    *,
    step: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Return polygon vertex (lon, lat) rings with shape (N, 4 * step)."""
    lon, lat = hp_ring.vertices(
        np.asarray(ipix, dtype=np.uint64),
        grid.depth,
        grid.ellipsoid,
        step=step,
    )
    return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)


def polygon_path(lons_row: np.ndarray, lats_row: np.ndarray) -> np.ndarray:
    """Build one Cartopy-safe polygon ring (handles dateline-crossing cells)."""
    lon = normalize_lon_180(lons_row)
    lat = np.asarray(lats_row, dtype=np.float64)
    if np.ptp(lon) > 180.0:
        lon = np.where(lon < 0.0, lon + 360.0, lon)
    # Cartopy PlateCarree expects [-180, 180] on most projections.
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return np.column_stack([lon, lat])


def polygon_paths(lons: np.ndarray, lats: np.ndarray) -> list[np.ndarray]:
    """Build vertex paths for :class:`matplotlib.collections.PolyCollection`."""
    return [polygon_path(row_lon, row_lat) for row_lon, row_lat in zip(lons, lats)]
