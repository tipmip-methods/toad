"""HEALPix cell polygon geometry for map rendering."""

from __future__ import annotations

import healpix_geo.ring as hp_ring
import numpy as np

from toad.healpix.convert import ipix_to_lonlat, normalize_lon_180
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


def _projection_center_lonlat(projection) -> tuple[float, float] | None:
    """Return ``(central_longitude, central_latitude)`` for centred projections."""
    import cartopy.crs as ccrs

    if isinstance(projection, ccrs.Orthographic):
        return (
            float(projection.proj4_params.get("lon_0", 0.0)),
            float(projection.proj4_params.get("lat_0", 0.0)),
        )
    return None


def visible_ipix_front_hemisphere(
    ipix: np.ndarray,
    grid: HealpixGrid,
    projection,
) -> np.ndarray:
    """Boolean mask (aligned with ``ipix``) for pixels on the projection's front hemisphere."""
    center = _projection_center_lonlat(projection)
    if center is None:
        return np.ones(ipix.shape, dtype=bool)

    clon, clat = center
    lats, lons = ipix_to_lonlat(ipix, grid, lon_convention="180")
    clat_r = np.deg2rad(clat)
    clon_r = np.deg2rad(clon)
    lat_r = np.deg2rad(lats)
    lon_r = np.deg2rad(lons)
    x = np.cos(lat_r) * np.cos(lon_r)
    y = np.cos(lat_r) * np.sin(lon_r)
    z = np.sin(lat_r)
    x0 = np.cos(clat_r) * np.cos(clon_r)
    y0 = np.cos(clat_r) * np.sin(clon_r)
    z0 = np.sin(clat_r)
    return (x * x0 + y * y0 + z * z0) > 0.0
