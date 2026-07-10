"""Lon/lat ↔ HEALPix index conversions via healpix-geo."""

from __future__ import annotations

import healpix_geo.ring as hp_ring
import numpy as np

from toad.healpix.grid import HealpixGrid


def normalize_lon_180(lon: np.ndarray) -> np.ndarray:
    """Wrap longitudes to [-180, 180] degrees."""
    lon_arr = np.asarray(lon, dtype=np.float64)
    return ((lon_arr + 180.0) % 360.0) - 180.0


def lonlat_to_ipix(
    lon: np.ndarray,
    lat: np.ndarray,
    grid: HealpixGrid,
) -> np.ndarray:
    """Map geographic coordinates to ring HEALPix pixel indices."""
    lon_in = normalize_lon_180(lon)
    lat_in = np.asarray(lat, dtype=np.float64)
    ipix = hp_ring.lonlat_to_healpix(
        lon_in,
        lat_in,
        grid.depth,
        grid.ellipsoid,
    )
    return np.asarray(ipix, dtype=np.int64)


def ipix_to_lonlat(
    ipix: np.ndarray,
    grid: HealpixGrid,
    *,
    lon_convention: str = "360",
) -> tuple[np.ndarray, np.ndarray]:
    """Map ring HEALPix pixel indices to (lat, lon) in degrees."""
    ipix_in = np.asarray(ipix, dtype=np.uint64)
    lon, lat = hp_ring.healpix_to_lonlat(ipix_in, grid.depth, grid.ellipsoid)
    lats = np.asarray(lat, dtype=np.float64)
    lons = np.asarray(lon, dtype=np.float64)
    if lon_convention == "360":
        lons = np.mod(lons, 360.0)
    elif lon_convention != "180":
        raise ValueError(
            f"lon_convention must be '360' or '180', got {lon_convention!r}."
        )
    return lats, lons
