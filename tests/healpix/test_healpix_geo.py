import numpy as np
import pytest

from toad.healpix import (
    HealpixGrid,
    build_ring1_spatial_edges,
    ipix_to_lonlat,
    k_ring_neighbourhood,
    lonlat_to_ipix,
    nside_to_depth,
    polygon_path,
)
from toad.regridding import HealPixRegridder


@pytest.mark.parametrize("nside", [4, 8, 16, 32])
def test_lonlat_ipix_roundtrip(nside: int):
    grid = HealpixGrid(nside=nside)
    lons = np.array([10.0, 180.0, 350.0, -45.0])
    lats = np.array([-45.0, 0.0, 45.0, 60.0])
    ipix = lonlat_to_ipix(lons, lats, grid)
    lats_out, lons_out = ipix_to_lonlat(ipix, grid)
    ipix_back = lonlat_to_ipix(lons_out, lats_out, grid)
    np.testing.assert_array_equal(ipix, ipix_back)


@pytest.mark.parametrize("nside", [8, 16])
def test_full_grid_centers_match_astropy(nside: int):
    astropy_healpix = pytest.importorskip("astropy_healpix")
    import astropy.units as u

    grid = HealpixGrid(nside=nside)
    pixels = np.arange(grid.npix, dtype=np.int64)
    lats, lons = ipix_to_lonlat(pixels, grid)
    lon_ap, lat_ap = astropy_healpix.healpix_to_lonlat(pixels, nside, order="ring")
    lons_ap = np.mod(lon_ap.to_value(u.deg), 360.0)
    lats_ap = lat_ap.to_value(u.deg)
    np.testing.assert_allclose(lats, lats_ap, rtol=0, atol=1e-10)
    np.testing.assert_allclose(lons, lons_ap, rtol=0, atol=1e-10)


def test_nside_depth_mapping():
    assert nside_to_depth(8) == 3
    assert nside_to_depth(1) == 0


def test_regridder_matches_healpix_module():
    regridder = HealPixRegridder(nside=8)
    lat = np.array([-45.0, 0.0, 45.0], dtype=np.float64)
    lon = np.array([10.0, 180.0, 350.0], dtype=np.float64)
    hp_idx = regridder.latlon_to_healpix(lat, lon)
    grid = HealpixGrid(nside=8)
    expected = lonlat_to_ipix(lon, lat, grid)
    np.testing.assert_array_equal(hp_idx, expected)


def test_ring1_neighbours_include_eight_neighbors(nside: int = 8):
    grid = HealpixGrid(nside=nside)
    pixel = 100
    nbrs = k_ring_neighbourhood(np.array([pixel], dtype=np.uint64), grid, ring=1)[0]
    assert pixel in nbrs
    assert len(set(int(x) for x in nbrs)) == 9


def test_polygon_path_unwraps_dateline():
    # Vertices like healpix-geo returns near the prime meridian / dateline.
    path = polygon_path(np.array([315.0, 0.0, 315.0, 270.0]), np.zeros(4))
    assert np.ptp(path[:, 0]) <= 180.0


def test_polygon_path_unwraps_negative_180():
    path = polygon_path(np.array([135.0, -180.0, 135.0, 90.0]), np.zeros(4))
    assert np.ptp(path[:, 0]) <= 180.0


def test_build_ring1_spatial_edges_undirected():
    rows, cols = build_ring1_spatial_edges(nside=8)
    assert rows.shape == cols.shape
    assert np.all(rows < cols)
