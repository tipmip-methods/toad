import gc

import numpy as np
import pytest
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD
from toad.regridding import HealPixRegridder


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up memory after each test. Important otherwise get bus errors on some machines."""
    yield
    gc.collect()


def test_healpix_hdbscan():
    """Test the HealPix HDBSCAN clustering method.

    Verifies that clustering runs correctly on coarsened synth_data with HealPix
    regridding. Cluster count has ±1 tolerance for HDBSCAN variability.

    Note:
        A RuntimeWarning about numpy.ndarray size is suppressed in pytest.ini
        (known HDBSCAN/numpy issue: https://github.com/scikit-learn-contrib/hdbscan/issues/457)
    """
    td = TOAD("tutorials/test_data/synth_data.nc")
    td.data = td.data.coarsen(lat=2, lon=2, boundary="trim").reduce(np.mean)

    td.compute_clusters(
        shift_threshold=0.5,
        method=HDBSCAN(min_cluster_size=25),
        overwrite=True,
        shift_selection="all",
    )

    N_clusters = len(td.get_cluster_ids(td.base_vars[0], exclude_noise=True))
    assert abs(N_clusters - 2) <= 1, f"Expected 2±1 clusters, got {N_clusters}"


def test_healpix_regridder_roundtrip_and_index_dtype():
    regridder = HealPixRegridder(nside=8)

    lat = np.array([-45.0, 0.0, 45.0], dtype=np.float64)
    lon = np.array([10.0, 180.0, 350.0], dtype=np.float64)

    hp_idx = regridder.latlon_to_healpix(lat, lon)

    assert hp_idx.dtype == np.int64
    assert hp_idx.shape == lat.shape

    lat0, lon0 = regridder.healpix_to_latlon(int(hp_idx[0]))
    assert isinstance(lat0, float)
    assert isinstance(lon0, float)
    assert -90.0 <= lat0 <= 90.0
    assert 0.0 <= lon0 < 360.0

    mapped = regridder.map_orig_to_regrid(np.column_stack([lat, lon]))
    np.testing.assert_array_equal(mapped, hp_idx)
