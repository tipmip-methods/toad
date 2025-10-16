import gc

import numpy as np
import pytest
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up memory after each test. Important otherwise get bus errors on some machines."""
    yield
    gc.collect()


@pytest.mark.parametrize(
    "lat,lon,min_cluster_size,shifts_threshold,shift_selection,expected_N_clusters",
    [
        (6, 6, 25, 0.5, "all", 7),  # First parameter set
        (6, 6, 25, 0.5, "local", 2),  # Second parameter set
    ],
)
def test_healpix_hdbscan(
    lat, lon, min_cluster_size, shifts_threshold, shift_selection, expected_N_clusters
):
    """Test the HealPix HDBSCAN clustering method.

    This test verifies the clustering of data using the HDBSCAN algorithm
    after coarsening the data based on specified latitude and longitude
    parameters. It also performs HealPix regridding to ensure the data
    is appropriately structured for clustering. It checks that the actual
    cluster counts and nside values match the expected results.

    Args:
        lat (int): Latitude coarsening factor.
        lon (int): Longitude coarsening factor.
        min_cluster_size (int): Minimum size of clusters to be identified.
        shifts_threshold (float): Threshold for filtering shifts.
        shift_selection (str): How shift values are selected for clustering.
        expected_N_clusters (int): Expected number of clusters.

    Note:
        This throws a warning the following warning which has been surpressed in pytest.ini: "RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility".
        This is due to a known issue with the HDBSCAN library and numpy. https://github.com/scikit-learn-contrib/hdbscan/issues/457#issuecomment-1004296870
    """

    # Setup
    td = TOAD("tutorials/test_data/global_mean_summer_tas.nc")
    td.data = td.data.coarsen(lat=lat, lon=lon, boundary="trim").reduce(np.mean)

    td.compute_clusters(
        shift_threshold=shifts_threshold,
        method=HDBSCAN(min_cluster_size=min_cluster_size),
        overwrite=True,
        shift_selection=shift_selection,
    )

    # Verify results
    N_clusters = len(td.get_cluster_ids(td.base_vars[0], exclude_noise=True))
    assert abs(N_clusters - expected_N_clusters) <= 2, (
        f"Expected {expected_N_clusters}±2, got {N_clusters}"
    )
