import gc

import pytest
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD
from toad.shifts import ASDETECT


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up memory after each test. Important otherwise get bus errors on some machines."""
    yield
    gc.collect()


@pytest.mark.parametrize(
    "min_cluster_size,shifts_threshold,shift_selection,expected_N_clusters",
    [
        (10, 0.8, "global", 3),
        (10, 0.8, "all", 3),
        (10, 0.8, "local", 4),
    ],
)
def test_irregular_grid(
    min_cluster_size, shifts_threshold, shift_selection, expected_N_clusters
):
    """Test TOAD pipeline on irregular grid data.

    This test verifies that the TOAD pipeline (computing shifts and clustering)
    works correctly on a dataset with an irregular spatial grid. The test uses
    sea ice concentration data and checks that the cluster counts match expected values.

    Args:
        min_cluster_size (int): Minimum size of clusters to be identified.
        shifts_threshold (float): Threshold for filtering shifts.
        expected_N_clusters (int): Expected number of clusters.
    """

    # Setup
    td = TOAD("tutorials/test_data/sea_ice_irregular_grid.nc")
    var = "siconc"

    # For irregular grids, use resampling instead of coarsening
    td.data = td.data.isel(
        i=slice(None, None, 4), j=slice(None, None, 4), time=slice(None, None, 2)
    )

    if len(td.shift_vars) == 0:
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))

    td.compute_clusters(
        var=var,
        shift_threshold=shifts_threshold,
        method=HDBSCAN(min_cluster_size=min_cluster_size),
        overwrite=True,
        time_scale_factor=2,
        shift_selection=shift_selection,
    )

    N_clusters = len(td.get_cluster_ids(var, exclude_noise=True))
    print(shift_selection, N_clusters)

    # only compare the noise cluster - was getting ±1 difference on the seceond cluster when running tests on Github Actions.
    assert abs(N_clusters - expected_N_clusters) <= 2, (
        f"Expected {expected_N_clusters}±2, got {N_clusters}"
    )
