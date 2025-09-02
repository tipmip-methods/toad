import pytest
from toad import TOAD

from sklearn.cluster import HDBSCAN  # type: ignore
from toad.shifts import ASDETECT


@pytest.fixture
def test_params():
    """Fixture providing parameters for the clustering test.
    Returns:
        dict: A dictionary containing:
            - min_cluster_size (int): Minimum size of clusters to be identified.
            - shifts_threshold (float): Threshold for filtering shifts.
            - expected_N_clusters (dict): Expected number of clusters for validation.
    """
    return {
        "min_cluster_size": 20,
        "shifts_threshold": 0.9,
        "expected_N_clusters": 2,
    }


@pytest.fixture
def toad_instance():
    return TOAD("tutorials/test_data/sea_ice_irregular_grid.nc")


def test_irregular_grid(test_params, toad_instance):
    """Test TOAD pipeline on irregular grid data.

    This test verifies that the TOAD pipeline (computing shifts and clustering)
    works correctly on a dataset with an irregular spatial grid. The test uses
    sea ice concentration data and checks that the cluster counts match expected values.

    Args:
        test_params (dict): Parameters for the test.
        toad_instance (TOAD): Instance of TOAD containing the data.
    """

    # Setup
    td = toad_instance
    var = "siconc"

    # For irregular grids, use resampling instead of coarsening
    td.data = td.data.isel(
        i=slice(None, None, 4), j=slice(None, None, 4), time=slice(None, None, 2)
    )

    td.compute_shifts(var, method=ASDETECT())

    td.compute_clusters(
        var=var,
        shift_threshold=test_params["shifts_threshold"],
        method=HDBSCAN(min_cluster_size=test_params["min_cluster_size"]),
        overwrite=True,
        time_scale_factor=1,
    )

    N_clusters = len(td.get_cluster_ids(var, exclude_noise=True))

    # only compare the noise cluster - was getting ±1 difference on the seceond cluster when running tests on Github Actions.
    assert N_clusters == test_params["expected_N_clusters"], (
        f"Expected {test_params['expected_N_clusters']}, got {N_clusters}"
    )
