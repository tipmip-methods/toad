import pytest
import numpy as np
from toad import TOAD

# from sklearn.cluster import HDBSCAN  # type: ignore
import fast_hdbscan


@pytest.fixture
def test_params():
    """Fixture providing parameters for the clustering test.
    Returns:
        dict: A dictionary containing:
            - lat (int): Latitude coarsening factor.
            - lon (int): Longitude coarsening factor.
            - min_cluster_size (int): Minimum size of clusters to be identified.
            - shifts_threshold (float): Threshold for filtering shifts.
            - expected_results (dict): Expected cluster counts for validation.
            - expected_nside (int): Expected nside value for the regridder.
    """
    return {
        "lat": 6,
        "lon": 6,
        "min_cluster_size": 25,
        "shifts_threshold": 0.5,
        "expected_results": {-1: 330239, 1: 2845, 0: 914, 2: 292, 3: 78},
    }


@pytest.fixture
def toad_instance():
    return TOAD("../tutorials/test_data/global_mean_summer_tas.nc")


def test_healpix_hdbscan(test_params, toad_instance):
    """Test the HealPix HDBSCAN clustering method.

    This test verifies the clustering of data using the HDBSCAN algorithm
    after coarsening the data based on specified latitude and longitude
    parameters. It also performs HealPix regridding to ensure the data
    is appropriately structured for clustering. It checks that the actual
    cluster counts and nside values match the expected results.

    Args:
        test_params (dict): Parameters for the test.
        toad_instance (TOAD): Instance of TOAD containing the data.

    Note:
        This warning caused by the HDBSCAN library may appear: "RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility".
        This is due to a known issue with the HDBSCAN library and numpy. https://github.com/scikit-learn-contrib/hdbscan/issues/457#issuecomment-1004296870
    """

    # Setup
    td = toad_instance
    td.data = td.data.coarsen(
        lat=test_params["lat"], lon=test_params["lon"], boundary="trim"
    ).mean()

    td.compute_clusters(
        "tas",
        shifts_filter_func=lambda x: np.abs(x) > test_params["shifts_threshold"],
        method=fast_hdbscan.HDBSCAN(min_cluster_size=test_params["min_cluster_size"]),
        overwrite=True,
    )

    # Verify results
    actual_counts = td.get_cluster_counts("tas")

    assert actual_counts == test_params["expected_results"], (
        f"Expected {test_params['expected_results']}, got {actual_counts}"
    )
