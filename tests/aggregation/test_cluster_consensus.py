import gc

import numpy as np
import pytest
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD
from toad.shifts import ASDETECT


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up memory after each test. Important otherwise get bus errors on some machines."""
    yield
    gc.collect()


def setup_irregular_grid():
    """Setup and coarsen irregular grid data."""
    td = TOAD("tutorials/test_data/sea_ice_irregular_grid.nc", time_dim="time")
    td.data = td.data.isel(
        i=slice(None, None, 4),
        j=slice(None, None, 4),
        time=slice(None, None, 2),
    )
    return td


def setup_native_grid():
    """Setup and coarsen native grid data."""
    td = TOAD("tutorials/test_data/garbe_2020_antarctica.nc", time_dim="GMST")
    td.data = td.data.coarsen(x=2, y=2, GMST=2, boundary="trim").reduce(np.mean)
    return td


def setup_regular_latlon_grid():
    """Setup and coarsen regular lat/lon grid data."""
    td = TOAD("tutorials/test_data/global_mean_summer_tas.nc", time_dim="time")
    td.data = td.data.coarsen(lat=10, lon=10, time=3, boundary="trim").reduce(np.mean)
    return td


@pytest.mark.parametrize(
    "setup_func,time_scale_factors,expected_min_clusters,expected_max_clusters,expected_mean_shift_time_range",
    [
        (
            setup_irregular_grid,
            [0.5, 1.0, 1.5, 2.0],
            5,
            7,
            (1870.0, 1920.0),
        ),
        (
            setup_native_grid,
            [0.25, 0.5, 1.0, 1.5],
            10,
            12,
            (1.5, 13.0),
        ),
        (
            setup_regular_latlon_grid,
            [0.5, 1.0, 1.5, 2.0],
            12,
            14,
            (2020.0, 2090.0),
        ),
    ],
)
def test_cluster_consensus(
    setup_func,
    time_scale_factors,
    expected_min_clusters,
    expected_max_clusters,
    expected_mean_shift_time_range,
):
    """Test cluster_consensus on different grid types.

    This test verifies that the cluster_consensus function works correctly
    on three different grid types:
    1. Irregular grid (sea ice data with i, j dimensions)
    2. Native grid (Antarctica data with x, y dimensions)
    3. Regular lat/lon grid (global temperature data that needs regridding)

    For each grid type:
    - Coarsens the dataset to make computation faster
    - Computes 4 clusterings using different time_scale_factors
    - Calls cluster_consensus to create consensus clusters
    - Validates that the output dataset contains valid masks
    - Validates that the summary dataframe matches the consensus clusters
    - Checks that mean_mean_shift_time values are valid

    Args:
        setup_func (callable): Function that returns a configured TOAD object.
        time_scale_factors (list): List of time_scale_factor values for clustering.
        expected_min_clusters (int): Minimum expected number of consensus clusters.
        expected_max_clusters (int): Maximum expected number of consensus clusters.
        expected_mean_shift_time_range (tuple): (min, max) expected range for mean_mean_shift_time.
    """
    # Setup
    td = setup_func()

    # Drop any existing cluster variables
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    td.data = td.data.drop_vars(td.shift_vars, errors="ignore")

    # Compute shifts if not present
    if len(td.shift_vars) == 0:
        var = td.base_vars[0]
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))

    # Compute 4 clusterings with different time_scale_factors
    for tsf in time_scale_factors:
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=10),
            time_scale_factor=tsf,
            shift_threshold=0.8,
        )

    # Verify we have 4 clusterings
    assert len(td.cluster_vars) == len(time_scale_factors), (
        f"Expected {len(time_scale_factors)} clusterings, got {len(td.cluster_vars)}"
    )

    # Call consensus clustering spatial function
    ds_consensus, summary_df = td.aggregation().cluster_consensus(
        min_consensus=0.5, top_n_clusters=10
    )

    # Assert that the dataset contains valid clusters
    assert "clusters" in ds_consensus, "clusters not in output dataset"
    assert "consistency" in ds_consensus, "consistency not in output dataset"

    clusters = ds_consensus["clusters"]
    consistency = ds_consensus["consistency"]

    # Assert that clusters has valid shape and values
    assert clusters.shape == consistency.shape, (
        "clusters and consistency must have the same shape"
    )

    # Assert that values are valid (including -1 for noise)
    assert np.all(np.isfinite(clusters.values) | (clusters.values == -1)), (
        "clusters contains invalid (non-finite) values"
    )

    # Assert that consistency values are valid (0-1 range or NaN)
    valid_consistency = np.isfinite(consistency.values)
    if np.any(valid_consistency):
        assert np.all(consistency.values[valid_consistency] >= 0), (
            "consistency contains negative values"
        )
        assert np.all(consistency.values[valid_consistency] <= 1), (
            "consistency contains values > 1"
        )

    # Get unique cluster IDs (excluding noise = -1)
    unique_clusters = np.unique(clusters.values)
    unique_clusters = unique_clusters[unique_clusters >= 0]

    # Assert that the summary dataframe contains the same number of clusters
    if len(unique_clusters) > 0:
        # Verify we have the expected number of clusters
        assert expected_min_clusters <= len(unique_clusters) <= expected_max_clusters, (
            f"Expected {expected_min_clusters}-{expected_max_clusters} clusters, "
            f"got {len(unique_clusters)}"
        )

        assert len(summary_df) == len(unique_clusters), (
            f"Summary dataframe has {len(summary_df)} clusters, "
            f"but clusters has {len(unique_clusters)} unique cluster IDs"
        )

        # Assert that all cluster IDs in summary match unique clusters
        summary_cluster_ids = set(summary_df["cluster_id"].values)
        unique_cluster_set = set(unique_clusters)
        assert summary_cluster_ids == unique_cluster_set, (
            f"Summary cluster IDs {summary_cluster_ids} do not match "
            f"unique clusters {unique_cluster_set}"
        )

        # Check that mean_mean_shift_time values are valid and within expected range
        if "mean_mean_shift_time" in summary_df.columns:
            mean_shift_times = summary_df["mean_mean_shift_time"].values
            assert np.all(np.isfinite(mean_shift_times)), (
                f"mean_mean_shift_time contains invalid values: {mean_shift_times}"
            )
            # Check that all mean shift times are within expected range
            min_time, max_time = expected_mean_shift_time_range
            assert np.all(
                (mean_shift_times >= min_time) & (mean_shift_times <= max_time)
            ), (
                f"mean_mean_shift_time values {mean_shift_times} are outside "
                f"expected range [{min_time}, {max_time}]"
            )
    else:
        # If no clusters found, verify this is expected (expected_min_clusters == 0)
        assert expected_min_clusters == 0, (
            f"Expected at least {expected_min_clusters} clusters, but none were found"
        )
        assert len(summary_df) == 0, (
            f"Expected empty summary dataframe when no clusters found, "
            f"but got {len(summary_df)} rows"
        )
