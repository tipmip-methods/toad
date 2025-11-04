import gc
from pathlib import Path

import matplotlib.pyplot as plt
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
        i=slice(None, None, 2),
        j=slice(None, None, 2),
        time=slice(None, None, 2),
    )
    return td


def setup_native_grid():
    """Setup and coarsen native grid data."""
    td = TOAD("tutorials/test_data/garbe_2020_antarctica.nc", time_dim="GMST")
    td.data = td.data.coarsen(x=4, y=4, GMST=3, boundary="trim").reduce(np.mean)
    return td


def setup_regular_latlon_grid():
    """Setup and coarsen regular lat/lon grid data."""
    td = TOAD("tutorials/test_data/global_mean_summer_tas.nc", time_dim="time")
    td.data = td.data.coarsen(lat=3, lon=3, time=3, boundary="trim").reduce(np.mean)
    return td


@pytest.mark.parametrize(
    "setup_func,time_scale_factors,expected_min_clusters,expected_max_clusters,expected_mean_shift_time,time_tolerance",
    [
        (
            setup_irregular_grid,
            [0.5, 1.0, 1.5, 2.0],
            4,
            6,
            1890.0,  # Typical value from [1910.7632, 1899.7142, 1887., 1873.4286]
            5.0,  # tolerance in years
        ),
        (
            setup_native_grid,
            [0.25, 0.5, 1.0, 1.5],
            4,
            6,
            7.5,  # Typical value from [1.9118391, 7.5021663, 7.4890475, 9.74135, 3.7066216, 2.5101]
            1.0,  # tolerance
        ),
        (
            setup_regular_latlon_grid,
            [0.5, 1.0, 1.5, 2.0],
            35,
            40,
            2028.0,  # Typical value from [2085., 2027.5714, 2085., 2028., 2028., 2029.5, 2030., 2028., 2026.5, 2026.5, 2028., 2026.5]
            10.0,  # tolerance in years
        ),
    ],
)
def test_cluster_consensus(
    setup_func,
    time_scale_factors,
    expected_min_clusters,
    expected_max_clusters,
    expected_mean_shift_time,
    time_tolerance,
    request,
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
        expected_mean_shift_time (float): Expected mean shift time value (None means skip check).
        time_tolerance (float): Tolerance for mean shift time comparison.
    """
    # Setup
    td = setup_func()

    # Drop any existing cluster variables
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")

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
        min_consensus=0.8, top_n_clusters=5
    )

    # Create plot with meaningful title based on configuration
    setup_name = setup_func.__name__.replace("setup_", "")
    title = (
        f"Consensus Clusters - {setup_name}\n"
        f"time_scale_factors={time_scale_factors}, "
        f"min_consensus=0.8, top_n_clusters=5\n"
        f"Found {len(np.unique(ds_consensus.clusters.values[ds_consensus.clusters.values >= 0]))} clusters"
    )

    fig, ax = plt.subplots(figsize=(10, 8))
    ds_consensus.clusters.plot(ax=ax)
    ax.set_title(title, fontsize=12)
    plt.tight_layout()

    # Save plot to artifacts directory
    artifacts_dir = Path("test_artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    # Generate filename based on test configuration
    # Use nodeid to get unique test identifier (includes parameters)
    test_id = request.node.name.replace("[", "_").replace("]", "").replace(":", "_")
    filename = f"consensus_clusters_{setup_name}_{test_id}.png"
    filepath = artifacts_dir / filename
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

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

        # Check that mean_mean_shift_time values are valid and match expected value (if provided)
        # Consensus clusters should always have valid transition times (at least one clustering
        # should have valid times for pixels in the consensus cluster)
        if "mean_mean_shift_time" in summary_df.columns:
            mean_shift_times = summary_df["mean_mean_shift_time"].values
            # All values should be finite - consensus clusters shouldn't have all-NaN transition times
            assert np.all(np.isfinite(mean_shift_times)), (
                f"mean_mean_shift_time contains invalid (NaN) values: {mean_shift_times}. "
                "Consensus clusters should always have valid transition times."
            )
            # Check if any cluster has mean time close to expected value (if provided)
            if expected_mean_shift_time is not None:
                differences = np.abs(mean_shift_times - expected_mean_shift_time)
                min_diff = np.min(differences)
                assert min_diff <= time_tolerance, (
                    f"No cluster found with mean_mean_shift_time within {time_tolerance} "
                    f"of expected {expected_mean_shift_time}. "
                    f"Actual values: {mean_shift_times}, "
                    f"min difference: {min_diff}"
                )
            else:
                # If expected value not provided, just print the values for debugging
                print(f"\nmean_mean_shift_times: {mean_shift_times}")
    else:
        # If no clusters found, verify this is expected (expected_min_clusters == 0)
        assert expected_min_clusters == 0, (
            f"Expected at least {expected_min_clusters} clusters, but none were found"
        )
        assert len(summary_df) == 0, (
            f"Expected empty summary dataframe when no clusters found, "
            f"but got {len(summary_df)} rows"
        )
