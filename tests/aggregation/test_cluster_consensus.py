import gc

import numpy as np
import pytest
import xarray as xr
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD
from toad.shifts import ASDETECT
from toad.utils import _attrs


def _latest_consensus_labels(td: TOAD):
    """Consensus label DataArray from the most recent :meth:`TOAD.compute_consensus` run."""
    names = td.consensus_cluster_vars
    assert names, "expected at least one consensus_cluster variable on td"
    return td.data[names[-1]]


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
    td = TOAD("tutorials/test_data/synth_data.nc", time_dim="time")
    td.data = td.data.coarsen(lat=3, lon=3, time=3, boundary="trim").reduce(np.mean)
    return td


def setup_synthetic_consensus_toad(
    cluster_fields: dict[str, np.ndarray],
    *,
    base_values: np.ndarray | None = None,
    add_latlon: bool = False,
    cluster_id_attrs: dict[str, np.ndarray] | None = None,
) -> TOAD:
    """Build a tiny deterministic TOAD object with manual cluster fields."""
    first = next(iter(cluster_fields.values()))
    T, y_len, x_len = first.shape
    coords = {
        "time": np.arange(T, dtype=np.float32),
        "y": np.arange(y_len, dtype=np.int32),
        "x": np.arange(x_len, dtype=np.int32),
    }
    if base_values is None:
        base_values = np.broadcast_to(
            (np.arange(T, dtype=np.float32) + 1.0).reshape(T, 1, 1),
            (T, y_len, x_len),
        ).copy()

    ds = xr.Dataset(
        {
            "foo": xr.DataArray(
                np.asarray(base_values, dtype=np.float32),
                coords=coords,
                dims=("time", "y", "x"),
            )
        }
    )

    if add_latlon:
        lat = np.broadcast_to(
            np.linspace(-20.0, 20.0, y_len, dtype=np.float32).reshape(y_len, 1),
            (y_len, x_len),
        )
        lon = np.broadcast_to(
            np.linspace(0.0, 30.0, x_len, dtype=np.float32).reshape(1, x_len),
            (y_len, x_len),
        )
        ds = ds.assign_coords(
            {
                "lat": (("y", "x"), lat),
                "lon": (("y", "x"), lon),
            }
        )

    for name, labels in cluster_fields.items():
        arr = np.asarray(labels, dtype=np.float32)
        cluster_ids = (
            cluster_id_attrs[name]
            if cluster_id_attrs is not None and name in cluster_id_attrs
            else np.unique(arr[np.isfinite(arr)].astype(np.int32))
        )
        da = xr.DataArray(arr, coords=coords, dims=("time", "y", "x"), name=name)
        da.attrs.update(
            {
                _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER,
                _attrs.BASE_VARIABLE: "foo",
                _attrs.CLUSTER_IDS: np.asarray(cluster_ids, dtype=np.int32),
            }
        )
        ds[name] = da

    return TOAD(ds, time_dim="time")


def test_filter_consensus_labels_min_size():
    """Post-hoc consensus filter drops clusters with small spatial footprint."""
    from toad.postprocessing.aggregation import _filter_consensus_labels_min_size

    coords = {"time": [0, 1], "y": [0, 1], "x": [0, 1, 2]}
    lab = np.full((2, 2, 3), -1, dtype=np.int64)
    lab[:, 0, 0] = 0  # cluster 0: one spatial cell, two timesteps (area 1)
    lab[:, 1, :] = 1  # cluster 1: three spatial cells × two timesteps (area 3)
    cons = np.ones_like(lab, dtype=np.float32) * 0.9
    da_lab = xr.DataArray(lab, coords=coords, dims=("time", "y", "x"))
    da_cons = xr.DataArray(cons, coords=coords, dims=("time", "y", "x"))
    out_lab, out_cons = _filter_consensus_labels_min_size(
        da_lab, da_cons, min_cluster_area=3, time_dim="time"
    )
    # id 0 -> area 1 (removed); id 1 -> area 3 (kept as new id 0)
    assert np.sum(out_lab.values == 0) == 6
    assert np.sum(out_lab.values == -1) == 6
    assert np.all(np.isnan(out_cons.values[out_lab.values == -1]))


def non_noise_cluster_ids(da_clusters: xr.DataArray) -> np.ndarray:
    """Return sorted non-noise cluster ids present in a consensus field."""
    values = np.asarray(da_clusters.values)
    valid = values[np.isfinite(values)]
    if valid.size == 0:
        return np.array([], dtype=np.int32)
    ids = np.unique(valid.astype(np.int32))
    return ids[ids >= 0]


@pytest.mark.parametrize(
    "setup_func,time_weights,expected_mean_shift_time,time_tolerance",
    [
        (
            setup_irregular_grid,
            [0.5, 1.0, 1.5, 2.0],
            1890.0,  # Typical value from [1910.7632, 1899.7142, 1887., 1873.4286]
            5.0,  # tolerance in years
        ),
        (
            setup_native_grid,
            [0.25, 0.5, 1.0, 1.5],
            7.5,  # Typical value from [1.9118391, 7.5021663, 7.4890475, 9.74135, 3.7066216, 2.5101]
            1.0,  # tolerance
        ),
        (
            setup_regular_latlon_grid,
            [0.5, 1.0, 1.5, 2.0],
            47.5,  # Typical value from synth_data.nc with 2 shifts [47.548386, 139.66667]
            100.0,  # tolerance in years (to cover both shift times)
        ),
    ],
)
def test_cluster_consensus(
    setup_func,
    time_weights,
    expected_mean_shift_time,
    time_tolerance,
    request,
):
    """Test spacetime cluster_consensus on different grid types.

    This test verifies that the cluster_consensus function works correctly
    on three different grid types:
    1. Irregular grid (sea ice data with i, j dimensions)
    2. Native grid (Antarctica data with x, y dimensions)
    3. Regular lat/lon grid (global temperature data)

    For each grid type:
    - Coarsens the dataset to make computation faster
    - Computes 4 clusterings using different time_weights
    - Calls cluster_consensus to create consensus clusters
    - Validates that the output dataset contains valid masks
    - Validates that the summary dataframe matches the consensus clusters
    - Checks that median_median_shift_time values are valid

    Args:
        setup_func (callable): Function that returns a configured TOAD object.
        time_weights (list): List of time_weight values for clustering.
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

    # Compute 4 clusterings with different time_weights
    for tsf in time_weights:
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=10),
            time_weight=tsf,
            shift_threshold=0.8,
        )

    # Verify we have 4 clusterings
    assert len(td.cluster_vars) == len(time_weights), (
        f"Expected {len(time_weights)} clusterings, got {len(td.cluster_vars)}"
    )

    # Call spacetime consensus clustering
    td.compute_consensus(
        min_consensus=0.8,
        temporal_tolerance=0,
        spatial_tolerance=0,
        top_n_clusters=5,
    )
    cv = td.consensus_cluster_vars[-1]
    summary_df = td.aggregate.consensus_summary(cv)
    assert "pooled_median_shift_time" in summary_df.columns
    assert "pooled_std_shift_time" in summary_df.columns

    assert cv in td.data, "consensus labels not in td.data"
    assert f"{cv}_consistency" in td.data, "consistency not in td.data"

    clusters = td.data[cv]
    consistency = td.data[f"{cv}_consistency"]
    assert td.time_dim in clusters.dims

    # Assert that clusters has valid shape and values
    assert clusters.shape == consistency.shape, (
        "clusters and consistency must have the same shape"
    )

    # Assert that values are valid: NaN (no shift), -1 (shift but not in consensus), or ids
    cvals = np.asarray(clusters.values, dtype=np.float64)
    assert np.all(np.isfinite(cvals) | (cvals == -1) | np.isnan(cvals)), (
        "clusters must be NaN, -1, or non-negative id"
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

    # Get unique cluster IDs (excluding noise = -1 and NaN = no shift)
    unique_clusters = np.unique(clusters.values)
    unique_clusters = unique_clusters[
        np.isfinite(unique_clusters) & (unique_clusters >= 0)
    ]

    # Assert that the summary dataframe contains the same number of clusters
    if len(unique_clusters) > 0:
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

        # Check that median_median_shift_time values are valid and match expected value (if provided)
        # Consensus clusters should always have valid transition times (at least one clustering
        # should have valid times for pixels in the consensus cluster)
        if "median_median_shift_time" in summary_df.columns:
            mean_shift_times = summary_df["median_median_shift_time"].values
            # All values should be finite - consensus clusters shouldn't have all-NaN transition times
            assert np.all(np.isfinite(mean_shift_times)), (
                f"median_median_shift_time contains invalid (NaN) values: {mean_shift_times}. "
                "Consensus clusters should always have valid transition times."
            )
            # Check if any cluster has median time close to expected value (if provided)
            if expected_mean_shift_time is not None:
                differences = np.abs(mean_shift_times - expected_mean_shift_time)
                min_diff = np.min(differences)
                assert min_diff <= time_tolerance, (
                    f"No cluster found with median_median_shift_time within {time_tolerance} "
                    f"of expected {expected_mean_shift_time}. "
                    f"Actual values: {mean_shift_times}, "
                    f"min difference: {min_diff}"
                )
            else:
                # If expected value not provided, just print the values for debugging
                print(f"\nmedian_median_shift_times: {mean_shift_times}")
    else:
        assert len(summary_df) == 0, (
            f"Expected empty summary dataframe when no clusters found, "
            f"but got {len(summary_df)} rows"
        )


def test_cluster_consensus_spacetime_native_grid():
    """Spacetime lattice consensus returns (time, y, x) on native (non-HealPix) grids."""
    td = setup_native_grid()
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        var = td.base_vars[0]
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))
    for tsf in (0.25, 0.5, 1.0, 1.5):
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=10),
            time_weight=tsf,
            shift_threshold=0.8,
        )
    time_dim = td.time_dim
    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=0,
        top_n_clusters=5,
        show_progress=False,
    )
    cv = td.consensus_cluster_vars[-1]
    summary_df = td.aggregate.consensus_summary(cv)
    dc = td.data[cv]
    assert time_dim in dc.dims
    assert dc.sizes[time_dim] == td.data[td.cluster_vars[0]].sizes[time_dim]
    assert dc.shape == td.data[f"{cv}_consistency"].shape
    assert summary_df is not None
    assert "cluster_id" in summary_df.columns
    assert "area" in summary_df.columns
    # Many timesteps can be all-noise under strict per-t agreement; require at least one
    # timestep with a non-noise pixel if any exist in the raw cluster fields.
    raw = td.data[td.cluster_vars[0]].values
    if np.any((raw >= 0) & np.isfinite(raw)):
        assert (dc.values >= 0).any()
        uc = np.unique(dc.values)
        uc = uc[uc >= 0]
        assert len(summary_df) == len(uc), (
            "spacetime summary must list every cluster id present in clusters"
        )


def test_spacetime_context_regular_latlon_defaults_to_native_grid():
    td = setup_regular_latlon_grid()

    context = td.aggregate._build_spacetime_consensus_context(
        sample=td.data[td.base_vars[0]],
        stitch_meridian=False,
    )

    assert context.n_space == context.y_len * context.x_len


def test_spacetime_context_curvilinear_uses_native_grid():
    """Irregular lat/lon layout still uses index 8-neighbour consensus graph."""
    td = setup_irregular_grid()
    context = td.aggregate._build_spacetime_consensus_context(
        sample=td.data[td.base_vars[0]],
        stitch_meridian=False,
    )
    assert context.n_space == context.y_len * context.x_len


def test_consensus_shift_time_distribution_spacetime():
    """consensus_shift_time_distribution matches summary transition pipeline."""
    td = setup_native_grid()
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        var = td.base_vars[0]
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))
    for tsf in (0.25, 0.5, 1.0, 1.5):
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=10),
            time_weight=tsf,
            shift_threshold=0.8,
        )
    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=0,
        top_n_clusters=5,
        show_progress=False,
    )
    dc = _latest_consensus_labels(td)
    summary_df = td.aggregate.consensus_summary(str(dc.name))
    dist_ds, df_cell = td.aggregate.consensus_shift_time_distribution(
        dc,
        shift_threshold=0.0,
    )
    assert "spatial_mean_transition_time" in dist_ds.data_vars
    assert "spatial_median_transition_time" in dist_ds.data_vars
    assert "spatial_std_transition_time" in dist_ds.data_vars
    assert dist_ds.sizes["cluster_var"] == len(td.cluster_vars)
    if len(summary_df) > 0 and len(df_cell) > 0:
        assert {"consensus_cluster_id", "cluster_var", "transition_time"} <= set(
            df_cell.columns
        )
        cid = int(summary_df["cluster_id"].iloc[0])
        sm = dist_ds["spatial_median_transition_time"].sel(
            consensus_cluster_id=float(cid)
        )
        recomputed = float(np.nanmedian(sm.values))
        tab = float(
            summary_df.loc[
                summary_df["cluster_id"] == cid, "median_median_shift_time"
            ].iloc[0]
        )
        assert np.isfinite(recomputed) and np.isfinite(tab)
        np.testing.assert_allclose(recomputed, tab, rtol=1e-4, atol=1e-4)


def test_consensus_shift_time_distributions_matches_long_dataframe():
    """Grouped shift-time samples should match the long-form dataframe exactly."""
    td = setup_native_grid()
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        var = td.base_vars[0]
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))
    for tsf in (0.25, 0.5, 1.0, 1.5):
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=10),
            time_weight=tsf,
            shift_threshold=0.8,
        )
    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=0,
        top_n_clusters=5,
        show_progress=False,
    )
    dc = _latest_consensus_labels(td)
    dist = td.aggregate.consensus_shift_time_distribution(
        dc,
        shift_threshold=0.0,
    )
    _, df_cell = dist
    grouped = td.aggregate.consensus_shift_time_distributions(
        dc,
        shift_threshold=0.0,
        distribution_result=dist,
    )

    expected_ids = sorted(df_cell["consensus_cluster_id"].unique().astype(int).tolist())
    assert sorted(grouped) == expected_ids
    for cid in expected_ids:
        expected = df_cell.loc[
            df_cell["consensus_cluster_id"] == cid, "transition_time"
        ].to_numpy(dtype=np.float64, copy=True)
        np.testing.assert_allclose(np.sort(grouped[cid]), np.sort(expected))


def test_consensus_shift_time_distribution_only_uses_supported_input_overlap():
    """Event-time samples should exactly match supported consensus voxels."""
    from toad.utils.cluster_consensus_utils import _consensus_input_support_mask

    td = setup_native_grid()
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        var = td.base_vars[0]
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))
    for tsf in (0.25, 0.5, 1.0, 1.5):
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=10),
            time_weight=tsf,
            shift_threshold=0.8,
        )
    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=0,
        top_n_clusters=5,
        show_progress=False,
    )
    dc = _latest_consensus_labels(td)
    _, df_cell = td.aggregate.consensus_shift_time_distribution(
        dc,
        shift_threshold=0.0,
    )

    for cvar in dc.attrs["cluster_vars"]:
        support_mask = _consensus_input_support_mask(
            td,
            dc,
            cvar,
            spatial_dims=tuple(td.space_dims),
            time_dim=td.time_dim,
        )
        support_labels = dc.where(support_mask, other=-1)
        time_broadcast = np.broadcast_to(
            np.asarray(td.numeric_time_values, dtype=np.float64).reshape((-1, 1, 1)),
            support_labels.shape,
        )
        expected_counts = {
            int(cid): int(((support_labels == cid).values).sum())
            for cid in np.unique(
                support_labels.values[
                    np.isfinite(support_labels.values) & (support_labels.values >= 0)
                ]
            )
        }
        observed = (
            df_cell.loc[df_cell["cluster_var"] == cvar]
            .groupby("consensus_cluster_id")
            .size()
            .to_dict()
        )
        assert observed == expected_counts

        for cid in expected_counts:
            expected_times = time_broadcast[(support_labels.values == cid)]
            observed_times = df_cell.loc[
                (df_cell["cluster_var"] == cvar)
                & (df_cell["consensus_cluster_id"] == cid),
                "transition_time",
            ].to_numpy(dtype=np.float64, copy=True)
            np.testing.assert_allclose(
                np.sort(observed_times),
                np.sort(expected_times.astype(np.float64, copy=False)),
            )


def test_consensus_cluster_timeseries_matches_manual_masked_mean():
    """Consensus-cluster timeseries should match manual masking and aggregation."""
    from toad.utils.cluster_consensus_utils import _consensus_input_support_mask

    td = setup_native_grid()
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        var = td.base_vars[0]
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))
    for tsf in (0.25, 0.5, 1.0, 1.5):
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=10),
            time_weight=tsf,
            shift_threshold=0.8,
        )
    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=0,
        top_n_clusters=5,
        show_progress=False,
    )
    dc = _latest_consensus_labels(td)
    summary_df = td.aggregate.consensus_summary(str(dc.name))
    assert len(summary_df) > 0
    cid = int(summary_df["cluster_id"].iloc[0])

    out = td.aggregate.consensus_cluster_timeseries(dc, cid, aggregation="mean")
    expected_keys = []
    for cvar in dc.attrs["cluster_vars"]:
        support_mask = _consensus_input_support_mask(
            td,
            dc,
            cvar,
            spatial_dims=tuple(td.space_dims),
            time_dim=td.time_dim,
        )
        if bool(((dc == cid) & support_mask).any().item()):
            expected_keys.append(cvar)
    assert sorted(out) == sorted(expected_keys)

    first_cluster_var = expected_keys[0]
    base_var = td.data[first_cluster_var].attrs[_attrs.BASE_VARIABLE]
    support_mask = _consensus_input_support_mask(
        td,
        dc,
        first_cluster_var,
        spatial_dims=tuple(td.space_dims),
        time_dim=td.time_dim,
    )
    footprint = ((dc == cid) & support_mask).any(dim=td.time_dim)
    expected = td._aggregate_spatial(td.data[base_var].where(footprint), method="mean")
    np.testing.assert_allclose(
        out[first_cluster_var].values,
        expected.values,
        rtol=1e-6,
        atol=1e-6,
        equal_nan=True,
    )


def test_consensus_cluster_timeseries_time_window_masks_outside_range():
    """keep_full_timeseries=False should trim to the consensus time span."""
    td = setup_native_grid()
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        var = td.base_vars[0]
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))
    for tsf in (0.25, 0.5, 1.0, 1.5):
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=10),
            time_weight=tsf,
            shift_threshold=0.8,
        )
    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=0,
        top_n_clusters=5,
        show_progress=False,
    )
    dc = _latest_consensus_labels(td)
    summary_df = td.aggregate.consensus_summary(str(dc.name))
    assert len(summary_df) > 0
    cid = int(summary_df["cluster_id"].iloc[0])
    cluster_mask = dc == cid
    active_idx = np.flatnonzero(cluster_mask.any(dim=td.space_dims).values)
    assert active_idx.size > 0

    out = td.aggregate.consensus_cluster_timeseries(
        dc,
        cid,
        aggregation="mean",
        keep_full_timeseries=False,
    )
    first_cluster_var = list(out)[0]
    vals = out[first_cluster_var].values
    assert np.all(np.isnan(vals[: active_idx[0]]))
    assert np.all(np.isnan(vals[active_idx[-1] + 1 :]))


def test_consensus_cluster_timeseries_skips_unsupported_inputs():
    """Only cluster vars with supported overlap should be returned."""
    from toad.utils.cluster_consensus_utils import _consensus_input_support_mask

    td = setup_native_grid()
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        var = td.base_vars[0]
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))
    for tsf in (0.25, 0.5, 1.0, 1.5):
        td.compute_clusters(
            method=HDBSCAN(min_cluster_size=10),
            time_weight=tsf,
            shift_threshold=0.8,
        )
    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=0,
        top_n_clusters=5,
        show_progress=False,
    )
    dc = _latest_consensus_labels(td)
    summary_df = td.aggregate.consensus_summary(str(dc.name))
    assert len(summary_df) > 0
    cid = int(summary_df["cluster_id"].iloc[0])

    out = td.aggregate.consensus_cluster_timeseries(dc, cid, aggregation="raw")
    expected = []
    for cvar in dc.attrs["cluster_vars"]:
        support_mask = _consensus_input_support_mask(
            td,
            dc,
            cvar,
            spatial_dims=tuple(td.space_dims),
            time_dim=td.time_dim,
        )
        if bool(((dc == cid) & support_mask).any().item()):
            expected.append(cvar)
    assert sorted(out) == sorted(expected)


def test_consensus_cluster_timeseries_normalize_max_scales_to_one():
    labels = np.zeros((3, 1, 2), dtype=np.float32)
    base_values = np.array([[[1.0, 2.0]], [[2.0, 4.0]], [[4.0, 8.0]]], dtype=np.float32)
    td = setup_synthetic_consensus_toad(
        {"c1": labels, "c2": labels}, base_values=base_values
    )

    td.compute_consensus(
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=0,
        show_progress=False,
    )
    dc = _latest_consensus_labels(td)
    summary_df = td.aggregate.consensus_summary(str(dc.name))

    assert len(summary_df) == 1
    out = td.aggregate.consensus_cluster_timeseries(
        dc,
        0,
        aggregation="mean",
        normalize="max",
    )
    for series in out.values():
        assert np.isclose(float(series.max()), 1.0)


def test_consensus_cluster_timeseries_normalize_max_each_scales_each_cell():
    labels = np.zeros((3, 1, 2), dtype=np.float32)
    base_values = np.array([[[1.0, 2.0]], [[2.0, 4.0]], [[4.0, 8.0]]], dtype=np.float32)
    td = setup_synthetic_consensus_toad(
        {"c1": labels, "c2": labels}, base_values=base_values
    )

    td.compute_consensus(
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=0,
        show_progress=False,
    )
    dc = _latest_consensus_labels(td)
    summary_df = td.aggregate.consensus_summary(str(dc.name))

    assert len(summary_df) == 1
    out = td.aggregate.consensus_cluster_timeseries(
        dc,
        0,
        aggregation="raw",
        normalize="max_each",
    )
    for series in out.values():
        assert "cell_xy" in series.dims
        np.testing.assert_allclose(
            series.max(dim=td.time_dim).values,
            np.ones(series.sizes["cell_xy"], dtype=np.float32),
        )


def test_build_spacetime_graph_edges_consecutive_time_chain():
    from toad.utils.cluster_consensus_utils import _build_spacetime_graph_edges

    sr = np.array([0], dtype=np.int64)
    sc = np.array([1], dtype=np.int64)
    u, v = _build_spacetime_graph_edges(5, 2, sr, sc)
    assert set(zip(u.tolist(), v.tolist())) == {
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
        (8, 9),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
        (5, 7),
        (6, 8),
        (7, 9),
    }


def test_dilate_cluster_labels_in_time():
    from toad.utils.cluster_consensus_utils import _dilate_cluster_labels_in_time

    labels = np.full((5, 2, 2), np.nan, dtype=np.float32)
    labels[2, 0, 0] = 0
    labels[1, 1, 1] = 1

    out = _dilate_cluster_labels_in_time(labels, np.array([0, 1]), temporal_tolerance=1)

    assert np.all(out[1:4, 0, 0] == 0)
    assert np.all(out[0:3, 1, 1] == 1)
    assert np.isnan(out[4, 1, 1])


def test_dilate_cluster_labels_in_time_marks_overlap_conflicts():
    from toad.utils.cluster_consensus_utils import _dilate_cluster_labels_in_time

    labels = np.full((5, 1, 1), np.nan, dtype=np.float32)
    labels[1, 0, 0] = 0
    labels[3, 0, 0] = 1

    out = _dilate_cluster_labels_in_time(labels, np.array([0, 1]), temporal_tolerance=1)

    np.testing.assert_array_equal(
        out[:, 0, 0], np.array([0, 0, -1, 1, 1], dtype=np.float32)
    )


def test_dilate_cluster_labels_in_time_ignores_disallowed_labels():
    from toad.utils.cluster_consensus_utils import _dilate_cluster_labels_in_time

    labels = np.full((5, 1, 1), np.nan, dtype=np.float32)
    labels[2, 0, 0] = 7

    out = _dilate_cluster_labels_in_time(labels, np.array([0, 1]), temporal_tolerance=1)

    assert np.isnan(out).all()


def test_dilate_cluster_labels_spacetime_spatial_hops():
    from toad.utils.cluster_consensus_utils import _dilate_cluster_labels_spacetime

    labels = np.full((1, 4), np.nan, dtype=np.float32)
    labels[0, 1] = 0
    rows = np.array([0, 1, 2], dtype=np.int64)
    cols = np.array([1, 2, 3], dtype=np.int64)

    out = _dilate_cluster_labels_spacetime(
        labels,
        np.array([0]),
        temporal_tolerance=0,
        spatial_tolerance=1,
        spatial_rows=rows,
        spatial_cols=cols,
    )

    np.testing.assert_array_equal(out[0], np.array([0, 0, 0, np.nan], dtype=np.float32))


def test_dilate_cluster_labels_spacetime_marks_spatial_overlap_conflicts():
    from toad.utils.cluster_consensus_utils import _dilate_cluster_labels_spacetime

    labels = np.full((1, 4), np.nan, dtype=np.float32)
    labels[0, 1] = 0
    labels[0, 2] = 1
    rows = np.array([0, 1, 2], dtype=np.int64)
    cols = np.array([1, 2, 3], dtype=np.int64)

    out = _dilate_cluster_labels_spacetime(
        labels,
        np.array([0, 1]),
        temporal_tolerance=0,
        spatial_tolerance=1,
        spatial_rows=rows,
        spatial_cols=cols,
    )

    np.testing.assert_array_equal(out[0], np.array([0, -1, -1, 1], dtype=np.float32))


def test_largest_cluster_ids_uses_actual_cluster_sizes():
    from toad.utils.cluster_consensus_utils import _largest_cluster_ids

    class FakeTD:
        def get_cluster_ids(self, var, exclude_noise=True):
            assert var == "demo_cluster"
            assert exclude_noise is True
            return np.array([7, 3, 11], dtype=np.int64)

        def get_cluster_counts(self, var, exclude_noise=True):
            assert var == "demo_cluster"
            assert exclude_noise is True
            return {3: 25, 11: 10, 7: 4}

    td = FakeTD()

    np.testing.assert_array_equal(
        _largest_cluster_ids(td, "demo_cluster", top_n_clusters=2),
        np.array([3, 11], dtype=np.int64),
    )


def test_native_edges_from_mask_stitches_meridian():
    from toad.utils.cluster_consensus_utils import _native_edges_from_mask

    mask = np.ones((2, 3), dtype=bool)
    flat_idx = np.arange(6, dtype=np.int64).reshape(2, 3)

    rows, cols = _native_edges_from_mask(mask, flat_idx, stitch_longitude=True)
    edges = set(zip(rows, cols))

    assert (0, 2) in edges
    assert (0, 5) in edges
    assert (2, 3) in edges


def test_cluster_consensus_stitch_meridian_merges_wraparound_components():
    labels = np.array([[[0, -1, -1, 0], [0, -1, -1, 0]]], dtype=np.float32)
    td = setup_synthetic_consensus_toad({"c1": labels, "c2": labels})

    td.compute_consensus(
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=0,
        stitch_meridian=False,
        show_progress=False,
        output_label="cons_no",
    )
    td.compute_consensus(
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=0,
        stitch_meridian=True,
        show_progress=False,
        output_label="cons_yes",
    )
    summary_no = td.aggregate.consensus_summary("cons_no")
    summary_yes = td.aggregate.consensus_summary("cons_yes")

    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_no"]), np.array([0, 1])
    )
    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_yes"]), np.array([0])
    )
    assert len(summary_no) == 2
    assert len(summary_yes) == 1


def test_cluster_consensus_spatial_tolerance_merges_offset_clusters():
    labels_a = np.array([[[0, 0, -1, -1]]], dtype=np.float32)
    labels_b = np.array([[[-1, -1, 0, 0]]], dtype=np.float32)
    td = setup_synthetic_consensus_toad({"c1": labels_a, "c2": labels_b})

    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=0,
        show_progress=False,
        output_label="cons_exact",
    )
    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=1,
        show_progress=False,
        output_label="cons_tol",
    )

    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_exact"]), np.array([0, 1])
    )
    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_tol"]), np.array([0])
    )


def test_cluster_consensus_temporal_tolerance_merges_offset_clusters_across_maps():
    labels_a = np.array([[[0]], [[0]], [[-1]], [[-1]]], dtype=np.float32)
    labels_b = np.array([[[-1]], [[-1]], [[0]], [[0]]], dtype=np.float32)
    td = setup_synthetic_consensus_toad({"c1": labels_a, "c2": labels_b})

    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=0,
        spatial_tolerance=0,
        min_cluster_area=None,
        show_progress=False,
        output_label="cons_exact",
    )
    td.compute_consensus(
        min_consensus=0.5,
        temporal_tolerance=1,
        spatial_tolerance=0,
        min_cluster_area=None,
        show_progress=False,
        output_label="cons_tol",
    )

    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_exact"]), np.array([0, 1])
    )
    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_tol"]), np.array([0])
    )


def test_cluster_consensus_top_n_clusters_filters_public_result():
    labels = np.array([[[5, 5, 5, 5, 0, 0]]], dtype=np.float32)
    td = setup_synthetic_consensus_toad(
        {"c1": labels, "c2": labels},
        cluster_id_attrs={
            "c1": np.array([0, 5, -1], dtype=np.int32),
            "c2": np.array([0, 5, -1], dtype=np.int32),
        },
    )

    td.compute_consensus(
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=0,
        show_progress=False,
        output_label="cons_all",
    )
    td.compute_consensus(
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=0,
        top_n_clusters=1,
        show_progress=False,
        output_label="cons_top1",
    )

    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_all"]), np.array([0, 1])
    )
    np.testing.assert_array_equal(
        non_noise_cluster_ids(td.data["cons_top1"]), np.array([0])
    )
    assert int((td.data["cons_all"].values >= 0).sum()) == 6
    assert int((td.data["cons_top1"].values >= 0).sum()) == 4


def test_cluster_consensus_all_noise_inputs_return_empty_result():
    labels = np.full((2, 1, 2), -1, dtype=np.float32)
    td = setup_synthetic_consensus_toad({"c1": labels, "c2": labels})

    td.compute_consensus(
        min_consensus=1.0,
        temporal_tolerance=0,
        spatial_tolerance=0,
        show_progress=False,
    )
    cv = td.consensus_cluster_vars[-1]
    summary_df = td.aggregate.consensus_summary(cv)

    assert np.all(td.data[cv].values == -1)
    assert np.all(td.data[f"{cv}_consistency"].values == 0)
    assert summary_df.empty


def test_cluster_consensus_returns_all_noise_when_threshold_removes_all_edges():
    labels_a = np.array([[[0, 0, -1, -1]]], dtype=np.float32)
    labels_b = np.array([[[-1, -1, 0, 0]]], dtype=np.float32)
    td = setup_synthetic_consensus_toad({"c1": labels_a, "c2": labels_b})

    td.compute_consensus(
        min_consensus=0.75,
        temporal_tolerance=0,
        spatial_tolerance=0,
        show_progress=False,
    )
    cv = td.consensus_cluster_vars[-1]
    summary_df = td.aggregate.consensus_summary(cv)

    assert np.all(td.data[cv].values == -1)
    assert np.all(td.data[f"{cv}_consistency"].values == 0)
    assert summary_df.empty


def test_compute_weighted_consensus_weighted_A_matches_duplicate_edges():
    from toad.utils.cluster_consensus_utils import _compute_weighted_consensus

    rows_v = np.array([0, 2], dtype=np.int64)
    cols_v = np.array([1, 3], dtype=np.int64)
    rows_a = np.array([0, 2], dtype=np.int64)
    cols_a = np.array([1, 3], dtype=np.int64)
    shape = (4, 4)

    W_dup = _compute_weighted_consensus(
        rows_v,
        cols_v,
        np.tile(rows_a, 3),
        np.tile(cols_a, 3),
        shape,
        min_consensus=0.3,
    )
    W_weighted = _compute_weighted_consensus(
        rows_v,
        cols_v,
        rows_a,
        cols_a,
        shape,
        min_consensus=0.3,
        data_A=np.full(rows_a.shape[0], 3.0, dtype=np.float32),
    )

    np.testing.assert_allclose(W_dup.toarray(), W_weighted.toarray())


def test_temporal_tolerance_is_local_rule_not_global_cluster_span_cap():
    from scipy.sparse.csgraph import connected_components

    from toad.utils.cluster_consensus_utils import (
        _build_spacetime_graph_edges,
        _compute_weighted_consensus,
        _dilate_cluster_labels_in_time,
    )

    labels = np.full((5, 1, 1), np.nan, dtype=np.float32)
    labels[0, 0, 0] = 0
    labels[2, 0, 0] = 0

    dil = _dilate_cluster_labels_in_time(labels, np.array([0]), temporal_tolerance=1)
    lab_flat = dil.reshape(-1)
    u, v = _build_spacetime_graph_edges(
        5, 1, np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    )
    ok_v = (lab_flat[u] == lab_flat[v]) & (lab_flat[u] >= 0) & np.isfinite(lab_flat[u])

    W = _compute_weighted_consensus(
        u[ok_v],
        v[ok_v],
        u,
        v,
        shape=(5, 5),
        min_consensus=1.0,
    )
    bin_adj = W.copy()
    bin_adj.data[:] = 1.0
    _, labels_cc = connected_components(
        bin_adj.maximum(bin_adj.T), directed=False, return_labels=True
    )

    active = np.flatnonzero(np.isfinite(dil[:, 0, 0]) & (dil[:, 0, 0] >= 0))
    assert active.tolist() == [0, 1, 2, 3]
    assert len(np.unique(labels_cc[active])) == 1
    assert active[-1] - active[0] > 1


def test_trim_spacetime_consensus_to_original_support_removes_tolerance_padding():
    from toad.utils.cluster_consensus_utils import (
        _trim_spacetime_consensus_to_original_support,
    )

    labels_flat = np.array([0, 0, 0, 0, -1], dtype=np.float32)
    cons_flat = np.array([0.4, 0.4, 0.4, 0.4, 0.0], dtype=np.float32)
    original_support = np.array([False, False, True, False, False], dtype=bool)

    labels_trim, cons_trim = _trim_spacetime_consensus_to_original_support(
        labels_flat, cons_flat, original_support
    )

    np.testing.assert_array_equal(
        labels_trim, np.array([-1, -1, 0, -1, -1], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        cons_trim, np.array([0.0, 0.0, 0.4, 0.0, 0.0], dtype=np.float32)
    )


def test_cluster_consensus_rejects_negative_tolerances():
    td = setup_native_grid()
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        var = td.base_vars[0]
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))
    td.compute_clusters(
        method=HDBSCAN(min_cluster_size=10),
        time_weight=0.5,
        shift_threshold=0.8,
    )
    with pytest.raises(ValueError, match="temporal_tolerance"):
        td.compute_consensus(
            min_consensus=0.5,
            temporal_tolerance=-1,
            spatial_tolerance=0,
            show_progress=False,
        )
    with pytest.raises(ValueError, match="spatial_tolerance"):
        td.compute_consensus(
            min_consensus=0.5,
            temporal_tolerance=0,
            spatial_tolerance=-1,
            show_progress=False,
        )


def test_cluster_consensus_spacetime_tolerances_run():
    td = setup_native_grid()
    td.data = td.data.drop_vars(td.cluster_vars, errors="ignore")
    if len(td.shift_vars) == 0:
        var = td.base_vars[0]
        td.compute_shifts(var, method=ASDETECT(ignore_nan_warnings=True))
    td.compute_clusters(
        method=HDBSCAN(min_cluster_size=10),
        time_weight=0.5,
        shift_threshold=0.8,
    )
    td.compute_consensus(
        min_consensus=0.5,
        top_n_clusters=5,
        temporal_tolerance=2,
        spatial_tolerance=1,
        show_progress=False,
    )
    cv = td.consensus_cluster_vars[-1]
    assert cv in td.data
    assert td.data[cv].attrs.get("temporal_tolerance") == 2
    assert td.data[cv].attrs.get("spatial_tolerance") == 1


def test_cluster_occurrence_rate_aggregation():
    """Per-cell share of run membership in [0,1] over :attr:`TOAD.cluster_vars`."""
    td = setup_regular_latlon_grid()
    td.drop_clusters()
    td.compute_clusters(method=HDBSCAN(min_cluster_size=10))
    da = td.aggregate.cluster_occurrence_rate()
    assert isinstance(da, xr.DataArray)
    assert (da >= 0).all() and (da <= 1).all()
    assert da.dims == tuple(td.data[td.cluster_vars[0]].dims[1:])
    assert "cluster_vars" in da.attrs


def test_label_field_shift_time_distributions():
    """Single 3D label map: grouped transition times for violin-style plots."""
    td = setup_regular_latlon_grid()
    td.drop_clusters()
    td.compute_clusters(method=HDBSCAN(min_cluster_size=10))
    cv = td.cluster_vars[0]
    dists = td.aggregate.label_shift_time_distributions(cv)
    assert isinstance(dists, dict)
    for _cid, arr in dists.items():
        assert arr.ndim == 1
        assert np.isfinite(arr).all() or arr.size == 0
    if dists:
        for k in dists:
            assert k >= 0
