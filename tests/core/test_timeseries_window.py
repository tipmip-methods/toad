import numpy as np
import xarray as xr

from toad import TOAD
from toad.utils import _attrs
from toad.utils.shift_selection_utils import _episode_overlap_mask_for_ts


def test_episode_overlap_mask_keeps_full_episode():
    ts = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    mask = _episode_overlap_mask_for_ts(ts, 0.5, win_start=4, win_end=4)
    assert mask.tolist() == [False, False, False, True, True, False, False]


def test_episode_overlap_mask_drops_non_overlapping_episode():
    ts = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    mask = _episode_overlap_mask_for_ts(ts, 0.5, win_start=4, win_end=4)
    assert mask[:6].tolist() == [False, False, False, True, True, False]
    assert not mask[6:].any()


def test_get_timeseries_shift_window_keeps_full_overlapping_episode():
    time = np.arange(8, dtype=float)
    lat = np.array([0.0])
    lon = np.array([0.0])

    var = np.full((8, 1, 1), 100.0)
    var[3:5, 0, 0] = 200.0
    dts = np.zeros((8, 1, 1), dtype=float)
    dts[3:5, 0, 0] = 1.0
    clusters = np.full((8, 1, 1), np.nan)
    clusters[4, 0, 0] = 0.0

    ds = xr.Dataset(
        {
            "test": (("time", "lat", "lon"), var),
            "test_dts": (("time", "lat", "lon"), dts),
            "test_dts_cluster": (("time", "lat", "lon"), clusters),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["test_dts"].attrs[_attrs.BASE_VARIABLE] = "test"
    ds["test_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT
    ds["test_dts_cluster"].attrs[_attrs.BASE_VARIABLE] = "test"
    ds["test_dts_cluster"].attrs[_attrs.SHIFTS_VARIABLE] = "test_dts"
    ds["test_dts_cluster"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_CLUSTER
    ds["test_dts_cluster"].attrs[_attrs.SHIFT_THRESHOLD] = 0.5
    ds["test_dts_cluster"].attrs[_attrs.CLUSTER_IDS] = np.array([0], dtype=int)

    td = TOAD(ds)
    ts = td.get_timeseries(
        "test",
        cluster_id=0,
        cluster_var="test_dts_cluster",
        aggregation="raw",
        timeseries_window="shift",
    )
    values = np.asarray(ts.values, dtype=float)
    assert np.isfinite(values[:, 3:5]).all()
    assert not np.isfinite(values[:, :3]).any()
    assert not np.isfinite(values[:, 5:]).any()


def test_get_timeseries_shift_window_excludes_non_overlapping_episodes():
    time = np.arange(10, dtype=float)
    lat = np.array([0.0])
    lon = np.array([0.0])

    var = np.full((10, 1, 1), 100.0)
    var[3:5, 0, 0] = 200.0
    var[7:9, 0, 0] = 300.0
    dts = np.zeros((10, 1, 1), dtype=float)
    dts[3:5, 0, 0] = 1.0
    dts[7:9, 0, 0] = 1.0
    clusters = np.full((10, 1, 1), np.nan)
    clusters[4, 0, 0] = 0.0

    ds = xr.Dataset(
        {
            "test": (("time", "lat", "lon"), var),
            "test_dts": (("time", "lat", "lon"), dts),
            "test_dts_cluster": (("time", "lat", "lon"), clusters),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["test_dts"].attrs[_attrs.BASE_VARIABLE] = "test"
    ds["test_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT
    ds["test_dts_cluster"].attrs[_attrs.BASE_VARIABLE] = "test"
    ds["test_dts_cluster"].attrs[_attrs.SHIFTS_VARIABLE] = "test_dts"
    ds["test_dts_cluster"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_CLUSTER
    ds["test_dts_cluster"].attrs[_attrs.SHIFT_THRESHOLD] = 0.5
    ds["test_dts_cluster"].attrs[_attrs.CLUSTER_IDS] = np.array([0], dtype=int)

    td = TOAD(ds)
    ts = td.get_timeseries(
        "test",
        cluster_id=0,
        cluster_var="test_dts_cluster",
        aggregation="raw",
        timeseries_window="shift",
    )
    values = np.asarray(ts.values, dtype=float)
    assert np.isfinite(values[:, 3:5]).all()
    assert not np.isfinite(values[:, 7:9]).any()


def test_get_timeseries_shift_window_without_cluster_id():
    time = np.arange(10, dtype=float)
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0])

    var = np.full((10, 2, 1), 100.0)
    var[3:5, 0, 0] = 200.0
    var[7:9, 1, 0] = 300.0
    dts = np.zeros((10, 2, 1), dtype=float)
    dts[3:5, 0, 0] = 1.0
    dts[7:9, 1, 0] = 1.0

    ds = xr.Dataset(
        {
            "test": (("time", "lat", "lon"), var),
            "test_dts": (("time", "lat", "lon"), dts),
        },
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["test_dts"].attrs[_attrs.BASE_VARIABLE] = "test"
    ds["test_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT
    ds["test_dts"].attrs[_attrs.SHIFT_THRESHOLD] = 0.5

    td = TOAD(ds)
    ts = td.get_timeseries(
        "test",
        cluster_id=None,
        aggregation="raw",
        timeseries_window="shift",
    )
    values = np.asarray(ts.values, dtype=float)
    assert np.isfinite(values[0, 3:5]).all()
    assert not np.isfinite(values[0, :3]).any()
    assert not np.isfinite(values[0, 5:]).any()
    assert np.isfinite(values[1, 7:9]).all()
    assert not np.isfinite(values[1, :7]).any()
