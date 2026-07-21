"""Tests for shared consensus view helpers."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from toad.utils.consensus_view import (
    build_simple_consensus_summary_df,
    collapse_consensus_for_map,
    consensus_cluster_ids,
    nside_from_npix,
)


def test_collapse_consensus_for_map_mode():
    labels = xr.DataArray(
        np.array(
            [
                [0, 0, -1],
                [0, 1, 1],
                [1, 1, -1],
            ],
            dtype=np.float64,
        ),
        dims=("time", "cell"),
    )
    collapsed = collapse_consensus_for_map(labels)
    assert collapsed.shape == (3,)
    assert collapsed[0] == 0
    assert collapsed[1] == 1
    assert collapsed[2] == 1


def test_consensus_cluster_ids():
    clusters_map = np.array([0, 0, 1, -1, np.nan])
    assert consensus_cluster_ids(clusters_map) == [0, 1]


def test_nside_from_npix():
    assert nside_from_npix(12 * 8**2) == 8


def test_build_simple_consensus_summary_df():
    # cluster 0 lives at (space 0,1); cluster 1 lives at (space 2,3). Rate values
    # differ per spacetime voxel and at pixels/times *outside* each cluster's own
    # footprint, so a correct mean_consensus_rate must mask by (time, space)
    # jointly -- not dilute with irrelevant voxels via a two-step spatial-then-
    # temporal collapse (see https://github.com/.../mean_consensus_rate bug).
    clusters = xr.DataArray(
        np.array(
            [
                [0, 0, 1, -1],
                [0, 0, 1, 1],
                [np.nan, 0, 1, -1],
            ],
            dtype=np.float64,
        ),
        dims=("time", "space"),
        coords={"time": [0, 1, 2], "space": [0, 1, 2, 3]},
        name="consensus_clusters",
    )
    rate = xr.DataArray(
        np.array(
            [
                [0.5, 0.75, 1.0, 0.25],
                [0.6, 0.8, 0.9, 0.7],
                [0.4, 0.65, 0.85, 0.3],
            ]
        ),
        dims=("time", "space"),
        coords={"time": [0, 1, 2], "space": [0, 1, 2, 3]},
    )
    shift_times = {0: np.array([1.0, 3.0]), 1: np.array([5.0])}
    summary = build_simple_consensus_summary_df(
        clusters,
        rate,
        shift_times,
        time_dim="time",
    )
    assert isinstance(summary, pd.DataFrame)
    assert list(summary.columns) == [
        "cluster_id",
        "size",
        "mean_consensus_rate",
        "mean_mean_shift_time",
        "std_mean_shift_time",
    ]
    row0 = summary.loc[summary["cluster_id"] == 0].iloc[0]
    assert row0["size"] == 2
    # rate at (t,s) in {(0,0),(0,1),(1,0),(1,1),(2,1)}: 0.5,0.75,0.6,0.8,0.65
    assert row0["mean_consensus_rate"] == pytest.approx(0.66)
    assert row0["mean_mean_shift_time"] == 2.0
    assert row0["std_mean_shift_time"] == pytest.approx(1.0)

    row1 = summary.loc[summary["cluster_id"] == 1].iloc[0]
    assert row1["size"] == 2
    # rate at (t,s) in {(0,2),(1,2),(1,3),(2,2)}: 1.0,0.9,0.7,0.85
    assert row1["mean_consensus_rate"] == pytest.approx(0.8625)
    assert row1["mean_mean_shift_time"] == 5.0
    assert np.isnan(row1["std_mean_shift_time"])
