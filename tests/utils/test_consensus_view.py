"""Tests for shared consensus view helpers."""

import numpy as np
import pandas as pd
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
    clusters_map = np.array([0, 0, 1, 1])
    rate_map = np.array([0.5, 0.75, 1.0, 0.25])
    shift_times = {0: np.array([1.0, 3.0]), 1: np.array([5.0])}
    summary = build_simple_consensus_summary_df(
        clusters_map,
        rate_map,
        shift_times,
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
    assert row0["mean_consensus_rate"] == 0.625
    assert row0["mean_mean_shift_time"] == 2.0
