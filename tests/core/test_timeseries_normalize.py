"""Tests for get_timeseries normalisation order."""

import numpy as np
import pytest
from sklearn.cluster import HDBSCAN  # type: ignore

from toad import TOAD


@pytest.fixture(scope="module")
def td_with_clusters():
    td = TOAD("tutorials/test_data/synth_data.nc")
    td.data = td.data.coarsen(lat=5, lon=5, boundary="trim").reduce(np.mean)
    td.drop_clusters()
    td.compute_clusters(method=HDBSCAN(min_cluster_size=5))
    return td


@pytest.mark.parametrize("normalize", ["max", "max_each"])
def test_normalized_aggregate_bands_are_ordered(td_with_clusters, normalize):
    """Min/median/max must be ordered at each time after normalise-then-aggregate."""
    td = td_with_clusters
    kwargs = {"var": "ts_dts_cluster", "cluster_id": 0, "normalize": normalize}

    median = td.get_timeseries(aggregation="median", **kwargs).values
    min_ts = td.get_timeseries(aggregation="min", **kwargs).values
    max_ts = td.get_timeseries(aggregation="max", **kwargs).values

    valid = np.isfinite(median) & np.isfinite(min_ts) & np.isfinite(max_ts)
    assert valid.any()
    assert np.all(min_ts[valid] <= median[valid] + 1e-12)
    assert np.all(median[valid] <= max_ts[valid] + 1e-12)
