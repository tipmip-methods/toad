import gc

import numpy as np
import pytest
import xarray as xr
from sklearn.cluster import DBSCAN, HDBSCAN  # type: ignore

from toad import TOAD
from toad.clustering import compute_clusters
from toad.postprocessing.member_support_consensus import sign_var_for_cluster_var
from toad.utils import _attrs


@pytest.fixture(autouse=True)
def cleanup_memory():
    """Clean up memory after each test."""
    yield
    gc.collect()


def _assert_clusters_sign_homogeneous(clusters_da, sign_da):
    """Every non-noise cluster label must have uniform shift sign on the sign grid."""
    c = clusters_da.values
    s = sign_da.values
    for cid in np.unique(c[np.isfinite(c) & (c >= 0)]):
        mask = c == cid
        signs = s[mask]
        signs = signs[np.isfinite(signs)]
        assert signs.size > 0
        assert np.all(signs > 0) or np.all(signs < 0), (
            f"Cluster {int(cid)} has mixed signs"
        )


def test_both_directions_split_by_sign_synthetic():
    """shift_direction='both' must not produce mixed-sign clusters."""
    time = np.arange(10, dtype=float)
    lat = np.array([0.0, 10.0])
    lon = np.array([0.0, 10.0])
    sh = np.zeros((10, 2, 2), dtype=float)
    # Pos/neg pairs at the same time and nearby in space would merge without sign split.
    sh[5, 0, 0] = 1.0
    sh[5, 0, 1] = 1.0
    sh[5, 1, 0] = -1.0
    sh[5, 1, 1] = -1.0

    ds = xr.Dataset(
        {"test_dts": (("time", "lat", "lon"), sh)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds["test_dts"].attrs[_attrs.BASE_VARIABLE] = "test"
    ds["test_dts"].attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_SHIFT

    td = TOAD(ds)
    td.data = compute_clusters(
        td,
        var="test_dts",
        method=DBSCAN(eps=500.0, min_samples=2),
        shift_threshold=0.5,
        shift_direction="both",
        shift_selection="all",
        disable_regridder=True,
        overwrite=True,
    )
    cluster_var = "test_dts_cluster"
    sign_var = sign_var_for_cluster_var(cluster_var)
    assert sign_var in td.data
    _assert_clusters_sign_homogeneous(td.data[cluster_var], td.data[sign_var])
    clusters = td.data[cluster_var].values
    signs = td.data[sign_var].values
    valid = clusters[np.isfinite(clusters) & (clusters >= 0)]
    if valid.size == 0:
        return
    for cid in np.unique(valid.astype(int)):
        mask = clusters == cid
        assert np.all(np.sign(signs[mask]) == np.sign(signs[mask][0]))


def test_both_directions_sign_homogeneous_on_synth_data():
    td = TOAD("tutorials/test_data/synth_data.nc")
    td.data = td.data.coarsen(lat=4, lon=4, boundary="trim").reduce(np.mean)

    td.compute_clusters(
        shift_threshold=0.5,
        method=HDBSCAN(min_cluster_size=10),
        shift_direction="both",
        shift_selection="all",
        disable_regridder=True,
        overwrite=True,
    )

    cluster_var = td.cluster_vars[0]
    sign_var = sign_var_for_cluster_var(cluster_var)
    assert sign_var in td.data
    _assert_clusters_sign_homogeneous(td.data[cluster_var], td.data[sign_var])
