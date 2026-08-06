import numpy as np
import xarray as xr

from toad import TOAD
from toad.postprocessing.healpix_member_support_consensus import (
    run_healpix_member_support_consensus,
)
from toad.postprocessing.member_support_consensus import (
    cluster_id_signs_from_map,
    cluster_id_signs_to_map,
)
from toad.utils import _attrs


class _Store:
    def __init__(self, data: xr.Dataset, time_dim: str):
        self.data = data
        self.time_dim = time_dim


def test_healpix_consensus_splits_opposite_sign_support():
    """Positive and negative model clusters must not merge into one consensus id."""
    T, nside, npix = 5, 4, 12 * 4**2
    coords = {"time": np.arange(T), "hp_pixel": np.arange(npix)}

    pos = np.full((T, npix), np.nan, dtype=np.float32)
    neg = np.full((T, npix), np.nan, dtype=np.float32)

    pos[2:4, 10:14] = 0
    neg[2:4, 12:16] = 0

    ds = xr.Dataset(
        {
            "model_a_cluster": (("time", "hp_pixel"), pos),
            "model_b_cluster": (("time", "hp_pixel"), neg),
        },
        coords=coords,
    )
    ds["model_a_cluster"].attrs[_attrs.CLUSTER_IDS] = np.array([0], dtype=int)
    ds["model_a_cluster"].attrs[_attrs.CLUSTER_ID_SIGNS] = cluster_id_signs_from_map(
        np.array([0]), {0: 1}
    )
    ds["model_b_cluster"].attrs[_attrs.CLUSTER_IDS] = np.array([0], dtype=int)
    ds["model_b_cluster"].attrs[_attrs.CLUSTER_ID_SIGNS] = cluster_id_signs_from_map(
        np.array([0]), {0: -1}
    )

    out, sign_by_id = run_healpix_member_support_consensus(
        _Store(ds, "time"),
        cluster_vars=["model_a_cluster", "model_b_cluster"],
        min_consensus=0.5,
        temporal_tolerance=1,
        spatial_tolerance=1,
        nside=nside,
        min_cluster_area=1,
        show_progress=False,
    )

    labels = out["clusters"].values
    assert labels.shape == (T, npix)
    valid = labels[np.isfinite(labels) & (labels >= 0)]
    assert valid.size > 0
    assert len(np.unique(valid)) >= 2

    cluster_ids = out["clusters"].attrs.get(_attrs.CLUSTER_IDS)
    id_signs = out["clusters"].attrs.get(_attrs.CLUSTER_ID_SIGNS)
    if id_signs is not None and cluster_ids is not None:
        sign_map = cluster_id_signs_to_map(cluster_ids, id_signs)
    else:
        sign_map = sign_by_id
    assert sign_map
    assert set(sign_map.values()) == {-1, 1}


def test_healpix_consensus_backward_compatible_without_sign_fields():
    """Exports without sign fields keep the legacy combined vote behaviour."""
    T, nside, npix = 5, 4, 12 * 4**2
    coords = {"time": np.arange(T), "hp_pixel": np.arange(npix)}
    a = np.full((T, npix), np.nan, dtype=np.float32)
    b = np.full((T, npix), np.nan, dtype=np.float32)
    a[2, 10:14] = 0
    a[3, 10:14] = 0
    b[2, 12:16] = 0
    b[3, 12:16] = 0
    ds = xr.Dataset(
        {
            "model_a_cluster": (("time", "hp_pixel"), a),
            "model_b_cluster": (("time", "hp_pixel"), b),
        },
        coords=coords,
    )
    out, sign_by_id = run_healpix_member_support_consensus(
        _Store(ds, "time"),
        cluster_vars=["model_a_cluster", "model_b_cluster"],
        min_consensus=0.5,
        temporal_tolerance=1,
        spatial_tolerance=1,
        nside=nside,
        min_cluster_area=1,
        show_progress=False,
    )
    assert np.any(out["clusters"].values >= 0)
    assert "_interim_sign_by_id" not in out["clusters"].attrs
    assert _attrs.CLUSTER_ID_SIGNS not in out["clusters"].attrs


def test_compute_consensus_splits_opposite_sign_support():
    """td.compute_consensus() must not merge opposite-sign cluster inputs."""
    T, y_len, x_len = 5, 8, 8
    coords = {
        "time": np.arange(T),
        "y": np.arange(y_len),
        "x": np.arange(x_len),
    }

    pos = np.full((T, y_len, x_len), np.nan, dtype=np.float32)
    neg = np.full((T, y_len, x_len), np.nan, dtype=np.float32)

    pos[2:4, 2:5, 2:5] = 0
    neg[2:4, 4:7, 4:7] = 0

    ds = xr.Dataset(
        {
            "model_a_cluster": (("time", "y", "x"), pos),
            "model_b_cluster": (("time", "y", "x"), neg),
        },
        coords=coords,
    )
    for cvar in ("model_a_cluster", "model_b_cluster"):
        ds[cvar].attrs.update(
            {
                _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER,
                _attrs.BASE_VARIABLE: "foo",
                _attrs.CLUSTER_IDS: np.array([0], dtype=np.int32),
            }
        )
    ds["model_a_cluster"].attrs[_attrs.CLUSTER_ID_SIGNS] = cluster_id_signs_from_map(
        np.array([0]), {0: 1}
    )
    ds["model_b_cluster"].attrs[_attrs.CLUSTER_ID_SIGNS] = cluster_id_signs_from_map(
        np.array([0]), {0: -1}
    )

    td = TOAD(ds, time_dim="time")
    td.compute_consensus(
        cluster_vars=["model_a_cluster", "model_b_cluster"],
        min_consensus=0.5,
        temporal_tolerance=1,
        spatial_tolerance=1,
        min_cluster_area=1,
        show_progress=False,
    )

    labels = _latest_consensus_labels(td).values
    valid = labels[np.isfinite(labels) & (labels >= 0)]
    assert valid.size > 0
    assert len(np.unique(valid)) >= 2

    consensus = _latest_consensus_labels(td)
    assert "_interim_sign_by_id" not in consensus.attrs
    sign_map = cluster_id_signs_to_map(
        consensus.attrs[_attrs.CLUSTER_IDS],
        consensus.attrs[_attrs.CLUSTER_ID_SIGNS],
    )
    assert sign_map
    assert set(sign_map.values()) == {-1, 1}


def _latest_consensus_labels(td: TOAD):
    names = td.consensus_cluster_vars
    assert names, "expected at least one consensus_cluster variable on td"
    return td.data[names[-1]]
