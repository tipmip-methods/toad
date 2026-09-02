import numpy as np
import pytest
import xarray as xr

from toad import TOAD
from toad.postprocessing.healpix_member_support_consensus import (
    run_healpix_member_support_consensus,
)
from toad.postprocessing.member_support_consensus import (
    build_sign_aware_consensus_labels,
    cluster_label_id_mapping,
    cluster_spatial_areas,
    relabel_by_id_mapping,
    sign_var_for_cluster_var,
    sorted_consensus_labels_by_area,
)
from toad.utils import _attrs


class _Store:
    def __init__(self, data: xr.Dataset, time_dim: str):
        self.data = data
        self.time_dim = time_dim


def _attach_cluster_sign_fields(
    ds: xr.Dataset,
    cluster_vars: list[str],
    sign_by_id: dict[int, int],
) -> None:
    """Write per-voxel ``{cluster_var}_sign`` grids for tests."""
    for cvar in cluster_vars:
        labels = np.asarray(ds[cvar].values)
        signs = np.full(labels.shape, np.nan, dtype=np.float32)
        for cid, sign in sign_by_id.items():
            mask = np.isfinite(labels) & (labels == cid)
            signs[mask] = float(sign)
        sign_var = sign_var_for_cluster_var(cvar)
        ds[sign_var] = (ds[cvar].dims, signs)


def _signs_by_cluster_id(labels: np.ndarray, signs: np.ndarray) -> dict[int, int]:
    valid = np.isfinite(labels) & (labels >= 0) & np.isfinite(signs)
    out: dict[int, int] = {}
    for cid in np.unique(labels[valid].astype(int)):
        cluster_signs = signs[labels == cid]
        cluster_signs = cluster_signs[np.isfinite(cluster_signs)]
        if cluster_signs.size:
            out[int(cid)] = int(cluster_signs[0])
    return out


def test_sign_aware_negative_only_consensus_rate_uses_neg_votes():
    """When only negative-sign voxels pass the threshold, rate must not be zeroed."""
    n_st = 10
    native_union = np.zeros(n_st, dtype=bool)
    native_union[5] = True
    native_pos = np.zeros(n_st, dtype=bool)
    native_neg = native_union.copy()
    votes_pos = np.zeros(n_st, dtype=np.int16)
    votes_neg = np.zeros(n_st, dtype=np.int16)
    votes_neg[5] = 4

    labels, rate, signs = build_sign_aware_consensus_labels(
        native_union=native_union,
        votes_pos=votes_pos,
        votes_neg=votes_neg,
        n_members=7,
        min_consensus=0.5,
        n_st=n_st,
        n_space=n_st,
        label_fn=_single_cluster_label_fn(n_st),
        sign_aware=True,
        native_pos=native_pos,
        native_neg=native_neg,
    )

    assert labels[5] == 0
    assert signs[5] == -1
    assert rate[5] == 4 / 7


def test_rate_credits_the_natively_present_sign():
    """A native negative voxel must not be reported at the positive support level."""
    n_st = 10
    native_pos = np.zeros(n_st, dtype=bool)
    native_neg = np.zeros(n_st, dtype=bool)
    native_neg[5] = True
    native_pos[6] = True
    native_neg[6] = True
    votes_pos = np.zeros(n_st, dtype=np.int16)
    votes_neg = np.zeros(n_st, dtype=np.int16)
    votes_pos[5] = 5
    votes_neg[5] = 1
    votes_pos[6] = 2
    votes_neg[6] = 3

    _, rate, _ = build_sign_aware_consensus_labels(
        native_union=native_pos | native_neg,
        votes_pos=votes_pos,
        votes_neg=votes_neg,
        n_members=8,
        min_consensus=0.5,
        n_st=n_st,
        n_space=n_st,
        label_fn=_no_cluster_label_fn(n_st),
        sign_aware=True,
        native_pos=native_pos,
        native_neg=native_neg,
    )

    assert rate[5] == 1 / 8
    assert rate[6] == 3 / 8


def test_negative_native_voxel_is_not_retained_under_positive_support():
    n_st = 10
    native_pos = np.zeros(n_st, dtype=bool)
    native_neg = np.zeros(n_st, dtype=bool)
    native_neg[5] = True
    votes_pos = np.zeros(n_st, dtype=np.int16)
    votes_pos[5] = 4
    votes_neg = np.zeros(n_st, dtype=np.int16)

    labels, _, signs = build_sign_aware_consensus_labels(
        native_union=native_pos | native_neg,
        votes_pos=votes_pos,
        votes_neg=votes_neg,
        n_members=7,
        min_consensus=0.5,
        n_st=n_st,
        n_space=n_st,
        label_fn=_single_cluster_label_fn(n_st),
        sign_aware=True,
        native_pos=native_pos,
        native_neg=native_neg,
    )

    assert not np.any(labels >= 0)
    assert not np.any(np.isfinite(signs))


def test_sign_aware_labels_require_sign_split_native_masks():
    n_st = 4
    native_union = np.ones(n_st, dtype=bool)
    votes = np.full(n_st, 4, dtype=np.int16)

    with pytest.raises(ValueError, match="native_pos"):
        build_sign_aware_consensus_labels(
            native_union=native_union,
            votes_pos=votes,
            votes_neg=votes,
            n_members=7,
            min_consensus=0.5,
            n_st=n_st,
            n_space=n_st,
            label_fn=_single_cluster_label_fn(n_st),
            sign_aware=True,
        )


def test_cluster_label_id_mapping_tracks_renumbering():
    before = np.array([-1, 0, 1, 1, 2, np.nan])
    after = np.array([-1, -1, 0, 0, 1, np.nan])
    assert cluster_label_id_mapping(before, after) == {1: 0, 2: 1}


def test_sorted_consensus_labels_by_area_ranks_footprint_over_duration():
    n_space = 6
    labels = np.array(
        [
            [0, -1, 1, 1, 1, -1],
            [0, -1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1],
            [0, -1, -1, -1, -1, -1],
        ],
        dtype=np.int64,
    ).ravel()

    assert cluster_spatial_areas(labels.reshape(-1, n_space)) == {0: 1, 1: 3}
    ranked, mapping = sorted_consensus_labels_by_area(labels, n_space=n_space)
    assert mapping == {1: 0, 0: 1}
    assert ranked.reshape(-1, n_space)[0].tolist() == [1, -1, 0, 0, 0, -1]


def test_relabel_by_id_mapping_preserves_noise_and_nan():
    labels = np.array([np.nan, -1.0, 0.0, 1.0])
    out = relabel_by_id_mapping(labels, {0: 1, 1: 0})
    assert np.isnan(out[0])
    assert out[1] == -1.0
    assert out[2] == 1.0
    assert out[3] == 0.0


def test_cluster_label_id_mapping_rejects_split_clusters():
    with pytest.raises(ValueError, match="maps to both"):
        cluster_label_id_mapping(np.array([0, 0]), np.array([0, 1]))


def _single_cluster_label_fn(n_st: int):
    def label_fn(keep: np.ndarray) -> np.ndarray:
        out = np.full(n_st, -1, dtype=np.int64)
        out[keep] = 0
        return out

    return label_fn


def _no_cluster_label_fn(n_st: int):
    def label_fn(_keep: np.ndarray) -> np.ndarray:
        return np.full(n_st, -1, dtype=np.int64)

    return label_fn


def test_healpix_consensus_splits_opposite_sign_support():
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
    _attach_cluster_sign_fields(ds, ["model_a_cluster"], {0: 1})
    _attach_cluster_sign_fields(ds, ["model_b_cluster"], {0: -1})

    out = run_healpix_member_support_consensus(
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
    signs = out["signs"].values
    valid = labels[np.isfinite(labels) & (labels >= 0)]
    assert valid.size > 0
    assert len(np.unique(valid)) >= 2
    sign_map = _signs_by_cluster_id(labels, signs)
    assert set(sign_map.values()) == {-1, 1}


def test_healpix_consensus_backward_compatible_without_sign_fields():
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
    out = run_healpix_member_support_consensus(
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
    assert "signs" not in out or not np.any(np.isfinite(out["signs"].values))


def test_compute_consensus_splits_opposite_sign_support():
    T, y_len, x_len = 5, 8, 8
    coords = {"time": np.arange(T), "y": np.arange(y_len), "x": np.arange(x_len)}

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
    _attach_cluster_sign_fields(ds, ["model_a_cluster"], {0: 1})
    _attach_cluster_sign_fields(ds, ["model_b_cluster"], {0: -1})

    td = TOAD(ds, time_dim="time")
    td.compute_consensus(
        cluster_vars=["model_a_cluster", "model_b_cluster"],
        min_consensus=0.5,
        temporal_tolerance=1,
        spatial_tolerance=1,
        min_cluster_area=1,
        show_progress=False,
    )

    consensus = _latest_consensus_labels(td)
    sign_var = f"{consensus.name}{_attrs.CONSENSUS_SIGN_SUFFIX}"
    labels = consensus.values
    signs = td.data[sign_var].values
    valid = labels[np.isfinite(labels) & (labels >= 0)]
    assert valid.size > 0
    assert len(np.unique(valid)) >= 2
    sign_map = _signs_by_cluster_id(labels, signs)
    assert set(sign_map.values()) == {-1, 1}


def test_healpix_cluster_signs_survive_min_cluster_area_renumbering():
    T, nside, npix = 12, 4, 12 * 4**2
    n_members = 4
    coords = {"time": np.arange(T), "hp_pixel": np.arange(npix)}

    fields = {}
    for i in range(n_members):
        labels = np.full((T, npix), np.nan, dtype=np.float32)
        labels[2:4, 8:12] = 0
        labels[0:10, 150] = 1
        fields[f"model_{i}_cluster"] = (("time", "hp_pixel"), labels)

    ds = xr.Dataset(fields, coords=coords)
    cluster_vars = list(fields)
    for cvar in cluster_vars:
        _attach_cluster_sign_fields(ds, [cvar], {0: -1, 1: 1})

    out = run_healpix_member_support_consensus(
        _Store(ds, "time"),
        cluster_vars=cluster_vars,
        min_consensus=0.5,
        temporal_tolerance=1,
        spatial_tolerance=1,
        nside=nside,
        min_cluster_area=3,
        show_progress=False,
    )

    labels = out["clusters"].values
    signs = out["signs"].values
    surviving = np.unique(labels[np.isfinite(labels) & (labels >= 0)]).astype(int)
    assert surviving.tolist() == [0]
    assert _signs_by_cluster_id(labels, signs) == {0: -1}
    assert np.all(labels[0:10, 150] < 0)
    assert np.all(labels[2:4, 8:12] == 0)


def test_compute_consensus_cluster_signs_survive_min_cluster_area_filter():
    T, y_len, x_len = 12, 10, 10
    n_members = 4
    coords = {"time": np.arange(T), "y": np.arange(y_len), "x": np.arange(x_len)}

    fields = {}
    for i in range(n_members):
        labels = np.full((T, y_len, x_len), np.nan, dtype=np.float32)
        labels[2:4, 1:3, 1:3] = 0
        labels[0:10, 8, 8] = 1
        fields[f"model_{i}_cluster"] = (("time", "y", "x"), labels)

    ds = xr.Dataset(fields, coords=coords)
    cluster_vars = list(fields)
    for cvar in cluster_vars:
        ds[cvar].attrs.update(
            {
                _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER,
                _attrs.BASE_VARIABLE: "foo",
                _attrs.CLUSTER_IDS: np.array([0, 1], dtype=np.int32),
            }
        )
        _attach_cluster_sign_fields(ds, [cvar], {0: -1, 1: 1})

    td = TOAD(ds, time_dim="time")
    td.compute_consensus(
        cluster_vars=cluster_vars,
        min_consensus=0.5,
        temporal_tolerance=1,
        spatial_tolerance=1,
        min_cluster_area=3,
        show_progress=False,
    )

    consensus = _latest_consensus_labels(td)
    sign_var = f"{consensus.name}{_attrs.CONSENSUS_SIGN_SUFFIX}"
    labels = consensus.values
    signs = td.data[sign_var].values
    assert _signs_by_cluster_id(labels, signs) == {0: -1}
    assert np.all(labels[0:10, 8, 8] < 0)
    assert np.all(labels[2:4, 1:3, 1:3] == 0)


def test_consensus_ids_are_ranked_by_area_not_voxel_count():
    T, y_len, x_len = 12, 10, 10
    n_members = 4
    coords = {"time": np.arange(T), "y": np.arange(y_len), "x": np.arange(x_len)}

    fields = {}
    for i in range(n_members):
        labels = np.full((T, y_len, x_len), np.nan, dtype=np.float32)
        labels[0:10, 8, 8] = 0
        labels[2:4, 1:3, 1:3] = 1
        fields[f"model_{i}_cluster"] = (("time", "y", "x"), labels)

    ds = xr.Dataset(fields, coords=coords)
    cluster_vars = list(fields)
    for cvar in cluster_vars:
        ds[cvar].attrs.update(
            {
                _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER,
                _attrs.BASE_VARIABLE: "foo",
                _attrs.CLUSTER_IDS: np.array([0, 1], dtype=np.int32),
            }
        )
        _attach_cluster_sign_fields(ds, [cvar], {0: 1, 1: -1})

    td = TOAD(ds, time_dim="time")
    td.compute_consensus(
        cluster_vars=cluster_vars,
        min_consensus=0.5,
        temporal_tolerance=1,
        spatial_tolerance=1,
        min_cluster_area=None,
        show_progress=False,
    )

    consensus = _latest_consensus_labels(td)
    sign_var = f"{consensus.name}{_attrs.CONSENSUS_SIGN_SUFFIX}"
    labels = consensus.values
    signs = td.data[sign_var].values
    assert np.all(labels[2:4, 1:3, 1:3] == 0)
    assert np.all(labels[0:10, 8, 8] == 1)
    assert _signs_by_cluster_id(labels, signs) == {0: -1, 1: 1}


def test_healpix_consensus_ids_are_ranked_by_area_not_voxel_count():
    T, nside, npix = 12, 4, 12 * 4**2
    n_members = 4
    coords = {"time": np.arange(T), "hp_pixel": np.arange(npix)}

    fields = {}
    for i in range(n_members):
        labels = np.full((T, npix), np.nan, dtype=np.float32)
        labels[0:10, 150] = 0
        labels[2:4, 8:12] = 1
        fields[f"model_{i}_cluster"] = (("time", "hp_pixel"), labels)

    ds = xr.Dataset(fields, coords=coords)
    cluster_vars = list(fields)
    for cvar in cluster_vars:
        _attach_cluster_sign_fields(ds, [cvar], {0: 1, 1: -1})

    out = run_healpix_member_support_consensus(
        _Store(ds, "time"),
        cluster_vars=cluster_vars,
        min_consensus=0.5,
        temporal_tolerance=1,
        spatial_tolerance=1,
        nside=nside,
        min_cluster_area=None,
        show_progress=False,
    )

    labels = out["clusters"].values
    signs = out["signs"].values
    assert np.all(labels[2:4, 8:12] == 0)
    assert np.all(labels[0:10, 150] == 1)
    assert _signs_by_cluster_id(labels, signs) == {0: -1, 1: 1}


def test_compute_consensus_ignores_opposite_sign_support():
    T, y_len, x_len = 5, 8, 8
    coords = {"time": np.arange(T), "y": np.arange(y_len), "x": np.arange(x_len)}

    def _event(y: int, x: int) -> np.ndarray:
        labels = np.full((T, y_len, x_len), np.nan, dtype=np.float32)
        labels[2, y, x] = 0
        return labels

    ds = xr.Dataset(
        {
            "model_a_cluster": (("time", "y", "x"), _event(4, 4)),
            "model_b_cluster": (("time", "y", "x"), _event(4, 3)),
            "model_c_cluster": (("time", "y", "x"), _event(4, 5)),
        },
        coords=coords,
    )
    signs = {"model_a_cluster": -1, "model_b_cluster": 1, "model_c_cluster": 1}
    for cvar, sign in signs.items():
        ds[cvar].attrs.update(
            {
                _attrs.VARIABLE_TYPE: _attrs.TYPE_CLUSTER,
                _attrs.BASE_VARIABLE: "foo",
                _attrs.CLUSTER_IDS: np.array([0], dtype=np.int32),
            }
        )
        _attach_cluster_sign_fields(ds, [cvar], {0: sign})

    td = TOAD(ds, time_dim="time")
    td.compute_consensus(
        cluster_vars=list(signs),
        min_consensus=0.5,
        temporal_tolerance=1,
        spatial_tolerance=1,
        min_cluster_area=1,
        show_progress=False,
    )

    labels = _latest_consensus_labels(td).values
    assert not np.any(labels[np.isfinite(labels)] >= 0)


def _latest_consensus_labels(td: TOAD):
    names = td.consensus_cluster_vars
    assert names, "expected at least one consensus_cluster variable on td"
    return td.data[names[-1]]
