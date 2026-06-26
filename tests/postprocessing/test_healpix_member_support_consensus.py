import numpy as np
import xarray as xr

from toad.postprocessing.healpix_member_support_consensus import (
    run_healpix_member_support_consensus,
)


class _Store:
    def __init__(self, data: xr.Dataset, time_dim: str):
        self.data = data
        self.time_dim = time_dim


def test_healpix_member_support_basic():
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
    labels = out["clusters"].values
    rate = out["rate"].values
    assert labels.shape == (T, npix)
    assert rate.shape == (T, npix)
    assert np.any(labels >= 0)
