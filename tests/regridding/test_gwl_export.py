"""Tests for categorical cluster-export -> GWL forward-binning."""

import numpy as np
import pytest
import xarray as xr

from toad.regridding.gwl_export import (
    _reduce_block,
    bin_export_to_gwl,
    export_on_continuous_gwl,
    remap_export_to_gwl,
)


class TestReduceBlock:
    def test_single_label_per_pixel_passes_through(self):
        block = np.array([[5.0, -1.0], [5.0, -1.0], [np.nan, -1.0]])
        out = _reduce_block(block)
        assert out[0] == 5.0
        assert np.isnan(out[1])

    def test_most_frequent_real_label_wins(self):
        block = np.array([[3.0], [3.0], [7.0]])
        out = _reduce_block(block)
        assert out[0] == 3.0

    def test_tie_breaks_to_lowest_id(self):
        block = np.array([[3.0], [7.0]])
        out = _reduce_block(block)
        assert out[0] == 3.0

    def test_all_noise_or_nan_gives_nan(self):
        block = np.array([[-1.0, np.nan], [np.nan, -1.0]])
        out = _reduce_block(block)
        assert np.all(np.isnan(out))


def _toy_mapping(y_lo=1900, y_hi=2000):
    years = np.arange(y_lo, y_hi + 1, dtype=float)
    gwl_axis = 2.0 * (years - y_lo) / (y_hi - y_lo)
    return xr.Dataset(
        {"gwl_axis": ("year", gwl_axis)},
        coords={"year": years},
    )


def test_remap_export_to_gwl_bins_labels_onto_gwl_grid():
    mapping = _toy_mapping()
    n_years = 101
    labels = np.full((n_years, 2), -1.0)
    labels[50, 0] = 4.0
    da = xr.DataArray(
        labels,
        dims=("time", "hp_pixel"),
        coords={"time": np.arange(1900, 2001), "hp_pixel": [0, 1]},
        name="cluster",
    )
    out = remap_export_to_gwl(da, mapping, gwl_step=0.1, gwl_max=2.0)
    assert out.dims == ("gwl", "hp_pixel")
    gwl_vals = out["gwl"].values
    idx_near_1 = np.argmin(np.abs(gwl_vals - 1.0))
    assert out.values[idx_near_1, 0] == 4.0
    assert np.all(np.isnan(out.values[:, 1]))


def test_remap_export_to_gwl_zero_based_time_needs_start_year():
    mapping = _toy_mapping()
    labels = np.full((101, 1), -1.0)
    labels[0, 0] = 1.0
    da = xr.DataArray(
        labels,
        dims=("time", "hp_pixel"),
        coords={"time": np.arange(0, 101), "hp_pixel": [0]},
        name="cluster",
    )
    with pytest.raises(ValueError, match="export_start_year"):
        remap_export_to_gwl(da, mapping, gwl_step=0.1, gwl_max=2.0)

    out = remap_export_to_gwl(
        da, mapping, export_start_year=1900, gwl_step=0.1, gwl_max=2.0
    )
    assert out.values[0, 0] == 1.0


def test_remap_export_to_gwl_no_overlap_raises():
    mapping = _toy_mapping(y_lo=1900, y_hi=2000)
    da = xr.DataArray(
        np.full((10, 1), -1.0),
        dims=("time", "hp_pixel"),
        coords={"time": np.arange(2100, 2110), "hp_pixel": [0]},
        name="cluster",
    )
    with pytest.raises(ValueError, match="no export year overlaps"):
        remap_export_to_gwl(da, mapping)


def test_remap_export_to_gwl_dataset_input_picks_cluster_var():
    mapping = _toy_mapping()
    labels = np.full((101, 1), -1.0)
    labels[50, 0] = 2.0
    da = xr.DataArray(
        labels,
        dims=("time", "hp_pixel"),
        coords={"time": np.arange(1900, 2001), "hp_pixel": [0]},
    )
    ds = da.to_dataset(name="cluster")
    out = remap_export_to_gwl(ds, mapping, gwl_step=0.1, gwl_max=2.0)
    assert isinstance(out, xr.Dataset)
    assert "cluster" in out.data_vars


def test_bin_export_to_gwl_from_continuous_gwl_axis():
    labels = np.full((50, 1), -1.0)
    labels[25, 0] = 3.0
    gwl = np.linspace(0.0, 2.0, 50)
    da = xr.DataArray(
        labels,
        dims=("time", "hp_pixel"),
        coords={
            "time": (
                "time",
                gwl,
                {"units": "degC", "long_name": "global warming level"},
            ),
            "hp_pixel": [0],
        },
        name="cluster",
    )
    assert export_on_continuous_gwl(da)
    out = bin_export_to_gwl(da, gwl_step=0.1, gwl_max=2.0)
    idx_near_1 = np.argmin(np.abs(out["gwl"].values - 1.0))
    assert out.values[idx_near_1, 0] == 3.0
