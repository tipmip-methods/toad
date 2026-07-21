"""
Forward-bin categorical cluster-label exports onto a common GWL grid.

This is the categorical counterpart to continuous ``resample_to_gwl`` in
``tipmip_gwl``: cluster ids must not be linearly interpolated. Instead, each
export timestep is placed in the GWL bin it reaches and labels in a bin are
reduced per pixel (non-noise wins; ties → most frequent, then lowest id).

Two entry points:

* :func:`remap_export_to_gwl` — export on calendar-year (or zero-based year) axis
* :func:`bin_export_to_gwl` — export already on a continuous GWL axis
  (e.g. after :func:`tipmip_gwl.relabel_to_gwl` in preprocessing)
"""

from __future__ import annotations

import numpy as np
import xarray as xr

DEFAULT_GWL_STEP = 0.02

__all__ = [
    "DEFAULT_GWL_STEP",
    "bin_export_to_gwl",
    "export_on_continuous_gwl",
    "remap_export_to_gwl",
]


def _gwl_grid(
    step: float = DEFAULT_GWL_STEP, gwl_max: float = 4.0, gwl_min: float = 0.0
):
    if step <= 0:
        raise ValueError(f"gwl step must be positive, got {step}")
    if gwl_max <= gwl_min:
        raise ValueError(
            f"gwl_max must exceed gwl_min, got gwl_max={gwl_max}, gwl_min={gwl_min}"
        )
    return np.arange(gwl_min, gwl_max + step / 2, step)


def _reduce_block(block: np.ndarray) -> np.ndarray:
    """Reduce a (n_years, n_pixels) block of labels to one label per pixel."""
    n_pix = block.shape[1]
    out = np.full(n_pix, np.nan, dtype=np.float32)

    valid = np.isfinite(block) & (block >= 0)
    any_valid = valid.any(axis=0)
    if not any_valid.any():
        return out

    first_idx = valid.argmax(axis=0)
    cols = np.arange(n_pix)
    out[any_valid] = block[first_idx, cols][any_valid]

    valid_cols = np.where(any_valid)[0]
    masked = np.where(valid[:, valid_cols], block[:, valid_cols], np.nan)
    vmin = np.nanmin(masked, axis=0)
    vmax = np.nanmax(masked, axis=0)
    multi = valid_cols[vmin != vmax]
    for p in multi:
        labels = block[valid[:, p], p].astype(np.int64)
        u, counts = np.unique(labels, return_counts=True)
        out[p] = float(u[np.argmax(counts)])
    return out


def export_on_continuous_gwl(export, label_var: str | None = None) -> bool:
    """True when an export's leading axis already holds GWL values (degC)."""
    da, _ = _select_label_da(export, label_var)
    time_dim = da.dims[0]
    coord = da[time_dim]
    units = str(coord.attrs.get("units", "")).lower()
    if "degc" in units:
        return True
    return coord.attrs.get("long_name") == "global warming level"


def _select_label_da(export, label_var: str | None = None) -> tuple[xr.DataArray, str]:
    if isinstance(export, xr.Dataset):
        if label_var is None:
            label_var = "cluster" if "cluster" in export.data_vars else None
        if label_var is None:
            data_vars = list(export.data_vars)
            if len(data_vars) != 1:
                raise ValueError(
                    "export is a Dataset with multiple variables; pass "
                    f"label_var=... (one of {data_vars})"
                )
            label_var = data_vars[0]
        da = export[label_var]
    else:
        da = export
        label_var = da.name or "cluster"
    return da, label_var


def _forward_bin_categorical_export(
    export,
    gwl_at_timestep: np.ndarray,
    *,
    label_var: str | None = None,
    gwl_step: float = DEFAULT_GWL_STEP,
    gwl_max: float = 4.0,
    gwl_attrs: dict | None = None,
    remap_method: str,
    history_suffix: str = "remapped to GWL by toad",
):
    """Forward-bin categorical labels from per-timestep GWL onto the common grid."""
    is_dataset = isinstance(export, xr.Dataset)
    da, label_var = _select_label_da(export, label_var)
    time_dim = da.dims[0]
    spatial_dims = da.dims[1:]
    grid = _gwl_grid(gwl_step, gwl_max)

    gwl_at_timestep = np.asarray(gwl_at_timestep, dtype=float)
    if gwl_at_timestep.shape != (da.sizes[time_dim],):
        raise ValueError(
            f"gwl_at_timestep must have length {da.sizes[time_dim]}, "
            f"got {gwl_at_timestep.shape}"
        )

    edges = np.concatenate(([-np.inf], (grid[:-1] + grid[1:]) / 2.0, [np.inf]))
    bin_idx = np.full(gwl_at_timestep.shape, -1, dtype=np.int64)
    valid = (
        np.isfinite(gwl_at_timestep)
        & (gwl_at_timestep >= grid[0])
        & (gwl_at_timestep <= grid[-1])
    )
    bin_idx[valid] = np.clip(
        np.digitize(gwl_at_timestep[valid], edges) - 1, 0, grid.size - 1
    )

    arr = np.asarray(da.values)
    flat = arr.reshape(arr.shape[0], -1)
    out = np.full((grid.size, flat.shape[1]), np.nan, dtype=np.float32)
    for b in range(grid.size):
        rows = np.where(bin_idx == b)[0]
        if rows.size:
            out[b] = _reduce_block(flat[rows])

    out = out.reshape((grid.size,) + arr.shape[1:])

    if gwl_attrs is None:
        gwl_attrs = {"long_name": "global warming level", "units": "degC"}

    coords = {"gwl": ("gwl", grid)}
    for d in spatial_dims:
        if d in da.coords:
            coords[d] = da[d]
    for name, c in da.coords.items():
        if name == time_dim or name in coords:
            continue
        if set(c.dims).issubset(set(spatial_dims)):
            coords[name] = c

    attrs = dict(da.attrs)
    attrs["remapped_to"] = "global warming level (gwl)"
    attrs["remap_method"] = remap_method
    result = xr.DataArray(
        out,
        dims=("gwl",) + spatial_dims,
        coords=coords,
        name=label_var,
        attrs=attrs,
    )
    result["gwl"].attrs.update(gwl_attrs)

    if is_dataset:
        ds_out = result.to_dataset(name=label_var)
        ds_out.attrs.update(export.attrs)
        ds_out.attrs["history"] = (
            ds_out.attrs.get("history", "") + f"; {history_suffix}"
        ).lstrip("; ")
        ds_out.attrs["gwl_step"] = gwl_step
        ds_out.attrs["gwl_max"] = gwl_max
        return ds_out
    return result


def bin_export_to_gwl(
    export,
    *,
    label_var: str | None = None,
    gwl_step: float = DEFAULT_GWL_STEP,
    gwl_max: float = 4.0,
):
    """Forward-bin a categorical export already on a continuous GWL axis."""
    da, _ = _select_label_da(export, label_var)
    time_dim = da.dims[0]
    gwl_vals = np.asarray(da[time_dim].values, dtype=float)
    return _forward_bin_categorical_export(
        export,
        gwl_vals,
        label_var=label_var,
        gwl_step=gwl_step,
        gwl_max=gwl_max,
        gwl_attrs=dict(da[time_dim].attrs),
        remap_method=(
            f"forward-binned from continuous GWL axis onto {gwl_step} degC grid "
            f"(0-{gwl_max} degC); per-pixel label reduction "
            "(non-noise wins, most frequent label, ties -> lowest id)"
        ),
        history_suffix="binned to GWL grid by toad",
    )


def remap_export_to_gwl(
    export,
    mapping: xr.Dataset,
    *,
    label_var: str | None = None,
    export_start_year: int | None = None,
    gwl_step: float = DEFAULT_GWL_STEP,
    gwl_max: float = 4.0,
):
    """Remap a categorical cluster export onto the common GWL grid by forward-binning."""
    da, label_var = _select_label_da(export, label_var)
    time_dim = da.dims[0]

    t_raw = np.asarray(da[time_dim].values)
    if export_start_year is not None:
        export_years = (t_raw - t_raw.min()).astype(float) + float(export_start_year)
    else:
        export_years = t_raw.astype(float)

    map_years = np.asarray(mapping["year"].values, dtype=float)
    gwl_axis = np.asarray(mapping["gwl_axis"].values, dtype=float)

    finite = np.isfinite(gwl_axis)
    if finite.sum() < 2:
        raise ValueError("mapping['gwl_axis'] has fewer than two finite values")
    yr_f = map_years[finite]
    gwl_f = gwl_axis[finite]
    y_lo, y_hi = yr_f.min(), yr_f.max()

    overlap = (export_years >= y_lo) & (export_years <= y_hi)
    if not overlap.any():
        hint = ""
        if export_start_year is None and float(t_raw.min()) < y_lo:
            hint = (
                " The export time axis looks zero-based; pass export_start_year="
                f"{int(round(y_lo))} (the mapping's rampup_start_year)."
            )
        raise ValueError(
            "no export year overlaps the mapping year range "
            f"[{int(y_lo)}, {int(y_hi)}] (export covers "
            f"[{export_years.min():.0f}, {export_years.max():.0f}]).{hint}"
        )

    gwl_of_year = np.full(export_years.shape, np.nan)
    gwl_of_year[overlap] = np.interp(export_years[overlap], yr_f, gwl_f)

    if "gwl" in mapping.coords:
        gwl_attrs = dict(mapping["gwl"].attrs)
    else:
        gwl_attrs = {"long_name": "global warming level", "units": "degC"}

    return _forward_bin_categorical_export(
        export,
        gwl_of_year,
        label_var=label_var,
        gwl_step=gwl_step,
        gwl_max=gwl_max,
        gwl_attrs=gwl_attrs,
        remap_method=(
            f"forward-binned by gwl_axis(year) onto {gwl_step} degC grid "
            f"(0-{gwl_max} degC); per-pixel label reduction "
            "(non-noise wins, most frequent label, ties -> lowest id)"
        ),
    )
