"""Shared map rendering for time-collapsed consensus label fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from toad.regridding.healpix import HealPixRegridder
from toad.utils import detect_latlon_names
from toad.utils.consensus_view import (
    collapse_consensus_for_map,
    infer_consensus_time_dim,
    resolve_healpix_nside,
)

if TYPE_CHECKING:
    from toad.plotting import MapStyle


def prepare_consensus_map_axes(
    ax: Axes | None,
    map_style: dict[str, Any] | MapStyle | None,
) -> tuple[Figure, Axes, Any, ccrs.Projection]:
    """Create or reuse map axes with projection and base map features."""
    from toad.plotting import _add_map_features, _normalize_map_style, get_projection

    config = _normalize_map_style(map_style)
    proj = (
        get_projection(config.projection)
        if config.projection is not None
        else ccrs.Mollweide()
    )

    if ax is None:
        fig, ax = plt.subplots(subplot_kw=dict(projection=proj))
        gax = cast(GeoAxes, ax)
        _add_map_features(gax, config)
        if config.extent is None and isinstance(proj, ccrs.Mollweide):
            gax.set_global()
        elif config.extent is not None:
            gax.set_extent(config.extent, crs=ccrs.PlateCarree())
        elif isinstance(proj, ccrs.SouthPolarStereo):
            gax.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
        elif isinstance(proj, ccrs.NorthPolarStereo):
            gax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
    else:
        fig = cast(Figure, ax.get_figure())

    return fig, ax, config, proj


def plot_healpix_cluster_labels_map(
    ax: Axes,
    lats: np.ndarray,
    lons: np.ndarray,
    clusters_map: np.ndarray,
    *,
    cmap: str = "tab10",
    s: float = 10,
    vmin: float | None = None,
    vmax: float | None = None,
    show_noise: bool = False,
    add_colorbar: bool = True,
    cluster_ids: Sequence[int] | None = None,
    colorbar_label: str = "Cluster ID",
    **kwargs: Any,
) -> Any:
    """Scatter-plot time-collapsed consensus labels on a HealPix grid."""
    cluster_mask = (clusters_map >= 0) & np.isfinite(clusters_map)
    if cluster_ids is not None:
        cluster_mask &= np.isin(clusters_map, list(cluster_ids))
    noise_mask = clusters_map == -1

    if show_noise:
        ax.scatter(
            lons[noise_mask],
            lats[noise_mask],
            c="lightgray",
            s=s,
            transform=ccrs.PlateCarree(),
            zorder=0,
            **kwargs,
        )

    sc = ax.scatter(
        lons[cluster_mask],
        lats[cluster_mask],
        c=clusters_map[cluster_mask],
        s=s,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        **kwargs,
    )
    if add_colorbar:
        plt.colorbar(sc, ax=ax, label=colorbar_label)
    return sc


def plot_native_cluster_labels_map(
    ax: Axes,
    clusters_map: np.ndarray,
    spatial_dims: Sequence[str],
    coords: dict[str, Any],
    dataset: xr.Dataset,
    proj: ccrs.Projection,
    *,
    cmap: str = "tab10",
    vmin: float | None = None,
    vmax: float | None = None,
    show_noise: bool = False,
    add_colorbar: bool = True,
    cluster_ids: Sequence[int] | None = None,
    colorbar_label: str = "Cluster ID",
    **kwargs: Any,
) -> Any:
    """Pcolormesh plot of time-collapsed consensus labels on a native grid."""
    da = xr.DataArray(
        clusters_map,
        dims=list(spatial_dims),
        coords={d: coords[d] for d in spatial_dims},
    )

    if cluster_ids is not None:
        cluster_ids_set = set(cluster_ids)
        da = da.where(
            da.isin(list(cluster_ids_set)) | (da == -1),
            other=np.nan,
        )
    if not show_noise:
        da = da.where(da >= 0, other=np.nan)

    lat_name, lon_name = detect_latlon_names(dataset)
    transform = ccrs.PlateCarree() if lat_name and lon_name else proj

    sc = da.plot.pcolormesh(
        ax=ax,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=add_colorbar,
        cbar_kwargs={"label": colorbar_label} if add_colorbar else {},
        transform=transform,
        **kwargs,
    )
    if add_colorbar and getattr(sc, "colorbar", None) is None:
        plt.colorbar(sc, ax=ax, label=colorbar_label)
    return sc


def plot_collapsed_consensus_labels_map(
    clusters_da: xr.DataArray,
    dataset: xr.Dataset,
    *,
    time_dim: str | None = None,
    nside: int | None = None,
    ax: Axes | None = None,
    map_style: dict[str, Any] | MapStyle | None = None,
    cmap: str = "tab10",
    s: float = 10,
    vmin: float | None = None,
    vmax: float | None = None,
    show_noise: bool = False,
    add_colorbar: bool = True,
    cluster_ids: Sequence[int] | None = None,
    colorbar_label: str = "Cluster ID",
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a time-collapsed consensus label field (HealPix or native grid)."""
    clusters_map = collapse_consensus_for_map(clusters_da, time_dim=time_dim)
    fig, ax, _config, proj = prepare_consensus_map_axes(ax, map_style)

    if "hp_pixel" in clusters_da.dims:
        npix = int(clusters_da.sizes["hp_pixel"])
        merged_attrs: dict[str, Any] = {}
        merged_attrs.update(dataset.attrs)
        merged_attrs.update(clusters_da.attrs)
        resolved_nside = resolve_healpix_nside(npix, nside=nside, attrs=merged_attrs)
        regridder = HealPixRegridder(nside=resolved_nside)
        lats, lons = regridder.pixels_to_latlon(np.arange(npix))
        plot_healpix_cluster_labels_map(
            ax,
            lats,
            lons,
            clusters_map,
            cmap=cmap,
            s=s,
            vmin=vmin,
            vmax=vmax,
            show_noise=show_noise,
            add_colorbar=add_colorbar,
            cluster_ids=cluster_ids,
            colorbar_label=colorbar_label,
            **kwargs,
        )
        return fig, ax

    resolved_time_dim = time_dim or infer_consensus_time_dim(clusters_da)
    if resolved_time_dim is None:
        raise ValueError("Could not infer time dimension for consensus labels.")
    spatial_dims = [d for d in clusters_da.dims if d != resolved_time_dim]
    if len(spatial_dims) < 2:
        raise ValueError("Native consensus_clusters must have at least 2 spatial dims.")

    plot_native_cluster_labels_map(
        ax,
        clusters_map,
        spatial_dims,
        {d: clusters_da.coords[d] for d in spatial_dims},
        dataset,
        proj,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_noise=show_noise,
        add_colorbar=add_colorbar,
        cluster_ids=cluster_ids,
        colorbar_label=colorbar_label,
        **kwargs,
    )
    return fig, ax
