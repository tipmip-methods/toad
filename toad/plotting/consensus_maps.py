"""Shared map rendering for time-collapsed consensus label fields."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, cast

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib import colors as mcolors
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from toad.healpix import HealpixGrid, ipix_to_lonlat, ipix_vertices, polygon_paths
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


def _healpix_label_masks(
    clusters_map: np.ndarray,
    *,
    show_noise: bool,
    cluster_ids: Sequence[int] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for cluster and noise pixels on a 1D hp_pixel map."""
    finite = np.isfinite(clusters_map)
    cluster_mask = finite & (clusters_map >= 0)
    if cluster_ids is not None:
        cluster_mask &= np.isin(clusters_map, list(cluster_ids))
    noise_mask = (
        finite & (clusters_map == -1)
        if show_noise
        else np.zeros_like(cluster_mask, dtype=bool)
    )
    return cluster_mask, noise_mask


def _ensure_geoaxes_view(ax: Axes) -> None:
    """PolyCollection does not set Cartopy limits; open unset axes before drawing."""
    if not isinstance(ax, GeoAxes):
        return
    x0, x1 = ax.get_xlim()
    if x1 - x0 <= 1.5:
        ax.set_global()


def _add_healpix_polygon_layer(
    ax: Axes,
    ipix: np.ndarray,
    values: np.ndarray | None,
    *,
    grid: HealpixGrid,
    edge_step: int,
    facecolor: str | None = None,
    cmap: str = "tab10",
    vmin: float | None = None,
    vmax: float | None = None,
    norm: mcolors.Normalize | None = None,
    zorder: int = 3,
    paths: list[np.ndarray] | None = None,
    **kwargs: Any,
) -> PolyCollection:
    """Add one PolyCollection layer for a subset of HEALPix pixels."""
    if ipix.size == 0:
        empty = PolyCollection([], transform=ccrs.PlateCarree())
        ax.add_collection(empty)
        return empty

    if paths is None:
        lons, lats = ipix_vertices(ipix, grid, step=edge_step)
        paths = polygon_paths(lons, lats)
    collection_kwargs: dict[str, Any] = {
        "transform": ccrs.PlateCarree(),
        "zorder": zorder,
        "linewidths": 0.0,
        "edgecolors": "none",
    }
    collection_kwargs.update(kwargs)

    if facecolor is not None:
        collection = PolyCollection(
            paths,
            facecolors=facecolor,
            **collection_kwargs,
        )
    else:
        if norm is None and (vmin is not None or vmax is not None):
            norm = mcolors.Normalize(
                vmin=-0.5 if vmin is None else vmin,
                vmax=9.5 if vmax is None else vmax,
            )
        collection = PolyCollection(
            paths,
            array=np.asarray(values, dtype=np.float64),
            cmap=cmap,
            norm=norm,
            **collection_kwargs,
        )
    ax.add_collection(collection)
    return collection


def _healpix_cluster_centroids(
    clusters_map: np.ndarray,
    ipix: np.ndarray,
    grid: HealpixGrid,
) -> dict[int, tuple[float, float]]:
    """Median pixel-centre (lon, lat) per cluster id on a collapsed HEALPix map."""
    lats, lons = ipix_to_lonlat(ipix, grid, lon_convention="180")
    labels = clusters_map[ipix].astype(np.int64, copy=False)
    positions: dict[int, tuple[float, float]] = {}
    for cluster_id in np.unique(labels):
        mask = labels == cluster_id
        positions[int(cluster_id)] = (
            float(np.median(lons[mask])),
            float(np.median(lats[mask])),
        )
    return positions


def _cluster_color_for_id(
    cluster_id: int,
    *,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    cluster_ids: np.ndarray,
) -> Any:
    """Match polygon fill colour for one discrete cluster id."""
    ids = np.asarray(cluster_ids, dtype=np.float64)
    vmin_use = float(ids.min()) - 0.5 if vmin is None else float(vmin)
    vmax_use = float(ids.max()) + 0.5 if vmax is None else float(vmax)
    norm = mcolors.Normalize(vmin=vmin_use, vmax=vmax_use)
    return plt.get_cmap(cmap)(norm(float(cluster_id)))


def _annotate_healpix_cluster_labels(
    ax: Axes,
    clusters_map: np.ndarray,
    ipix: np.ndarray,
    *,
    grid: HealpixGrid,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
) -> None:
    """Place cluster-id labels at median pixel centres (same style as cluster_map)."""
    from toad.plotting import _cluster_annotate

    if ipix.size == 0:
        return
    centroids = _healpix_cluster_centroids(clusters_map, ipix, grid)
    label_ids = np.sort(np.unique(clusters_map[ipix].astype(np.int64)))
    for cluster_id in label_ids.tolist():
        lon, lat = centroids[int(cluster_id)]
        if not np.isfinite(lon) or not np.isfinite(lat):
            continue
        cluster_color = _cluster_color_for_id(
            int(cluster_id),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cluster_ids=label_ids,
        )
        _cluster_annotate(
            ax,
            lon,
            lat,
            int(cluster_id),
            cluster_color,
            transform=ccrs.PlateCarree(),
        )


def plot_healpix_cluster_labels_map(
    ax: Axes,
    clusters_map: np.ndarray,
    *,
    nside: int,
    cmap: str = "tab10",
    vmin: float | None = None,
    vmax: float | None = None,
    show_noise: bool = False,
    add_colorbar: bool = True,
    cluster_ids: Sequence[int] | None = None,
    colorbar_label: str = "Cluster ID",
    edge_step: int = 1,
    add_labels: bool = True,
    **kwargs: Any,
) -> PolyCollection:
    """Plot time-collapsed consensus labels as HEALPix cell polygons."""
    grid = HealpixGrid(nside=nside)
    cluster_mask, noise_mask = _healpix_label_masks(
        clusters_map,
        show_noise=show_noise,
        cluster_ids=cluster_ids,
    )

    if np.any(noise_mask):
        _add_healpix_polygon_layer(
            ax,
            np.flatnonzero(noise_mask),
            values=None,
            grid=grid,
            edge_step=edge_step,
            facecolor="lightgray",
            zorder=0,
            **kwargs,
        )

    ipix = np.flatnonzero(cluster_mask)
    collection = _add_healpix_polygon_layer(
        ax,
        ipix,
        clusters_map[ipix],
        grid=grid,
        edge_step=edge_step,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        zorder=3,
        **kwargs,
    )

    if add_colorbar and ipix.size > 0:
        plt.colorbar(collection, ax=ax, label=colorbar_label)
    _ensure_geoaxes_view(ax)
    if add_labels and ipix.size > 0:
        _annotate_healpix_cluster_labels(
            ax,
            clusters_map,
            ipix,
            grid=grid,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
    return collection


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
    edge_step: int = 1,
    add_labels: bool | None = None,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a time-collapsed consensus label field (HealPix or native grid)."""
    _ = s  # kept for API compatibility; HEALPix maps use filled polygons.
    clusters_map = collapse_consensus_for_map(clusters_da, time_dim=time_dim)
    fig, ax, config, proj = prepare_consensus_map_axes(ax, map_style)
    if add_labels is None:
        add_labels = config.add_labels

    if "hp_pixel" in clusters_da.dims:
        npix = int(clusters_da.sizes["hp_pixel"])
        merged_attrs: dict[str, Any] = {}
        merged_attrs.update(dataset.attrs)
        merged_attrs.update(clusters_da.attrs)
        resolved_nside = resolve_healpix_nside(npix, nside=nside, attrs=merged_attrs)
        if clusters_map.shape != (npix,):
            clusters_map = np.asarray(clusters_map).reshape(-1)
        plot_healpix_cluster_labels_map(
            ax,
            clusters_map,
            nside=resolved_nside,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            show_noise=show_noise,
            add_colorbar=add_colorbar,
            cluster_ids=cluster_ids,
            colorbar_label=colorbar_label,
            edge_step=edge_step,
            add_labels=add_labels,
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
