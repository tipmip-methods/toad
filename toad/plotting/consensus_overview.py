"""Shared two-panel consensus overview (map + shift-time panel)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, Tuple, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import FigureBase

import cartopy.crs as ccrs

from toad.utils import detect_latlon_names

if TYPE_CHECKING:
    from toad.mma import MMA
    from toad.plotting import Plotter


def plot_consensus_overview(
    plotter: Plotter,
    consensus_var: str | None = None,
    cluster_ids: Optional[Union[int, list[int], np.ndarray, range]] = None,
    *,
    kind: Literal["medians", "violins"] = "medians",
    spread: Literal["iqr", "std"] = "std",
    figsize: Optional[Tuple[float, float]] = None,
    width_ratios: Tuple[float, float] = (1.25, 1.0),
    wspace: float = 0.28,
    cmap: Union[str, ListedColormap, None] = None,
    colorbar_shrink: float = 0.38,
    colorbar_pad: float = 0.025,
    colorbar_aspect: float = 28.0,
    colorbar_label: Optional[str] = None,
    map_style: Optional[dict[str, Any] | Any] = None,
    show_legend: bool = True,
    ylabel: Optional[str] = None,
    seed: Optional[int] = None,
    show_sum: bool = False,
    show_total: bool = True,
    total_color: str = "#666666",
    bw_method: float = 0.18,
    shift_time_distributions: dict[int, np.ndarray] | None = None,
    **kwargs: Any,
) -> Tuple[FigureBase, Any, Axes]:
    """Render the shared consensus overview figure.

    When ``shift_time_distributions`` is provided (MMA path), the right panel uses
    those pooled export-derived samples. Otherwise shift times come from
    :class:`toad.postprocessing.Aggregation` on the wrapped :class:`Plotter` TOAD.
    """
    from toad.plotting import (
        _add_map_features,
        _normalize_map_style,
        default_cmap,
        get_projection,
    )

    if kind not in ("medians", "violins"):
        raise ValueError(
            f"consensus_overview: unknown kind {kind!r}; expected 'medians' or 'violins'."
        )
    if shift_time_distributions is not None and kind == "medians":
        raise ValueError(
            "MMA-style shift-time overviews support kind='violins' only. "
            "Use a TOAD with input cluster variables for kind='medians'."
        )

    if cmap is None:
        cmap = default_cmap

    consensus_var_resolved = plotter.td._resolve_consensus_var(consensus_var)
    da = plotter.td.data[consensus_var_resolved]

    if kind == "medians":
        dist_ds, _ = plotter.td.aggregate.consensus_shift_time_distribution(da)
        if (
            len(dist_ds.data_vars) == 0
            or "spatial_median_transition_time" not in dist_ds
        ):
            raise ValueError(
                "No per-model spatial median shift times available; check consensus labels "
                "and input cluster variables."
            )
        sm = dist_ds["spatial_median_transition_time"]
        ids_all = np.asarray(sm.coords["consensus_cluster_id"].values, dtype=np.int64)
    elif shift_time_distributions is not None:
        if not shift_time_distributions:
            raise ValueError(
                "No transition-time samples for violin plot; check consensus labels "
                "and MMA export files."
            )
        ids_all = np.array(sorted(shift_time_distributions.keys()), dtype=np.int64)
    else:
        dists = plotter.td.aggregate.consensus_shift_time_distributions(da)
        if not dists:
            raise ValueError(
                "No transition-time samples for violin plot; check consensus labels "
                "and input cluster variables."
            )
        ids_all = np.array(sorted(dists.keys()), dtype=np.int64)

    if cluster_ids is not None:
        if isinstance(cluster_ids, int):
            wanted = {int(cluster_ids)}
        else:
            wanted = {int(x) for x in cluster_ids}
        plot_ids = np.array([i for i in ids_all if int(i) in wanted], dtype=np.int64)
    else:
        plot_ids = ids_all.copy()
    if plot_ids.size == 0:
        raise ValueError("No consensus cluster ids left after filtering.")
    plot_ids.sort()

    _figsize = figsize if figsize is not None else (12.0, 5.2)
    config = _normalize_map_style(map_style)
    lat_name, lon_name = detect_latlon_names(plotter.td.data)
    has_latlon = lat_name is not None and lon_name is not None
    if config.projection is None:
        projection_obj = get_projection("plate_carree") if has_latlon else None
    else:
        projection_obj = get_projection(config.projection)
    if projection_obj is None:
        projection_obj = get_projection("mollweide")

    fig = plt.figure(figsize=_figsize)
    gs = fig.add_gridspec(1, 2, width_ratios=list(width_ratios))
    ax_map = cast(GeoAxes, fig.add_subplot(gs[0, 0], projection=projection_obj))
    ax_right = fig.add_subplot(gs[0, 1])

    _add_map_features(ax_map, config)
    if config.extent is None:
        if ax_map.projection == ccrs.SouthPolarStereo():
            ax_map.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
        elif ax_map.projection == ccrs.NorthPolarStereo():
            ax_map.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
        else:
            ax_map.set_global()
    else:
        ax_map.set_extent(config.extent, crs=ccrs.PlateCarree())
    ax_map.set_frame_on(config.map_frame)

    map_ids = [int(x) for x in plot_ids.tolist()]
    use_labels_map = "hp_pixel" in da.dims
    if use_labels_map:
        map_out = plotter.consensus_labels_map(
            consensus_var,
            ax=ax_map,
            cluster_ids=map_ids,
            cmap=cmap if isinstance(cmap, str) else "tab10",
            map_style=map_style,
            add_colorbar=True,
            colorbar_label=colorbar_label or "Cluster ID",
        )
    else:
        map_out = plotter.consensus_map(
            consensus_var,
            cluster_ids=map_ids,
            ax=ax_map,
            cmap=cmap,
            colorbar_shrink=colorbar_shrink,
            colorbar_pad=colorbar_pad,
            colorbar_aspect=colorbar_aspect,
            colorbar_label=colorbar_label,
            map_style=map_style,
        )
    if map_out[1] is None:
        plt.close(fig)
        raise ValueError(
            "consensus map produced no axes; check consensus cluster ids against the dataset."
        )

    if kind == "medians":
        median_kw = dict(kwargs)
        median_kw.pop("ax", None)
        median_kw.pop("figsize", None)
        if "summary_cluster_cmap" not in median_kw:
            median_kw["summary_cluster_cmap"] = cmap

        plotter.consensus_shift_times_medians(
            consensus_var=consensus_var,
            cluster_ids=cluster_ids,
            spread=spread,
            ax=ax_right,
            show_legend=show_legend,
            ylabel=ylabel,
            seed=seed,
            tight_layout=False,
            **median_kw,
        )
    else:
        violin_kw = dict(kwargs)
        violin_kw.pop("ax", None)
        violin_kw.pop("figsize", None)
        for _k in ("show_sum", "show_total", "total_color", "bw_method"):
            violin_kw.pop(_k, None)
        violin_cmap = violin_kw.pop("cmap", cmap)
        plotter.consensus_shift_times_violins(
            consensus_var=consensus_var,
            cluster_ids=cluster_ids,
            ax=ax_right,
            cmap=violin_cmap,
            show_legend=show_legend,
            ylabel=ylabel,
            seed=seed,
            show_sum=show_sum,
            show_total=show_total,
            total_color=total_color,
            bw_method=bw_method,
            tight_layout=False,
            shift_time_distributions=shift_time_distributions,
            **violin_kw,
        )

    fig.subplots_adjust(wspace=wspace)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94 if show_legend else 1.0))

    return fig, ax_map, ax_right


class MMAPlotView:
    """Plotting API for :class:`toad.MMA` (mirrors :class:`Plotter` consensus methods)."""

    def __init__(self, mma: MMA) -> None:
        self._mma = mma

    def _plotter(self) -> Plotter:
        from toad import TOAD
        from toad.plotting import Plotter

        return Plotter(
            TOAD(self._mma.data, time_dim=self._mma._time_dim, log_level="CRITICAL")
        )

    def consensus_overview(
        self,
        consensus_var: str | None = None,
        cluster_ids: Optional[Union[int, list[int], np.ndarray, range]] = None,
        **kwargs: Any,
    ) -> Tuple[FigureBase, Any, Axes]:
        """Same API as :meth:`Plotter.consensus_overview` for MMA consensus results."""
        _ = consensus_var  # inferred from MMA consensus dataset
        return plot_consensus_overview(
            self._plotter(),
            cluster_ids=cluster_ids,
            shift_time_distributions=self._mma.get_shift_times_per_consensus_cluster(
                numeric=True
            ),
            **kwargs,
        )
