import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_hex, to_rgba, to_rgb
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.figure
from matplotlib.axes import Axes
from typing import Union, Tuple, Optional, List, Any, overload, Literal
from dataclasses import dataclass

_projection_map = {
    "plate_carree": ccrs.PlateCarree(),
    "north_pole": ccrs.NorthPolarStereo(),
    "south_pole": ccrs.SouthPolarStereo(),
    "global": ccrs.Robinson(),
    "mollweide": ccrs.Mollweide(),
}


@dataclass
class PlotConfig:
    resolution: str = "110m"
    coastline_linewidth: float = 0.5
    border_linewidth: float = 0.25
    grid_labels: bool = True
    grid_style: str = "--"
    grid_width: float = 0.5
    grid_color: str = "gray"
    grid_alpha: float = 0.5
    borders: bool = True
    projection: str = "plate_carree"
    map_frame: bool = True


@dataclass
class ToadColors:
    green = "#6F9F50"
    green_light = "#BCCDB3"
    green_dark = "#43712C"
    yellow = "#F1E0B0"
    primary = green_dark
    secondary = green_light
    tertiary = yellow


class TOADPlotter:
    def __init__(self, td, config: Optional[PlotConfig] = None):
        from toad import TOAD

        self.td: TOAD = td
        self.default_config = config if config is not None else PlotConfig()

    # Overloads are used for type hinting
    @overload
    def map(
        self,
        nrows: Literal[1] = 1,
        ncols: Literal[1] = 1,
        projection: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        figsize: Optional[Tuple[float, float]] = None,
        height_ratios: Optional[List[float]] = None,
    ) -> Tuple[matplotlib.figure.Figure, Axes]: ...

    @overload
    def map(
        self,
        nrows: int,
        ncols: int = 1,
        projection: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        figsize: Optional[Tuple[float, float]] = None,
        height_ratios: Optional[List[float]] = None,
    ) -> Tuple[matplotlib.figure.Figure, np.ndarray]: ...

    def map(
        self,
        nrows: int = 1,
        ncols: int = 1,
        projection: Optional[str] = None,
        config: Optional[PlotConfig] = None,
        figsize: Optional[Tuple[float, float]] = None,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        subplot_spec: Optional[Any] = None,
    ) -> Tuple[matplotlib.figure.Figure, Union[Axes, np.ndarray]]:
        """Create map plots with standard features.

        Args:
            nrows: Number of rows in subplot grid
            ncols: Number of columns in subplot grid
            projection: Map projection to use
            config: Plot configuration
            figsize: Figure size (width, height) in inches
            height_ratios: List of height ratios for subplots
            width_ratios: List of width ratios for subplots
            subplot_spec: A gridspec subplot spec to place the map in
        """
        config = config if config else self.default_config
        projection = projection if projection else config.projection

        if projection not in _projection_map:
            raise ValueError(f"Invalid projection '{projection}'")

        if subplot_spec is not None:
            # Create map in existing figure using subplot_spec
            fig = plt.gcf()
            ax = fig.add_subplot(subplot_spec, projection=_projection_map[projection])
            axs = ax
        else:
            # Create new figure with subplots
            gridspec_kw = {}
            if height_ratios:
                gridspec_kw["height_ratios"] = height_ratios
            if width_ratios:
                gridspec_kw["width_ratios"] = width_ratios

            fig, axs = plt.subplots(
                nrows,
                ncols,
                figsize=figsize,
                subplot_kw={"projection": _projection_map[projection]},
                gridspec_kw=gridspec_kw if gridspec_kw else None,
            )

        # Ensure axs is always an array for consistent iteration
        axs_array = np.array(axs, ndmin=2)

        # Add map features and set extent for all axes
        for ax in axs_array.flat:
            self._add_map_features(ax, config)
            ax.set_frame_on(config.map_frame)

            if projection == "south_pole":
                ax.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
            elif projection == "north_pole":
                ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())

        # Return single axis or array
        if axs_array.size == 1:
            return fig, axs_array[0, 0]
        else:
            return fig, np.squeeze(axs)

    def _replace_ax_projection(
        self,
        fig: matplotlib.figure.Figure,
        axs: Union[np.ndarray, Axes],
        row: int,
        col: int,
        projection: str,
    ) -> Union[np.ndarray, Axes]:
        """
        Replace the subplot at the given row and column of axs with a map projection
        """
        if isinstance(axs, Axes):
            # For single Axes, create 1x1 array
            axs = np.array([[axs]])
        else:
            axs = np.array(axs, ndmin=2)

        axs[row, col].remove()
        axs[row, col] = fig.add_subplot(
            axs.shape[0],
            axs.shape[1],
            row * axs.shape[1] + col + 1,
            projection=_projection_map[projection],
        )

        # Return single Axes if input was single Axes
        if isinstance(axs, Axes):
            return axs[0, 0]
        return axs

    def _remove_spines(
        self,
        axs: Union[Axes, List[Axes], np.ndarray],
        spines: Union[List[str], str, np.ndarray] = ["top", "right", "bottom", "left"],
    ):
        if isinstance(axs, Axes):
            axs = np.asarray([axs])

        if isinstance(spines, str):
            spines = np.asarray([spines])

        for ax in axs:
            for s in spines:
                ax.spines[s].set_visible(False)

    def _remove_ticks(
        self, axs: Union[Axes, List[Axes], np.ndarray], keep_x=False, keep_y=False
    ):
        if isinstance(axs, Axes):
            axs = np.asarray([axs])

        for ax in axs:
            if not keep_x:
                ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
                [label.set_visible(False) for label in ax.get_xticklabels()]
            if not keep_y:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)
                [label.set_visible(False) for label in ax.get_yticklabels()]

    def frame_plot_with_rows(
        self,
        rows: int,
        h_ratios: Optional[List[int]] = None,
        no_ticks: bool = False,
        alternate_label_side: bool = True,
        ax_padding_y: float = 0.2,
        axs: Optional[
            Union[List[Axes], np.ndarray]
        ] = None,  # New parameter for existing axes
        **kwargs,
    ) -> Tuple[Optional[matplotlib.figure.Figure], np.ndarray]:
        fig = None
        if axs is None:
            fig, axs = plt.subplots(
                nrows=rows,
                ncols=1,
                sharex=True,
                gridspec_kw={"height_ratios": h_ratios},
                **kwargs,
            )

        # set height ratios
        plt.subplots_adjust(hspace=0)
        for ax in axs:
            ax.margins(y=ax_padding_y)

        self._remove_spines(axs[0], "bottom")
        self._remove_ticks(axs[0], keep_y=True)

        for ax in axs[1:-1]:
            self._remove_spines(ax, ["top", "bottom"])
            self._remove_ticks(ax, keep_y=True)

        if alternate_label_side:
            for ax in axs[1::2]:
                self._remove_ticks(ax)
                ax.yaxis.tick_right()
                ax.tick_params(axis="y", which="both", labelleft=False, labelright=True)
                ax.yaxis.set_label_position("right")

        self._remove_spines(axs[-1], "top")
        if fig:
            return fig, axs
        else:
            return axs

    def alternating_axis_rows(
        self,
        rows: int,
        h_ratios: Optional[List[int]] = None,
        ax_padding_y: float = 0.2,
        axs: Optional[
            Union[List[Axes], np.ndarray]
        ] = None,  # New parameter for existing axes
        **kwargs,
    ) -> Tuple[Optional[matplotlib.figure.Figure], np.ndarray]:
        fig = None
        if axs is None:
            fig, axs = plt.subplots(
                nrows=rows,
                ncols=1,
                sharex=True,
                gridspec_kw={"height_ratios": h_ratios},
                **kwargs,
            )

        # set height ratios
        plt.subplots_adjust(hspace=0)
        for ax in axs:
            ax.margins(y=ax_padding_y)

        # Even plots
        for i in range(len(axs)):
            ax = axs[i]

            self._remove_spines(ax, "top")
            if i % 2 == 0:
                self._remove_spines(ax, "right")
            else:
                self._remove_spines(ax, "left")
            if i < len(axs) - 1:
                self._remove_spines(ax, "bottom")

        # odd plots
        for ax in axs[1::2]:
            ax.yaxis.tick_right()
            ax.tick_params(axis="y", which="both", labelleft=False, labelright=True)
            ax.yaxis.set_label_position("right")

        self._remove_ticks(axs[:-1], keep_y=True)
        return fig, axs

    def _add_map_features(self, ax: Axes, config: PlotConfig) -> None:
        """Add standard map features to an axes.

        Args:
            ax: Matplotlib axes with cartopy projection
            config: Plot configuration, uses default if None
        """
        ax.coastlines(
            resolution=config.resolution, linewidth=config.coastline_linewidth
        )

        if config.borders:
            ax.add_feature(
                cfeature.BORDERS, linestyle="-", linewidth=config.border_linewidth
            )

        if config.grid_labels:
            ax.gridlines(
                draw_labels=config.grid_labels,
                linewidth=config.grid_width,
                color=config.grid_color,
                alpha=config.grid_alpha,
                linestyle=config.grid_style,
            )

    def _cluster_annotate(
        self,
        ax: Axes,
        x: float,
        y: float,
        cluster_id: int,
        acol: str,
        scale: float = 1,
        relative_coords: bool = False,
    ):
        black_or_white = get_high_constrast_text_color(acol)
        t = ax.annotate(
            text=str(cluster_id),
            xy=(x, y),
            xycoords="axes fraction" if relative_coords else "data",
            annotation_clip=False,
            color=black_or_white,
            zorder=100,
            fontweight="semibold",
            ha="center",
            va="center",
            fontsize=6 + 4 * scale,
        )
        t.set_bbox(
            dict(
                facecolor=acol,
                alpha=1,
                edgecolor=black_or_white,
                boxstyle="round,pad=0.2,rounding_size=0.2",  # adjust rounding_size to control corner radius
            )
        )

    def cluster_map_contour(
        self,
        var: str,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = None,
        projection: Optional[str] = None,
        ax: Optional[Axes] = None,
        color: Optional[str] = None,
        cmap: Union[str, ListedColormap] = "tab20",
        add_labels: bool = True,
        **kwargs: Any,
    ) -> Tuple[matplotlib.figure.Figure, Axes]:
        return self.cluster_map(
            var,
            cluster_ids,
            projection,
            ax,
            color,
            cmap,
            only_contour=True,
            add_labels=add_labels,
        )

    def cluster_map(
        self,
        var: str,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = None,
        projection: Optional[str] = None,
        ax: Optional[Axes] = None,
        color: Optional[Union[str, Tuple, List[Union[str, Tuple]]]] = None,
        cmap: Union[str, ListedColormap] = "tab20",
        add_contour=True,
        only_contour=False,
        add_labels: bool = True,
        unclustered_color: Optional[str] = None,
        remaining_clusters_color: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[matplotlib.figure.Figure, Axes]:
        """Plot one or multiple clusters on a map.

        Args:
            var: Variable name for which clusters have been computed
            cluster_ids: Single cluster ID or list of cluster IDs to plot. Defaults to all clusters
            ax: Matplotlib axes to plot on. Creates new figure if None
            color: Color for cluster visualization. Can be:
                - A single color (str, hex, RGB tuple) to use for all clusters
                - A list of colors to use for each cluster
            cmap: Colormap for multiple clusters. Used only if color is None
            add_contour: If True, add contour lines around clusters
            only_contour: If True, only plot contour lines (no fill)
            add_labels: If True, add cluster ID labels
            **kwargs: Additional arguments passed to xarray.plot

        Returns:
            Tuple of (figure, axes)

        Raises:
            ValueError: If no clusters found for given variable
        """
        clusters = self.td.get_clusters(var)
        if clusters is None:
            raise ValueError(f"No clusters found for variable {var}")

        if ax is None:
            fig, ax = self.map(projection=projection)
        else:
            fig = None

        # Plot all clusters (except -1) if no cluster_ids passed
        all_cluster_ids = clusters.cluster_ids
        cluster_ids = (
            cluster_ids
            if cluster_ids is not None
            else all_cluster_ids[all_cluster_ids != -1]
        )

        # Check that we have a valid cluster_ids value
        if not isinstance(cluster_ids, (int, list, np.ndarray, range)):
            raise TypeError("cluster_ids must be int, list, np.ndarray, range, or None")

        # Convert single cluster_id to list for consistent handling
        if isinstance(cluster_ids, int):
            single_plot = True
            cluster_ids = [cluster_ids]
        else:
            single_plot = False
            cluster_ids = list(cluster_ids)  # Convert to list for consistent indexing

        # Create color list for each cluster
        if color is not None:
            # If color is a list, use it directly (one color per cluster)
            if (
                isinstance(color, (list, tuple))
                and len(color) > 1
                and not all(isinstance(c, (int, float)) for c in color)
            ):
                color_list = color
                if len(color_list) < len(cluster_ids):
                    # Repeat colors if needed
                    color_list = color_list * (len(cluster_ids) // len(color_list) + 1)
                color_list = color_list[
                    : len(cluster_ids)
                ]  # Trim to match cluster_ids length
            else:
                # Single color for all clusters
                color_list = [color] * len(cluster_ids)
        else:
            # Use colormap to generate colors
            if isinstance(cmap, str):
                base_cmap = plt.get_cmap(cmap)
                color_list = [base_cmap(i) for i in np.linspace(0, 1, len(cluster_ids))]
            elif isinstance(cmap, ListedColormap):
                # Extract colors from the ListedColormap
                cmap_colors = cmap.colors
                # Repeat colors if needed
                if len(cmap_colors) < len(cluster_ids):
                    cmap_colors = cmap_colors * (
                        len(cluster_ids) // len(cmap_colors) + 1
                    )
                color_list = cmap_colors[: len(cluster_ids)]
            else:
                raise TypeError("cmap must be a string or ListedColormap")

        # Create a ListedColormap for each cluster
        cmap_list = [ListedColormap([c]) for c in color_list]

        for i, id in enumerate(cluster_ids):
            # Skip if not in cluster ids
            if id not in all_cluster_ids:
                continue

            # Get the colormap for this cluster
            cluster_cmap = cmap_list[i]

            # Get mask for clustered or unclustered cells
            mask = (
                self.td.get_permanent_unclustered_mask(var)
                if id == -1
                else self.td.get_spatial_cluster_mask(var, id)
            )

            plot_params = {
                "ax": ax,
                "cmap": cluster_cmap,
                "add_colorbar": False,
                "transform": ccrs.PlateCarree()
                if "lat" in self.td.space_dims
                else None,
                **kwargs,
            }

            if not only_contour:
                # Don't plot values outside mask: FALSE -> np.nan
                mask.where(mask, np.nan).plot(
                    **plot_params,
                )

            if only_contour or add_contour:
                if add_contour:
                    # Make contour color darker
                    contour_color = cluster_cmap.colors[0]
                    color_rgba = to_rgba(contour_color)
                    darker_color = (
                        color_rgba[0] * 0.8,
                        color_rgba[1] * 0.8,
                        color_rgba[2] * 0.8,
                        color_rgba[3],
                    )
                    plot_params["cmap"] = ListedColormap([darker_color])

                mask.plot.contour(
                    levels=1,
                    linewidths=2,
                    **plot_params,
                )

            if add_labels:
                # returns space_dims[0, 1], so y, x or lon, lat
                y, x = self.td.cluster_stats(var).space.median(id)
                # TODO: Use a peak density pos instead of the median (case you have a circular cluster...)
                self._cluster_annotate(ax, x, y, id, cluster_cmap.colors[0])

            if single_plot:
                ax.set_title(f"{var}_cluster {id}")

        # Plot remaining clusters
        if remaining_clusters_color:
            remaining_cluster_ids = [  # get unplotted clusters ids (except -1)
                int(id) for id in all_cluster_ids if id not in cluster_ids and id != -1
            ]
            mask = self.td.get_spatial_cluster_mask(var, remaining_cluster_ids)
            mask = mask.where(mask > 0, np.nan)
            mask.plot(
                cmap=ListedColormap([remaining_clusters_color]),
                add_colorbar=False,
                ax=ax,
            )

        # Plot unclustered cells
        if unclustered_color:
            unclustered_mask = self.td.get_permanent_unclustered_mask(var).where(
                ~self.td.data[var].isnull().all(dim=self.td.time_dim), 0
            )  # unclustered cells with data
            unclustered_mask = unclustered_mask.where(unclustered_mask > 0, np.nan)
            unclustered_mask.plot(
                cmap=ListedColormap([unclustered_color]),
                add_colorbar=False,
                ax=ax,
            )

        return fig, ax

    # def _filter_existing_cluster_ids()

    def cluster_maps(
        self,
        var: str,
        cluster_ids: Union[List[int], np.ndarray, range],
        ncols: int = 5,
        color: Optional[str] = None,
        projection: Optional[str] = None,
        width: float = 12,
        row_height: float = 2.5,
        **kwargs: Any,
    ):
        """Plot individual clusters on separate maps.

        Args:
            var: Variable name for which clusters have been computed
            max_clusters: Maximum number of clusters to plot
            ncols: Number of columns in subplot grid
            color: Color for cluster visualization
            south_pole: If True, use South Pole projection
            **kwargs: Additional arguments passed to plot_cluster_on_map

        Raises:
            ValueError: If no clusters found for given variable
        """
        cluster_counts = self.td.get_cluster_counts(var)
        if cluster_counts is None:
            raise ValueError(f"No clusters found for variable {var}")

        # Filter cluster_ids to only include existing clusters
        cluster_ids = [id for id in cluster_ids if id in self.td.get_cluster_ids(var)]

        nrows = int(np.ceil(len(cluster_ids) / ncols))

        fig, axs = self.map(
            nrows,
            ncols,
            projection=projection,
            figsize=(width, nrows * row_height),
        )

        # Plot clusters
        for i, cluster_id in enumerate(cluster_ids):
            ax = axs.flat[i]
            self.cluster_map(
                var, ax=ax, cluster_ids=int(cluster_id), color=color, **kwargs
            )
            ax.set_title(
                f"id {cluster_id} with {cluster_counts[cluster_id]} members",
                fontsize=10,
            )

    def cluster_timeseries(
        self,
        var: str,
        cluster_ids: Union[int, List[int], np.ndarray, range],
        plot_var: Optional[str] = None,
        ax: Optional[Axes] = None,
        color: Optional[str] = None,
        cmap: Union[str, ListedColormap] = "coolwarm",
        alpha: float = 0.1,
        add_legend: bool = True,
        max_trajectories: int = 1_000,
        plot_stats: bool = False,
        plot_stats_legend: bool = True,
        full_timeseries: bool = True,
        cluster_highlight_color: Optional[str] = None,
        cluster_highlight_alpha: float = 0.5,
        cluster_highlight_linewidth: float = 0.5,
        **plot_kwargs: Any,
    ) -> Tuple[Optional[matplotlib.figure.Figure], Axes]:
        """Plot the time series of a cluster.

        Args:
            var: Variable name for which clusters have been computed
            cluster_id: ID of cluster to plot
            ax: Matplotlib axes to plot on. Creates new figure if None
            max_trajectories: Maximum number of trajectories to plot
            **plot_kwargs: Additional arguments passed to plot

        Returns:
            Self for method chaining

        Raises:
            ValueError: If no clusters found for given variable
        """

        # Filter cluster_ids to only include existing clusters
        cluster_ids = [id for id in cluster_ids if id in self.td.get_cluster_ids(var)]

        plot_var = plot_var if plot_var is not None else var

        create_new_ax = ax is None
        fig = None
        if create_new_ax:
            fig, ax = plt.subplots()

        for i, id in enumerate(cluster_ids):
            # Get color
            if color:
                id_color = color
            if not color:
                if len(cluster_ids) == 1:
                    id_color = ToadColors.primary
                else:
                    id_color = get_cmap_seq(stops=len(cluster_ids), cmap=cmap)[i]

            cells = self.td.get_cluster_timeseries(
                plot_var, id, cluster_var=var, keep_full_timeseries=full_timeseries
            )

            if cells is None:
                raise ValueError(f"No timeseries found for cluster {id}")

            # Limit the number of trajectories to plot
            max_trajectories = np.min([max_trajectories, len(cells)])

            # Shuffle the cell to get a random sample
            order = np.arange(len(cells))
            np.random.shuffle(order)
            order = order[:max_trajectories]

            for i, idx in enumerate(order):
                add_label = f"id={id}" if (add_legend and i == 0) else "__nolegend__"
                cells[idx].plot(
                    ax=ax, color=id_color, alpha=alpha, label=add_label, **plot_kwargs
                )

            if plot_stats:
                stats = self.td.cluster_stats(var).time.all_stats(id)
                ax.axvspan(stats["start"], stats["end"], color="#eee", label="Duration")
                ax.axvspan(
                    stats["iqr_90"][0],
                    stats["iqr_90"][1],
                    color="#ddd",
                    label=r"90% IQR" if plot_stats_legend else "__nolegend__",
                )
                ax.axvspan(
                    stats["iqr_50"][0],
                    stats["iqr_50"][1],
                    color="#ccc",
                    label=r"50% IQR" if plot_stats_legend else "__nolegend__",
                )
                ax.axvline(
                    x=stats["membership_peak"],
                    color="red",
                    linestyle="--",
                    label="Peak Time" if plot_stats_legend else "__nolegend__",
                    lw=1,
                )
            if plot_stats_legend or add_legend:
                legend = ax.legend(
                    frameon=False,
                )
                for handle in legend.get_lines():
                    handle.set_alpha(1.0)
                # # Make sure text is visible regardless of color
                # for text in legend.get_texts():
                #     text.set_color("black")

            if cluster_highlight_color:
                cells = self.td.get_cluster_timeseries(
                    var, id, keep_full_timeseries=False
                )
                for ts in cells:
                    ax.plot(
                        ts.time.values,
                        ts.values,
                        color=cluster_highlight_color,
                        alpha=cluster_highlight_alpha,
                        lw=cluster_highlight_linewidth,
                    )

        if len(cluster_ids) == 1:
            if max_trajectories < len(cells):
                ax.set_title(
                    f"Random sample of {max_trajectories} from total {len(cells)} cell for {var} in cluster {cluster_ids[0]}"
                )
            else:
                ax.set_title(
                    f"{len(cells)} timeseries for {var} in cluster {cluster_ids[0]}"
                )

        return fig, ax

    def cluster_aggregate(
        self,
        cluster_var: str,  # Variable used for clustering
        cluster_ids: Union[List[int], np.ndarray, range],
        plot_var: Optional[str] = None,  # Variable to plot (defaults to cluster_var)
        ax: Optional[Axes] = None,
        color: Optional[str] = None,
        cmap: Union[str, ListedColormap] = "coolwarm",
        alpha: float = 0.1,
        plot_stats: bool = True,
        plot_stats_legend: bool = True,
        full_timeseries: bool = True,
        cluster_highlight_color: Optional[str] = None,
        cluster_highlight_alpha: float = 0.5,
        cluster_highlight_linewidth: float = 0.5,
        add_legend: bool = True,
        **plot_kwargs: Any,
    ) -> Tuple[Optional[matplotlib.figure.Figure], Axes]:
        """Plot the time series of a cluster.

        Hatching represents 68% IQR
        Dark color shade represents the duration of the cluster
        Curve represents the mean of the cluster

        Args:
            cluster_var: Variable name for which clusters have been computed
            plot_var: Variable name to plot (if different from cluster_var)
            cluster_ids: List of cluster IDs to plot
            ax: Matplotlib axes to plot on. Creates new figure if None
            **plot_kwargs: Additional arguments passed to plot

        Returns:
            Tuple of (figure, axes)
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # Use cluster_var for clustering but plot_var (or cluster_var if None) for visualization
        plot_var = plot_var if plot_var is not None else cluster_var

        # Filter cluster_ids to only include existing clusters
        found_cluster_ids = self.td.get_cluster_ids(cluster_var)
        cluster_ids = [id for id in cluster_ids if id in found_cluster_ids]

        for i, id in enumerate(cluster_ids):
            # Get color
            if color:
                id_color = color
            if not color:
                if len(cluster_ids) == 1:
                    id_color = ToadColors.primary
                else:
                    id_color = get_cmap_seq(stops=len(cluster_ids), cmap=cmap)[i]
            # Get the timeseries for plot_var using the cluster mask from cluster_var
            mean = self.td.get_cluster_timeseries(
                plot_var, id, aggregation="mean", cluster_var=cluster_var
            )
            perc_start = self.td.get_cluster_timeseries(
                plot_var,
                id,
                aggregation="percentile",
                percentile=0.01,
                cluster_var=cluster_var,
            )
            perc_end = self.td.get_cluster_timeseries(
                plot_var,
                id,
                aggregation="percentile",
                percentile=0.99,
                cluster_var=cluster_var,
            )

            black_or_white = get_high_constrast_text_color(id_color)
            mean.plot(ax=ax, color=black_or_white if plot_stats else id_color, lw=1)
            ax.fill_between(
                self.td.data.time,
                perc_start,
                perc_end,
                color=id_color,
                alpha=0.5 if plot_stats else 1.0,
            )

            if plot_stats:
                # Note: We use cluster_var for stats since they're based on the clustering
                stats = self.td.cluster_stats(cluster_var).time.all_stats(id)

                # Plot duration of cluster
                label = f"id={id}" if len(cluster_ids) == 1 else f"id={id}"
                label = label if add_legend else "__nolegend__"
                ax.fill_between(
                    self.td.data.time,
                    perc_start,
                    perc_end,
                    where=(self.td.data.time >= stats["start"])
                    & (self.td.data.time <= stats["end"]),
                    facecolor=id_color,
                    alpha=1,
                    label=label,
                )

                # Plot 68% IQR of clusterw
                ax.fill_between(
                    self.td.data.time,
                    perc_start,
                    perc_end,
                    where=(self.td.data.time >= stats["iqr_68"][0])
                    & (self.td.data.time <= stats["iqr_68"][1]),
                    hatch="//",
                    facecolor="none",
                    edgecolor=black_or_white,
                    alpha=0.5 if black_or_white == "#ffffff" else 0.25,
                    # label="68% IQR",
                )

                # ax.axvline(
                #     x=stats["membership_peak"],
                #     color=black_or_white,
                #     linestyle="--",
                #     # label="Peak Time",
                #     lw=1,
                # )

                # ax.axvline(
                #     x=stats["steepest_gradient"],
                #     color="red",
                #     linestyle="-.",
                #     label="Steepest Gradient",
                #     lw=1,
                # )

            if add_legend:
                legend = ax.legend(frameon=False)
                # Make all lines in legend black
                for handle in legend.get_lines():
                    handle.set_color("black")
                # for handle in legend.get_patches():
                #     handle.set_edgecolor("black")  # set hatches to visible color

        ax.set_title(f"{plot_var} for clusters from {cluster_var} {cluster_ids}")
        return fig, ax

    def cluster_evolution(
        self,
        cluster_var: str,
        cluster_id: int,
        plot_var: Optional[str] = None,
        ncols: int = 5,
        snapshots: int = 5,
        projection: Optional[str] = None,
    ) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
        """Plot the time series of a cluster on a map.

        Args:
            var: Variable name for which clusters have been computed
            cluster_id: ID of cluster to plot
            ncols: Number of columns in subplot grid
            snapshots: Number of snapshots to plot
        """

        # Use cluster_var for clustering but plot_var (or cluster_var if None) for visualization
        plot_var = plot_var if plot_var is not None else cluster_var

        start, end = (
            self.td.cluster_stats(cluster_var).time.start(cluster_id),
            self.td.cluster_stats(cluster_var).time.end(cluster_id),
        )
        times = np.linspace(start, end, snapshots)
        da = self.td.apply_cluster_mask(cluster_var, plot_var, cluster_id).sel(
            time=times, method="nearest"
        )
        nplots = len(da)
        nrows = int(np.ceil(nplots / ncols))
        fig, axs = self.map(nrows, ncols, projection=projection)

        # hide superfluous axes
        for ax in axs.flat[nplots:]:
            ax.set_visible(False)

        for i in range(nplots):
            da[i].plot(add_colorbar=False, ax=axs.flat[i])
            axs.flat[i].set_title(f"{self.td.time_dim} = {times[i]:.2f}", fontsize=10)
        plt.tight_layout()
        return fig, axs

    def cluster_overview(
        self,
        cluster_var: str,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = None,
        plot_var: Optional[str] = None,
        projection: Optional[str] = None,
        figsize: tuple = (12, 6),
        width_ratios: List[float] = [1, 1],
        height_ratios: Optional[List[float]] = None,
        map_kwargs: dict = {},
        timeseries_kwargs: dict = {},
        timeseries_ylabel: bool = False,
        cmap: str = "tab20",
        wspace: float = 0.1,
        hspace: float = 0.1,
        vertical: bool = False,
        n_timeseries_col: int = 1,
    ) -> Tuple[matplotlib.figure.Figure, dict]:
        """Create a combined plot with one map and multiple time series."""

        if not cluster_ids:
            cluster_ids = self.td.get_cluster_ids(cluster_var)
        elif isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]  # Convert single int to list

        # Filter cluster_ids to only include existing clusters
        found_cluster_ids = self.td.get_cluster_ids(cluster_var)
        cluster_ids = [id for id in cluster_ids if id in found_cluster_ids]

        if len(cluster_ids) == 0:
            raise ValueError("No clusters found for variable", cluster_var)

        # if n_timeseries_col not in [1, 2]:
        #     raise ValueError("n_timeseries_col must be 1 or 2")

        # Calculate layout dimensions
        n_ts = len(cluster_ids)
        n_ts_rows = int(np.ceil(n_ts / n_timeseries_col))

        # Create figure with constrained_layout
        fig = plt.figure(figsize=figsize, constrained_layout=True)

        if vertical:
            # Create main gridspec that spans the whole figure
            main_gs = fig.add_gridspec(
                nrows=2,
                ncols=1,
                height_ratios=height_ratios if height_ratios else [1, 1],
                hspace=hspace,
            )

            # Create map in top gridspec
            _, map_ax = self.map(
                nrows=1, ncols=1, subplot_spec=main_gs[0], projection=projection
            )

            # Create timeseries grid in bottom gridspec
            gs = main_gs[1].subgridspec(
                nrows=n_ts_rows,
                ncols=n_timeseries_col,
                hspace=hspace,
                wspace=wspace if n_timeseries_col > 1 else 0,
            )
        else:
            # Create main gridspec that spans the whole figure
            main_gs = fig.add_gridspec(
                nrows=1,
                ncols=2,
                width_ratios=width_ratios,
            )

            # Create map in left column
            _, map_ax = self.map(
                nrows=1, ncols=1, subplot_spec=main_gs[0, 0], projection=projection
            )

            # Create timeseries grid in right column
            gs = main_gs[0, 1].subgridspec(
                nrows=n_ts_rows,
                ncols=n_timeseries_col,
                hspace=hspace,
                wspace=wspace if n_timeseries_col > 1 else 0,
            )

        # Plot map
        colors = get_cmap_seq(stops=len(cluster_ids), cmap=cmap)
        self.cluster_map(
            cluster_var,
            cluster_ids=cluster_ids,
            color=colors[0] if len(colors) == 1 else colors,
            ax=map_ax,
            **map_kwargs,
        )

        # Create and plot timeseries
        ts_axes = []
        for i in range(n_ts):
            row = i // n_timeseries_col
            col = i % n_timeseries_col
            ax = fig.add_subplot(gs[row, col])
            ts_axes.append(ax)

            # Plot timeseries
            self.cluster_aggregate(
                cluster_var=cluster_var,
                plot_var=plot_var,
                cluster_ids=[cluster_ids[i]],
                color=colors[i],
                ax=ax,
                **timeseries_kwargs,
            )
            ax.axhline(0, ls="--", lw=0.25, color="k")
            ax.set_title("")
            # self._cluster_annotate(
            # ax, 1.0, 1.0, cluster_ids[i], colors[i], relative_coords=True
            # )
            # ax.text(
            #     0.5,
            #     1.0,
            #     f"Duration: {self.td.cluster_stats(cluster_var).time.duration_timesteps(cluster_ids[i])} timesteps, "
            #     f"Footprint: {self.td.cluster_stats(cluster_var).space.footprint_cumulative_area(cluster_ids[i])} cells, "
            #     f"Total members: {self.td.get_cluster_counts(cluster_var)[cluster_ids[i]]}",
            #     ha="center",
            #     va="bottom",
            #     fontsize=8,
            #     transform=ax.transAxes,
            # )

            if not timeseries_ylabel:
                ax.set_ylabel("")

            # Handle axis cleanup
            if (vertical and row < n_ts_rows - 1) or (not vertical and i < n_ts - 1):
                ax.set_xlabel("")
                self._remove_spines(ax, ["right", "top", "bottom"])
            else:
                self._remove_spines(ax, ["right", "top"])

            if (
                i < n_ts - 1
                and (vertical and row < n_ts_rows - 1)
                or (not vertical and i < n_ts - 1)
            ):
                self._remove_ticks(ax, keep_y=True)

        # Hide any empty subplots
        for i in range(n_ts, n_ts_rows * n_timeseries_col):
            row = i // n_timeseries_col
            col = i % n_timeseries_col
            ax = fig.add_subplot(gs[row, col])
            ax.set_visible(False)
        return fig, {"map": map_ax, "timeseries": ts_axes}

    def shifts_distribution(self, figsize=(15, 10)):
        """Plot histograms showing the distribution of shifts for each shift variable."""
        fig, axs = plt.subplots(nrows=self.td.shift_vars.size, figsize=figsize)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        if len(axs) > 1:
            self._remove_ticks(axs[:-1])
            self._remove_ticks(axs[-1], keep_x=True)
            self._remove_spines(axs[:-1])
            self._remove_spines(axs[-1], spines=["left", "right", "top"])
        for i in range(self.td.shift_vars.size):
            axs[i].hist(
                self.td.get_shifts(self.td.shift_vars[i]).values.flatten(),
                range=(-1, 1),
                bins=20,
                density=True,
            )
            axs[i].set_ylabel(
                self.td.shift_vars[i], rotation=0, ha="right", va="center"
            )
        return fig, axs


# end of TOADPlotter


def get_max_index(pos, n_rows=None):
    """Helper function to get the maximum index from a position spec."""
    if isinstance(pos, slice):
        if pos.stop is not None:
            return pos.stop
        return (n_rows - 1) if n_rows is not None else 0
    return pos


def get_high_constrast_text_color(color: Union[tuple, str]) -> str:
    try:
        brightness = (
            sum(
                to_rgb(color)[i] * factor
                for i, factor in enumerate([0.299, 0.587, 0.114])
            )
            * 255
        )
        return "#ffffff" if brightness < 128 else "#000000"
    except ValueError:
        print(f"Error converting {color} to RGB")
        return "#000000"


def get_cmap_seq(start=0, end=-1, stops=10, cmap="coolwarm", reverse=False):
    cmap = plt.get_cmap(cmap)
    end = (
        end if end != -1 else cmap.N
    )  # Use cmap.N to get the number of colors in the colormap
    cycle_index = np.linspace(start, end - 1, stops, dtype=int)
    colors = cmap(cycle_index)  # Generate colors using the indices
    if reverse:
        colors = colors[::-1]
    colors = [to_hex(color) for color in colors]
    return colors
