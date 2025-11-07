from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple, Union, overload

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap, to_hex, to_rgb, to_rgba
from matplotlib.patches import Rectangle

from toad.utils import _attrs, detect_latlon_names, is_regular_grid

_projection_map = {
    "plate_carree": ccrs.PlateCarree(),
    "north_pole": ccrs.NorthPolarStereo(),
    "north_polar_stereo": ccrs.NorthPolarStereo(),
    "south_pole": ccrs.SouthPolarStereo(),
    "south_polar_stereo": ccrs.SouthPolarStereo(),
    "global": ccrs.Robinson(),
    "robinson": ccrs.Robinson(),
    "mollweide": ccrs.Mollweide(),
}

default_cmap = "tab20b"


@dataclass
class PlotConfig:
    """Configuration for map plotting parameters.

    This dataclass contains all the configuration options for creating maps
    with TOADPlotter, including coastline, grid, and projection settings.
    """

    resolution: str = "110m"
    coastline_linewidth: float = 0.5
    border_linewidth: float = 0.25
    grid_labels: bool = False
    grid_lines: bool = True
    grid_style: str = "--"
    grid_width: float = 0.5
    grid_color: str = "gray"
    grid_alpha: float = 0.5
    borders: bool = True
    projection: str = "plate_carree"
    map_frame: bool = True
    continent_shading: bool = False
    continent_shading_color: str = "lightgray"
    ocean_shading: bool = False
    ocean_shading_color: str = "lightgray"


@dataclass
class ToadColors:
    green = "#6F9F50"
    green_light = "#BCCDB3"
    green_dark = "#43712C"
    yellow = "#F1E0B0"
    # primary = green_dark
    primary = "#000000"
    secondary = green_light
    tertiary = yellow


class TOADPlotter:
    """Plotting utilities for TOAD objects.

    The TOADPlotter class provides methods for creating publication-ready visualizations
    of TOAD data, including maps, timeseries, and statistical plots.

    Args:
        td: TOAD object containing the data to plot
        config: Optional PlotConfig object with plotting preferences. If None, uses defaults.
    """

    def __init__(self, td, config: Optional[PlotConfig] = None):
        from toad import TOAD

        self.td: TOAD = td
        self.default_config = config if config is not None else PlotConfig()

    def __call__(self, config=None, **kwargs):
        """Return a new TOADPlotter with updated configuration."""
        config = config if config else self.default_config
        config.__dict__.update(kwargs)
        return TOADPlotter(self.td, config=config)

    # Overloads are used for type hinting
    @overload
    def map(
        self,
        nrows: Literal[1] = 1,
        ncols: Literal[1] = 1,
        projection: Optional[str | ccrs.Projection] = None,
        config: Optional[PlotConfig] = None,
        figsize: Optional[Tuple[float, float]] = None,
        height_ratios: Optional[List[float]] = None,
    ) -> Tuple[matplotlib.figure.Figure, Axes]: ...

    @overload
    def map(
        self,
        nrows: int,
        ncols: int = 1,
        projection: Optional[str | ccrs.Projection] = None,
        config: Optional[PlotConfig] = None,
        figsize: Optional[Tuple[float, float]] = None,
        height_ratios: Optional[List[float]] = None,
    ) -> Tuple[matplotlib.figure.Figure, np.ndarray]: ...

    def map(
        self,
        nrows: int = 1,
        ncols: int = 1,
        projection: Optional[str | ccrs.Projection] = None,
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

        if isinstance(projection, str) and projection not in _projection_map:
            raise ValueError(
                f"Invalid projection '{projection}'. Please choose between {list(_projection_map.keys())}"
            )
        elif isinstance(projection, ccrs.Projection):
            projection = projection
        else:
            projection = _projection_map[projection]

        if subplot_spec is not None:
            # Create map in existing figure using subplot_spec
            fig = plt.gcf()
            ax = fig.add_subplot(subplot_spec, projection=projection)
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
                subplot_kw={"projection": projection},
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
        projection: str | ccrs.Projection,
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
            projection=projection,
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
        return fig, axs

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
        # Add continent shading
        if config.continent_shading:
            # TODO p2: continent needs same resolution as coastlines
            ax.add_feature(
                cfeature.LAND,
                facecolor=config.continent_shading_color,
                alpha=0.5,
            )

        if config.ocean_shading:
            ax.add_feature(
                cfeature.OCEAN,
                facecolor=config.ocean_shading_color,
                alpha=0.5,
            )

        ax.coastlines(
            resolution=config.resolution, linewidth=config.coastline_linewidth
        )

        if config.borders:
            ax.add_feature(
                cfeature.BORDERS, linestyle="-", linewidth=config.border_linewidth
            )

        if config.grid_lines:
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
        transform: Optional[ccrs.Projection] = None,
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
            fontsize=4 + 4 * scale,
            transform=transform,
        )
        t.set_bbox(
            dict(
                facecolor=acol,
                alpha=1,
                edgecolor=black_or_white,
                boxstyle="round,pad=0.2,rounding_size=0.2",  # adjust rounding_size to control corner radius
            )
        )

    def cluster_map(
        self,
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = range(10),
        projection: Optional[str | ccrs.Projection] = None,
        ax: Optional[Axes] = None,
        color: Optional[Union[str, Tuple, List[Union[str, Tuple]]]] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        add_contour: bool = True,
        only_contour: bool = False,
        add_labels: bool = True,
        remaining_clusters_cmap: Optional[Union[str, Colormap]] = "jet",
        remaining_clusters_legend_pos: Optional[Tuple[float, float]] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[matplotlib.figure.Figure], Axes]:
        """Plot one or multiple clusters on a map.

        Args:
            var: Base variable name (e.g. 'temperature', will look for
                        'temperature_cluster') or custom cluster variable name. If None, TOAD will attempt to infer which variable to use.
                A ValueError is raised if the variable cannot be uniquely determined.
            cluster_ids: Single cluster ID or list of cluster IDs to plot.
                         Defaults to first 10 clusters (except -1) if None.
            projection: Projection to use for the map. Uses default if None. Can be a string or a cartopy projection object.
            ax: Matplotlib axes to plot on. Creates new figure if None.
            color: Color for cluster visualization. Can be:
                - A single color (str, hex, RGB tuple) to use for all clusters.
                - A list of colors to use for each cluster. Overrides cmap.
            cmap: Colormap for multiple clusters. Used only if color is None.
            add_contour: If True, add contour lines around clusters.
            only_contour: If True, only plot contour lines (no fill).
            add_labels: If True, add cluster ID labels using a geometrically
                        central point (`central_point_for_labeling`).
            remaining_clusters_cmap: Colormap for remaining clusters. Can be:
                - A string (e.g., "jet", "viridis") to use a built-in colormap.
                - A matplotlib colormap object.
            remaining_clusters_legend_pos: Tuple of (x, y) position of the remaining clusters legend. If None, the legend is placed automatically.
            **kwargs: Additional arguments passed to xarray.plot methods
                      (e.g., `plot`, `plot.contour`).

        Returns:
            Tuple of (figure, axes). Figure is None if ax was provided.

        Raises:
            ValueError: If no clusters found for given variable.
            TypeError: If `cluster_ids` is not an int, list, ndarray, range, or None,
                       or if `cmap` is not a string or ListedColormap.
        """

        var = self.td._get_base_var_if_none(var)
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
                cmap_colors: list = cmap.colors  # type: ignore
                # Repeat colors if needed
                if len(cmap_colors) < len(cluster_ids):
                    cmap_colors = cmap_colors * (
                        len(cluster_ids) // len(cmap_colors) + 1
                    )
                color_list = cmap_colors[: len(cluster_ids)]

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

            # prepare common plot parameters
            plot_params = {
                "ax": ax,
                "cmap": cluster_cmap,
                "add_colorbar": False,
                "alpha": 0.75,
                "transform": None,
                **kwargs,
            }

            lat_name, lon_name = detect_latlon_names(self.td.data)
            has_latlon = lat_name is not None and lon_name is not None

            # plot on lat/lon coordinates if available
            if has_latlon:
                plot_params["x"] = lon_name
                plot_params["y"] = lat_name
                plot_params["transform"] = ccrs.PlateCarree()

            if not only_contour:
                # Don't plot values outside mask: FALSE -> np.nan
                mask.where(mask, np.nan).plot(
                    **plot_params,
                )

            # contour plots don't work for irregular grids
            if (only_contour or add_contour) and is_regular_grid(self.td.data):
                if add_contour:
                    # Make contour color darker
                    contour_color = cluster_cmap.colors[0]  # type: ignore
                    color_rgba = to_rgba(contour_color)  # type: ignore
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
                # Uses the point furthest from the cluster edge for robust labeling
                y, x = self.td.cluster_stats(var).space.central_point_for_labeling(id)
                if np.isnan(x) or np.isnan(y):
                    # Get median coordinates as fallback
                    y, x = self.td.cluster_stats(var).space.footprint_median(id)

                if not (np.isnan(x) or np.isnan(y)):
                    self._cluster_annotate(
                        ax,
                        x,
                        y,
                        id,
                        cluster_cmap.colors[0],
                        transform=plot_params["transform"],
                    )  # type: ignore
                else:
                    print(
                        f"Warning: Could not find valid label position for cluster {id}"
                    )

            if single_plot:
                ax.set_title(f"{var}_cluster {id}")

        # Plot remaining clusters
        if remaining_clusters_cmap:
            remaining_cluster_ids = [  # get unplotted clusters ids (except -1)
                int(id) for id in all_cluster_ids if id not in cluster_ids and id != -1
            ]
            if len(remaining_cluster_ids) > 0:
                mask = self.td.get_cluster_mask(var, remaining_cluster_ids)
                cl = self.td.get_clusters(var).where(mask)

                plot_params["cmap"] = remaining_clusters_cmap
                cl.max(dim=self.td.time_dim).plot(
                    **plot_params,
                )  # type: ignore

                # Pass the colormap to the legend function
                self._add_gradient_legend(
                    ax,
                    remaining_cluster_ids[0],
                    remaining_cluster_ids[-1],
                    var=var,
                    legend_pos=remaining_clusters_legend_pos,
                    cmap=plt.get_cmap(remaining_clusters_cmap)
                    if isinstance(remaining_clusters_cmap, str)
                    else remaining_clusters_cmap,
                )
        return fig, ax

    def cluster_maps(
        self,
        var: str | None = None,
        cluster_ids: Union[List[int], np.ndarray, range] = range(5),
        ncols: int = 5,
        color: Optional[str] = None,
        projection: Optional[str | ccrs.Projection] = None,
        width: float = 12,
        row_height: float = 2.5,
        **kwargs: Any,
    ) -> None:
        """Plot individual clusters on separate maps in a grid layout.

        Args:
            var: Variable name for which clusters have been computed. If None, TOAD will attempt to infer which variable to use.
                A ValueError is raised if the variable cannot be uniquely determined.
            cluster_ids: List, range, or array of cluster IDs to plot.
            ncols: Number of columns in the subplot grid.
            color: Single color to use for all cluster visualizations. Passed to `cluster_map`.
            projection: Map projection to use for each subplot. Uses default if None. Can be a string or a cartopy projection object.
            width: Total width of the figure in inches.
            row_height: Height of each row in the subplot grid in inches.
            **kwargs: Additional arguments passed down to `self.cluster_map` for each plot.

        Returns:
            None: This function creates a plot but does not return any values.

        Raises:
            ValueError: If no clusters found for the given variable `var`.
        """
        var = self.td._get_base_var_if_none(var)
        cluster_counts = self.td.get_cluster_counts(var)
        if cluster_counts is None:
            raise ValueError(f"No clusters found for variable {var}")

        # Filter cluster_ids to only include existing clusters
        cluster_ids = self.filter_by_existing_clusters(cluster_ids, var)

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
                var,
                ax=ax,
                cluster_ids=int(cluster_id),
                color=color,
                remaining_clusters_cmap=None,
                **kwargs,
            )
            ax.set_title(
                f"id {cluster_id} with {cluster_counts[cluster_id]} members",
                fontsize=10,
            )

    def cluster_timeseries(
        self,
        var: str | None = None,
        cluster_ids: Union[int, List[int], np.ndarray, range] = range(5),
        plot_var: Optional[str] = None,
        ax: Optional[Axes] = None,
        color: Optional[str] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        alpha: float = 0.1,
        normalize: Optional[Literal["first", "max", "last"]] = None,
        add_legend: bool = True,
        max_trajectories: int = 1_000,
        plot_stats: bool = False,
        full_timeseries: bool = True,
        cluster_highlight_color: Optional[str] = None,
        cluster_highlight_alpha: float = 0.5,
        cluster_highlight_linewidth: float = 0.5,
        **plot_kwargs: Any,
    ) -> Tuple[Optional[matplotlib.figure.Figure], Axes]:
        """Plot the time series of one or multiple clusters.

        Args:
            var: Variable name for which clusters have been computed. If None, TOAD will attempt to infer which variable to use.
                A ValueError is raised if the variable cannot be uniquely determined.
            cluster_ids: ID or list of IDs of clusters to plot.
            plot_var: Variable name to plot (if different from var). Defaults to var.
            ax: Matplotlib axes to plot on. Creates new figure if None.
            color: Single color to use for all plotted clusters. Overrides cmap.
            cmap: Colormap to use if plotting multiple clusters and color is None.
            alpha: Alpha transparency for individual time series lines.
            normalize: Method to normalize timeseries ('first', 'max', 'last'). Defaults to None.
            add_legend: If True, add a legend indicating cluster IDs.
            max_trajectories: Maximum number of individual trajectories to plot per cluster.
            plot_stats: If True, add vertical spans indicating cluster duration and IQR.
            full_timeseries: If True, plot the full timeseries for each cell. If False,
                only plot the segment belonging to the cluster.
            cluster_highlight_color: Color to highlight the actual cluster segment
                when full_timeseries is True.
            cluster_highlight_alpha: Alpha for the cluster highlight segment.
            cluster_highlight_linewidth: Line width for the cluster highlight segment.
            **plot_kwargs: Additional arguments passed to xarray.plot for each trajectory.

        Returns:
            Tuple of (figure, axes). Figure is None if ax was provided.

        Raises:
            ValueError: If no timeseries found for a given cluster ID.
        """

        # Filter cluster_ids to only include existing clusters
        var = self.td._get_base_var_if_none(var)
        cluster_ids = self.filter_by_existing_clusters(cluster_ids, var)

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
                plot_var,
                id,
                cluster_var=var,
                keep_full_timeseries=full_timeseries,
                normalize=normalize,
            )

            if cells is None:
                raise ValueError(f"No timeseries found for cluster {id}")

            # Limit the number of trajectories to plot
            max_trajectories = np.min([max_trajectories, len(cells)])

            # Shuffle the cell to get a random sample
            order = np.arange(len(cells))
            np.random.shuffle(order)
            order = order[:max_trajectories]

            for plot_idx, cell_idx in enumerate(order):
                add_label = (
                    f"id={id}" if (add_legend and plot_idx == 0) else "__nolegend__"
                )
                cells[cell_idx].plot(
                    ax=ax, color=id_color, alpha=alpha, label=add_label, **plot_kwargs
                )

            if plot_stats:
                start = self.td.cluster_stats(var).time.start(id)
                end = self.td.cluster_stats(var).time.end(id)
                iqr_68 = self.td.cluster_stats(var).time.iqr_68(id)
                ax.axvspan(start, end, color="#eee", label="Duration")
                ax.axvspan(
                    iqr_68[0],
                    iqr_68[1],
                    color="#ccc",
                    label=r"68% IQR",
                )
            if add_legend:
                legend = ax.legend(
                    frameon=False,
                )
                for handle in legend.get_lines():
                    handle.set_alpha(1.0)

            if cluster_highlight_color:
                cells = self.td.get_cluster_timeseries(
                    var, id, keep_full_timeseries=False, normalize=normalize
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
        cluster_var: str | None = None,
        cluster_ids: Union[List[int], np.ndarray, range] = range(5),
        plot_var: Optional[str] = None,
        ax: Optional[Axes] = None,
        color: Optional[str] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        median_linewidth: float = 3,
        mean_linewidth: float = 3,
        normalize: Optional[Literal["first", "max", "last"]] = None,
        add_legend: bool = True,
        plot_cluster_range: bool = True,
        plot_cluster_68iqr: bool = True,
        plot_cluster_95iqr: bool = False,
        plot_cluster_median: bool = True,
        plot_cluster_mean: bool = False,
        plot_cluster_iqr: Optional[tuple[float, float]] = None,
        alpha: float = 0.4,
        plot_shift_range: bool = True,
        plot_largest_gradient: bool = True,
    ) -> tuple[Optional[matplotlib.figure.Figure], Axes]:
        """Plot aggregated time series statistics for one or multiple clusters.

        TODO p2: make this function faster!!
        TODO p2: merge this function with cluster_timeseries()

        Plots median and/or mean lines along with shaded interquartile ranges (default: full range and 68% IQR).
        The shift indicator shows the temporal extent of each cluster by plotting horizontal lines at different shades:
        - The light shaded line spans the full duration of the cluster (from first to last occurrence)
        - The darker shaded line shows the 68% interquartile range (IQR) duration, which represents the core period when the cluster is most active

        Args:
            cluster_var: Variable name for which clusters have been computed. If None, TOAD will attempt to infer which variable to use.
                A ValueError is raised if the variable cannot be uniquely determined.
            cluster_ids: List of cluster IDs to plot.
            plot_var: Variable name to plot (if different from cluster_var). Defaults to cluster_var.
            ax: Matplotlib axes to plot on. Creates new figure if None.
            color: Single color to use for all plotted clusters. Overrides cmap.
            cmap: Colormap to use if plotting multiple clusters and color is None.
            median_linewidth: Linewidth for the median curve.
            mean_linewidth: Linewidth for the mean curve.
            normalize: Method to normalize timeseries ('first', 'max', 'last'). Defaults to None.
            add_legend: If True, add a legend indicating cluster IDs.
            plot_cluster_range: If True, plot the full range (min to max) as a shaded area.
            plot_cluster_68iqr: If True, plot the 68% IQR (16th to 84th percentile) as a shaded area.
            plot_cluster_95iqr: If True, plot the 95% IQR (2.5th to 97.5th percentile) as a shaded area.
            plot_cluster_median: If True, plot the median timeseries curve.
            plot_cluster_mean: If True, plot the mean timeseries curve.
            plot_cluster_iqr: Tuple of (start_percentile, end_percentile) for a custom IQR shaded area.
            alpha: Alpha transparency for the shaded IQR areas.
            plot_shift_range: If True, adds shaded regions that indicate the cluster's temporal extent (start/end),
                and draws a vertical line marking the point of steepest change within the cluster (largest gradient of
                the cluster median timeseries)

        Returns:
            Tuple of (figure, axes). Figure is None if ax was provided.
        """
        fig = None
        if ax is None:
            fig, ax = plt.subplots()

        # Use cluster_var for clustering but plot_var (or cluster_var if None) for visualization
        cluster_var = self.td._get_base_var_if_none(cluster_var)
        plot_var = plot_var if plot_var is not None else cluster_var

        # Filter cluster_ids to only include existing clusters
        cluster_ids = self.filter_by_existing_clusters(cluster_ids, cluster_var)

        for i, id in enumerate(cluster_ids):
            if color:
                id_color = color
            if not color:
                if len(cluster_ids) == 1:
                    id_color = ToadColors.primary
                else:
                    id_color = get_cmap_seq(stops=len(cluster_ids), cmap=cmap)[i]

            def plot_iqr(percentile_start, percentile_end):
                # Use original time values for plotting if available, otherwise use numeric values

                ax.fill_between(
                    self.td.data[self.td.time_dim].values,
                    self.td.get_cluster_timeseries(
                        plot_var,
                        id,
                        aggregation="percentile",
                        percentile=percentile_start,
                        cluster_var=cluster_var,
                        normalize=normalize,
                    ),
                    self.td.get_cluster_timeseries(
                        plot_var,
                        id,
                        aggregation="percentile",
                        percentile=percentile_end,
                        cluster_var=cluster_var,
                        normalize=normalize,
                    ),
                    color=id_color,
                    alpha=alpha,
                )

            if plot_cluster_range:
                plot_iqr(0.00001, 0.999999)

            if plot_cluster_68iqr:
                plot_iqr(0.16, 0.84)

            if plot_cluster_95iqr:
                plot_iqr(0.025, 0.975)

            if plot_cluster_iqr:
                plot_iqr(plot_cluster_iqr[0], plot_cluster_iqr[1])

            if plot_cluster_mean:
                self.td.get_cluster_timeseries(
                    plot_var,
                    id,
                    aggregation="mean",
                    cluster_var=cluster_var,
                    normalize=normalize,
                ).plot(
                    ax=ax,
                    color=id_color,
                    lw=mean_linewidth,
                    label=f"id={id}",
                )

            if plot_cluster_median:
                self.td.get_cluster_timeseries(
                    plot_var,
                    id,
                    aggregation="median",
                    cluster_var=cluster_var,
                    normalize=normalize,
                ).plot(ax=ax, color=id_color, lw=median_linewidth, label=f"id={id}")

            if plot_shift_range:
                start = self.td.cluster_stats(cluster_var).time.start(id)
                end = self.td.cluster_stats(cluster_var).time.end(id)
                ax.axvspan(start, end, color=id_color, alpha=0.25, zorder=-100)

            if plot_largest_gradient:
                largest_gradient = self.td.cluster_stats(
                    cluster_var
                ).time.steepest_gradient(id)
                ax.axvline(
                    largest_gradient, ls="--", color="k", lw=1.0, zorder=100, alpha=0.25
                )

            # TODO p2: something like the max(median(_dts))
            # if plot_detection_signal_peak:
            #     self.td.shift_vars_for_var(cluster_var)

            if add_legend:
                ax.legend(frameon=False)

        ax.set_title(f"{plot_var} for clusters from {cluster_var} {cluster_ids}")
        return fig, ax

    def cluster_cummulative(
        self,
        cluster_var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = None,
        plot_var: Optional[str] = None,
        ax: Optional[Axes] = None,
        color: Optional[str] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        figsize: Optional[Tuple[float, float]] = None,
        remaining_clusters_color: Optional[str] = "gray",
    ) -> Tuple[Optional[matplotlib.figure.Figure], Axes]:
        """Plot the cumulative sum of the timeseries for one or multiple clusters.

        When specific cluster_ids are provided, remaining clusters will be grouped together
        and shown as a single layer at the bottom of the plot.
        """

        # Use cluster_var for clustering but plot_var (or cluster_var if None) for visualization
        cluster_var = self.td._get_base_var_if_none(cluster_var)
        plot_var = plot_var if plot_var is not None else cluster_var

        # Get all valid cluster IDs (excluding -1)
        # If cluster_ids specified, separate into selected and remaining clusters
        if cluster_ids is not None:
            if isinstance(cluster_ids, int):
                cluster_ids = [cluster_ids]
            selected_cluster_ids = self.filter_by_existing_clusters(
                cluster_ids, cluster_var
            )
            remaining_cluster_ids = [
                cid
                for cid in self.td.get_cluster_ids(cluster_var)
                if cid not in selected_cluster_ids
            ]
        else:
            selected_cluster_ids = self.td.get_cluster_ids(cluster_var)
            remaining_cluster_ids = []

        # Get timeseries for selected clusters
        series_list = []
        for cid in selected_cluster_ids:
            series = self.td.get_cluster_timeseries(
                plot_var, cluster_var=cluster_var, cluster_id=cid, aggregation="sum"
            )
            series_list.append(series)

        # First create colors using original order (to match cluster_map)
        if color:
            colors = [color] * len(selected_cluster_ids)
        else:
            if isinstance(cmap, str):
                base_cmap = plt.get_cmap(cmap)
                colors = [
                    base_cmap(i) for i in np.linspace(0, 1, len(selected_cluster_ids))
                ]

        # Create a mapping of cluster IDs to their colors
        color_map = dict(zip(selected_cluster_ids, colors))

        # Sort clusters by shift time (Early shifts first should be on top)
        shift_times = [
            self.td.cluster_stats(cluster_var).time.start(cid)
            for cid in selected_cluster_ids
        ]
        sorted_indices = np.argsort(shift_times)[
            ::-1
        ]  # Add [::-1] to reverse the order
        series_list = [series_list[i] for i in sorted_indices]
        selected_cluster_ids = [selected_cluster_ids[i] for i in sorted_indices]

        # Reorder colors to match the sorted clusters while maintaining original color assignments
        colors = [color_map[cid] for cid in selected_cluster_ids]

        # If there are remaining clusters, add their combined timeseries at the bottom
        if remaining_cluster_ids:
            remaining_series = None
            for cid in remaining_cluster_ids:
                series = self.td.get_cluster_timeseries(
                    plot_var, cluster_var=cluster_var, cluster_id=cid, aggregation="sum"
                )
                if remaining_series is None:
                    remaining_series = series
                else:
                    remaining_series += series

            if remaining_series is not None and remaining_clusters_color is not None:
                # Get the highest cluster ID from the individually plotted clusters
                max_plotted_id = max(selected_cluster_ids)

                # Append to the end instead of inserting at the beginning
                series_list.append(remaining_series)  # Add to the end of the stack
                colors.append(remaining_clusters_color)  # Add color to the end
                selected_cluster_ids.append(
                    f">{max_plotted_id}"
                )  # Add label to the end

        # Create figure and axes if not provided
        fig = None
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        # Stack the areas for clusters
        ax.stackplot(
            series_list[0][self.td.time_dim].values,
            [s.values for s in series_list],
            labels=[f"Cluster {cid}" for cid in selected_cluster_ids],
            colors=colors,
        )

        # Add unclustered cells mirrored below y=0
        unclustered = self.td.get_cluster_timeseries(
            plot_var, cluster_var=cluster_var, cluster_id=-1, aggregation="sum"
        )

        ax.fill_between(
            unclustered[self.td.time_dim].values,
            0,
            -unclustered.values,
            color="lightgray",
            label="Unclustered",
            alpha=0.7,
        )

        # Add horizontal line at y=0
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        if len(selected_cluster_ids) < 25:  # don't show legend if too many clusters
            ax.legend(loc="upper right", ncols=2)

        # Set labels from xarray metadata
        time_var = series_list[0][self.td.time_dim]
        x_label = (
            time_var.attrs.get("long_name", None)
            or time_var.attrs.get("standard_name", None)
            or time_var.name
        )
        x_units = time_var.attrs.get("units", "")
        if x_units:
            x_label = f"{x_label} ({x_units})"

        data_var = self.td.data[plot_var]
        y_label = (
            data_var.attrs.get("long_name", None)
            or data_var.attrs.get("standard_name", None)
            or data_var.name
        )
        y_units = data_var.attrs.get("units", "")
        if y_units:
            y_label = f"Cummulative {y_label} ({y_units})"

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        plt.tight_layout()

        return fig, ax

    def cluster_evolution(
        self,
        cluster_var: str | None = None,
        cluster_id: int = 0,
        plot_var: str | None = None,
        ncols: int = 5,
        snapshots: int = 5,
        projection: str | ccrs.Projection | None = None,
    ) -> Tuple[matplotlib.figure.Figure, np.ndarray]:
        """Plot spatial snapshots of a cluster's evolution over time.

        Takes multiple snapshots of the variable `plot_var` (masked by the
        cluster defined in `cluster_var`) at different time steps
        within the cluster's duration and plots them on separate maps.

        Args:
            cluster_var: Variable name used for clustering. If None, TOAD will attempt to infer which variable to use.
                A ValueError is raised if the variable cannot be uniquely determined.
            cluster_id: ID of the specific cluster to visualize.
            plot_var: Variable name whose data to plot within the cluster mask.
                      Defaults to `cluster_var` if None.
            ncols: Number of columns in the subplot grid for the snapshots.
            snapshots: Number of time snapshots to plot across the cluster's duration.
            projection: Map projection to use for the snapshot maps. Uses default if None. Can be a string or a cartopy projection object.

        Returns:
            Tuple[matplotlib.figure.Figure, np.ndarray]: The figure and the array of axes
            containing the snapshot plots.
        """

        cluster_var = self.td._get_base_var_if_none(cluster_var)

        # Use cluster_var for clustering but plot_var (or cluster_var if None) for visualization
        plot_var = plot_var if plot_var is not None else cluster_var

        start, end = (
            self.td.cluster_stats(cluster_var).time.start(cluster_id),
            self.td.cluster_stats(cluster_var).time.end(cluster_id),
        )
        times = np.linspace(start, end, snapshots)
        da = self.td.apply_cluster_mask(cluster_var, plot_var, cluster_id).sel(
            **{self.td.time_dim: times}, method="nearest"
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
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = range(5),
        map_var: Optional[str] = None,
        timeseries_var: Optional[str] = None,
        plot_shifts: bool = False,
        projection: Optional[str | ccrs.Projection] = None,
        figsize: tuple = (12, 6),
        width_ratios: List[float] = [1, 1],
        height_ratios: Optional[List[float]] = None,
        map_kwargs: dict = {},
        timeseries_kwargs: dict = {},
        timeseries_ylabel: bool = False,
        cmap: str = default_cmap,
        wspace: float = 0.1,
        hspace: float = 0.1,
        vertical: bool = False,
        n_timeseries_col: int = 1,
        plot_all_clusters_on_map: bool = True,
    ) -> Tuple[matplotlib.figure.Figure, dict]:
        """Create a combined plot with a cluster map and aggregated time series.

        Combination of a `cluster_map` and `cluster_aggregate`.

        Args:
            var: Variable name used for clustering. If None, TOAD will attempt to infer which variable to use.
                A ValueError is raised if the variable cannot be uniquely determined.
            cluster_ids: ID or list of IDs of clusters to plot. Defaults to all
                         clusters found for `var`.
            map_var: Variable name whose data to plot in the map.
                     Defaults to `var` if None.
            timeseries_var: Variable name whose data to plot in the timeseries.
                            Defaults to base variable of the cluster variable.
            projection: Map projection for the cluster map. Uses default if None. Can be a string or a cartopy projection object.
            figsize: Overall figure size (width, height) in inches.
            width_ratios: List of relative widths for map vs. timeseries section
                          (used in horizontal layout).
            height_ratios: List of relative heights for map vs. timeseries section
                           (used in vertical layout).
            map_kwargs: Dictionary of keyword arguments passed to `cluster_map`.
            timeseries_kwargs: Dictionary of keyword arguments passed to
                               `cluster_aggregate` for each timeseries plot.
            timeseries_ylabel: If True, show y-axis label on the timeseries plots.
            cmap: Colormap used to color clusters consistently across map and
                  timeseries plots.
            wspace: Width space between timeseries subplots (if n_timeseries_col > 1).
            hspace: Height space between map/timeseries (vertical) or timeseries rows.
            vertical: If True, arrange map above timeseries plots. Otherwise, map
                      is placed to the left.
            n_timeseries_col: Number of columns for the timeseries subplot grid.

        Returns:
            Tuple containing:
                - fig: The matplotlib Figure object.
                - axes_dict: A dictionary containing the map axes and a list of
                  timeseries axes, e.g., {'map': map_ax, 'timeseries': [ts_ax1, ts_ax2,...]}.
        """

        var = self.td._get_base_var_if_none(var)

        if not cluster_ids:
            cluster_ids = self.td.get_cluster_ids(var)
        elif isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]  # Convert single int to list

        # Filter cluster_ids to only include existing clusters
        cluster_ids = self.filter_by_existing_clusters(cluster_ids, var)

        if map_var is None:
            map_var = var

        if len(cluster_ids) == 0:
            raise ValueError("No clusters found for variable", var)

        # Get base variable from clusters attrs
        if timeseries_var is None:
            timeseries_var = self.td.get_clusters(var).attrs[_attrs.BASE_VARIABLE]
        if plot_shifts:
            timeseries_var = self.td.get_clusters(var).attrs[_attrs.SHIFTS_VARIABLE]

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

        # Don't plot remaining clusters on map if not requested
        if not plot_all_clusters_on_map:
            map_kwargs["remaining_clusters_cmap"] = None

        # Plot map
        colors = get_cmap_seq(stops=len(cluster_ids), cmap=cmap)
        self.cluster_map(
            map_var,
            cluster_ids=cluster_ids,
            color=colors[0] if len(colors) == 1 else colors,
            ax=map_ax,
            **map_kwargs,
        )

        # Create and plot timeseries
        ts_axes = []
        y_label = ""
        for i in range(n_ts):
            row = i // n_timeseries_col
            col = i % n_timeseries_col
            ax = fig.add_subplot(gs[row, col])
            ts_axes.append(ax)

            # Plot timeseries
            self.cluster_aggregate(
                cluster_var=var,
                plot_var=timeseries_var,
                cluster_ids=[cluster_ids[i]],
                color=colors[i],
                ax=ax,
                **timeseries_kwargs,
            )
            ax.axhline(0, ls="--", lw=0.25, color="k")
            ax.set_title("")

            if not timeseries_ylabel:
                y_label = ax.get_ylabel()
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

        # set title of time series axes
        ts_axes[0].set_title(
            f"{len(cluster_ids)} {'largest ' if len(cluster_ids) < len(self.td.get_cluster_ids(var)) else ''}"
            + f"clusters{' in ' + y_label if y_label != '' else ''}"
        )

        return fig, {"map": map_ax, "timeseries": ts_axes}

    def shifts_distribution(
        self, figsize: Optional[tuple] = None, yscale: str = "log", bins=20
    ):
        """Plot histograms showing the distribution of shifts for each shift variable."""

        if figsize is None:
            figsize = (15, 2 * len(self.td.shift_vars))

        fig, axs = plt.subplots(nrows=len(self.td.shift_vars), figsize=figsize)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        if len(axs) > 1:
            self._remove_ticks(axs[:-1], keep_y=True)
            self._remove_spines(axs[:-1], spines=["right", "top"])

        self._remove_spines(axs[-1], spines=["right", "top"])

        for i in range(len(self.td.shift_vars)):
            axs[i].hist(
                self.td.get_shifts(self.td.shift_vars[i]).values.flatten(),
                range=(-1, 1),
                bins=bins,
            )
            axs[i].set_ylabel(
                f"#{self.td.shift_vars[i]}", rotation=0, ha="right", va="center"
            )
            axs[i].set_yscale(yscale)
        return fig, axs

    def filter_by_existing_clusters(
        self, cluster_ids: Union[int, List[int], np.ndarray, range], var: str
    ) -> List[int]:
        """Filter cluster_ids to only include existing clusters."""

        if isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]

        return [
            id
            for id in cluster_ids
            if id in self.td.get_cluster_ids(var, exclude_noise=False)
        ]

    def _add_gradient_legend(
        self,
        ax: matplotlib.axes.Axes,
        start: int,
        end: int,
        legend_pos: Optional[Tuple[float, float]] = None,
        legend_size: Tuple[float, float] = (0.05, 0.02),
        label_text: Optional[str] = None,
        fontsize: int = 7,
        cmap: Optional[Union[str, Colormap]] = None,
        var: Optional[str] = None,
    ):
        """Add a custom gradient legend to a plot.

        This method adds a gradient legend to visualize cluster IDs from start to end.
        The legend can be automatically positioned based on the variable data or
        manually positioned using legend_pos.

        Args:
            ax: The matplotlib axes to add the legend to
            start: Starting cluster ID for the gradient
            end: Ending cluster ID for the gradient
            legend_pos: Optional tuple of (x, y) coordinates in axes fraction units
                for legend placement. If None, position is determined automatically.
            legend_size: Tuple of (width, height) for the legend size in axes fraction units.
                Defaults to (0.05, 0.02).
            label_text: Optional text label for the legend. If None, no label is added.
            fontsize: Font size for legend text. Defaults to 7.
            cmap: Optional colormap to use for the gradient. If None, uses the colormap
                from the last plotted image or defaults to viridis.
            var: Variable name used for optimal legend positioning when legend_pos is None.
                If None, uses projection-based default positions.

        Returns:
            None
        """

        # Handle automatic positioning
        if legend_pos is None:
            if var is not None:
                legend_pos = self.find_optimal_legend_position(ax, var, legend_size)
            else:
                # Fallback to projection-based positioning
                import cartopy.crs as ccrs

                if hasattr(ax, "projection") and isinstance(
                    ax.projection, ccrs.Projection
                ):
                    if isinstance(ax.projection, ccrs.PlateCarree):
                        legend_pos = (0.75, 0.95)  # top-right
                    else:
                        legend_pos = (0.02, 0.95)  # top-left
                else:
                    legend_pos = (0.02, 0.95)

        # Get colormap
        if cmap is None:
            # Try to get colormap from the last image in the axes
            images = [
                child for child in ax.get_children() if hasattr(child, "get_cmap")
            ]
            if images:
                cmap = images[-1].get_cmap()
            else:
                cmap = plt.cm.viridis  # fallback

        legend_x, legend_y = legend_pos
        legend_width, legend_height = legend_size

        # Check if we have a single cluster (start == end)
        is_single_cluster = start == end

        if is_single_cluster:
            # For single cluster, use a solid color square (middle of the colormap)
            single_color = cmap(0.5)  # Use middle color of the colormap

            rect = Rectangle(
                (legend_x, legend_y),
                legend_width,
                legend_height,
                facecolor=single_color,
                edgecolor="black",
                linewidth=0.5,
                clip_on=False,
                transform=ax.transAxes,
                zorder=1000,
            )
            ax.add_patch(rect)

            # Label for single cluster
            if label_text is None:
                label_text = f"{start}"
        else:
            # Create the gradient effect using multiple thin rectangles
            n_segments = 50
            segment_width = legend_width / n_segments

            for i in range(n_segments):
                color_val = i / (n_segments - 1)
                color = cmap(color_val)
                rect = Rectangle(
                    (legend_x + i * segment_width, legend_y),
                    segment_width,
                    legend_height,
                    facecolor=color,
                    edgecolor="none",
                    clip_on=False,
                    transform=ax.transAxes,
                    zorder=1000,
                )
                ax.add_patch(rect)

            # Add border around the gradient
            border_rect = Rectangle(
                (legend_x, legend_y),
                legend_width,
                legend_height,
                facecolor="none",
                edgecolor="black",
                linewidth=0.5,
                clip_on=False,
                transform=ax.transAxes,
                zorder=1000,
            )
            ax.add_patch(border_rect)

            # Label for multiple clusters
            if label_text is None:
                label_text = f"{start}-{end}"

        # Add text label
        ax.text(
            legend_x + legend_width + 0.01,
            legend_y + legend_height / 2,
            label_text,
            transform=ax.transAxes,
            verticalalignment="center",
            fontsize=fontsize,
            clip_on=False,
        )

    def find_optimal_legend_position(
        self, ax, var, legend_size=(0.05, 0.02), margin=0.02
    ):
        """
        Simple approach: find the corner with the least cluster coverage.
        """
        try:
            # Get all cluster IDs and create a combined mask
            cluster_ids = self.td.get_cluster_ids(var)
            valid_clusters = cluster_ids[cluster_ids != -1]

            if len(valid_clusters) == 0:
                return (margin, 1.0 - margin - legend_size[1])  # default top-left

            # Get combined spatial mask for all clusters
            combined_mask = None
            for cluster_id in valid_clusters:
                mask = self.td.get_spatial_cluster_mask(var, cluster_id)
                if combined_mask is None:
                    combined_mask = mask
                else:
                    combined_mask = combined_mask | mask

            # Define corner regions (each corner gets 25% of the space)
            y_coords = combined_mask[self.td.space_dims[0]]
            x_coords = combined_mask[self.td.space_dims[1]]

            y_mid = (y_coords.max() + y_coords.min()) / 2
            x_mid = (x_coords.max() + x_coords.min()) / 2

            # Count cluster pixels in each corner
            corners = {}
            corners["top-left"] = combined_mask.where(
                (y_coords >= y_mid) & (x_coords <= x_mid)
            ).sum()
            corners["top-right"] = combined_mask.where(
                (y_coords >= y_mid) & (x_coords >= x_mid)
            ).sum()
            corners["bottom-left"] = combined_mask.where(
                (y_coords <= y_mid) & (x_coords <= x_mid)
            ).sum()
            corners["bottom-right"] = combined_mask.where(
                (y_coords <= y_mid) & (x_coords >= x_mid)
            ).sum()

            # Find corner with minimum clusters
            best_corner = min(corners, key=lambda k: float(corners[k]))

            # Convert to legend positions
            legend_width, legend_height = legend_size
            positions = {
                "top-left": (margin, 1.0 - margin - legend_height),
                "top-right": (
                    1.0 - margin - legend_width,
                    1.0 - margin - legend_height,
                ),
                "bottom-left": (margin, margin),
                "bottom-right": (1.0 - margin - legend_width, margin),
            }

            return positions[best_corner]

        except Exception:
            # Fallback to projection-based default
            import cartopy.crs as ccrs

            if hasattr(ax, "projection") and isinstance(
                ax.projection, ccrs.PlateCarree
            ):
                return (
                    1.0 - margin - legend_size[0],
                    1.0 - margin - legend_size[1],
                )  # top-right
            else:
                return (margin, 1.0 - margin - legend_size[1])  # top-left


# end of TOADPlotter


# This function appears unused in this file. Consider removing if not used elsewhere.
# def get_max_index(pos, n_rows=None):
#     """Helper function to get the maximum index from a position spec."""
#     if isinstance(pos, slice):
#         if pos.stop is not None:
#             return pos.stop
#         return (n_rows - 1) if n_rows is not None else 0
#     return pos


def get_high_constrast_text_color(color: Union[tuple, str]) -> str:
    """Determines whether black or white text provides better contrast against a given background color.

    Args:
        color: The background color (matplotlib-compatible string or RGB tuple).

    Returns:
        '#ffffff' (white) or '#000000' (black) for the text color.
        Defaults to black if the color conversion fails.
    """
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


def get_cmap_seq(
    cmap: Colormap | str,
    start: int = 0,
    end: int = -1,
    stops: int = 10,
    reverse: bool = False,
) -> List[str]:
    """Extracts a sequence of distinct colors from a matplotlib colormap.

    Args:
        cmap: Name of the matplotlib colormap.
        start: Starting index within the colormap.
        end: Ending index within the colormap. Defaults to the end of the cmap.
        stops: The number of distinct colors to extract.
        reverse: If True, reverse the order of the extracted colors.

    Returns:
        A list of color hex codes.
    """
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
