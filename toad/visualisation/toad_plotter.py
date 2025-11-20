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
            projection: Map projection to use. If None, uses default projection if lat/lon
                        coordinates exist, otherwise creates regular (non-geographic) axes.
            config: Plot configuration
            figsize: Figure size (width, height) in inches
            height_ratios: List of height ratios for subplots
            width_ratios: List of width ratios for subplots
            subplot_spec: A gridspec subplot spec to place the map in
        """
        config = config if config else self.default_config

        # Check if data has lat/lon coordinates
        lat_name, lon_name = detect_latlon_names(self.td.data)
        has_latlon = lat_name is not None and lon_name is not None

        # Determine if we should use a projection
        # Use projection if: explicitly provided OR (has lat/lon AND projection not explicitly None)
        use_projection = False
        projection_obj = None
        projection_str = None

        if projection is not None:
            # User explicitly provided a projection - use it
            use_projection = True
            if isinstance(projection, str):
                projection_str = projection
                if projection not in _projection_map:
                    raise ValueError(
                        f"Invalid projection '{projection}'. Please choose between {list(_projection_map.keys())}"
                    )
                projection_obj = _projection_map[projection]
            else:
                # It's already a ccrs.Projection object
                projection_obj = projection
                # Try to find the string name for extent setting
                projection_class = projection.__class__
                for name, proj in _projection_map.items():
                    if proj.__class__ == projection_class:
                        projection_str = name
                        break
        elif has_latlon:
            # No explicit projection but has lat/lon - use default
            use_projection = True
            projection_str = config.projection
            projection_obj = _projection_map[projection_str]
        # else: no projection, no lat/lon - use regular Axes (use_projection = False)

        if subplot_spec is not None:
            # Create map in existing figure using subplot_spec
            fig = plt.gcf()
            if use_projection:
                ax = fig.add_subplot(subplot_spec, projection=projection_obj)
            else:
                ax = fig.add_subplot(subplot_spec)
            axs = ax
        else:
            # Create new figure with subplots
            gridspec_kw = {}
            if height_ratios:
                gridspec_kw["height_ratios"] = height_ratios
            if width_ratios:
                gridspec_kw["width_ratios"] = width_ratios

            subplot_kw = {}
            if use_projection:
                subplot_kw["projection"] = projection_obj

            fig, axs = plt.subplots(
                nrows,
                ncols,
                figsize=figsize,
                subplot_kw=subplot_kw if subplot_kw else None,
                gridspec_kw=gridspec_kw if gridspec_kw else None,
            )

        # Ensure axs is always an array for consistent iteration
        axs_array = np.array(axs, ndmin=2)

        # Add map features and set extent for all axes (only if using projection)
        for ax in axs_array.flat:
            if use_projection:
                self._add_map_features(ax, config)
                ax.set_frame_on(config.map_frame)

                if projection_str == "south_pole":
                    ax.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
                elif projection_str == "north_pole":
                    ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())
            else:
                # Regular axes - just set frame
                ax.set_frame_on(config.map_frame)

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
        return fig, np.array(axs)

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
        return fig, np.array(axs)

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
        contour_linewidth: float = 1.5,
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
            contour_linewidth: Linewidth for contour lines. Defaults to 1.5.
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
                **kwargs,
            }

            lat_name, lon_name = detect_latlon_names(self.td.data)
            has_latlon = lat_name is not None and lon_name is not None

            # Check if axes is a GeoAxes (has projection)
            is_geoaxes = hasattr(ax, "projection") and ax.projection is not None

            # plot on lat/lon coordinates if available
            if has_latlon:
                plot_params["x"] = lon_name
                plot_params["y"] = lat_name
                plot_params["transform"] = ccrs.PlateCarree()
            elif is_geoaxes:
                # GeoAxes but no lat/lon - use spatial dimensions explicitly
                # This ensures xarray.plot uses the correct dimensions
                space_dims = self.td.space_dims
                if len(space_dims) >= 2:
                    plot_params["x"] = space_dims[1]  # x/lon dimension
                    plot_params["y"] = space_dims[0]  # y/lat dimension
                # Don't set transform - let xarray handle it based on the GeoAxes projection
            else:
                # Regular axes, no lat/lon - use spatial dimensions explicitly
                # This ensures xarray.plot uses the correct dimensions
                space_dims = self.td.space_dims
                if len(space_dims) >= 2:
                    plot_params["x"] = space_dims[1]  # x/lon dimension
                    plot_params["y"] = space_dims[0]  # y/lat dimension

            if not only_contour:
                # Don't plot values outside mask: FALSE -> np.nan
                # Use pcolormesh explicitly for regular axes to ensure proper coordinate handling
                if not has_latlon and not is_geoaxes:
                    mask.where(mask, np.nan).plot.pcolormesh(
                        **plot_params,
                    )
                else:
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
                    linewidths=contour_linewidth,
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
                        transform=plot_params.get("transform"),
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

    def _create_timeseries_layout(
        self,
        fig: matplotlib.figure.Figure,
        n_clusters: int,
        n_subplots_col: int,
        subplot_spec: Any = None,
        hspace: float = 0.1,
        wspace: float = 0.1,
    ) -> List[Axes]:
        """Create subplot layout for timeseries plots.

        Args:
            fig: Figure to create subplots in
            n_clusters: Number of clusters (subplots to create)
            n_subplots_col: Number of columns in subplot grid
            subplot_spec: Optional gridspec subplot spec (if None, creates new gridspec)
            hspace: Height space between subplots
            wspace: Width space between subplots

        Returns:
            List of axes for timeseries plots.
        """
        n_ts_rows = int(np.ceil(n_clusters / n_subplots_col))
        ts_axes_list: List[Axes] = []

        if subplot_spec is not None:
            # Use provided subplot_spec (when map exists)
            gs = subplot_spec.subgridspec(
                nrows=n_ts_rows,
                ncols=n_subplots_col,
                hspace=hspace,
                wspace=wspace if n_subplots_col > 1 else 0,
            )
        else:
            # Create new gridspec (when no map)
            gs = fig.add_gridspec(
                nrows=n_ts_rows,
                ncols=n_subplots_col,
                hspace=hspace,
                wspace=wspace if n_subplots_col > 1 else 0,
            )

        # Create timeseries axes
        for i in range(n_clusters):
            row = i // n_subplots_col
            col = i % n_subplots_col
            ts_ax = fig.add_subplot(gs[row, col])
            ts_axes_list.append(ts_ax)

        # Hide any empty subplots
        for i in range(n_clusters, n_ts_rows * n_subplots_col):
            row = i // n_subplots_col
            col = i % n_subplots_col
            empty_ax = fig.add_subplot(gs[row, col])
            empty_ax.set_visible(False)

        return ts_axes_list

    def timeseries(
        self,
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = None,
        plot_var: Optional[str] = None,
        ax: Optional[Axes] = None,
        color: Optional[str] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        normalize: Optional[Literal["max", "max_each"]] | str = None,
        add_legend: bool = True,
        # Individual trajectories
        plot_individual: bool = True,
        max_trajectories: int = 1_000,
        individual_alpha: float = 0.5,
        individual_linewidth: float = 0.5,
        full_timeseries: bool = True,
        cluster_highlight_color: Optional[str] = None,
        cluster_highlight_alpha: float = 0.5,
        cluster_highlight_linewidth: float = 0.5,
        plot_shifts: bool = False,  # If True, plot shifts variable in timeseries
        # Aggregated statistics
        plot_median: bool = False,
        plot_mean: bool = False,
        median_linewidth: float = 3,
        mean_linewidth: float = 3,
        # Shaded regions
        plot_range: bool = False,  # Full range (min to max)
        plot_std_range: bool = False,  # 68% interquartile range (16th to 84th percentile)
        range_alpha: float = 0.2,
        std_alpha: float = 0.4,
        # Shift duration
        plot_shift_duration: bool = True,
        shift_duration_color: Optional[str] = None,  # Uses cluster color if None
        shift_duration_alpha: float = 0.25,
        # Subplot layout
        subplots: bool = False,  # If True, create one subplot per cluster
        n_subplots_col: int = 1,  # Number of columns for subplot grid
        # Map options
        map: bool = False,  # If True, add a map alongside timeseries
        map_var: Optional[str] = None,  # Variable for map (defaults to var)
        map_projection: Optional[str | ccrs.Projection] = None,
        map_add_contour: bool = True,
        map_only_contour: bool = False,
        map_add_labels: bool = True,
        map_remaining_clusters_cmap: Optional[Union[str, Colormap]] = "jet",
        map_remaining_clusters_legend_pos: Optional[Tuple[float, float]] = None,
        plot_all_clusters_on_map: bool = True,
        # Layout (only used when map=True or subplots=True)
        vertical: bool = False,  # Map above/below timeseries
        width_ratios: Tuple[float, float] = (1.0, 1.0),  # For horizontal layout
        height_ratios: Optional[Tuple[float, float]] = None,  # For vertical layout
        figsize: tuple = (12, 6),
        wspace: float = 0.1,
        hspace: float = 0.1,
        timeseries_ylabel: bool = False,  # Only relevant for subplots
        **plot_kwargs: Any,
    ) -> Tuple[Optional[matplotlib.figure.Figure], Union[Axes, List[Axes], dict]]:
        """Plot time series from clusters or all data.

        This function allows flexible plotting of individual trajectories, aggregated statistics
        (median/mean), shaded regions (full range and std range), and shift duration indicators.
        If no cluster_ids are provided, plots all timeseries from the dataset.

        Can optionally create separate subplots for each cluster and/or add a map alongside.

        Args:
            var: Variable name for which clusters have been computed. If None, TOAD will attempt
                to infer which variable to use. A ValueError is raised if the variable cannot be
                uniquely determined.
            cluster_ids: ID or list of IDs of clusters to plot. If None, plots all timeseries
                from the dataset (no clustering).
            plot_var: Variable name to plot (if different from var). Defaults to var.
            ax: Matplotlib axes to plot on. Creates new figure if None. Ignored if subplots=True or map=True.
            color: Single color to use for all plotted clusters. Overrides cmap.
            cmap: Colormap to use if plotting multiple clusters and color is None.
            normalize: Method to normalize timeseries ('max', 'max_each'). Defaults to None.
            add_legend: If True, add a legend indicating cluster IDs.
            plot_individual: If True, plot individual cell trajectories.
            max_trajectories: Maximum number of individual trajectories to plot (per cluster if
                cluster_ids provided, or total if plotting all data).
            individual_alpha: Alpha transparency for individual time series lines. Defaults to 0.5.
            individual_linewidth: Linewidth for individual time series lines. Defaults to 0.5.
            full_timeseries: If True, plot the full timeseries for each cell. If False,
                only plot the segment belonging to the cluster.
            cluster_highlight_color: Color to highlight the actual cluster segment
                when full_timeseries is True.
            cluster_highlight_alpha: Alpha for the cluster highlight segment.
            cluster_highlight_linewidth: Line width for the cluster highlight segment.
            plot_shifts: If True, plot shifts variable in timeseries instead of base variable.
            plot_median: If True, plot the median timeseries curve.
            plot_mean: If True, plot the mean timeseries curve.
            median_linewidth: Linewidth for the median curve.
            mean_linewidth: Linewidth for the mean curve.
            plot_range: If True, plot the full range (min to max) as a shaded area.
            plot_std_range: If True, plot the 68% interquartile range (16th to 84th percentile) as a shaded area.
            range_alpha: Alpha transparency for the full range shaded area.
            std_alpha: Alpha transparency for the IQR shaded area.
            plot_shift_duration: If True, adds horizontal shading indicating the cluster's
                temporal extent (start to end). Only applies when cluster_ids are provided.
            shift_duration_color: Color for shift duration shading. Uses cluster color if None.
            shift_duration_alpha: Alpha for the shift duration shading.
            subplots: If True, create one subplot per cluster. Defaults to False.
                Note: Subplots are automatically created when map=True with multiple clusters.
            n_subplots_col: Number of columns for subplot grid when subplots=True. Must be > 0.
            map: If True, add a map alongside timeseries. Defaults to False.
            map_var: Variable name whose data to plot in the map. Defaults to var if None.
            map_projection: Map projection for the cluster map. Uses default if None.
            map_add_contour: If True, add contour lines around clusters.
            map_only_contour: If True, only plot contour lines (no fill).
            map_add_labels: If True, add cluster ID labels using a geometrically central point.
            map_remaining_clusters_cmap: Colormap for remaining clusters.
            map_remaining_clusters_legend_pos: Tuple of (x, y) position of the remaining clusters legend.
            plot_all_clusters_on_map: If True, plot all clusters on the map. If False, only plot selected clusters.
            vertical: If True, arrange map above timeseries plots. Otherwise, map is placed to the left.
            width_ratios: Tuple of relative widths for map vs. timeseries section (used in horizontal layout).
            height_ratios: Optional tuple of relative heights for map vs. timeseries section (used in vertical layout).
            figsize: Overall figure size (width, height) in inches. Used when subplots=True or map=True.
            wspace: Width space between timeseries subplots (if n_subplots_col > 1).
            hspace: Height space between map/timeseries (vertical) or timeseries rows.
            timeseries_ylabel: If True, show y-axis label on the timeseries plots. Only relevant for subplots.
            **plot_kwargs: Additional arguments passed to xarray.plot for each trajectory.

        Returns:
            Tuple of (figure, axes).
            - If single plot: (figure, Axes)
            - If subplots=True: (figure, List[Axes])
            - If map=True: (figure, dict) with keys 'map' and 'timeseries' (which may be Axes or List[Axes])
            Figure is None if ax was provided and subplots=False and map=False.

        Raises:
            ValueError: If no timeseries found for a given cluster ID, or if nothing is set to plot.
        """

        # Handle case when no cluster_ids provided - plot all data
        plot_all_data = cluster_ids is None

        if plot_all_data:
            # Treat as single pseudo-cluster with id=None
            cluster_ids_list: List[Optional[int]] = [None]
            single_plot = True
            var = self.td._get_base_var_if_none(var)
        else:
            # Filter cluster_ids to only include existing clusters
            var = self.td._get_base_var_if_none(var)
            cluster_ids = self.filter_by_existing_clusters(cluster_ids, var)

            # Check if we have any clusters to plot
            if len(cluster_ids) == 0:
                raise ValueError(f"No valid clusters found for variable {var}")

            # Convert single cluster_id to list for consistent handling
            if isinstance(cluster_ids, int):
                single_plot = True
                cluster_ids_list = [cluster_ids]
            else:
                single_plot = False
                cluster_ids_list = list(cluster_ids)

        # Determine plot_var for timeseries
        if plot_var is None:
            plot_var = var
        plot_var = self.td._get_base_var_if_none(plot_var)

        # Handle map setup and determine plot_var for timeseries when map=True
        map_ax = None
        if map:
            if plot_all_data:
                raise ValueError(
                    "Cannot plot map when cluster_ids is None (plotting all data)"
                )
            if map_var is None:
                map_var = var
            # Get base variable from clusters attrs for timeseries if plot_var wasn't explicitly set
            # (i.e., if it equals var, meaning user didn't specify a different variable)
            if plot_var == var:
                plot_var = self.td.get_clusters(var).attrs[_attrs.BASE_VARIABLE]
            if plot_shifts:
                plot_var = self.td.get_clusters(var).attrs[_attrs.SHIFTS_VARIABLE]

        # Validate that something will be plotted
        has_individual = plot_individual
        has_aggregate = plot_median or plot_mean or plot_range or plot_std_range
        if not has_individual and not has_aggregate:
            raise ValueError(
                "Nothing to plot: set at least one of plot_individual, plot_median, "
                "plot_mean, plot_range, or plot_std_range to True."
            )

        # Validate n_subplots_col
        if n_subplots_col <= 0:
            raise ValueError(f"n_subplots_col must be > 0, got {n_subplots_col}")

        # Determine if we need subplots (either explicitly requested or when map=True with multiple clusters)
        use_subplots = subplots or (map and len(cluster_ids_list) > 1)

        # Setup figure and axes layout
        fig = None
        ts_axes_list: List[Axes] = []

        if map or use_subplots:
            # Create figure with constrained_layout
            fig = plt.figure(figsize=figsize, constrained_layout=True)

            if map:
                # Create map first, then timeseries subplots
                if vertical:
                    main_gs = fig.add_gridspec(
                        nrows=2,
                        ncols=1,
                        height_ratios=list(height_ratios) if height_ratios else [1, 1],
                        hspace=hspace,
                    )
                    _, map_ax = self.map(
                        nrows=1,
                        ncols=1,
                        subplot_spec=main_gs[0],
                        projection=map_projection,
                    )
                    ts_subplot_spec = main_gs[1]
                else:
                    main_gs = fig.add_gridspec(
                        nrows=1,
                        ncols=2,
                        width_ratios=list(width_ratios),
                    )
                    _, map_ax = self.map(
                        nrows=1,
                        ncols=1,
                        subplot_spec=main_gs[0, 0],
                        projection=map_projection,
                    )  # type: ignore
                    ts_subplot_spec = main_gs[0, 1]  # type: ignore

                # Create timeseries subplots in remaining space
                ts_axes_list = self._create_timeseries_layout(
                    fig=fig,
                    n_clusters=len(cluster_ids_list),
                    n_subplots_col=n_subplots_col,
                    subplot_spec=ts_subplot_spec,
                    hspace=hspace,
                    wspace=wspace,
                )
            else:
                # Only subplots, no map
                ts_axes_list = self._create_timeseries_layout(
                    fig=fig,
                    n_clusters=len(cluster_ids_list),
                    n_subplots_col=n_subplots_col,
                    hspace=hspace,
                    wspace=wspace,
                )
        else:
            # Single plot - use provided ax or create new one
            create_new_ax = ax is None
            if create_new_ax:
                fig, ax = plt.subplots()
            ts_axes_list = [ax]  # Use single ax for consistency

        # Get colors for clusters (used for both map and timeseries)
        colors = None
        if map or (use_subplots and len(cluster_ids_list) > 1):
            colors = get_cmap_seq(stops=len(cluster_ids_list), cmap=cmap)

        # Plot map if requested
        if map and map_ax is not None:
            # Don't plot remaining clusters on map if not requested
            remaining_clusters_cmap = (
                None if not plot_all_clusters_on_map else map_remaining_clusters_cmap
            )
            # Filter out None values for cluster_ids
            cluster_ids_for_map = [id for id in cluster_ids_list if id is not None]
            # Determine color for map: single color if one cluster, list if multiple, or use provided color
            map_color: Optional[Union[str, Tuple, List[Union[str, Tuple]]]] = None
            if colors:
                if len(colors) == 1:
                    map_color = colors[0]
                else:
                    # Cast List[str] to List[Union[str, Tuple]] for type checker
                    map_color = colors  # type: ignore[assignment]
            elif color:
                map_color = color
            self.cluster_map(
                map_var,
                cluster_ids=cluster_ids_for_map,
                color=map_color,
                ax=map_ax,
                add_contour=map_add_contour,
                only_contour=map_only_contour,
                add_labels=map_add_labels,
                remaining_clusters_cmap=remaining_clusters_cmap,
                remaining_clusters_legend_pos=map_remaining_clusters_legend_pos,
                **plot_kwargs,
            )

        # Single unified loop for both all-data and clustered plotting
        y_label = ""
        for i, id in enumerate(cluster_ids_list):
            # Get the axes for this cluster
            # When subplots=False, all clusters use the same axis
            if use_subplots:
                current_ax = ts_axes_list[i]
            else:
                current_ax = ts_axes_list[0]

            # Get color
            if color:
                id_color = color
            else:
                if len(cluster_ids_list) == 1:
                    id_color = ToadColors.primary
                else:
                    id_color = (
                        colors[i]
                        if colors
                        else get_cmap_seq(stops=len(cluster_ids_list), cmap=cmap)[i]
                    )

            # Use cluster color for shift duration if not specified
            shift_color = (
                shift_duration_color if shift_duration_color is not None else id_color
            )

            # Determine if this is a real cluster (id is not None)
            is_real_cluster = id is not None

            # Prepare get_cluster_timeseries kwargs
            ts_kwargs = {
                "var": plot_var,
                "cluster_id": id,
                "normalize": normalize,
            }
            if is_real_cluster:
                ts_kwargs["cluster_var"] = var

            # Plot aggregated statistics first (so they appear behind individual trajectories)
            # Plot full range (min to max)
            if plot_range:
                min_ts = self.td.get_cluster_timeseries(
                    aggregation="min",
                    **ts_kwargs,
                )
                max_ts = self.td.get_cluster_timeseries(
                    aggregation="max",
                    **ts_kwargs,
                )
                current_ax.fill_between(
                    self.td.data[self.td.time_dim].values,
                    min_ts,
                    max_ts,
                    color=id_color,
                    alpha=range_alpha,
                    zorder=0,
                )

            # Plot 68% interquartile range (16th to 84th percentile)
            if plot_std_range:
                p16_ts = self.td.get_cluster_timeseries(
                    aggregation="percentile",
                    percentile=0.16,
                    **ts_kwargs,
                )
                p84_ts = self.td.get_cluster_timeseries(
                    aggregation="percentile",
                    percentile=0.84,
                    **ts_kwargs,
                )
                current_ax.fill_between(
                    self.td.data[self.td.time_dim].values,
                    p16_ts,
                    p84_ts,
                    color=id_color,
                    alpha=std_alpha,
                    zorder=1,
                )

            # Plot mean
            if plot_mean:
                if not is_real_cluster:
                    label = "mean"
                else:
                    label = f"#{id}"
                self.td.get_cluster_timeseries(
                    aggregation="mean",
                    **ts_kwargs,
                ).plot(
                    ax=current_ax,
                    color=id_color,
                    lw=mean_linewidth,
                    label=label if add_legend else "__nolegend__",
                    zorder=3,
                )  # type: ignore

            # Plot median
            if plot_median:
                if not is_real_cluster:
                    label = "median"
                else:
                    label = f"#{id}"
                self.td.get_cluster_timeseries(
                    aggregation="median",
                    **ts_kwargs,
                ).plot(
                    ax=current_ax,
                    color=id_color,
                    lw=median_linewidth,
                    label=label if add_legend else "__nolegend__",
                    zorder=3,
                )  # type: ignore

            # Plot shift duration (horizontal shading) - only for real clusters
            if plot_shift_duration and is_real_cluster:
                # Add a bit more padding on top and bottom by using ymin and ymax arguments
                current_ax.axvspan(
                    self.td.cluster_stats(var).time.start(id),
                    self.td.cluster_stats(var).time.end(id),
                    color=shift_color if shift_color else id_color,
                    alpha=shift_duration_alpha,
                    zorder=-100,
                )

            # Plot individual trajectories
            if plot_individual:
                individual_ts_kwargs = {
                    **ts_kwargs,
                    "aggregation": "raw",
                }
                if is_real_cluster:
                    individual_ts_kwargs["keep_full_timeseries"] = full_timeseries

                cells = self.td.get_cluster_timeseries(**individual_ts_kwargs)

                if cells is None:
                    if is_real_cluster:
                        raise ValueError(f"No timeseries found for cluster {id}")
                    else:
                        raise ValueError(f"No timeseries found for {plot_var}")

                # Limit the number of trajectories to plot
                max_trajectories_actual = np.min([max_trajectories, len(cells)])

                # Shuffle the cell to get a random sample
                order = np.arange(len(cells))
                np.random.shuffle(order)
                order = order[:max_trajectories_actual]

                for plot_idx, cell_idx in enumerate(order):
                    if is_real_cluster:
                        # Add label on first trajectory if legend is enabled
                        # For single plot: add label on first trajectory of each cluster
                        # For subplots: don't add label to line (we'll use ax.text instead)
                        if add_legend and plot_idx == 0 and not use_subplots:
                            # Label each cluster (only for single plot, not subplots)
                            add_label = f"#{id}"
                        else:
                            add_label = "__nolegend__"
                    else:
                        add_label = "__nolegend__"
                    cells[cell_idx].plot(
                        ax=current_ax,
                        color=id_color,
                        alpha=individual_alpha,
                        lw=individual_linewidth,
                        label=add_label,
                        **plot_kwargs,
                    )

                if cluster_highlight_color and is_real_cluster:
                    # Reuse cells if already fetched with keep_full_timeseries=False
                    # Otherwise fetch separately
                    if not full_timeseries:
                        cells_highlight = cells
                    else:
                        highlight_ts_kwargs = {
                            **ts_kwargs,
                            "aggregation": "raw",
                            "keep_full_timeseries": False,
                        }
                        cells_highlight = self.td.get_cluster_timeseries(
                            **highlight_ts_kwargs
                        )
                    for ts in cells_highlight:
                        ts.plot(
                            ax=current_ax,
                            color=cluster_highlight_color,
                            alpha=cluster_highlight_alpha,
                            lw=cluster_highlight_linewidth,
                        )

            # Handle axis cleanup for subplots
            if use_subplots:
                current_ax.axhline(0, ls="--", lw=0.25, color="k")
                current_ax.set_title("")

                if not timeseries_ylabel:
                    if i == 0:
                        y_label = current_ax.get_ylabel()
                    current_ax.set_ylabel("")

                # Determine if this subplot is in the bottom row of its column
                n_ts = len(cluster_ids_list)
                # If multiple columns, check if this is the bottom subplot in its column
                # Otherwise, check if it's the last subplot overall
                if n_subplots_col > 1:
                    # Check if next subplot in same column exists
                    is_bottom_in_column = i + n_subplots_col >= n_ts
                else:
                    # Single column: just check if it's the last subplot
                    is_bottom_in_column = i == n_ts - 1

                # Handle axis cleanup
                if not is_bottom_in_column:
                    current_ax.set_xlabel("")
                    self._remove_spines(current_ax, ["right", "top", "bottom"])
                else:
                    self._remove_spines(current_ax, ["right", "top"])

                if not is_bottom_in_column:
                    self._remove_ticks(current_ax, keep_y=True)

            # Handle legend
            if add_legend:
                if use_subplots:
                    # For subplots: use ax.text to add cluster ID label (no color needed)
                    if is_real_cluster:
                        # Position text in upper right corner using axes coordinates
                        current_ax.text(
                            1.02,
                            1.02,
                            f"#{id}",
                            ha="right",
                            va="top",
                            transform=current_ax.transAxes,
                        )
                else:
                    # For single plot: use legend
                    # Check if there are any labeled artists before calling legend()
                    handles, labels = current_ax.get_legend_handles_labels()
                    has_labels = any(
                        label and not label.startswith("_") for label in labels
                    )

                    if has_labels:
                        # Single plot: only show legend on the last iteration
                        # Position in upper right corner
                        if i == len(cluster_ids_list) - 1:
                            legend = current_ax.legend(frameon=False, loc="upper right")
                            for handle in legend.get_lines():
                                handle.set_alpha(1.0)

        # Set title for subplots (only for first subplot when map=True)
        if map and use_subplots and len(ts_axes_list) > 0:
            # Only label as "largest" if clusters are consecutive starting at 0
            cluster_ids_int = [id for id in cluster_ids_list if id is not None]
            if cluster_ids_int:
                is_zero_indexed = (
                    cluster_ids_int
                    == list(
                        range(
                            cluster_ids_int[0],
                            cluster_ids_int[0] + len(cluster_ids_int),
                        )
                    )
                    and cluster_ids_int[0] == 0
                )
                if (
                    len(cluster_ids_int) < len(self.td.get_cluster_ids(var))
                    and is_zero_indexed
                ):
                    title = f"{len(cluster_ids_int)} largest clusters"
                else:
                    title = "clusters"
                if y_label != "":
                    title += f" in {y_label}"
                ts_axes_list[0].set_title(title)

        # Set title for single plot (not subplots)
        if not use_subplots:
            is_all_data = plot_all_data or (
                len(cluster_ids_list) == 1 and cluster_ids_list[0] is None
            )
            current_ax = ts_axes_list[0]

            if is_all_data:
                # Title for all data case
                if plot_individual:
                    # Get cell count for title
                    cells = self.td.get_cluster_timeseries(
                        plot_var,
                        cluster_id=None,
                        aggregation="raw",
                        normalize=normalize,
                    )
                    if cells is not None:
                        max_trajectories_actual = np.min([max_trajectories, len(cells)])
                        if max_trajectories_actual < len(cells):
                            current_ax.set_title(
                                f"Random sample of {max_trajectories_actual} from total {len(cells)} timeseries for {plot_var}"
                            )
                        else:
                            current_ax.set_title(
                                f"{len(cells)} timeseries for {plot_var}"
                            )
                    else:
                        current_ax.set_title(f"{plot_var} timeseries")
                else:
                    current_ax.set_title(f"{plot_var} timeseries")
            else:
                # Title for clusters case
                if has_aggregate and not plot_individual:
                    # Aggregated statistics only
                    current_ax.set_title(
                        f"{plot_var} for clusters from {var} {cluster_ids_list}"
                    )
                elif plot_individual and single_plot:
                    # Individual trajectories for single cluster - get cell count for title
                    cells = self.td.get_cluster_timeseries(
                        plot_var,
                        cluster_ids_list[0],
                        cluster_var=var,
                        aggregation="raw",
                        keep_full_timeseries=full_timeseries,
                        normalize=normalize,
                    )
                    if cells is not None:
                        max_trajectories_actual = np.min([max_trajectories, len(cells)])
                        if max_trajectories_actual < len(cells):
                            current_ax.set_title(
                                f"Random sample of {max_trajectories_actual} from total {len(cells)} cell for {var} in cluster {cluster_ids_list[0]}"
                            )
                        else:
                            current_ax.set_title(
                                f"{len(cells)} timeseries for {var} in cluster {cluster_ids_list[0]}"
                            )
                elif plot_individual and not single_plot:
                    # Multiple clusters with individual trajectories
                    current_ax.set_title(
                        f"{plot_var} trajectories for clusters from {var} {cluster_ids_list}"
                    )
                elif plot_individual and has_aggregate:
                    # Both individual and aggregate
                    current_ax.set_title(
                        f"{plot_var} for clusters from {var} {cluster_ids_list}"
                    )

        # Return appropriate values
        if map:
            # Return dict with map and timeseries axes
            if use_subplots:
                return fig, {"map": map_ax, "timeseries": ts_axes_list}
            else:
                return fig, {"map": map_ax, "timeseries": ts_axes_list[0]}
        elif use_subplots:
            # Return list of axes
            return fig, ts_axes_list
        else:
            # Return single axes
            return fig, ts_axes_list[0]

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
        normalize: Optional[Literal["max", "max_each"]] | str = None,
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

        .. deprecated::
            This function is now a wrapper around `cluster_timeseries` with `mode="aggregate"`.
            Consider using `cluster_timeseries` directly for more flexibility.

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
            normalize: Method to normalize timeseries ('max', 'max_each'). Defaults to None.
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
        # Map old parameter names to new parameter names
        return self.timeseries(
            var=cluster_var,
            cluster_ids=cluster_ids,
            plot_var=plot_var,
            ax=ax,
            color=color,
            cmap=cmap,
            normalize=normalize,
            add_legend=add_legend,
            mode="aggregate",
            plot_aggregate=True,
            plot_individual=False,
            plot_median=plot_cluster_median,
            plot_mean=plot_cluster_mean,
            median_linewidth=median_linewidth,
            mean_linewidth=mean_linewidth,
            plot_range=plot_cluster_range,
            plot_68iqr=plot_cluster_68iqr,
            plot_95iqr=plot_cluster_95iqr,
            plot_custom_iqr=plot_cluster_iqr,
            iqr_alpha=alpha,
            plot_shift_range=plot_shift_range,
            plot_largest_gradient=plot_largest_gradient,
        )

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
        mode: Literal["individual", "aggregate", "both"] | str = "individual",
        projection: Optional[str | ccrs.Projection] = None,
        figsize: tuple = (12, 6),
        width_ratios: List[float] = [1, 1],
        height_ratios: Optional[List[float]] = None,
        timeseries_ylabel: bool = False,
        cmap: str = default_cmap,
        wspace: float = 0.1,
        hspace: float = 0.1,
        vertical: bool = False,
        n_timeseries_col: int = 1,
        plot_all_clusters_on_map: bool = True,
        # cluster_map parameters (prefixed with map_)
        map_add_contour: bool = True,
        map_only_contour: bool = False,
        map_add_labels: bool = True,
        map_remaining_clusters_cmap: Optional[Union[str, Colormap]] = "jet",
        map_remaining_clusters_legend_pos: Optional[Tuple[float, float]] = None,
        # cluster_timeseries parameters (prefixed with ts_)
        ts_normalize: Optional[Literal["max", "max_each"]] | str = None,
        ts_add_legend: bool = True,
        ts_plot_individual: bool = True,
        ts_max_trajectories: int = 1_000,
        ts_individual_alpha: Optional[float] = None,
        ts_alpha: Optional[
            float
        ] = None,  # Backward compatibility: maps to individual_alpha
        ts_full_timeseries: bool = True,
        ts_cluster_highlight_color: Optional[str] = None,
        ts_cluster_highlight_alpha: float = 0.5,
        ts_cluster_highlight_linewidth: float = 0.5,
        ts_plot_median: bool = False,
        ts_plot_mean: bool = False,
        ts_median_linewidth: float = 3,
        ts_mean_linewidth: float = 3,
        ts_plot_range: bool = False,
        ts_plot_std_range: bool = False,
        ts_range_alpha: float = 0.2,
        ts_std_alpha: float = 0.4,
        ts_plot_shift_duration: bool = True,
        ts_shift_duration_color: Optional[str] = None,
        ts_shift_duration_alpha: float = 0.25,
        ts_color: Optional[str] = None,
        **kwargs: Any,
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
            plot_shifts: If True, plot shifts variable in timeseries instead of base variable.
            mode: Plotting mode - "individual", "aggregate", or "both". Defaults to "individual".
            projection: Map projection for the cluster map. Uses default if None. Can be a string or a cartopy projection object.
            figsize: Overall figure size (width, height) in inches.
            width_ratios: List of relative widths for map vs. timeseries section
                          (used in horizontal layout).
            height_ratios: List of relative heights for map vs. timeseries section
                           (used in vertical layout).
            timeseries_ylabel: If True, show y-axis label on the timeseries plots.
            cmap: Colormap used to color clusters consistently across map and
                  timeseries plots.
            wspace: Width space between timeseries subplots (if n_timeseries_col > 1).
            hspace: Height space between map/timeseries (vertical) or timeseries rows.
            vertical: If True, arrange map above timeseries plots. Otherwise, map
                      is placed to the left.
            n_timeseries_col: Number of columns for the timeseries subplot grid.
            plot_all_clusters_on_map: If True, plot all clusters on the map. If False, only plot selected clusters.
            map_add_contour: If True, add contour lines around clusters.
            map_only_contour: If True, only plot contour lines (no fill).
            map_add_labels: If True, add cluster ID labels using a geometrically central point.
            map_remaining_clusters_cmap: Colormap for remaining clusters. Can be a string (e.g., "jet", "viridis") or a matplotlib colormap object.
            map_remaining_clusters_legend_pos: Tuple of (x, y) position of the remaining clusters legend. If None, the legend is placed automatically.
            ts_normalize: Method to normalize timeseries ('max', 'max_each'). Defaults to None.
            ts_add_legend: If True, add a legend indicating cluster IDs.
            ts_plot_individual: If True, plot individual trajectories.
            ts_max_trajectories: Maximum number of individual trajectories to plot per cluster.
            ts_individual_alpha: Alpha transparency for individual time series lines. Defaults to 0.1.
            ts_alpha: (Deprecated, use ts_individual_alpha) Backward compatibility parameter that maps to individual_alpha.
            ts_full_timeseries: If True, plot the full timeseries for each cell. If False,
                only plot the segment belonging to the cluster.
            ts_cluster_highlight_color: Color to highlight the actual cluster segment
                when ts_full_timeseries is True.
            ts_cluster_highlight_alpha: Alpha for the cluster highlight segment.
            ts_cluster_highlight_linewidth: Line width for the cluster highlight segment.
            ts_plot_median: If True, plot the median timeseries curve.
            ts_plot_mean: If True, plot the mean timeseries curve.
            ts_median_linewidth: Linewidth for the median curve.
            ts_mean_linewidth: Linewidth for the mean curve.
            ts_plot_range: If True, plot the full range (min to max) as a shaded area.
            ts_plot_std_range: If True, plot the 68% interquartile range (16th to 84th percentile) as a shaded area.
            ts_range_alpha: Alpha transparency for the full range shaded area.
            ts_std_alpha: Alpha transparency for the std range (IQR) shaded area.
            ts_plot_shift_duration: If True, adds horizontal shading that indicates the cluster's temporal extent (start/end).
            ts_shift_duration_color: Color for the shift duration shading. Uses cluster color if None.
            ts_shift_duration_alpha: Alpha transparency for the shift duration shading.
            ts_color: Single color to use for all plotted clusters. Overrides cmap.
            **kwargs: Additional arguments passed to xarray.plot methods (for both map and timeseries plots).

        Returns:
            Tuple containing:
                - fig: The matplotlib Figure object.
                - axes_dict: A dictionary containing the map axes and a list of
                  timeseries axes, e.g., {'map': map_ax, 'timeseries': [ts_ax1, ts_ax2,...]}.
        """
        # Handle backward compatibility for ts_alpha -> ts_individual_alpha
        individual_alpha_val = (
            ts_individual_alpha
            if ts_individual_alpha is not None
            else (ts_alpha if ts_alpha is not None else 0.1)
        )

        # Handle default cluster_ids (range(5) is the default, so we check if it's explicitly None)
        # Note: range(5) is truthy, so we can't use `if not cluster_ids`
        # The timeseries function will handle the default properly

        # Call timeseries with map=True and subplots=True
        return self.timeseries(
            var=var,
            cluster_ids=cluster_ids,
            plot_var=timeseries_var,
            color=ts_color,
            cmap=cmap,
            normalize=ts_normalize,
            add_legend=ts_add_legend,
            plot_individual=ts_plot_individual,
            max_trajectories=ts_max_trajectories,
            individual_alpha=individual_alpha_val,
            full_timeseries=ts_full_timeseries,
            cluster_highlight_color=ts_cluster_highlight_color,
            cluster_highlight_alpha=ts_cluster_highlight_alpha,
            cluster_highlight_linewidth=ts_cluster_highlight_linewidth,
            plot_median=ts_plot_median,
            plot_mean=ts_plot_mean,
            median_linewidth=ts_median_linewidth,
            mean_linewidth=ts_mean_linewidth,
            plot_range=ts_plot_range,
            plot_std_range=ts_plot_std_range,
            range_alpha=ts_range_alpha,
            std_alpha=ts_std_alpha,
            plot_shift_duration=ts_plot_shift_duration,
            shift_duration_color=ts_shift_duration_color,
            shift_duration_alpha=ts_shift_duration_alpha,
            subplots=True,  # Always use subplots for cluster_overview
            n_subplots_col=n_timeseries_col,
            map=True,
            map_var=map_var,
            plot_shifts=plot_shifts,
            map_projection=projection,
            map_add_contour=map_add_contour,
            map_only_contour=map_only_contour,
            map_add_labels=map_add_labels,
            map_remaining_clusters_cmap=map_remaining_clusters_cmap,
            map_remaining_clusters_legend_pos=map_remaining_clusters_legend_pos,
            plot_all_clusters_on_map=plot_all_clusters_on_map,
            vertical=vertical,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            figsize=figsize,
            wspace=wspace,
            hspace=hspace,
            timeseries_ylabel=timeseries_ylabel,
            **kwargs,
        )

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
        fontsize: int = 8,
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
            fontsize: Font size for legend text. Defaults to 8.
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
                # legend_pos = self.find_optimal_legend_position(ax, var, legend_size)
                legend_pos = (
                    0.01,
                    -0.07,
                )  # TODO p2 find more robust way to place legend
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
                label_text = f"#{start}-{end}"

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
