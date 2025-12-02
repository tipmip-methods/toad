from dataclasses import dataclass
from typing import Any, List, Literal, Optional, Tuple, Union, cast, overload

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap, to_hex, to_rgb, to_rgba
from matplotlib.patches import Rectangle

from toad.utils import _attrs, detect_latlon_names, is_regular_grid

__all__ = ["Plotter", "MapStyle"]

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


def get_projection(projection: str | ccrs.Projection) -> ccrs.Projection:
    """Get a cartopy projection object from a string name or return the projection.

    Args:
        projection: Either a string name of a projection (e.g., "plate_carree", "north_pole")
            or a cartopy Projection object. Valid string names are: "plate_carree",
            "north_pole", "north_polar_stereo", "south_pole", "south_polar_stereo",
            "global", "robinson", "mollweide".

    Returns:
        A cartopy Projection object.

    Raises:
        ValueError: If projection is a string but not a valid projection name.
        TypeError: If projection is neither a string nor a Projection object.
    """
    if isinstance(projection, str):
        if projection not in _projection_map:
            raise ValueError(
                f"Invalid projection: {projection}. Please choose between {list(_projection_map.keys())} or provide a ccrs.Projection object."
            )
        return _projection_map[projection]
    elif isinstance(projection, ccrs.Projection):
        return projection
    else:
        raise TypeError(f"Invalid projection: {projection}")


default_cmap = "tab20b"


@dataclass
class MapStyle:
    """Configuration for map styling parameters.

    This dataclass contains all the configuration options for styling maps
    with Plotter, including coastline, grid, and projection settings.
    """

    resolution: Literal["110m", "50m", "10m"] | str = "110m"
    coastline_linewidth: float = 0.5
    border_linewidth: float = 0.25
    grid_labels: bool = False
    grid_lines: bool = True
    grid_style: str = "--"
    grid_width: float = 0.5
    grid_color: str = "gray"
    grid_alpha: float = 0.5
    borders: bool = True
    projection: Optional[str | ccrs.Projection] = (
        None  # if lat/lon PlateCarree is used by default
    )
    extent: Optional[Tuple[float, float, float, float]] = None
    map_frame: bool = True
    continent_shading: bool = False
    continent_shading_color: str = "lightgray"
    ocean_shading: bool = False
    ocean_shading_color: str = "lightgray"

    # Cluster map visualization options
    plot_contour: bool = True
    plot_fill: bool = True
    add_labels: bool = True
    contour_linewidth: float = 1.5


def _normalize_map_style(
    map_style: Optional[Union[MapStyle, dict]] = None,
) -> MapStyle:
    """Normalize map_style to MapStyle.

    Args:
        map_style: Either a MapStyle object, a dict with MapStyle fields, or None.

    Returns:
        MapStyle object. If map_style is None, returns default MapStyle().
        If map_style is a dict, creates MapStyle from it (missing keys use defaults).
    """
    if map_style is None:
        return MapStyle()
    elif isinstance(map_style, MapStyle):
        return map_style
    elif isinstance(map_style, dict):
        # Create MapStyle from dict, missing keys will use defaults
        return MapStyle(**map_style)
    else:
        raise TypeError(
            f"map_style must be MapStyle, dict, or None, got {type(map_style)}"
        )


class Plotter:
    """Plotting utilities for TOAD objects.

    The Plotter class provides methods for creating publication-ready visualizations
    of TOAD data, including maps, timeseries, and statistical plots.

    Args:
        td: TOAD object containing the data to plot
    """

    def __init__(self, td):
        from toad import TOAD

        self.td: TOAD = td

    # Overloads are used for type hinting
    @overload
    def map(
        self,
        nrows: Literal[1] = 1,
        ncols: Literal[1] = 1,
        *,
        figsize: Optional[Tuple[float, float]] = None,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        subplot_spec: Any = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
    ) -> Tuple[matplotlib.figure.Figure, Axes]: ...

    @overload
    def map(
        self,
        nrows: int,
        ncols: int = 1,
        *,
        figsize: Optional[Tuple[float, float]] = None,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        subplot_spec: Any = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
    ) -> Tuple[matplotlib.figure.Figure, np.ndarray]: ...

    @overload
    def map(
        self,
        nrows: int,
        ncols: int,
        *,
        figsize: Optional[Tuple[float, float]] = None,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        subplot_spec: Any = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
    ) -> Tuple[matplotlib.figure.Figure, Axes]: ...

    def map(
        self,
        nrows: int = 1,
        ncols: int = 1,
        *,
        figsize: Optional[Tuple[float, float]] = None,
        height_ratios: Optional[List[float]] = None,
        width_ratios: Optional[List[float]] = None,
        subplot_spec: Optional[Any] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
    ) -> Tuple[matplotlib.figure.Figure, Union[Axes, np.ndarray]]:
        """Create map plots with standard features.

        Args:
            nrows: Number of rows in subplot grid
            ncols: Number of columns in subplot grid
            figsize: Figure size (width, height) in inches. If None, not set (matplotlib default).
            height_ratios: List of height ratios for subplots
            width_ratios: List of width ratios for subplots
            subplot_spec: A gridspec subplot spec to place the map in
            map_style: Map style configuration. Can be a MapStyle object, a dict with
                     MapStyle fields, or None (uses defaults). If dict, missing keys use defaults.
                     The projection is set via map_style.projection.

        Returns:
            Tuple of (figure, axes). If nrows=1 and ncols=1, returns a single Axes.
            Otherwise, returns a numpy array of axes.
        """
        # Normalize map_style to MapStyle
        config = _normalize_map_style(map_style)

        # Check if data has lat/lon coordinates
        lat_name, lon_name = detect_latlon_names(self.td.data)
        has_latlon = lat_name is not None and lon_name is not None

        # Determine if we should use a projection
        projection_obj = None

        if config.projection is None:
            # Default to PlateCarree for lat/lon if projection not specified
            if has_latlon:
                projection_obj = get_projection("plate_carree")
        else:
            projection_obj = get_projection(config.projection)

        if subplot_spec is not None:
            # Create map in existing figure using subplot_spec
            fig = plt.gcf()
            ax = fig.add_subplot(subplot_spec, projection=projection_obj)
            axs = ax
        else:
            # Create new figure with subplots
            gridspec_kw = {}
            if height_ratios:
                gridspec_kw["height_ratios"] = height_ratios
            if width_ratios:
                gridspec_kw["width_ratios"] = width_ratios

            subplot_kw = {"projection": projection_obj}

            fig, axs = plt.subplots(
                nrows,
                ncols,
                figsize=figsize,
                subplot_kw=subplot_kw,
                gridspec_kw=gridspec_kw if gridspec_kw else None,
            )

        # Ensure axs is always an array for consistent iteration
        axs_array = np.array(axs, ndmin=2)

        # Add map features and set extent for all axes (only if using projection)
        for ax in axs_array.flat:
            if hasattr(ax, "projection"):
                _add_map_features(ax, config)

                # Set extent if not specified
                if config.extent is None:
                    if ax.projection == ccrs.SouthPolarStereo():
                        ax.set_extent([-180, 180, -90, -65], crs=ccrs.PlateCarree())
                    elif ax.projection == ccrs.NorthPolarStereo():
                        ax.set_extent([-180, 180, 65, 90], crs=ccrs.PlateCarree())

            # toggle frame on/off
            ax.set_frame_on(config.map_frame)

        # Return single axis or array
        if axs_array.size == 1:
            return fig, axs_array[0, 0]
        else:
            return fig, np.squeeze(axs_array)

    @overload
    def cluster_map(
        self,
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = range(9),
        *,
        ax: Optional[Axes] = None,
        color: Optional[Union[str, Tuple, List[Union[str, Tuple]]]] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        map_cmap_other: Optional[Union[str, Colormap]] = "jet",
        include_all_clusters: bool = True,
        subplots: Literal[False] = False,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[matplotlib.figure.Figure], Axes]: ...

    @overload
    def cluster_map(
        self,
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = range(9),
        *,
        ax: Optional[Axes] = None,
        color: Optional[Union[str, Tuple, List[Union[str, Tuple]]]] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        map_cmap_other: Optional[Union[str, Colormap]] = "jet",
        include_all_clusters: bool = True,
        subplots: Literal[True] = True,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[matplotlib.figure.Figure], np.ndarray]: ...

    def cluster_map(
        self,
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = range(9),
        *,
        ax: Optional[Axes] = None,
        color: Optional[Union[str, Tuple, List[Union[str, Tuple]]]] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        map_cmap_other: Optional[Union[str, Colormap]] = "jet",
        include_all_clusters: bool = True,
        subplots: bool = False,
        ncols: int = 3,
        figsize: Optional[Tuple[float, float]] = None,
        map_style: Optional[Union[MapStyle, dict]] = None,
        **kwargs: Any,
    ) -> Tuple[Optional[matplotlib.figure.Figure], Union[Axes, np.ndarray]]:
        """Plot one or multiple clusters on a map.

        Args:
            var: Base variable name (e.g. 'temperature', will look for
                        'temperature_cluster') or custom cluster variable name. If None, TOAD will attempt to infer which variable to use.
                A ValueError is raised if the variable cannot be uniquely determined.
            cluster_ids: Single cluster ID or list of cluster IDs to plot.
                         Defaults to range(9) (clusters 0-8) if not provided.
            map_style: Map style configuration. Can be a MapStyle object, a dict with
                     MapStyle fields, or None (uses defaults). If dict, missing keys use defaults.
                     Controls projection, grid, borders, and cluster visualization options
                     (plot_contour, plot_fill, add_labels, contour_linewidth).
            ax: Matplotlib axes to plot on. Creates new figure if None. Cannot be used with subplots=True.
            color: Color for cluster visualization. Can be:
                - A single color (str, hex, RGB tuple) to use for all clusters.
                - A list of colors to use for each cluster. Overrides cmap.
            cmap: Colormap for multiple clusters. Used only if color is None.
            map_cmap_other: Colormap for remaining clusters. Can be:
                - A string (e.g., "jet", "viridis") to use a built-in colormap.
                - A matplotlib colormap object.
            include_all_clusters: If True, plot all clusters on the map. If False, only plot selected clusters.
                Defaults to True.
            subplots: If True, plot each cluster on its own subplot. Defaults to False.
            figsize: Figure size when subplots=True. Defaults to (12, 3 * nrows).
            ncols: Number of columns in subplot grid when subplots=True. Defaults to 3.
            **kwargs: Additional arguments passed to xarray.plot methods
                      (e.g., `plot`, `plot.contour`).

        Returns:
            Tuple of (figure, axes). Figure is None if ax was provided.
            When subplots=True, axes is a numpy array of axes.

        Raises:
            ValueError: If no clusters found for given variable, or if ax is provided with subplots=True.
            TypeError: If `cluster_ids` is not an int, list, ndarray, range, or None,
                       or if `cmap` is not a string or ListedColormap.
        """
        # Normalize map_style to MapStyle
        config = _normalize_map_style(map_style)

        # Get cluster visualization options from map_style
        plot_contour = config.plot_contour
        plot_fill = config.plot_fill
        add_labels = config.add_labels
        contour_linewidth = config.contour_linewidth

        assert plot_fill or plot_contour, (
            "plot_fill and plot_contour cannot both be False"
        )

        # plot_contour is not supported on irregular grids
        if plot_contour and not is_regular_grid(self.td.data):
            raise ValueError(
                "plot_contour is not supported on irregular grids. Set plot_contour=False or use a regular grid."
            )

        var = self.td._get_base_var_if_none(var)
        # get_clusters raises ValueError if no clusters found, so no need to check for None
        clusters_obj = self.td.get_clusters(var)

        # Check for incompatible parameters
        if subplots and ax is not None:
            raise ValueError(
                "Cannot use ax parameter with subplots=True. Set ax=None when using subplots."
            )

        # Plot all clusters (except -1) if no clusters passed
        all_cluster_ids = clusters_obj.cluster_ids
        cluster_ids = (
            cluster_ids
            if cluster_ids is not None
            else all_cluster_ids[all_cluster_ids != -1]
        )

        # Check that we have a valid clusters value
        if not isinstance(cluster_ids, (int, list, np.ndarray, range)):
            raise TypeError("clusters must be int, list, np.ndarray, range, or None")

        # Convert single cluster_id to list for consistent handling
        if isinstance(cluster_ids, int):
            single_plot = True
            cluster_ids = [cluster_ids]
        else:
            single_plot = False
            cluster_ids = list(cluster_ids)  # Convert to list for consistent indexing

        # Filter out cluster IDs that don't exist
        valid_cluster_ids = [id for id in cluster_ids if id in all_cluster_ids]
        if len(valid_cluster_ids) == 0:
            raise ValueError(f"No valid clusters found in clusters for variable {var}")

        # Setup subplots if requested
        if subplots:
            n_clusters = len(valid_cluster_ids)
            nrows = int(np.ceil(n_clusters / ncols))
            if figsize is None:
                figsize = (12, 3 * nrows)
            fig, axs = self.map(
                nrows=nrows,
                ncols=ncols,
                figsize=figsize,
                map_style=config,
            )
            # Ensure axs is always an array for consistent iteration
            axs_array: Optional[np.ndarray] = np.array(axs, ndmin=2)
        else:
            if ax is None:
                fig, ax = self.map(figsize=figsize, map_style=config)
            else:
                fig = None
            axs_array = None

        # Create color list for each cluster (based on valid_cluster_ids)
        n_valid = len(valid_cluster_ids)
        if color is not None:
            # If color is a list, use it directly (one color per cluster)
            if (
                isinstance(color, (list, tuple))
                and len(color) > 1
                and not all(isinstance(c, (int, float)) for c in color)
            ):
                color_list = color
                if len(color_list) < n_valid:
                    # Repeat colors if needed
                    color_list = color_list * (n_valid // len(color_list) + 1)
                color_list = color_list[
                    :n_valid
                ]  # Trim to match valid_cluster_ids length
            else:
                # Single color for all clusters
                color_list = [color] * n_valid
        else:
            # Use colormap to generate colors
            if isinstance(cmap, str):
                base_cmap = plt.get_cmap(cmap)
                color_list = [base_cmap(i) for i in np.linspace(0, 1, n_valid)]
            elif isinstance(cmap, ListedColormap):
                # Extract colors from the ListedColormap
                cmap_colors: list = cmap.colors  # type: ignore
                # Repeat colors if needed
                if len(cmap_colors) < n_valid:
                    cmap_colors = cmap_colors * (n_valid // len(cmap_colors) + 1)
                color_list = cmap_colors[:n_valid]

        # Create a ListedColormap for each cluster
        cmap_list = [ListedColormap([c]) for c in color_list]

        for i, id in enumerate(valid_cluster_ids):
            # Select the appropriate axis for this cluster
            if subplots:
                # Calculate subplot index
                row = i // ncols
                col = i % ncols
                current_ax: Axes = axs_array[row, col]  # type: ignore
            else:
                # ax is guaranteed to be set at this point (created if None)
                assert ax is not None, "ax should be set when subplots=False"
                current_ax = ax

            # Get the colormap for this cluster
            cluster_cmap = cmap_list[i]

            # Get mask for clustered or unclustered cells
            mask = (
                self.td.get_cluster_mask_permanent_noise(var)
                if id == -1
                else self.td.get_cluster_mask_spatial(var, id)
            )

            # prepare common plot parameters
            plot_params = {
                "ax": current_ax,
                "cmap": cluster_cmap,
                "add_colorbar": False,
                "alpha": 0.75,
                **kwargs,
            }

            lat_name, lon_name = detect_latlon_names(self.td.data)
            has_latlon = lat_name is not None and lon_name is not None

            # Check if axes is a GeoAxes (has projection)
            projection_attr = getattr(current_ax, "projection", None)
            is_geoaxes = projection_attr is not None

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

            if plot_fill:
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
            if plot_contour and is_regular_grid(self.td.data):
                # Make contour color darker (use color_list directly to avoid type issues)
                contour_color = cast(Any, color_list[i])
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
                    linewidths=contour_linewidth,
                    **plot_params,
                )

            if add_labels:
                # returns space_dims[0, 1], so y, x or lon, lat
                # Uses the point furthest from the cluster edge for robust labeling
                y, x = self.td.stats(var).space.central_point_for_labeling(id)
                if np.isnan(x) or np.isnan(y):
                    # Get median coordinates as fallback
                    y, x = self.td.stats(var).space.footprint_median(id)

                if not (np.isnan(x) or np.isnan(y)):
                    # Use color_list directly to avoid type issues with cluster_cmap.colors
                    # Cast to Any since color_list[i] can be str or tuple (from colormap)
                    cluster_color = cast(Any, color_list[i])
                    _cluster_annotate(
                        current_ax,
                        x,
                        y,
                        id,
                        cluster_color,  # type: ignore[arg-type]
                        transform=plot_params.get("transform"),
                    )  # type: ignore
                else:
                    print(
                        f"Warning: Could not find valid label position for cluster {id}"
                    )

            # Set title for subplots or single plot
            if subplots:
                current_ax.set_title(f"{var}_cluster {id}")
            elif single_plot:
                current_ax.set_title(f"{var}_cluster {id}")

        # Plot remaining clusters (only when not using subplots)
        if map_cmap_other and not subplots and include_all_clusters:
            # ax is guaranteed to be set at this point when subplots=False
            assert ax is not None, "ax should be set when subplots=False"
            remaining_cluster_ids = [  # get unplotted clusters ids (except -1)
                int(id)
                for id in all_cluster_ids
                if id not in valid_cluster_ids and id != -1
            ]
            if len(remaining_cluster_ids) > 0:
                mask = self.td.get_cluster_mask(var, remaining_cluster_ids)
                cl = self.td.get_clusters(var).where(mask)

                plot_params["cmap"] = map_cmap_other
                plot_params["ax"] = ax  # Use the single ax for remaining clusters
                cl.max(dim=self.td.time_dim).plot(
                    **plot_params,
                )  # type: ignore

                # Pass the colormap to the legend function
                _add_gradient_legend(
                    ax,
                    remaining_cluster_ids[0],
                    remaining_cluster_ids[-1],
                    var=var,
                    cmap=plt.get_cmap(map_cmap_other)
                    if isinstance(map_cmap_other, str)
                    else map_cmap_other,
                )

        # Return appropriate axes based on subplots setting
        if subplots:
            assert axs_array is not None, "axs_array should be set when subplots=True"
            if axs_array.size > 1:
                return fig, np.squeeze(axs_array)  # type: ignore
            else:
                return fig, axs_array[0, 0]  # type: ignore
        else:
            assert ax is not None, "ax should be set when subplots=False"
        return fig, ax

    def timeseries(
        self,
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = None,
        *,
        timeseries_var: Optional[str] = None,
        ax: Optional[Axes] = None,
        color: Optional[str] = None,
        cmap: Union[str, ListedColormap] = default_cmap,
        normalize: Optional[Literal["max", "max_each"]] | str = None,
        add_legend: bool = True,
        # Individual trajectories
        plot_trajectories: bool = True,
        max_trajectories: int = 1_000,
        trajectories_sample_seed=0,
        trajectories_alpha: float = 0.5,
        trajectories_linewidth: float = 0.5,
        full_timeseries: bool = True,
        highlight_color: Optional[str] = None,
        highlight_alpha: float = 0.5,
        highlight_linewidth: float = 0.5,
        plot_dts: bool = False,  # If True, plot shifts variable in timeseries
        # Aggregated statistics
        plot_median: bool = False,
        plot_mean: bool = False,
        median_linewidth: float = 3,
        mean_linewidth: float = 3,
        # Shaded regions
        plot_trajectory_range: bool = False,  # Full range (min to max)
        plot_trajectory_std: bool = False,  # 68% interquartile range (16th to 84th percentile)
        trajectory_range_alpha: float = 0.2,
        trajectory_std_alpha: float = 0.4,
        # Shift duration
        plot_shift_indicator: bool = True,
        shift_indicator_color: Optional[str] = None,  # Uses cluster color if None
        shift_indicator_alpha: float = 0.25,
        # Map options
        plot_map: bool = False,
        map_var: Optional[str] = None,
        map_cmap_other: Optional[Union[str, Colormap]] = "jet",
        map_include_all_clusters: bool = True,
        # Subplot layout
        subplots: bool = False,  # If True, create one subplot per cluster
        ncols: int = 1,  # Number of columns for subplot grid
        figsize: Optional[Tuple[float, float]] = None,
        wspace: float = 0.1,
        hspace: float = 0.1,
        show_ylabels: bool = False,  # Only relevant for subplots
        vertical: bool = False,  # Only relevant when plot_map=True
        width_ratios: Tuple[float, float] = (
            1.0,
            1.0,
        ),  # Only relevant when plot_map=True
        height_ratios: Optional[
            Tuple[float, float]
        ] = None,  # Only relevant when plot_map=True
        map_style: Optional[Union[MapStyle, dict]] = None,
        **plot_kwargs: Any,
    ) -> Tuple[Optional[matplotlib.figure.Figure], Union[Axes, List[Axes], dict]]:
        """Plot time series from clusters or all data.

        This function allows flexible plotting of individual trajectories, aggregated statistics
        (median/mean), shaded regions (full range and IQR), and shift duration indicators.
        If no clusters are provided, plots all timeseries from the dataset.

        Can optionally create separate subplots for each cluster, and optionally include a map
        showing cluster spatial locations alongside the timeseries.

        Args:
            var: Base variable or cluster variable. If None, TOAD will attempt
                to infer which variable to use. A ValueError is raised if the variable cannot be
                uniquely determined.
            cluster_ids: ID or list of IDs of clusters to plot. If None, plots all timeseries
                from the dataset (no clustering). Cannot be None if plot_map=True.
            timeseries_var: Variable name to plot (if different from var). Defaults to var.
            ax: Matplotlib axes to plot on. Creates new figure if None. Ignored if subplots=True
                or plot_map=True.
            color: Single color to use for all plotted clusters. Overrides cmap.
            cmap: Colormap to use if plotting multiple clusters and color is None.
            normalize: Method to normalize timeseries ('max', 'max_each'). Defaults to None.
            add_legend: If True, add a legend indicating cluster IDs.
            plot_trajectories: If True, plot individual cell trajectories.
            max_trajectories: Maximum number of individual trajectories to plot (per cluster if
                clusters provided, or total if plotting all data).
            trajectories_sample_seed: Seed for the random number generator used to sample trajectories. Defaults to 0.
            trajectories_alpha: Alpha transparency for individual time series lines. Defaults to 0.5.
            trajectories_linewidth: Linewidth for individual time series lines. Defaults to 0.5.
            full_timeseries: If True, plot the full timeseries for each cell. If False,
                only plot the segment belonging to the cluster.
            highlight_color: Color to highlight the actual cluster segment
                when full_timeseries is True.
            highlight_alpha: Alpha for the cluster highlight segment.
            highlight_linewidth: Line width for the cluster highlight segment.
            plot_dts: If True, plot shifts variable in timeseries instead of base variable.
            plot_median: If True, plot the median timeseries curve.
            plot_mean: If True, plot the mean timeseries curve.
            median_linewidth: Linewidth for the median curve.
            mean_linewidth: Linewidth for the mean curve.
            plot_trajectory_range: If True, plot the full range (min to max) as a shaded area.
            plot_trajectory_std: If True, plot the 68% interquartile range (16th to 84th percentile) as a shaded area.
            trajectory_range_alpha: Alpha transparency for the full range shaded area.
            trajectory_std_alpha: Alpha transparency for the IQR shaded area.
            plot_shift_indicator: If True, adds horizontal shading indicating the cluster's
                temporal extent (start to end). Only applies when clusters are provided.
            shift_indicator_color: Color for shift duration shading. Uses cluster color if None.
            shift_indicator_alpha: Alpha for the shift duration shading.
            plot_map: If True, include a map showing cluster spatial locations alongside timeseries.
                Defaults to False.
            map_var: Variable name whose data to plot in the map. Defaults to var if None. Only used if plot_map=True.
            map_cmap_other: Colormap for remaining clusters on map. Only used if plot_map=True.
            map_include_all_clusters: If True, plot all clusters on the map. If False, only plot selected clusters.
                Only used if plot_map=True.
            subplots: If True, create one subplot per cluster. Defaults to False. If plot_map=True and multiple
                clusters, subplots are automatically enabled.
            ncols: Number of columns for subplot grid when subplots=True. Defaults to 1.
            figsize: Figure size (width, height) in inches. Used when subplots=True or plot_map=True.
            wspace: Width space between timeseries subplots (if ncols > 1).
            hspace: Height space between timeseries rows.
            show_ylabels: If True, show y-axis label on the timeseries plots. Only relevant for subplots.
            vertical: If True, arrange map above timeseries plots. Otherwise, map is placed to the left.
                Only used if plot_map=True.
            width_ratios: Tuple of relative widths for map vs. timeseries section (used in horizontal layout).
                Only used if plot_map=True.
            height_ratios: Optional tuple of relative heights for map vs. timeseries section (used in vertical layout).
                Only used if plot_map=True.
            **plot_kwargs: Additional arguments passed to xarray.plot for each trajectory.

        Returns:
            Tuple of (figure, axes).
            - If plot_map=False and single plot: (figure, Axes)
            - If plot_map=False and subplots=True: (figure, List[Axes])
            - If plot_map=True: (figure, dict) with keys 'map' and 'timeseries'
            Figure is None if ax was provided and subplots=False and plot_map=False.

        Raises:
            ValueError: If no timeseries found for a given cluster ID, if nothing is set to plot,
                if cluster_ids is None when plot_map=True, or if plotting all data when plot_map=True.
        """
        # Validate plot_map requirements
        if plot_map:
            if cluster_ids is None:
                raise ValueError(
                    "cluster_ids cannot be None when plot_map=True. Provide at least one cluster ID."
                )
            if ax is not None:
                raise ValueError(
                    "Cannot use ax parameter when plot_map=True. Set ax=None when using plot_map."
                )

        # Parse cluster IDs
        cluster_ids_list, single_plot, plot_all_data = self._parse_cluster_ids(
            cluster_ids, var
        )
        var = self.td._get_base_var_if_none(var)

        if plot_map and plot_all_data:
            raise ValueError(
                "Cannot plot map when cluster_ids is None (plotting all data). "
                "Set plot_map=False or provide cluster_ids."
            )

        if plot_map and map_var is None:
            map_var = var

        # Infer plot variable (pass map=plot_map)
        timeseries_var = self._infer_plot_var(
            var, timeseries_var, plot_dts, map=plot_map
        )

        # Validate that something will be plotted
        has_individual = plot_trajectories
        has_aggregate = (
            plot_median or plot_mean or plot_trajectory_range or plot_trajectory_std
        )
        if not has_individual and not has_aggregate:
            raise ValueError(
                "Nothing to plot: set at least one of plot_trajectories, plot_median, "
                "plot_mean, plot_trajectory_range, or plot_trajectory_std to True."
            )

        # Validate ncols
        if ncols <= 0:
            raise ValueError(f"ncols must be > 0, got {ncols}")

        # Determine if we need subplots
        # When plot_map=True and multiple clusters, automatically enable subplots
        if plot_map and len(cluster_ids_list) > 1:
            use_subplots = True
        else:
            use_subplots = subplots

        # Setup figure and axes layout
        fig, ts_axes_list, map_ax = self._setup_timeseries_axes(
            map=plot_map,
            use_subplots=use_subplots,
            cluster_ids_list=cluster_ids_list,
            n_subplots_col=ncols,
            figsize=figsize,
            vertical=vertical if plot_map else False,
            width_ratios=width_ratios if plot_map else (1.0, 1.0),
            height_ratios=height_ratios if plot_map else None,
            hspace=hspace,
            wspace=wspace,
            ax=ax if not plot_map else None,
            map_style=map_style,
        )

        # Get colors for clusters
        colors = self._assign_cluster_colors(
            cluster_ids_list, color, cmap, map=plot_map, use_subplots=use_subplots
        )

        # Plot map if requested
        if plot_map:
            assert map_ax is not None, "map_ax should be set when plot_map=True"
            self._plot_timeseries_map(
                map_var=map_var,
                cluster_ids_list=cluster_ids_list,
                map_ax=map_ax,
                colors=colors,
                color=color,
                map_include_all_clusters=map_include_all_clusters,
                map_cmap_other=map_cmap_other,
                map_style=map_style,
                **plot_kwargs,
            )

        # Single unified loop for both all-data and clustered plotting
        y_label = ""
        for i, id in enumerate(cluster_ids_list):
            # Get the axes for this cluster
            current_ax = ts_axes_list[i] if use_subplots else ts_axes_list[0]

            # Get color for this cluster
            id_color = self._get_cluster_color(i, cluster_ids_list, color, colors, cmap)

            # Use cluster color for shift duration if not specified
            shift_color = (
                shift_indicator_color if shift_indicator_color is not None else id_color
            )

            # Plot aggregated statistics first (so they appear behind individual trajectories)
            if plot_trajectory_range:
                self._plot_trajectory_range_band(
                    current_ax=current_ax,
                    plot_var=timeseries_var,
                    var=var,
                    cluster_id=id,
                    id_color=id_color,
                    range_alpha=trajectory_range_alpha,
                    normalize=normalize,
                    time_dim=self.td.time_dim,
                )

            if plot_trajectory_std:
                self._plot_iqr_band(
                    current_ax=current_ax,
                    plot_var=timeseries_var,
                    var=var,
                    cluster_id=id,
                    id_color=id_color,
                    iqr_alpha=trajectory_std_alpha,
                    normalize=normalize,
                    time_dim=self.td.time_dim,
                )

            if plot_mean:
                self._plot_mean_curve(
                    current_ax=current_ax,
                    plot_var=timeseries_var,
                    var=var,
                    cluster_id=id,
                    id_color=id_color,
                    mean_linewidth=mean_linewidth,
                    add_legend=add_legend,
                    normalize=normalize,
                )

            if plot_median:
                self._plot_median_curve(
                    current_ax=current_ax,
                    plot_var=timeseries_var,
                    var=var,
                    cluster_id=id,
                    id_color=id_color,
                    median_linewidth=median_linewidth,
                    add_legend=add_legend,
                    normalize=normalize,
                )

            # Plot shift duration (horizontal shading) - only for real clusters
            if plot_shift_indicator and id is not None:
                self._plot_shift_indicator(
                    current_ax=current_ax,
                    var=var,
                    cluster_id=id,
                    shift_color=shift_color,
                    shift_indicator_alpha=shift_indicator_alpha,
                )

            # Plot individual trajectories
            cells = None
            if plot_trajectories:
                cells = self._plot_individual_trajectories(
                    current_ax=current_ax,
                    plot_var=timeseries_var,
                    var=var,
                    cluster_id=id,
                    id_color=id_color,
                    trajectory_alpha=trajectories_alpha,
                    trajectory_linewidth=trajectories_linewidth,
                    max_trajectories=max_trajectories,
                    trajectories_sample_seed=trajectories_sample_seed,
                    full_timeseries=full_timeseries,
                    normalize=normalize,
                    add_legend=add_legend,
                    use_subplots=use_subplots,
                    **plot_kwargs,
                )

                if highlight_color and id is not None:
                    self._highlight_cluster_segments(
                        current_ax=current_ax,
                        plot_var=timeseries_var,
                        var=var,
                        cluster_id=id,
                        highlight_color=highlight_color,
                        highlight_alpha=highlight_alpha,
                        highlight_linewidth=highlight_linewidth,
                        full_timeseries=full_timeseries,
                        normalize=normalize,
                        cells=cells,
                    )

            # Handle axis cleanup for subplots
            if use_subplots:
                y_label = self._cleanup_subplot_axes(
                    current_ax=current_ax,
                    i=i,
                    cluster_ids_list=cluster_ids_list,
                    n_subplots_col=ncols,
                    timeseries_ylabel=show_ylabels,
                )

            # Handle legend
            self._apply_legend(
                current_ax=current_ax,
                cluster_id=id,
                add_legend=add_legend,
                use_subplots=use_subplots,
                i=i,
                cluster_ids_list=cluster_ids_list,
            )

        # Set title
        self._set_timeseries_title(
            ts_axes_list=ts_axes_list,
            map=plot_map,
            use_subplots=use_subplots,
            plot_all_data=plot_all_data,
            cluster_ids_list=cluster_ids_list,
            plot_var=timeseries_var,
            var=var,
            plot_individual=plot_trajectories,
            has_aggregate=has_aggregate,
            single_plot=single_plot,
            max_trajectories=max_trajectories,
            full_timeseries=full_timeseries,
            normalize=normalize,
            y_label=y_label,
        )

        # Return appropriate values using helper function
        return self._package_timeseries_result(
            fig=fig,
            map=plot_map,
            use_subplots=use_subplots,
            map_ax=map_ax,
            ts_axes_list=ts_axes_list,
        )

    def overview(
        self,
        var: str | None = None,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]] = range(6),
        map_style: Optional[Union[MapStyle, dict]] = None,
        mode: Literal["timeseries", "aggregated"] = "timeseries",
        **kwargs: Any,
    ) -> Tuple[Optional[matplotlib.figure.Figure], dict]:
        """Create an overview plot with map and timeseries for clusters.

        This is a convenience method that creates a combined visualization showing
        both the spatial distribution of clusters on a map and their corresponding
        timeseries. It automatically enables subplots and map display.

        Args:
            var: Base variable or cluster variable. If None, TOAD will attempt
                to infer which variable to use. A ValueError is raised if the variable cannot be
                uniquely determined.
            cluster_ids: ID or list of IDs of clusters to plot. Defaults to range(6) (clusters 0-5).
            map_style: Map style configuration. Can be a MapStyle object, a dict with
                MapStyle fields, or None (uses defaults). If dict, missing keys use defaults.
            mode: Visualization mode. "timeseries" shows individual trajectories,
                "aggregated" shows statistical summaries (median, range, IQR).
            **kwargs: Additional arguments passed to timeseries() method.

        Returns:
            Tuple of (figure, dict) with keys 'map' and 'timeseries'.
            - 'map': Axes for the map plot
            - 'timeseries': List of axes for timeseries subplots (one per cluster)
        """
        result = self.timeseries(
            var=var,
            cluster_ids=cluster_ids,
            plot_map=True,
            subplots=True,
            map_style=map_style,
            plot_trajectory_std=mode == "aggregated",
            plot_trajectories=mode == "timeseries",
            plot_trajectory_range=mode == "aggregated",
            plot_median=mode == "aggregated",
            **kwargs,
        )
        return cast(Tuple[Optional[matplotlib.figure.Figure], dict], result)

    def shifts_distribution(
        self, figsize: Optional[tuple] = None, yscale: str = "log", bins=20
    ):
        """Plot histograms showing the distribution of shifts for each shift variable.

        Args:
            figsize: Figure size (width, height) in inches. If None, defaults to
                (12, 2 * number of shift variables).
            yscale: Scale for the y-axis. Defaults to "log".
            bins: Number of bins for the histogram. Defaults to 20.

        Returns:
            Tuple of (figure, axes). Axes is a numpy array of axes (one per shift variable).
        """

        if figsize is None:
            figsize = (12, 2 * len(self.td.shift_vars))

        fig, axs = plt.subplots(nrows=len(self.td.shift_vars), figsize=figsize)
        if not isinstance(axs, np.ndarray):
            axs = np.array([axs])

        if len(axs) > 1:
            _remove_ticks(axs[:-1], keep_y=True)
            _remove_spines(axs[:-1], spines=["right", "top"])

        _remove_spines(axs[-1], spines=["right", "top"])

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

    def _parse_cluster_ids(
        self,
        cluster_ids: Optional[Union[int, List[int], np.ndarray, range]],
        var: Optional[str],
    ) -> Tuple[List[Optional[int]], bool, bool]:
        """Parse and validate cluster_ids input.

        Args:
            cluster_ids: Cluster IDs to parse
            var: Variable name for clusters

        Returns:
            Tuple of (cluster_ids_list, single_plot, plot_all_data)
        """
        plot_all_data = cluster_ids is None

        if plot_all_data:
            # Treat as single pseudo-cluster with id=None
            cluster_ids_list: List[Optional[int]] = [None]
            single_plot = True
            var = self.td._get_base_var_if_none(var)
        else:
            # Filter cluster_ids to only include existing clusters
            var = self.td._get_base_var_if_none(var)
            cluster_ids = _filter_by_existing_clusters(self.td, cluster_ids, var)

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

        return cluster_ids_list, single_plot, plot_all_data

    def _infer_plot_var(
        self,
        var: Optional[str],
        plot_var: Optional[str],
        plot_shifts: bool,
        map: bool,
    ) -> str:
        """Infer the plot variable for timeseries.

        Args:
            var: Base variable name
            plot_var: Explicitly provided plot variable
            plot_shifts: Whether to plot shifts variable
            map: Whether map is being plotted

        Returns:
            The inferred plot variable name
        """
        # Determine plot_var for timeseries
        if plot_var is None:
            plot_var = var
        plot_var = self.td._get_base_var_if_none(plot_var)

        # Handle map setup and determine plot_var for timeseries when map=True
        if map and var is not None:
            # Get base variable from clusters attrs for timeseries if plot_var wasn't explicitly set
            # (i.e., if it equals var, meaning user didn't specify a different variable)
            clusters_obj = self.td.get_clusters(var)
            if plot_var == var:
                plot_var = clusters_obj.attrs[_attrs.BASE_VARIABLE]
            if plot_shifts:
                plot_var = clusters_obj.attrs[_attrs.SHIFTS_VARIABLE]

        if plot_var is None:
            raise ValueError("Failed to infer plot_var")
        return plot_var

    def _setup_timeseries_axes(
        self,
        map: bool,
        use_subplots: bool,
        cluster_ids_list: List[Optional[int]],
        n_subplots_col: int,
        figsize: Optional[Tuple[float, float]],
        vertical: bool,
        width_ratios: Tuple[float, float],
        height_ratios: Optional[Tuple[float, float]],
        hspace: float,
        wspace: float,
        ax: Optional[Axes],
        map_style: Optional[Union[MapStyle, dict]] = None,
    ) -> Tuple[
        Optional[matplotlib.figure.Figure],
        List[Axes],
        Optional[Axes],
    ]:
        """Setup figure and axes layout for timeseries plots.

        Args:
            map: Whether to include a map
            use_subplots: Whether to use subplots
            cluster_ids_list: List of cluster IDs to plot
            n_subplots_col: Number of columns for subplot grid
            figsize: Figure size
            vertical: Whether to arrange map vertically
            width_ratios: Width ratios for horizontal layout
            height_ratios: Height ratios for vertical layout
            hspace: Height space between subplots
            wspace: Width space between subplots
            ax: Optional existing axes
            map_style: Map style configuration

        Returns:
            Tuple of (figure, timeseries_axes_list, map_ax)
        """
        fig = None
        ts_axes_list: List[Axes] = []
        map_ax = None

        if map or use_subplots:
            # Create figure with constrained_layout
            if figsize is None:
                figsize = (12, 6)
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
                        map_style=map_style,
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
                        map_style=map_style,
                    )  # type: ignore
                    ts_subplot_spec = main_gs[0, 1]  # type: ignore

                # Create timeseries subplots in remaining space
                ts_axes_list = _create_timeseries_layout(
                    fig=fig,
                    n_clusters=len(cluster_ids_list),
                    n_subplots_col=n_subplots_col,
                    subplot_spec=ts_subplot_spec,
                    hspace=hspace,
                    wspace=wspace,
                )
            else:
                # Only subplots, no map
                ts_axes_list = _create_timeseries_layout(
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

        return fig, ts_axes_list, map_ax

    def _assign_cluster_colors(
        self,
        cluster_ids_list: List[Optional[int]],
        color: Optional[str],
        cmap: Union[str, ListedColormap],
        map: bool,
        use_subplots: bool,
    ) -> Optional[List[str]]:
        """Assign colors to clusters.

        Args:
            cluster_ids_list: List of cluster IDs
            color: Single color override
            cmap: Colormap to use
            map: Whether map is being plotted
            use_subplots: Whether subplots are being used

        Returns:
            List of colors or None if single color should be used
        """
        colors = None
        if map or (use_subplots and len(cluster_ids_list) > 1):
            colors = _get_cmap_seq(stops=len(cluster_ids_list), cmap=cmap)
        return colors

    def _get_cluster_color(
        self,
        i: int,
        cluster_ids_list: List[Optional[int]],
        color: Optional[str],
        colors: Optional[List[str]],
        cmap: Union[str, ListedColormap],
    ) -> str:
        """Get color for a specific cluster.

        Args:
            i: Index of cluster in cluster_ids_list
            cluster_ids_list: List of cluster IDs
            color: Single color override
            colors: Pre-computed color list
            cmap: Colormap to use

        Returns:
            Color string for the cluster
        """
        if color:
            return color
        else:
            if len(cluster_ids_list) == 1:
                return "black"
            else:
                if colors:
                    return colors[i]
                else:
                    return _get_cmap_seq(stops=len(cluster_ids_list), cmap=cmap)[i]

    def _plot_timeseries_map(
        self,
        map_var: Optional[str],
        cluster_ids_list: List[Optional[int]],
        map_ax: Axes,
        colors: Optional[List[str]],
        color: Optional[str],
        map_include_all_clusters: bool,
        map_cmap_other: Optional[Union[str, Colormap]],
        map_style: Optional[Union[MapStyle, dict]],
        **plot_kwargs: Any,
    ) -> None:
        """Plot map alongside timeseries.

        Args:
            map_var: Variable for map
            cluster_ids_list: List of cluster IDs to plot on map
            map_ax: Axes for map
            colors: Pre-computed color list
            color: Single color override
            map_include_all_clusters: Whether to plot all clusters
            map_cmap_other: Colormap for remaining clusters
            map_style: Map style configuration (controls plot_contour, plot_fill, add_labels)
            **plot_kwargs: Additional plot arguments
        """
        # Don't plot remaining clusters on map if not requested
        map_cmap_other = None if not map_include_all_clusters else map_cmap_other
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
            map_cmap_other=map_cmap_other,
            map_style=map_style,
            **plot_kwargs,
        )

    def _plot_trajectory_range_band(
        self,
        current_ax: Axes,
        plot_var: str,
        var: str,
        cluster_id: Optional[int],
        id_color: str,
        range_alpha: float,
        normalize: Optional[Literal["max", "max_each"]] | str,
        time_dim: str,
    ) -> None:
        """Plot full range (min to max) as shaded area.

        Args:
            current_ax: Axes to plot on
            plot_var: Variable to plot
            var: Base variable name
            cluster_id: Cluster ID (None for all data)
            id_color: Color for the band
            range_alpha: Alpha transparency
            normalize: Normalization method
            time_dim: Time dimension name
        """
        ts_kwargs = {
            "var": plot_var,
            "cluster_id": cluster_id,
            "normalize": normalize,
        }
        if cluster_id is not None:
            ts_kwargs["cluster_var"] = var

        min_ts = self.td.get_cluster_timeseries(
            aggregation="min",
            **ts_kwargs,
        )
        max_ts = self.td.get_cluster_timeseries(
            aggregation="max",
            **ts_kwargs,
        )
        current_ax.fill_between(
            self.td.data[time_dim].values,
            min_ts,
            max_ts,
            color=id_color,
            alpha=range_alpha,
            zorder=0,
        )

    def _plot_iqr_band(
        self,
        current_ax: Axes,
        plot_var: str,
        var: str,
        cluster_id: Optional[int],
        id_color: str,
        iqr_alpha: float,
        normalize: Optional[Literal["max", "max_each"]] | str,
        time_dim: str,
        percentile_lower: float = 0.16,
        percentile_upper: float = 0.84,
        fill_zorder: int = 1,
    ) -> None:
        """Plot interquartile range as shaded area between two percentiles.

        Args:
            current_ax: Axes to plot on
            plot_var: Variable to plot
            var: Base variable name
            cluster_id: Cluster ID (None for all data)
            id_color: Color for the band
            iqr_alpha: Alpha transparency
            normalize: Normalization method
            time_dim: Time dimension name
            percentile_lower: Lower percentile for band (default 0.16)
            percentile_upper: Upper percentile for band (default 0.84)
            fill_zorder: Z-order for fill_between (default 1)
        """
        ts_kwargs = {
            "var": plot_var,
            "cluster_id": cluster_id,
            "normalize": normalize,
        }
        if cluster_id is not None:
            ts_kwargs["cluster_var"] = var

        p_low_ts = self.td.get_cluster_timeseries(
            aggregation="percentile",
            percentile=percentile_lower,
            **ts_kwargs,
        )
        p_up_ts = self.td.get_cluster_timeseries(
            aggregation="percentile",
            percentile=percentile_upper,
            **ts_kwargs,
        )
        current_ax.fill_between(
            self.td.data[time_dim].values,
            p_low_ts,
            p_up_ts,
            color=id_color,
            alpha=iqr_alpha,
            zorder=fill_zorder,
        )

    def _plot_mean_curve(
        self,
        current_ax: Axes,
        plot_var: str,
        var: str,
        cluster_id: Optional[int],
        id_color: str,
        mean_linewidth: float,
        add_legend: bool,
        normalize: Optional[Literal["max", "max_each"]] | str,
    ) -> None:
        """Plot mean timeseries curve.

        Args:
            current_ax: Axes to plot on
            plot_var: Variable to plot
            var: Base variable name
            cluster_id: Cluster ID (None for all data)
            id_color: Color for the curve
            mean_linewidth: Line width
            add_legend: Whether to add legend
            normalize: Normalization method
        """
        ts_kwargs = {
            "var": plot_var,
            "cluster_id": cluster_id,
            "normalize": normalize,
        }
        if cluster_id is not None:
            ts_kwargs["cluster_var"] = var

        if cluster_id is None:
            label = "mean"
        else:
            label = f"#{cluster_id}"

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

    def _plot_median_curve(
        self,
        current_ax: Axes,
        plot_var: str,
        var: str,
        cluster_id: Optional[int],
        id_color: str,
        median_linewidth: float,
        add_legend: bool,
        normalize: Optional[Literal["max", "max_each"]] | str,
    ) -> None:
        """Plot median timeseries curve.

        Args:
            current_ax: Axes to plot on
            plot_var: Variable to plot
            var: Base variable name
            cluster_id: Cluster ID (None for all data)
            id_color: Color for the curve
            median_linewidth: Line width
            add_legend: Whether to add legend
            normalize: Normalization method
        """
        ts_kwargs = {
            "var": plot_var,
            "cluster_id": cluster_id,
            "normalize": normalize,
        }
        if cluster_id is not None:
            ts_kwargs["cluster_var"] = var

        if cluster_id is None:
            label = "median"
        else:
            label = f"#{cluster_id}"

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

    def _plot_shift_indicator(
        self,
        current_ax: Axes,
        var: str,
        cluster_id: int,
        shift_color: str,
        shift_indicator_alpha: float,
    ) -> None:
        """Plot shift duration as horizontal shading.

        Args:
            current_ax: Axes to plot on
            var: Base variable name
            cluster_id: Cluster ID
            shift_color: Color for shading
            shift_indicator_alpha: Alpha transparency
        """
        current_ax.axvspan(
            self.td.stats(var).time.start(cluster_id),
            self.td.stats(var).time.end(cluster_id),
            color=shift_color,
            alpha=shift_indicator_alpha,
            zorder=-100,
        )

    def _plot_individual_trajectories(
        self,
        current_ax: Axes,
        plot_var: str,
        var: str,
        cluster_id: Optional[int],
        id_color: str,
        trajectory_alpha: float,
        trajectory_linewidth: float,
        max_trajectories: int,
        trajectories_sample_seed: int,
        full_timeseries: bool,
        normalize: Optional[Literal["max", "max_each"]] | str,
        add_legend: bool,
        use_subplots: bool,
        **plot_kwargs: Any,
    ) -> Optional[Any]:
        """Plot individual cell trajectories.

        Args:
            current_ax: Axes to plot on
            plot_var: Variable to plot
            var: Base variable name
            cluster_id: Cluster ID (None for all data)
            id_color: Color for trajectories
            trajectory_alpha: Alpha transparency
            trajectory_linewidth: Line width
            max_trajectories: Maximum number of trajectories
            trajectories_sample_seed: Seed for random sampling
            full_timeseries: Whether to plot full timeseries
            normalize: Normalization method
            add_legend: Whether to add legend
            use_subplots: Whether using subplots
            **plot_kwargs: Additional plot arguments
        """
        is_real_cluster = cluster_id is not None

        individual_ts_kwargs = {
            "var": plot_var,
            "cluster_id": cluster_id,
            "normalize": normalize,
            "aggregation": "raw",
        }
        if is_real_cluster:
            individual_ts_kwargs["cluster_var"] = var
            individual_ts_kwargs["keep_full_timeseries"] = full_timeseries

        cells = self.td.get_cluster_timeseries(**individual_ts_kwargs)

        if cells is None:
            if is_real_cluster:
                raise ValueError(f"No timeseries found for cluster {cluster_id}")
            else:
                raise ValueError(f"No timeseries found for {plot_var}")

        # Limit the number of trajectories to plot
        max_trajectories_actual = np.min([max_trajectories, len(cells)])

        # Shuffle the cell to get a random sample
        order = np.arange(len(cells))
        np.random.seed(trajectories_sample_seed)
        np.random.shuffle(order)
        order = order[:max_trajectories_actual]

        for plot_idx, cell_idx in enumerate(order):
            if is_real_cluster:
                # Add label on first trajectory if legend is enabled
                # For single plot: add label on first trajectory of each cluster
                # For subplots: don't add label to line (we'll use ax.text instead)
                if add_legend and plot_idx == 0 and not use_subplots:
                    # Label each cluster (only for single plot, not subplots)
                    add_label = f"#{cluster_id}"
                else:
                    add_label = "__nolegend__"
            else:
                add_label = "__nolegend__"
            cells[cell_idx].plot(
                ax=current_ax,
                color=id_color,
                alpha=trajectory_alpha,
                lw=trajectory_linewidth,
                label=add_label,
                **plot_kwargs,
            )

        return cells

    def _highlight_cluster_segments(
        self,
        current_ax: Axes,
        plot_var: str,
        var: str,
        cluster_id: int,
        highlight_color: str,
        highlight_alpha: float,
        highlight_linewidth: float,
        full_timeseries: bool,
        normalize: Optional[Literal["max", "max_each"]] | str,
        cells: Optional[Any] = None,
    ) -> None:
        """Highlight cluster segments when full_timeseries is True.

        Args:
            current_ax: Axes to plot on
            plot_var: Variable to plot
            var: Base variable name
            cluster_id: Cluster ID
            highlight_color: Color for highlight
            highlight_alpha: Alpha transparency
            highlight_linewidth: Line width
            full_timeseries: Whether full timeseries was plotted
            normalize: Normalization method
            cells: Optional pre-fetched cells (when full_timeseries=False)
        """
        if not full_timeseries:
            # Reuse cells if already fetched with keep_full_timeseries=False
            if cells is not None:
                cells_highlight = cells
            else:
                return
        else:
            highlight_ts_kwargs = {
                "var": plot_var,
                "cluster_id": cluster_id,
                "cluster_var": var,
                "normalize": normalize,
                "aggregation": "raw",
                "keep_full_timeseries": False,
            }
            cells_highlight = self.td.get_cluster_timeseries(**highlight_ts_kwargs)

        for ts in cells_highlight:
            ts.plot(
                ax=current_ax,
                color=highlight_color,
                alpha=highlight_alpha,
                lw=highlight_linewidth,
            )

    def _cleanup_subplot_axes(
        self,
        current_ax: Axes,
        i: int,
        cluster_ids_list: List[Optional[int]],
        n_subplots_col: int,
        timeseries_ylabel: bool,
    ) -> str:
        """Clean up axes for subplots.

        Args:
            current_ax: Current axes
            i: Index of current subplot
            cluster_ids_list: List of cluster IDs
            n_subplots_col: Number of columns
            timeseries_ylabel: Whether to show y-label

        Returns:
            y_label string (empty if not first subplot or timeseries_ylabel is True)
        """
        y_label = ""
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
            _remove_spines(current_ax, ["right", "top", "bottom"])
        else:
            _remove_spines(current_ax, ["right", "top"])

        if not is_bottom_in_column:
            _remove_ticks(current_ax, keep_y=True)

        return y_label

    def _apply_legend(
        self,
        current_ax: Axes,
        cluster_id: Optional[int],
        add_legend: bool,
        use_subplots: bool,
        i: int,
        cluster_ids_list: List[Optional[int]],
    ) -> None:
        """Apply legend or cluster ID label to axes.

        Args:
            current_ax: Axes to add legend to
            cluster_id: Cluster ID (None for all data)
            add_legend: Whether to add legend
            use_subplots: Whether using subplots
            i: Index of current cluster
            cluster_ids_list: List of all cluster IDs
        """
        if not add_legend:
            return

        is_real_cluster = cluster_id is not None

        if use_subplots:
            # For subplots: use ax.text to add cluster ID label (no color needed)
            if is_real_cluster:
                # Position text in upper right corner using axes coordinates
                current_ax.text(
                    1.02,
                    1.02,
                    f"#{cluster_id}",
                    ha="right",
                    va="top",
                    transform=current_ax.transAxes,
                )
        else:
            # For single plot: use legend
            # Check if there are any labeled artists before calling legend()
            handles, labels = current_ax.get_legend_handles_labels()
            has_labels = any(label and not label.startswith("_") for label in labels)

            if has_labels:
                # Single plot: only show legend on the last iteration
                # Position in upper right corner
                if i == len(cluster_ids_list) - 1:
                    legend = current_ax.legend(frameon=False, loc="upper right")
                    for handle in legend.get_lines():
                        handle.set_alpha(1.0)

    def _set_timeseries_title(
        self,
        ts_axes_list: List[Axes],
        map: bool,
        use_subplots: bool,
        plot_all_data: bool,
        cluster_ids_list: List[Optional[int]],
        plot_var: str,
        var: str,
        plot_individual: bool,
        has_aggregate: bool,
        single_plot: bool,
        max_trajectories: int,
        full_timeseries: bool,
        normalize: Optional[Literal["max", "max_each"]] | str,
        y_label: str,
    ) -> None:
        """Set title for timeseries plots.

        Args:
            ts_axes_list: List of timeseries axes
            map: Whether map is included
            use_subplots: Whether using subplots
            plot_all_data: Whether plotting all data
            cluster_ids_list: List of cluster IDs
            plot_var: Variable being plotted
            var: Base variable name
            plot_individual: Whether plotting individual trajectories
            has_aggregate: Whether plotting aggregated statistics
            single_plot: Whether single plot mode
            max_trajectories: Maximum trajectories
            full_timeseries: Whether full timeseries mode
            normalize: Normalization method
            y_label: Y-axis label text
        """
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
                    if plot_var is None:
                        raise ValueError("Failed to infer plot_var")
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
                    if plot_var is None:
                        raise ValueError("Failed to infer plot_var")
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

    def _package_timeseries_result(
        self,
        fig: Optional[matplotlib.figure.Figure],
        map: bool,
        use_subplots: bool,
        map_ax: Optional[Axes],
        ts_axes_list: List[Axes],
    ) -> Tuple[Optional[matplotlib.figure.Figure], Union[Axes, List[Axes], dict]]:
        """Package return values for timeseries method.

        Args:
            fig: Figure object
            map: Whether map is included
            use_subplots: Whether using subplots
            map_ax: Map axes
            ts_axes_list: List of timeseries axes

        Returns:
            Tuple of (figure, axes) with appropriate structure
        """
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

    # ----- DEPRECATED FUNCTIONS -----

    def cluster_overview(
        self,
        **kwargs: Any,
    ):
        """Deprecated function: use td.plot.overview() instead."""
        raise DeprecationWarning("Use td.plot.overview() instead")


# end of Plotter


def _filter_by_existing_clusters(
    td, cluster_ids: Union[int, List[int], np.ndarray, range], var: str
) -> List[int]:
    """Filter cluster_ids to only include existing clusters.

    Args:
        td: TOAD object containing cluster data.
        cluster_ids: Single cluster ID or list/array/range of cluster IDs to filter.
        var: Variable name for clusters.

    Returns:
        List of cluster IDs that exist in the dataset (excluding noise cluster -1).
    """

    if isinstance(cluster_ids, int):
        cluster_ids = [cluster_ids]

    return [
        id for id in cluster_ids if id in td.get_cluster_ids(var, exclude_noise=False)
    ]


def _get_high_constrast_text_color(color: Union[tuple, str]) -> str:
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


def _get_cmap_seq(
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


def _remove_spines(
    axs: Union[Axes, List[Axes], np.ndarray],
    spines: Union[List[str], str, np.ndarray] = ["top", "right", "bottom", "left"],
):
    """Remove spines (borders) from matplotlib axes.

    Args:
        axs: Single axes, list of axes, or numpy array of axes.
        spines: Spine(s) to remove. Can be a single string, list of strings, or array.
            Valid values: "top", "right", "bottom", "left". Defaults to all four.
    """
    if isinstance(axs, Axes):
        axs = np.asarray([axs])

    if isinstance(spines, str):
        spines = np.asarray([spines])

    for ax in axs:
        for s in spines:
            ax.spines[s].set_visible(False)


# Not used...
def _replace_ax_projection(
    fig: matplotlib.figure.Figure,
    axs: Union[np.ndarray, Axes],
    row: int,
    col: int,
    projection: str | ccrs.Projection,
) -> Union[np.ndarray, Axes]:
    """Replace the subplot at the given row and column of axs with a map projection.

    Args:
        fig: Matplotlib figure containing the subplots.
        axs: Array of axes or single Axes object.
        row: Row index of the subplot to replace.
        col: Column index of the subplot to replace.
        projection: Cartopy projection to use (string name or Projection object).

    Returns:
        Updated axes array or single Axes, matching the input type.
    """
    # Remember if input was a single Axes before conversion
    was_single_axes = isinstance(axs, Axes)

    if was_single_axes:
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
    if was_single_axes:
        return axs[0, 0]
    return axs


def _remove_ticks(axs: Union[Axes, List[Axes], np.ndarray], keep_x=False, keep_y=False):
    """Remove tick marks and labels from matplotlib axes.

    Args:
        axs: Single axes, list of axes, or numpy array of axes.
        keep_x: If True, keep x-axis ticks. Defaults to False.
        keep_y: If True, keep y-axis ticks. Defaults to False.
    """
    if isinstance(axs, Axes):
        axs = np.asarray([axs])

    for ax in axs:
        if not keep_x:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
            [label.set_visible(False) for label in ax.get_xticklabels()]
        if not keep_y:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            [label.set_visible(False) for label in ax.get_yticklabels()]


def _add_map_features(ax: GeoAxes, config: MapStyle) -> None:
    """Add standard map features to an axes.

    Args:
        ax: Matplotlib axes with cartopy projection
        config: Plot configuration
    """
    # Add continent shading
    if config.continent_shading:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "land",
                config.resolution,
                facecolor=config.continent_shading_color,
                edgecolor="none",
                alpha=0.5,
            )
        )

    if config.ocean_shading:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical",
                "ocean",
                config.resolution,
                facecolor=config.ocean_shading_color,
                edgecolor="none",
                alpha=0.5,
            )
        )

    ax.coastlines(resolution=config.resolution, linewidth=config.coastline_linewidth)

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
    ax: Axes,
    x: float,
    y: float,
    cluster_id: int,
    acol: str,
    scale: float = 1,
    relative_coords: bool = False,
    transform: Optional[ccrs.Projection] = None,
):
    """Annotate a cluster on a map with its ID number.

    Args:
        ax: Matplotlib axes to annotate on.
        x: X coordinate for annotation (in data coordinates unless relative_coords=True).
        y: Y coordinate for annotation (in data coordinates unless relative_coords=True).
        cluster_id: Cluster ID number to display.
        acol: Background color for the annotation box.
        scale: Scale factor for font size. Defaults to 1.
        relative_coords: If True, x and y are in axes fraction coordinates (0-1).
            If False, x and y are in data coordinates. Defaults to False.
        transform: Optional cartopy projection transform for data coordinates.
            Only used when relative_coords=False.
    """
    black_or_white = _get_high_constrast_text_color(acol)
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


def _create_timeseries_layout(
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


def _add_gradient_legend(
    ax: Axes,
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
            legend_pos = (
                0.01,
                -0.07,
            )
        else:
            # Fallback to projection-based positioning
            import cartopy.crs as ccrs

            projection = getattr(ax, "projection", None)
            if projection is not None and isinstance(projection, ccrs.Projection):
                if isinstance(projection, ccrs.PlateCarree):
                    legend_pos = (0.75, 0.95)  # top-right
                else:
                    legend_pos = (0.02, 0.95)  # top-left
            else:
                legend_pos = (0.02, 0.95)

    # Get colormap
    if cmap is None:
        # Try to get colormap from the last image in the axes
        images = [child for child in ax.get_children() if hasattr(child, "get_cmap")]
        if images:
            get_cmap_method = getattr(images[-1], "get_cmap", None)
            if get_cmap_method is not None:
                cmap = get_cmap_method()
            else:
                cmap = plt.get_cmap("viridis")  # fallback
        else:
            cmap = plt.get_cmap("viridis")  # fallback

    # Normalize cmap to Colormap (convert string to Colormap if needed)
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # Ensure cmap is a Colormap (fallback if somehow still None or invalid)
    if cmap is None or not isinstance(cmap, Colormap):
        cmap = plt.get_cmap("viridis")

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
