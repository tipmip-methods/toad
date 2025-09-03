from toad.utils import all_functions
import inspect
import numpy as np
from scipy.ndimage import distance_transform_edt


class ClusterSpaceStats:
    """Class containing functions for calculating space-related statistics for clusters, such as mean, median, std, etc."""

    def __init__(self, toad, var):
        """Initialize ClusterSpaceStats.

        Args:
            toad: TOAD object
            var: Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
        """
        self.td = toad
        self.var = var

    def _get_cluster_coordinate_values_spacetime(self, cluster_id):
        """Get coordinate arrays for space-time cluster analysis, preferring lat/lon."""
        from toad.utils import detect_latlon_names

        lat_name, lon_name = detect_latlon_names(self.td.data)
        has_latlon = lat_name is not None and lon_name is not None

        if has_latlon:
            # Use lat/lon coordinates instead of dimension coordinates
            y_coords = self.td.apply_cluster_mask(self.var, lat_name, cluster_id)
            x_coords = self.td.apply_cluster_mask(self.var, lon_name, cluster_id)
            return y_coords, x_coords
        else:
            # Fallback to dimension coordinates
            y_coords = self.td.apply_cluster_mask(
                self.var, self.td.space_dims[0], cluster_id
            )
            x_coords = self.td.apply_cluster_mask(
                self.var, self.td.space_dims[1], cluster_id
            )
            return y_coords, x_coords

    def _get_cluster_coordinate_values(self, cluster_id):
        """Get coordinate arrays for spatial footprint analysis, preferring lat/lon."""
        from toad.utils import detect_latlon_names

        lat_name, lon_name = detect_latlon_names(self.td.data)
        has_latlon = lat_name is not None and lon_name is not None

        if has_latlon:
            # Use lat/lon coordinates (works for both 1D regular and 2D irregular grids)
            spatial_mask = self.td.get_spatial_cluster_mask(self.var, cluster_id)

            if self.td.data[lat_name].ndim == 2:
                # 2D coordinates (irregular grid)
                lat_values = self.td.data[lat_name].where(spatial_mask)
                lon_values = self.td.data[lon_name].where(spatial_mask)
            else:
                # 1D coordinates (regular grid) - apply mask to get subset
                lat_values = self.td.apply_spatial_cluster_mask(
                    self.var, lat_name, cluster_id
                )
                lon_values = self.td.apply_spatial_cluster_mask(
                    self.var, lon_name, cluster_id
                )

            return lat_values, lon_values
        else:
            # Fallback to dimension coordinates when lat/lon not available
            y_coords = self.td.apply_spatial_cluster_mask(
                self.var, self.td.space_dims[0], cluster_id
            )
            x_coords = self.td.apply_spatial_cluster_mask(
                self.var, self.td.space_dims[1], cluster_id
            )
            return y_coords, x_coords

    def mean(self, cluster_id):
        """Returns the mean of the spatial coordinates across space and time."""
        y_coords, x_coords = self._get_cluster_coordinate_values_spacetime(cluster_id)
        return (float(y_coords.mean()), float(x_coords.mean()))

    def median(self, cluster_id):
        """Returns the median of the spatial coordinates across space and time."""
        y_coords, x_coords = self._get_cluster_coordinate_values_spacetime(cluster_id)
        return (float(y_coords.median()), float(x_coords.median()))

    def std(self, cluster_id):
        """Returns the standard deviation of the spatial coordinates across space and time."""
        y_coords, x_coords = self._get_cluster_coordinate_values_spacetime(cluster_id)
        return (float(y_coords.std()), float(x_coords.std()))

    def footprint_mean(self, cluster_id):
        """Returns the mean of the spatial coordinates of the cluster footprint."""
        y_coords, x_coords = self._get_cluster_coordinate_values(cluster_id)
        return (float(y_coords.mean()), float(x_coords.mean()))

    def footprint_median(self, cluster_id):
        """Returns the median of the spatial coordinates of the cluster footprint."""
        y_coords, x_coords = self._get_cluster_coordinate_values(cluster_id)
        return (float(y_coords.median()), float(x_coords.median()))

    def footprint_std(self, cluster_id):
        """Returns the standard deviation of the spatial coordinates of the cluster footprint."""
        y_coords, x_coords = self._get_cluster_coordinate_values(cluster_id)
        return (float(y_coords.std()), float(x_coords.std()))

    def central_point_for_labeling(self, cluster_id) -> tuple[float, float]:
        """Calculates a central point within the cluster's spatial footprint suitable for labeling.

        This method uses the Euclidean Distance Transform to find the point within
        the cluster footprint that is furthest from any edge (the "pole of
        inaccessibility"). This ensures the point is robustly inside the cluster shape,
        even for complex geometries like rings or C-shapes.

        Args:
            cluster_id: The ID of the cluster to analyze.

        Returns:
            A tuple containing the (y, x) coordinates of the calculated central point.
            Returns (np.nan, np.nan) if the footprint is empty.
        """
        # Get the 2D spatial footprint mask (ensure it's boolean or 0/1)
        spatial_mask = self.td.get_spatial_cluster_mask(self.var, cluster_id)

        # Ensure the mask is boolean or integer type for distance transform
        if spatial_mask.dtype != bool and not np.issubdtype(
            spatial_mask.dtype, np.integer
        ):
            # Attempt conversion, assuming non-zero means True
            spatial_mask = spatial_mask > 0
        elif spatial_mask.dtype == bool:
            # Keep as is
            pass
        else:  # Integer type
            spatial_mask = spatial_mask > 0  # Convert to boolean

        # Handle empty mask case
        if not spatial_mask.any():
            return (np.nan, np.nan)

        # Calculate the distance transform
        # distance_transform_edt returns distances for non-zero elements from the nearest zero element
        distance_map = distance_transform_edt(spatial_mask.values)  # Pass numpy array

        # Find the index (row, col) of the maximum distance
        # np.argmax flattens the array first, so we need unravel_index
        max_dist_idx_flat = np.argmax(distance_map)
        max_dist_idx_unraveled = np.unravel_index(max_dist_idx_flat, distance_map.shape)  # type: ignore

        # Use consistent coordinate system
        from toad.utils import detect_latlon_names

        lat_name, lon_name = detect_latlon_names(self.td.data)
        has_latlon = lat_name is not None and lon_name is not None

        if has_latlon and self.td.data[lat_name].ndim == 2:
            # 2D lat/lon coordinates - get actual geographic values
            lat_2d = self.td.data[lat_name].values
            lon_2d = self.td.data[lon_name].values
            y_label_coord = lat_2d[max_dist_idx_unraveled[0], max_dist_idx_unraveled[1]]
            x_label_coord = lon_2d[max_dist_idx_unraveled[0], max_dist_idx_unraveled[1]]
        elif has_latlon:
            # 1D lat/lon coordinates
            lat_coords = spatial_mask[lat_name].values  # type: ignore
            lon_coords = spatial_mask[lon_name].values  # type: ignore
            y_label_coord = lat_coords[max_dist_idx_unraveled[0]]
            x_label_coord = lon_coords[max_dist_idx_unraveled[1]]
        else:
            # Fallback to dimension coordinates
            y_coords = spatial_mask[self.td.space_dims[0]].values  # type: ignore
            x_coords = spatial_mask[self.td.space_dims[1]].values  # type: ignore
            y_label_coord = y_coords[max_dist_idx_unraveled[0]]
            x_label_coord = x_coords[max_dist_idx_unraveled[1]]

        return (float(y_label_coord), float(x_label_coord))

    def footprint_cumulative_area(self, cluster_id) -> int:
        """Returns the total number of spatial cells that were ever touched by the cluster."""
        return int(self.td.get_spatial_cluster_mask(self.var, cluster_id).sum().sum())

    def all_stats(self, cluster_id) -> dict:
        """Return all cluster stats"""
        dict = {}
        for method_name in all_functions(self):
            if (
                not method_name.startswith("all_stats")
                and not method_name.startswith("_")
                and len(inspect.signature(getattr(self, method_name)).parameters) == 1
            ):
                dict[method_name] = getattr(self, method_name)(cluster_id)
        return dict
