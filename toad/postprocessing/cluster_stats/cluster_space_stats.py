from toad.utils import all_functions
import inspect
import numpy as np
from scipy.ndimage import distance_transform_edt


class ClusterSpaceStats:
    """Class containing functions for calculating space-related statistics for clusters, such as mean, median, std, etc."""

    def __init__(self, toad, var):
        """
        >> Args:
            toad : (TOAD)
                TOAD object
            var : (str)
                Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
        """

        self.td = toad
        self.var = var
        # Initialize other necessary attributes

    # Define space-related statistics methods

    def mean(self, cluster_id):
        """Returns the mean of the first and second spatial dimension of all cells in the cluster across space and time"""
        return (
            float(
                self.td.apply_cluster_mask(
                    self.var, self.td.space_dims[0], cluster_id
                ).mean()
            ),
            float(
                self.td.apply_cluster_mask(
                    self.var, self.td.space_dims[1], cluster_id
                ).mean()
            ),
        )

    def median(self, cluster_id):
        """Returns the median of the first and second spatial dimension of all cells in the cluster across space and time"""
        return (
            float(
                self.td.apply_cluster_mask(
                    self.var, self.td.space_dims[0], cluster_id
                ).median()
            ),
            float(
                self.td.apply_cluster_mask(
                    self.var, self.td.space_dims[1], cluster_id
                ).median()
            ),
        )

    def std(self, cluster_id):
        """Returns the standard deviation of the first and second spatial dimension of all cells in the cluster across space and time"""
        return (
            float(
                self.td.apply_cluster_mask(
                    self.var, self.td.space_dims[0], cluster_id
                ).std()
            ),
            float(
                self.td.apply_cluster_mask(
                    self.var, self.td.space_dims[1], cluster_id
                ).std()
            ),
        )

    def footprint_mean(self, cluster_id):
        """Returns the mean of the first and second spatial dimension of the footprint of the cluster, i.e. the collection of spatial cells that were ever touched by the cluster."""
        return (
            float(
                self.td.apply_spatial_cluster_mask(
                    self.var, self.td.space_dims[0], cluster_id
                ).mean()
            ),
            float(
                self.td.apply_spatial_cluster_mask(
                    self.var, self.td.space_dims[1], cluster_id
                ).mean()
            ),
        )

    def footprint_median(self, cluster_id):
        """Returns the median of the first and second spatial dimension of the footprint of the cluster, i.e. the collection of spatial cells that were ever touched by the cluster."""
        return (
            float(
                self.td.apply_spatial_cluster_mask(
                    self.var, self.td.space_dims[0], cluster_id
                ).median()
            ),
            float(
                self.td.apply_spatial_cluster_mask(
                    self.var, self.td.space_dims[1], cluster_id
                ).median()
            ),
        )

    def footprint_std(self, cluster_id):
        """Returns the standard deviation of the first and second spatial dimension of the footprint of the cluster, i.e. the collection of spatial cells that were ever touched by the cluster."""
        return (
            float(
                self.td.apply_spatial_cluster_mask(
                    self.var, self.td.space_dims[0], cluster_id
                ).std()
            ),
            float(
                self.td.apply_spatial_cluster_mask(
                    self.var, self.td.space_dims[1], cluster_id
                ).std()
            ),
        )

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
        max_dist_idx_unraveled = np.unravel_index(max_dist_idx_flat, distance_map.shape)

        # Get the coordinate labels corresponding to the index
        # We need the coordinate arrays from the spatial_mask DataArray
        y_coords = spatial_mask[self.td.space_dims[0]].values
        x_coords = spatial_mask[self.td.space_dims[1]].values

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
                and len(inspect.signature(getattr(self, method_name)).parameters) == 1
            ):
                dict[method_name] = getattr(self, method_name)(cluster_id)
        return dict
