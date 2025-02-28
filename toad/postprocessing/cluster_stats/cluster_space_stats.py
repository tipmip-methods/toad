from toad.utils import all_functions
import inspect


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
