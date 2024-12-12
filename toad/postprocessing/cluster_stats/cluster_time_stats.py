import numpy as np
from toad.utils import all_functions
import inspect

class ClusterTimeStats:
    """Class containing functions for calculating time-related statistics for clusters, such as start time, peak time, etc."""

    def __init__(self, toad, var):
        """
        Args:
            toad (TOAD): TOAD object
            var (str): Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
        """
        self.td = toad
        self.var = var

    def start(self, cluster_id) -> float:
        """Return the start time of the cluster"""
        return float(self.td.apply_cluster_mask(self.var, self.td.time_dim, cluster_id).min())

    def end(self, cluster_id) -> float:
        """Return the end time of the cluster"""
        return float(self.td.apply_cluster_mask(self.var, self.td.time_dim, cluster_id).max())

    def duration(self, cluster_id) -> float:
        """Return duration of the cluster in time."""
        return float(float(self.end(cluster_id) - self.start(cluster_id)))
    
    def peak(self, cluster_id) -> float:
        """Return the time of the largest cluster temporal density"""
        ctd = self.td.get_cluster_spatial_density(self.var, cluster_id)
        return float(ctd[self.td.time_dim][ctd.argmax()].values)

    def peak_density(self, cluster_id) -> float:
        """Return the largest cluster temporal density"""
        ctd = self.td.get_cluster_spatial_density(self.var, cluster_id)
        return float(ctd.max().values)


    def iqr(self, cluster_id, lower_quantile: float, upper_quantile: float) -> tuple[float, float]:
        """Get start and end time of the specified interquantile range of the cluster temporal density.
        
        Args:
            cluster_id: ID of the cluster
            lower_quantile: Lower bound of the interquantile range (0-1)
            upper_quantile: Upper bound of the interquantile range (0-1)
            
        Returns:
            tuple: (start_time, end_time) of the interquantile range
        """
        ctd = self.td.get_cluster_spatial_density(self.var, cluster_id)
        cum_dist = ctd.cumsum()
        
        # Find both quantiles
        lower_time = ctd[self.td.time_dim].where(cum_dist >= lower_quantile * cum_dist[-1], drop=True)
        upper_time = ctd[self.td.time_dim].where(cum_dist >= upper_quantile * cum_dist[-1], drop=True)
        
        # Get first occurrence of each quantile
        lower = float(lower_time[0].values if lower_time.size > 0 else np.nan)
        upper = float(upper_time[0].values if upper_time.size > 0 else np.nan)
        
        return (lower, upper)

    def iqr_50(self, cluster_id) -> tuple[float, float]:
        """Get start and end time of the 50% interquantile range of the cluster temporal density"""
        return self.iqr(cluster_id, 0.25, 0.75)

    def iqr_90(self, cluster_id) -> tuple[float, float]:
        """Get start and end time of the 90% interquantile range of the cluster temporal density"""
        return self.iqr(cluster_id, 0.05, 0.95)
    
    def mean(self, cluster_id) -> float:
        """ Return mean time value of the cluster. """
        return float(self.td.apply_cluster_mask(self.var, self.td.time_dim, cluster_id).mean())

    def median(self, cluster_id) -> float:
        """Return median of the active time values weighted by spatial extent"""
        return float(self.td.apply_cluster_mask(self.var, self.td.time_dim, cluster_id).median())

    def std(self, cluster_id) -> float:
        """Return standard deviation of the active time values weighted by spatial extent"""
        return float(self.td.apply_cluster_mask(self.var, self.td.time_dim, cluster_id).std())


    # TODO implement
    # - cluster_mean_membership_duration
    # - cluster_median_membership_duration
    # - cluster_std_membership_duration

    def all_stats(self, cluster_id) -> dict:
        """Return all cluster stats"""
        dict = {}
        for method_name in all_functions(self):
            if not method_name.startswith('all_stats') and len(inspect.signature(getattr(self, method_name)).parameters) == 1:
                dict[method_name] = getattr(self, method_name)(cluster_id)
        return dict


    # Global cluster stats ========================================================

    # TODO make a decision about these global stats, if we want them they need to be updated, possibly put in further subclass .global

    # @property
    # def global_mean_time(self) -> float:
    #     """Return mean of the times during which clusters are active"""
    #     return float(np.concatenate([self._cluster_active_time(cluster_id).values for cluster_id in self.cluster_ids if cluster_id != -1]).mean())
        
    # @property
    # def global_median_time(self) -> float:
    #     """Return median of the times during which clusters are active"""
    #     return float(np.median(np.concatenate([self._cluster_active_time(cluster_id).values for cluster_id in self.cluster_ids if cluster_id != -1])))

    # @property
    # def global_std_time(self) -> float:
    #     """Return standard deviation of the times during which clusters are active"""
    #     return float(np.concatenate([self._cluster_active_time(cluster_id).values for cluster_id in self.cluster_ids if cluster_id != -1]).std())

    # @property
    # def global_mean_start_time(self) -> float:
    #     """Return mean of the start time of all clusters"""
    #     start_times = [self.cluster_start_time(cluster_id) for cluster_id in self.cluster_ids if cluster_id != -1]
    #     return float(np.mean(start_times))

    # @property
    # def global_mean_end_time(self) -> float:
    #     """Return mean of the end time of all clusters"""
    #     end_times = [self.cluster_end_time(cluster_id) for cluster_id in self.cluster_ids if cluster_id != -1]
    #     return float(np.mean(end_times))

    # @property
    # def global_median_start_time(self) -> float:
    #     """Return median of the start time of all clusters"""
    #     start_times = [self.cluster_start_time(cluster_id) for cluster_id in self.cluster_ids if cluster_id != -1]
    #     return float(np.median(start_times))
    
    # @property
    # def global_median_end_time(self) -> float:
    #     """Return median of the end time of all clusters"""
    #     end_times = [self.cluster_end_time(cluster_id) for cluster_id in self.cluster_ids if cluster_id != -1]
    #     return float(np.median(end_times))

    # @property
    # def global_std_start_time(self) -> float:
    #     """Return standard deviation of the start time of all clusters"""
    #     start_times = [self.cluster_start_time(cluster_id) for cluster_id in self.cluster_ids if cluster_id != -1]
    #     return float(np.std(start_times))

    # @property
    # def global_std_end_time(self) -> float:
    #     """Return standard deviation of the end time of all clusters"""
    #     end_times = [self.cluster_end_time(cluster_id) for cluster_id in self.cluster_ids if cluster_id != -1]
    #     return float(np.std(end_times))


    # def all_global_stats(self) -> dict:
    #     """Return all global stats"""
    #     dict = {}
    #     for method_name in dir(self):
    #         if method_name.startswith('global_'):
    #             attr = getattr(self, method_name)
    #             dict[method_name] = attr() if callable(attr) else attr
    #     return dict
