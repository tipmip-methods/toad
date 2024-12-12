from toad.postprocessing.cluster_stats.cluster_time_stats import ClusterTimeStats
from toad.postprocessing.cluster_stats.cluster_space_stats import ClusterSpaceStats
from toad.postprocessing.cluster_stats.cluster_general_stats import ClusterGeneralStats

class ClusterStats:
<<<<<<< HEAD
<<<<<<< HEAD
    """Interface to access specialized statistics calculators for clusters: time, space, and general metrics."""

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

    @property
    def time(self):
        """Access time-related statistics for clusters."""
        return ClusterTimeStats(self.td, self.var)

    @property
    def space(self):
        """Access space-related statistics for clusters."""
        return ClusterSpaceStats(self.td, self.var)

    @property
    def general(self):
        """Access general statistics for clusters."""
        return ClusterGeneralStats(self.td, self.var)
=======
=======
    """Interface to access specialized statistics calculators for clusters: time, space, and general metrics."""

>>>>>>> ba8e9d6 (Clean up docstrings)
    def __init__(self, toad, var):
        """
        Args:
            toad (TOAD): TOAD object
            var (str): Base variable name (e.g. 'temperature', will look for 'temperature_cluster') or custom cluster variable name.
        """
        self.td = toad
        self.var = var

    @property
    def time(self):
        """Access time-related statistics for clusters."""
        return ClusterTimeStats(self.td, self.var)

    @property
    def space(self):
        """Access space-related statistics for clusters."""
        return ClusterSpaceStats(self.td, self.var)

    @property
    def general(self):
<<<<<<< HEAD
        if self._general_stats is None:
            self._general_stats = ClusterGeneralStats(self.td, self.var)
        return self._general_stats
>>>>>>> 7d33054 ([Breaking changes] Refactored timeseries and Clustering + stats)
=======
        """Access general statistics for clusters."""
        return ClusterGeneralStats(self.td, self.var)
>>>>>>> ba8e9d6 (Clean up docstrings)
    


"""
    TODO: implement all stats
    Stats for clusters
    Time related:
        Individual clusters:
            - mean_time
            - median_time
            - std_time
            - peak_time
            - start_time
            - end_time
            - duration_time
            - iqr_50_start
            - iqr_50_end
            - iqr_50_duration
            - iqr_90_start
            - iqr_90_end
            - iqr_90_duration
            - peak_time (when is the cluster at its largest)
            - peak_magnitude
            - peak_probability
            - mean_membership_duration
            - median_membership_duration
            - std_membership_duration
            - iqr_50_membership_duration
            - iqr_90_membership_duration
        For all clusters:
            - mean_time
            - median_time
            - std_time
            - peak_time (when do we have most clusters)
            - start_time
            - end_time
            - iqr_50_start
            - iqr_50_end
            - iqr_90_start
            - iqr_90_end
    Space related:
        Individual clusters:
            - mean_x
            - mean_y
            - median_x
            - median_y
            - std_x
            - std_y
            - density (function of time)
            - density_std
            - density_mean
            - peak_density
            - area (function of time)
            - area_std
            - area_mean
            - area_median
        For all clusters:
            - density_mean (function of time)
            - density_std (function of time)
            - area_mean (function of time)
            - area_std (function of time)
    Other stats:
        - cluster_score
        - cluster_score_fit
        - cluster_size
        
    """
