from toad.postprocessing.cluster_stats.cluster_time_stats import ClusterTimeStats
from toad.postprocessing.cluster_stats.cluster_space_stats import ClusterSpaceStats
from toad.postprocessing.cluster_stats.cluster_general_stats import ClusterGeneralStats


class ClusterStats:
    """Interface to access specialized statistics calculators for clusters: time, space, and general metrics."""

    def __init__(self, toad, var):
        """Initialize the ClusterStats object.

        Args:
            toad (TOAD): TOAD object
            var (str): Base variable name (e.g. 'temperature', will look for 'temperature_cluster')
                or custom cluster variable name.
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
