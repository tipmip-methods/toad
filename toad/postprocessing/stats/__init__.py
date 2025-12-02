from toad.postprocessing.stats.general import GeneralStats
from toad.postprocessing.stats.space import SpaceStats
from toad.postprocessing.stats.time import TimeStats

__all__ = ["TimeStats", "SpaceStats", "GeneralStats", "Stats"]


class Stats:
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
        return TimeStats(self.td, self.var)

    @property
    def space(self):
        """Access space-related statistics for clusters."""
        return SpaceStats(self.td, self.var)

    @property
    def general(self):
        """Access general statistics for clusters."""
        return GeneralStats(self.td, self.var)
