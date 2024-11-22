from abc import ABC, abstractmethod

# Abstract class for clustering methods
class ClusteringMethod(ABC):
    @abstractmethod
    def apply(self, coords, weights):
        """Abstract method for clustering."""
        pass