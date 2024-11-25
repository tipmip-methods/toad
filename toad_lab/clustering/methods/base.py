from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Optional

# Abstract class for clustering methods
class ClusteringMethod(ABC):
    @abstractmethod
    def apply(self, coords: np.ndarray, weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Dict]:
        """Apply the clustering method to the input coordinates.

        Args:
            coords (np.ndarray): A list of the coordinates to be clustered, e.g. (time, x, y)
            weights (Optional[np.ndarray]): Importance weights for each data point.

        Returns:
            Tuple[np.ndarray, Dict]: A tuple containing:
                - np.ndarray: A 1D NumPy array of cluster labels for each data point, 
                  where -1 indicates unclustered points.
                - Dict: A dictionary summarizing the method parameters used, suitable 
                  for storing as metadata or documentation.
        """
        pass