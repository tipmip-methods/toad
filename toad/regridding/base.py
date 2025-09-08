from abc import ABC, abstractmethod

import numpy as np


class BaseRegridder(ABC):
    """Abstract base class for regridding methods."""

    def __init__(self):
        self.df = None  # Holds regridded data with coordinates and values
        self.original_coords = None  # Store original coordinates for regridding back

    @abstractmethod
    def regrid(
        self, coords: np.ndarray, weights: np.ndarray, space_dims_size: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Regrid data to new coordinate system.

        Args:
            coords: 3dArray of coordinates (time, lon, lat) in that order
            weights: 1dArray of weights
            space_dims_size: Tuple of (nlat, nlon) sizes of the original grid dimensions
        Returns:
            3dArray of coordinates (time, lon, lat) in that order
            1dArray of weights
        """
        pass

    @abstractmethod
    def regrid_clusters_back(self, cluster_labels: np.ndarray) -> np.ndarray:
        """
        Map cluster labels back to original coordinates.

        Args:
            cluster_labels: Array of cluster labels for regridded points

        Returns:
            Array of cluster labels for original coordinates
        """
        pass
