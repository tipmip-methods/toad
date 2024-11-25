"""dbscan algorithm for clustering

Uses the scipy dbscan clustering algorithm and a taylored preprocessing
pipeline.

Created: October 22, ??
Refactored: Nov, 2024
"""

import numpy as np
import sklearn.cluster
from toad_lab.clustering.methods.base import ClusteringMethod


class DBSCAN(ClusteringMethod):
    """
    DBSCAN clustering algorithm applicable to coordinate data in 1d-array 

    Args:
        eps (float): The maximum distance between two samples for them to be 
            considered as part of the same cluster, defining the neighborhood 
            radius for DBSCAN.
        min_samples (int): The minimum number of samples required in a neighborhood 
            for a point to be considered a core point, determining the density 
            threshold for clusters.
        metric (str, optional): The distance metric to use for clustering. Defaults to 'euclidean'.
    """

    def __init__(self, 
                eps : float,
                min_samples : int, 
                metric="euclidean"
        ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric

    def apply(self, coords: np.ndarray, weights=None):
        """
        Apply the DBSCAN algorithm to cluster the provided coordinate data.

        Args:
            coords (np.ndarray): A list of the coordinates to be clustered, e.g. (time, x, y)
            weights (np.ndarray, optional): Importance weights for each data point. 
                Not used in HDBSCAN but included for compatibility with other clustering methods.

        Returns:
            np.ndarray: A 1D NumPy array of cluster labels for each data point, 
                where -1 indicates noise points.
            dict: A dictionary summarizing the HDBSCAN parameters used, suitable 
                for storing as metadata or documentation.

        TODO: find way to make this work with haversine distance for lat/lon data
        
        """
        # Fit clusters with HDBSCAN
        clustering = sklearn.cluster.DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
        )
        cluster_labels_array = clustering.fit_predict(coords, sample_weight=weights).astype(float)

        # Metadata details
        method_details = {
            "method": "dbscan",
            "params": {
                "eps": self.eps,
                "min_samples": self.min_samples,
                "metric": self.metric,
            },
        }

        return cluster_labels_array, method_details