"""HDBSCAN algorithm for clustering

Uses the Hierarchical Density-Based Spatial Clustering of Applications with Noise (HDBSCAN) algorithm to cluster data points.

Created: Nov, 2024 (Jakob)
"""

from toad.clustering.methods.base import ClusteringMethod
import hdbscan as hdbscan_lib # hdbscan library
import numpy as np


class HDBSCAN(ClusteringMethod):
    """
    A class to apply the HDBSCAN clustering algorithm to coordinate data.

    This function uses the HDBSCAN algorithm to identify clusters in the given 
    coordinate data using Euclidean distance. 

    Args:
        min_cluster_size (int): The minimum size of clusters. Controls the smallest 
            grouping that HDBSCAN will consider as a cluster.
        min_samples (int, optional): The minimum number of samples in a neighborhood 
            for a point to be considered a core point. If not provided, it defaults 
            to the same value as `min_cluster_size`.
        cluster_selection_epsilon (float, optional): The distance threshold for 
            merging clusters during the cluster selection phase. Defaults to 0.0.
            Higher values yield fewer but larger clusters.
        metric (str, optional): The distance metric to use for clustering. Defaults to 'euclidean'.
    """

    def __init__(self, 
                 min_cluster_size, 
                 min_samples=None, 
                 cluster_selection_epsilon=0.0, 
                 metric="euclidean"
        ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric

    def apply(self, coords: np.ndarray, weights=None):
        """
        Apply the HDBSCAN algorithm to cluster the provided coordinate data.

        Args:
            coords (np.ndarray): A list of the coordinates to be clustered, e.g. (time, x, y)
            weights (np.ndarray, optional): Importance weights for each data point. 
                Not used in HDBSCAN but included for compatibility with abstract clustering class.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: A 1D NumPy array of cluster labels for each data point, 
                  where -1 indicates noise points.
                - dict: A dictionary summarizing the HDBSCAN parameters used, suitable 
                  for storing as metadata or documentation.
        
        TODO: find way to make this work with haversine distance for lat/lon data
        """
        # Fit clusters with HDBSCAN
        clustering = hdbscan_lib.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            allow_single_cluster=True,
        )
        cluster_labels_array = clustering.fit_predict(coords).astype(float)

        # Metadata details
        method_params = {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "cluster_selection_epsilon": self.cluster_selection_epsilon,
        }

        return cluster_labels_array, method_params