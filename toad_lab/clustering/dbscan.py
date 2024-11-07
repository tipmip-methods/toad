"""dbscan algorithm for clustering

Uses the scipy dbscan clustering algorithm and a taylored preprocessing
pipeline.

Created: October 22, ??
Refactored: Nov, 2024
"""

import numpy as np
from sklearn.cluster import DBSCAN

def dbscan(
        coords: np.array,
        weights: np.array,
        eps : float,
        min_samples : int,
    ):
    """Apply the DBSCAN clustering algorithm to 1D coordinate data.

    Args:
        coords (np.array): A 1D NumPy array representing the coordinates 
            of the data points to be clustered.
        weights (np.array): A 1D NumPy array of the same length as `coords`, 
            representing the importance or weight of each data point. Higher 
            weights make points more influential in forming clusters.
        eps (float): The maximum distance between two samples for them to be 
            considered as part of the same cluster, defining the neighborhood 
            radius for DBSCAN.
        min_samples (int): The minimum number of samples required in a neighborhood 
            for a point to be considered a core point, determining the density 
            threshold for clusters.

    Returns:
        tuple: A tuple containing:
            - np.array: A 1D NumPy array of cluster labels for each data point, 
              where -1 indicates noise points.
            - str: A string summarizing the DBSCAN parameters used, suitable 
              for storing as metadata or documentation.

    TODO: find way to make this work with haversine distance for lat/lon data
    """

    #  this works with euclidean distance, but we need to use haversine for lat/lon, but HDBscan can't do 3d then
    metric = "euclidean"

    # Fit clusters with DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
    cluster_labels_array = clustering.fit_predict(coords, sample_weight=weights).astype(float)

    # method details to be stored in the dataset attributes
    method_details = f'dbscan (eps={eps}, min_samples={min_samples})'

    return cluster_labels_array, method_details