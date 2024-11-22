from hdbscan import HDBSCAN
import numpy as np

def hdbscan(
    coords: np.ndarray,
    min_cluster_size : int,
    min_samples : int = None,
    cluster_selection_epsilon : float = 0.0,
    **kwargs # needed to allow for unused weights to be passed too all clustering methods
    ):
    """Apply the HDBSCAN clustering algorithm to coordinate data.

    This function uses the HDBSCAN algorithm to identify clusters in the given 
    coordinate data using Euclidean distance. 

    Args:
        coords (np.ndarray): A 2D NumPy array of shape (n_samples, n_features) 
            representing the coordinates of the data points to be clustered.
        min_cluster_size (int): The minimum size of clusters. This parameter 
            controls the smallest grouping that HDBSCAN will consider as a cluster.
        min_samples (int, optional): The minimum number of samples in a neighborhood 
            for a point to be considered a core point. If not provided, it defaults 
            to the same value as `min_cluster_size`.
        cluster_selection_epsilon (float, optional): The distance threshold for 
            merging clusters during the cluster selection phase. Defaults to 0.0.
            Higher value gives fewer but larger clusters.
        **kwargs: Additional keyword arguments to maintain compatibility with 
            other clustering methods. This allows unused parameters, such as weights, 
            to be passed without causing errors.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: A 1D NumPy array of cluster labels for each data point, 
                where -1 indicates noise points.
            - str: A string summarizing the HDBSCAN parameters used, suitable 
                for storing as metadata or documentation.
                
                
    TODO: find way to make this work with haversine distance for lat/lon data
    """
     
    #  this works with euclidean distance, but we need to use haversine for lat/lon, but HDBscan can't do 3d then
    metric = "euclidean"

    # Fit clusters with HDBSCAN
    clustering = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, cluster_selection_epsilon=cluster_selection_epsilon, allow_single_cluster=True)
    cluster_labels_array = clustering.fit_predict(coords).astype(float)

    # method details to be stored in the dataset attributes
    method_details = f'hdbscan (min_cluster_size={min_cluster_size}, min_samples={min_samples})'

    return cluster_labels_array, method_details
