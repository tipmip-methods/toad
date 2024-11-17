"""CSPA aggregation module

Simple approach to aggregate various clusterings into one ensemble clustering based on the 
Cluster-based Similarity Partitioning Algorithm (CSPA) from
 A. Strehl and J. Ghosh, "Cluster ensembles -- a knowledge reuse framework for combining multiple partitions," 
 Journal of Machine Learning Research, vol. 3, pp. 583-617, 2002.

October 24
"""
import xarray as xr
import numpy as np
from scipy.sparse import dok_matrix
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

def aggregate(data: xr.Dataset, 
              coocurrence_threshold: float = 0.5, 
              distance_threshold: float = 0.5, 
              plot_dendrogram: bool = True,
              first_dim: str = "time", second_dim: str = "latitude", third_dim: str = "longitude", 
              block_size: int = 1000) -> xr.Dataset:
    """
    Perform Cluster-based Similarity Partitioning Algorithm (CSPA) on an xarray.Dataset
    with multiple clustering results and optionally plot the dendrogram.
    
    Parameters:
    - data: xarray.Dataset containing multiple clusterings and with 3 dimensions
    - coocurrence_threshold: threshold for co-cluster occurrence matrix (default: 0.5)
    - distance_threshold: maximum distance for hierarchical clustering merges (default: 0.5)
    - plot_dendrogram: whether to plot the dendrogram to help guide distance_threshold
    - first_dim: name of the first dimension
    - second_dim: name of the second dimension
    - third_dim: name of the third dimension
    - block_size: number of samples to process in each block for memory efficiency (default: 1000)
    
    Returns:
    - xr.Dataset: Updated dataset with CSPA clustering as a new variable
    """
    # Determine clusterings
    cluster_vars = list(data.data_vars)

    # Create a mask to identify points that have at least one valid cluster across all variables
    combined_mask = np.zeros(data[cluster_vars[0]].shape, dtype=bool)
    for var in cluster_vars:
        combined_mask |= (data[var].values != -1)  # Assuming -1 represents unclustered/no cluster

    # Flatten the mask to get valid data points
    valid_indices = np.where(combined_mask.flatten())[0]
    
    if valid_indices.size == 0:
        raise ValueError("No valid data points found with co-occurrence across the clusterings.")
    

    # Initialize a list to store cluster labels for each variable (flattened)
    all_labels = []

    for var in cluster_vars:
        # Flatten the 3D data to a 1D array
        cluster_labels = data[var].values.flatten()
        filtered_labels = cluster_labels[valid_indices]
        all_labels.append(filtered_labels)

    # Stack the cluster labels into a 2D matrix where each column is a flattened clustering variable
    label_matrix = np.vstack(all_labels).T  # Shape: (num_samples, num_clusters)

    # Compute the co-occurrence matrix using dot product
    # This creates a (num_samples, num_samples) matrix of co-occurrence counts
    co_occurrence_matrix = np.dot(label_matrix, label_matrix.T)  # This is a dense matrix

    # Normalize to get the similarity matrix
    similarity_matrix = co_occurrence_matrix / len(cluster_vars)
    similarity_matrix = similarity_matrix.toarray()

    # Threshold the data
    similarity_matrix[similarity_matrix < coocurrence_threshold] = 0

    # Convert similarity to dissimilarity (1 - similarity)
    dissimilarity_matrix = 1 - similarity_matrix

    # Perform hierarchical clustering on the dissimilarity matrix using linkage from scipy
    linkage_matrix = linkage(dissimilarity_matrix, method='average')

    # Plot dendrogram if requested
    if plot_dendrogram:
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix, no_labels=True)
        plt.title("Dendrogram for CSPA Clustering")
        plt.xlabel("Sample index")
        plt.ylabel("Dissimilarity")
        plt.show()

    # Perform hierarchical clustering using sklearn with the distance threshold
    clustering_model = AgglomerativeClustering(
        n_clusters=None,
        affinity='precomputed',
        linkage='average',
        distance_threshold=distance_threshold
    )

    # Fit the model on the dissimilarity matrix
    cluster_labels = clustering_model.fit_predict(dissimilarity_matrix)

    # Create an output array filled with -1 (indicating invalid points)
    output_labels = np.full(data[first_dim].size * data[second_dim].size * data[third_dim].size, -1)
    
    # Assign the cluster labels back to the valid indices
    output_labels[valid_indices] = cluster_labels

    # Reshape the labels back to the original dimensions
    cspa_labels = output_labels.reshape(data[first_dim].size, data[second_dim].size, data[third_dim].size)

    # Create a new xr.Dataset including the original variables and CSPA result
    cspa_dataset = data.copy()
    cspa_dataset['cspa_clustering'] = ((first_dim, second_dim, third_dim), cspa_labels)

    return cspa_dataset


# import xarray as xr
# import numpy as np
# from scipy.sparse import coo_matrix
# from sklearn.cluster import AgglomerativeClustering
# from scipy.cluster.hierarchy import dendrogram, linkage
# import matplotlib.pyplot as plt

# def aggregate(data: xr.Dataset, 
#                 coocurrence_threshold: float = 0.5, 
#                 distance_threshold: float = 0.5, 
#                 plot_dendrogram: bool = True,
#                 first_dim: str = "time", second_dim: str = "latitude", third_dim: str = "longitude", 
# ) -> xr.Dataset:
#     """
#     Perform Cluster-based Similarity Partitioning Algorithm (CSPA) on an xarray.Dataset
#     with multiple clustering results and optionally plot the dendrogram.
    
#     Parameters:
#     - data: xarray.Dataset containing multiple clusterings and with 3 dimensions
#     - coocurrence_threshold: threshold for co-cluster occurrence matrix (default: 0.5)
#     - distance_threshold: maximum distance for hierarchical clustering merges (default: 0.5)
#     - plot_dendrogram: whether to plot the dendrogram to help guide distance_threshold
#     - first_dim: name of the first dimension
#     - second_dim: name of the second dimension
#     - third_dim: name of the third dimension 
    
#     Returns:
#     - xr.Dataset: Updated dataset with CSPA clustering as a new variable
#     """
#     # Determine clustering variables
#     cluster_vars = list(data.data_vars)
    
#     # Get dimensions
#     num_first = data.dims[first_dim]
#     num_second = data.dims[second_dim]
#     num_third = data.dims[third_dim]
    
#     # Total number of samples (time * lat * lon)
#     num_samples = num_first * num_second * num_third
    
#     # List to hold non-zero entries for sparse co-occurrence matrix
#     rows, cols, values = [], [], []
    
#     # Build the co-occurrence matrix efficiently using flattening and vectorized operations
#     for var in cluster_vars:
#         cluster_labels = data[var].values
#         cluster_labels_flat = cluster_labels.reshape(-1)
        
#         # Vectorized comparison using broadcasting
#         for i in range(num_samples):
#             mask = (cluster_labels_flat == cluster_labels_flat[i])  # Find all matching labels
#             rows.extend([i] * np.sum(mask))  # Extend by index i
#             cols.extend(np.where(mask)[0])  # Append indices where matches occur
#             values.extend([1] * np.sum(mask))  # Add 1 for each match

#     # Create a sparse co-occurrence matrix
#     co_occurrence_matrix = coo_matrix((values, (rows, cols)), shape=(num_samples, num_samples))
    
#     # Normalize to get similarity matrix
#     similarity_matrix = co_occurrence_matrix / len(cluster_vars)
    
#     # Binarize the similarity matrix based on threshold
#     similarity_matrix = similarity_matrix.toarray()  # Convert to dense for thresholding
#     similarity_matrix[similarity_matrix < coocurrence_threshold] = 0
#     similarity_matrix[similarity_matrix >= coocurrence_threshold] = 1
    
#     # Convert similarity to dissimilarity (1 - similarity)
#     dissimilarity_matrix = 1 - similarity_matrix
    
#     # Perform hierarchical clustering on the dissimilarity matrix using linkage from scipy
#     linkage_matrix = linkage(dissimilarity_matrix, method='average')
    
#     # Plot dendrogram if requested
#     if plot_dendrogram:
#         plt.figure(figsize=(10, 7))
#         dendrogram(linkage_matrix, no_labels=True)
#         plt.title("Dendrogram for CSPA Clustering")
#         plt.xlabel("Sample index")
#         plt.ylabel("Dissimilarity")
#         plt.show()
    
#     # Perform hierarchical clustering using sklearn with the distance threshold
#     clustering_model = AgglomerativeClustering(
#         n_clusters=None,
#         affinity='precomputed',
#         linkage='average',
#         distance_threshold=distance_threshold  # Use the provided distance threshold
#     )
    
#     # Fit the model on the dissimilarity matrix
#     cluster_labels = clustering_model.fit_predict(dissimilarity_matrix)
    
#     # Reshape the labels back to the dimensions 
#     cspa_labels = cluster_labels.reshape(num_first, num_second, num_third)
    
#     # Create a new xr.Dataset including the original variables and CSPA result
#     cspa_dataset = data.copy()
#     cspa_dataset['cspa_clustering'] = ((first_dim, second_dim, third_dim), cspa_labels)
    
#     return cspa_dataset


