"""ACE aggregation module

Adaptive Clustering Ensemble method from: 
Alqurashi, T., & Wang, W. (2018). Clustering ensemble method. 
International Journal of Machine Learning and Cybernetics, 10(6), 1227–1246. https://doi.org/10.1007/s13042-017-0756-7

November 24
"""
import numpy as np
import xarray as xr

# ACE Algorithm Implementation
def aggregate(data: xr.Dataset, n_final_clusters, alpha1=0.8, alpha2=0.7, delta_alpha=0.1, noise_threshold = 0.5):
    """
    Adaptive Clustering Ensemble (ACE) Algorithm.

    Parameters:
    - clusterings (xarray.Dataset): Dataset where each variable represents a clustering.
    - n_final_clusters (int): Desired number of final clusters.
    - alpha1 (float): Initial similarity threshold for merging clusters.
    - alpha2 (float): Certainty threshold for resolving uncertain objects.
    - delta_alpha (float): Step size for adapting alpha1.

    Returns:
    - xarray.DataArray: Final clustering labels for each object.
    """
    # Step 1: Binary Transformation
    binary_matrix, valid_objects = transform_to_binary(clusterings, noise_threshold)

    # Step 2: Generate Consensus Clusters
    consensus_clusters = generate_consensus_clusters(binary_matrix, n_final_clusters, alpha1, delta_alpha)

    # Step 3: Resolve Uncertain Assignments
    ace_clustering = resolve_uncertain_objects(binary_matrix, consensus_clusters, valid_objects, alpha2)

    # Add ACE clustering result as a new variable in the dataset
    clusterings["ace_clustering"] = xr.DataArray(ace_clustering, dims=["object"], coords={"object": clusterings.object})

    return clusterings

def transform_to_binary(clusterings, threshold=0.5):
    """
    Transform the clustering dataset into a binary membership matrix.
    Objects that are noise in more than the given threshold of clusterings are excluded.
    
    Parameters:
    - clusterings: xarray.Dataset with each variable as a clustering.
    - threshold: Proportion of clusterings where an object must not be noise to be considered valid.
    
    Returns:
    - membership_matrix: Binary membership matrix (objects x clusters).
    - valid_objects: Boolean array indicating whether each object is valid.
    """
    data = []
    num_clusterings = len(clusterings.data_vars)
    noise_counts = np.zeros(clusterings.object.size, dtype=int)

    # Flatten each clustering and count noise occurrences
    for var in clusterings.data_vars:
        clustering = clusterings[var].values.flatten()
        mask = clustering != -1
        noise_counts += ~mask  # Increment noise count for objects labeled as -1
        unique_clusters = np.unique(clustering[mask])
        binary_repr = np.eye(len(unique_clusters))[np.searchsorted(unique_clusters, clustering[mask])]
        data.append(binary_repr)

    # Compute valid objects based on the noise threshold
    valid_objects = (num_clusterings - noise_counts) / num_clusterings >= threshold

    # Concatenate valid cluster binary representations
    membership_matrix = np.hstack(data) if data else np.empty((len(valid_objects), 0))

    return membership_matrix, valid_objects


# Generate Consensus Clusters
def generate_consensus_clusters(binary_matrix, n_final_clusters, alpha1, delta_alpha):
    """
    Generate consensus clusters by merging clusters iteratively based on similarity.
    Each cluster is represented as an array of object indices.

    :param binary_matrix: Binary membership matrix (objects x clusters).
    :param n_final_clusters: Desired number of final clusters.
    :param alpha1: Initial similarity threshold for merging clusters.
    :param delta_alpha: Step size for decreasing alpha1.
    :return: List of arrays, where each array contains indices of objects in a consensus cluster.
    """
    # Calculate similarity matrix using set correlation
    similarity_matrix = calculate_similarity_matrix(binary_matrix)

    # Initialize each cluster as a separate group of object indices
    clusters = [np.where(binary_matrix[:, i])[0] for i in range(binary_matrix.shape[1])]

    # Iteratively merge clusters until the desired number of clusters is achieved
    while len(clusters) > n_final_clusters:
        max_sim = -1  # Track the maximum similarity found in the current iteration
        merge_candidates = None

        # Search for the most similar pair of clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # Use the precomputed similarity matrix
                sim = similarity_matrix[i, j]
                if sim > max_sim and sim >= alpha1:
                    max_sim = sim
                    merge_candidates = (i, j)

        # If no clusters can be merged, lower the threshold
        if merge_candidates is None:
            alpha1 -= delta_alpha
            continue

        # Merge the most similar clusters
        i, j = merge_candidates
        clusters[i] = np.union1d(clusters[i], clusters[j])  # Merge clusters
        clusters.pop(j)  # Remove the merged cluster

        # Update similarity matrix for the newly merged cluster
        similarity_matrix = update_similarity_matrix(similarity_matrix, clusters, i)

    return clusters


# Calculate Set Correlation
def calculate_set_correlation(cluster1, cluster2, total_objects):
    """
    Compute the set correlation between two clusters.
    :param cluster1: Array of indices representing the first cluster.
    :param cluster2: Array of indices representing the second cluster.
    :param total_objects: Total number of objects in the dataset.
    :return: Set correlation score.
    """
    # Compute intersection size
    intersection_size = len(np.intersect1d(cluster1, cluster2))
    
    # Compute cluster sizes
    size1 = len(cluster1)
    size2 = len(cluster2)

    # Compute numerator and denominator for set correlation
    numerator = intersection_size - (size1 * size2) / total_objects
    denominator = np.sqrt(size1 * size2 * (total_objects - size1) * (total_objects - size2))

    # Return normalized set correlation
    return numerator / denominator if denominator != 0 else 0

# Calculate Similarity Matrix using Set Correlation
def calculate_similarity_matrix(binary_matrix):
    """
    Calculate the cluster similarity matrix using set correlation.
    :param binary_matrix: Binary membership matrix.
    :return: Symmetric similarity matrix (Sc).
    """
    num_clusters = binary_matrix.shape[1]
    total_objects = binary_matrix.shape[0]
    
    similarity_matrix = np.zeros((num_clusters, num_clusters))
    
    for i in range(num_clusters):
        for j in range(i, num_clusters):
            cluster1 = np.where(binary_matrix[:, i])[0]
            cluster2 = np.where(binary_matrix[:, j])[0]
            
            similarity = calculate_set_correlation(cluster1, cluster2, total_objects)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity  # Symmetry
    
    return similarity_matrix

def update_similarity_matrix(similarity_matrix, clusters, merged_index):
    """
    Efficiently update the similarity matrix after two clusters are merged.
    :param similarity_matrix: Existing similarity matrix.
    :param clusters: List of current clusters (after merging).
    :param merged_index: Index of the newly merged cluster.
    :return: Updated similarity matrix.
    """
    total_objects = similarity_matrix.shape[0]
    num_clusters = len(clusters)

    # Create a new similarity matrix with one fewer cluster
    new_sim_matrix = np.zeros((num_clusters, num_clusters))

    for i in range(num_clusters):
        for j in range(i, num_clusters):
            if i == j:
                new_sim_matrix[i, j] = 1  # Self-similarity
            elif i == merged_index or j == merged_index:
                # Recalculate similarity involving the newly merged cluster
                new_sim_matrix[i, j] = calculate_set_correlation(clusters[i], clusters[j], total_objects)
                new_sim_matrix[j, i] = new_sim_matrix[i, j]  # Symmetry
            else:
                # Copy existing similarity values for unaffected pairs
                old_i = i if i < merged_index else i + 1
                old_j = j if j < merged_index else j + 1
                new_sim_matrix[i, j] = similarity_matrix[old_i, old_j]
                new_sim_matrix[j, i] = new_sim_matrix[i, j]

    return new_sim_matrix


# Resolve Uncertain Assignments
def resolve_uncertain_objects(binary_matrix, clusters, valid_objects, alpha2):
    """
    Resolve uncertain assignments for objects, retaining noise labels for unclustered objects.
    """
    final_labels = -1 * np.ones(binary_matrix.shape[0], dtype=int)

    for i, cluster in enumerate(clusters):
        for obj in cluster:
            if final_labels[obj] == -1:
                final_labels[obj] = i
            
            else: # if object appears to be assigned to multiple clusters

                # initialise
                max_similarity = 0
                best_cluster = -1

                for j, cluster in enumerate(clusters):
                    similarity = binary_matrix[obj, cluster].mean() # calculate mean similarity of the object to each cluster
                    if similarity > max_similarity and similarity >= alpha2:
                        max_similarity = similarity
                        best_cluster = j 
                if best_cluster != -1: 
                    final_labels[obj] = best_cluster #  choose the cluster with highest similarity that exceeds alph2 threshold
                    # if no cluster satisfies the threshold, the object is labeled as noise 

    # Reset noise objects to -1
    final_labels[~valid_objects] = -1

    return final_labels
