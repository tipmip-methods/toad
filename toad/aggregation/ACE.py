"""ACE aggregation module

Adaptive Clustering Ensemble method from: 
Alqurashi, T., & Wang, W. (2018). Clustering ensemble method. 
International Journal of Machine Learning and Cybernetics, 10(6), 1227–1246. https://doi.org/10.1007/s13042-017-0756-7

November 24
"""
import numpy as np
import xarray as xr

# ACE Algorithm Implementation
def aggregate(data: xr.Dataset, n_final_clusters, alpha1=0.8, alpha2=0.7, delta_alpha=0.1, noise_threshold = 0.5,
              first_dimension = "latitude", second_dimension = "longitude", third_dimension = "time",):
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
    # Step 1: Flatten dataset
    flattened_data = data.stack(object=(first_dimension, second_dimension, third_dimension))

    # Step 2: Binary Transformation
    binary_matrix, valid_objects = transform_to_binary(flattened_data, noise_threshold)

    # Step 3: Generate Consensus Clusters
    consensus_clusters = generate_consensus_clusters(binary_matrix, n_final_clusters, alpha1, delta_alpha)

    # Step 4: Resolve Uncertain Assignments
    ace_clustering = resolve_uncertain_objects(binary_matrix, consensus_clusters, valid_objects, alpha2)

     # Step 5: Reshape ACE clustering back to original dimensions
    reshaped_ace_clustering = xr.DataArray(
        ace_clustering.reshape(data.sizes[first_dimension], data.sizes[second_dimension], data.sizes[third_dimension]),
        coords={dim: data[dim] for dim in data.dims},
        dims=data.dims,
    )

    data_ace = data.copy()
    # Step 6: Add ACE clustering as a new variable in the dataset
    data_ace["ace_clustering"] = reshaped_ace_clustering

    return data_ace

def transform_to_binary(clusterings, threshold=0.5):
    """
    Transform the clustering dataset into a binary membership matrix.
    All objects are included in the matrix, and objects labeled as noise (-1)
    are represented with a row of zeros for the respective clustering.

    Parameters:
    - clusterings: xarray.Dataset with clustering variables.
    - threshold: Proportion of clusterings where an object must not be noise.

    Returns:
    - membership_matrix: Binary membership matrix (objects x clusters).
    - valid_objects: Boolean array indicating whether each object is valid.
    """
    data = []
    num_clusterings = len(clusterings.data_vars)
    noise_counts = np.zeros(clusterings.sizes["object"], dtype=int)

    for var in clusterings.data_vars:
        clustering = clusterings[var].values.flatten()

        # Mask valid (non-noise) objects
        mask = clustering != -1
        noise_counts += ~mask

        # Create binary representation only for valid clusters (exclude -1)
        unique_clusters = np.unique(clustering[mask])  # Unique clusters, excluding -1
        binary_repr = np.zeros((len(clustering), len(unique_clusters)), dtype=int)

        # Fill binary matrix for valid clusters
        for i, cluster_id in enumerate(unique_clusters):
            binary_repr[clustering == cluster_id, i] = 1  # Set 1 where the object belongs to this cluster

        # If the object is noise (-1), its row remains all zeros
        data.append(binary_repr)

    # Concatenate binary matrices along the columns (each clustering adds more columns)
    membership_matrix = np.hstack(data)

    # Compute valid objects based on the noise threshold
    valid_objects = (num_clusterings - noise_counts) / num_clusterings >= threshold

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

def resolve_uncertain_objects(binary_matrix, clusters, valid_objects, alpha2):
    """
    Assign objects to clusters, resolving uncertainty while minimizing cluster quality impact.
    
    Parameters:
    - binary_matrix: Binary membership matrix (objects x clusters).
    - clusters: List of clusters, where each cluster contains object indices.
    - valid_objects: Boolean array indicating valid objects (not noise).
    - alpha2: Certainty threshold for assigning objects to clusters.

    Returns:
    - Array of final cluster labels for all objects.
    """
    # Initialize final labels (-1 for noise)
    final_labels = -1 * np.ones(binary_matrix.shape[0], dtype=int)

    # Calculate membership similarity (Sx)
    cluster_membership = np.zeros((binary_matrix.shape[0], len(clusters)))
    for i, cluster in enumerate(clusters):
        for obj in cluster:
            cluster_membership[obj, i] += 1
    cluster_membership /= cluster_membership.sum(axis=1, keepdims=True)

    # Assign certain objects based on alpha2
    max_similarity = cluster_membership.max(axis=1)
    best_cluster = cluster_membership.argmax(axis=1)

    for obj in range(len(final_labels)):
        if valid_objects[obj] and max_similarity[obj] >= alpha2:
            final_labels[obj] = best_cluster[obj]

    # Handle uncertain objects
    uncertain_objects = (valid_objects) & (max_similarity < alpha2)
    for obj in np.where(uncertain_objects)[0]:
        min_quality_loss = float("inf")
        best_cluster = -1

        # Find the best cluster for the object
        for cluster_idx in range(len(clusters)):
            temp_clusters = [list(cluster) for cluster in clusters]  # Ensure temp_clusters is a list of lists
            temp_clusters[cluster_idx].append(obj)

            # Calculate quality (variance of similarities in the cluster)
            cluster_quality = np.var(cluster_membership[temp_clusters[cluster_idx], cluster_idx])

            if cluster_quality < min_quality_loss:
                min_quality_loss = cluster_quality
                best_cluster = cluster_idx

        # Assign the object to the best cluster
        if best_cluster != -1:
            final_labels[obj] = best_cluster

    return final_labels



# def resolve_uncertain_objects(binary_matrix, clusters, valid_objects, alpha2):
#     """
#     Resolve uncertain assignments for objects, retaining noise labels for unclustered objects.
#     """
#     # Initialize all objects as noise (-1)
#     final_labels = -1 * np.ones(binary_matrix.shape[0], dtype=int)

#     for cluster_index, cluster in enumerate(clusters):
#         for obj in cluster:
#             if obj < 0 or obj >= binary_matrix.shape[0]:
#                 continue  # Skip invalid object indices

#             # If object already assigned, resolve conflict
#             if final_labels[obj] != -1:
#                 max_similarity = 0
#                 best_cluster = -1

#                 for j, candidate_cluster in enumerate(clusters):
#                     if j >= binary_matrix.shape[1]:
#                         continue  # Avoid out-of-bounds access
#                     similarity = binary_matrix[obj, candidate_cluster].mean()  # Calculate similarity
#                     if similarity > max_similarity and similarity >= alpha2:
#                         max_similarity = similarity
#                         best_cluster = j
                
#                 if best_cluster != -1:
#                     final_labels[obj] = best_cluster  # Assign to best cluster
#             else:
#                 final_labels[obj] = cluster_index

#     # Ensure noise objects remain labeled as -1
#     final_labels[~valid_objects] = -1

#     return final_labels
