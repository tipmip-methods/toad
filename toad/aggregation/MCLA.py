import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

def aggregate(
    data: xr.Dataset, k=3, method='medoids', hierarchical=False, dendrogram_threshold=None
):
    """
    Perform meta-clustering and assign meta-cluster labels to each object in the dataset.
    Additionally, return a dataset of chosen medoids.
    
    - ds: xarray.Dataset where each DataArray represents a clustering.
    - k: Number of final aggregated clusters (ignored if hierarchical=True and dendrogram_threshold is used).
    - method: Clustering method for meta-clustering ('medoids' or 'kmeans').
    - hierarchical: Whether to use hierarchical clustering instead of k-means/medoids.
    - dendrogram_threshold: If using hierarchical clustering, threshold to cut the dendrogram.
    Returns:
      - Updated xarray.Dataset with a new DataArray for meta-cluster labels.
      - xarray.Dataset containing the chosen medoids and their locations.
    """
    all_medoids = []
    medoid_details = []  # Store medoid coordinates and cluster IDs
    point_to_centroid_mapping = []

    # Process each clustering array in the dataset
    for name, da in data.items():
        print(f"Processing {name}...")

        cluster_labels = da.values  # Get the cluster labels
        coords = [da.coords[dim].values for dim in da.dims]  # Dynamically retrieve coords based on dims

        # Compute medoids for the current clustering
        medoids, medoid_indices = compute_medoids(cluster_labels, coords)
        all_medoids.append(medoids)

        # Add medoid details
        for i, medoid in enumerate(medoids):
            medoid_details.append({
                "cluster": i,
                **{dim: medoid[j] for j, dim in enumerate(da.dims)},
                "dataset": name
            })

        # Map each point to its cluster medoid index
        for cluster_id in np.unique(cluster_labels):
            if cluster_id == -1:  # Skip noise
                continue
            cluster_indices = np.where(cluster_labels == cluster_id)
            medoid_idx = len(all_medoids) - 1  # Current medoid index
            for idx in zip(*cluster_indices):
                point_to_centroid_mapping.append((idx, medoid_idx))
    
    # Combine all medoids
    combined_medoids = np.vstack(all_medoids)

    # Perform meta-clustering
    if hierarchical:
        print("Performing hierarchical clustering...")
        linkage_matrix = linkage(combined_medoids, method='ward')
        dendrogram(linkage_matrix)
        plt.title("Dendrogram of Combined Clusters")
        plt.xlabel("Cluster Index")
        plt.ylabel("Distance")
        plt.show()

        if dendrogram_threshold is not None:
            meta_labels = fcluster(linkage_matrix, t=dendrogram_threshold, criterion='distance')
        else:
            raise ValueError("Please provide a dendrogram_threshold for hierarchical clustering.")
    else:
        if method == 'medoids':
            print("Performing K-medoids clustering...")
            clusterer = KMedoids(n_clusters=k, random_state=42)
        elif method == 'kmeans':
            from sklearn.cluster import KMeans
            print("Performing K-means clustering...")
            clusterer = KMeans(n_clusters=k, random_state=42)
        else:
            raise ValueError("Invalid clustering method. Use 'medoids' or 'kmeans'.")

        meta_labels = clusterer.fit_predict(combined_medoids)

    # Map meta-cluster labels back to the original dataset
    meta_cluster_array = np.full_like(list(data.values())[0], fill_value=-1, dtype=int)

    for point, centroid_idx in point_to_centroid_mapping:
        meta_cluster_array[point] = meta_labels[centroid_idx]

    # Add meta-cluster labels to the dataset
    output = data.copy()
    output['meta_clusters'] = (list(data.dims), meta_cluster_array)

    # Create an xarray.Dataset for medoids
    medoid_coords = {
        "medoid_index": range(len(medoid_details)),
        **{dim: [m[dim] for m in medoid_details] for dim in data.dims},
        "cluster": [m["cluster"] for m in medoid_details],
        "dataset": [m["dataset"] for m in medoid_details],
    }

    medoids_ds = xr.Dataset(
        {
            "medoid_cluster": (["medoid_index"], medoid_coords["cluster"]),
            **{dim: (["medoid_index"], medoid_coords[dim]) for dim in data.dims},
            "dataset": (["medoid_index"], medoid_coords["dataset"]),
        },
        coords={"medoid_index": medoid_coords["medoid_index"]}
    )

    return output, medoids_ds


def compute_medoids(cluster_labels, coords):
    """
    Compute medoids for 3D DBSCAN clusters by flattening the space.
    - cluster_labels: N-dimensional array of DBSCAN cluster labels.
    - coords: List of coordinate arrays corresponding to cluster_labels' dimensions.
    Returns:
      - Medoid coordinates for each cluster.
      - Medoid indices (in the flattened array).
    """
    cluster_labels = cluster_labels.flatten()  # Flatten the cluster labels
    coords = np.stack([coord.flatten() for coord in coords], axis=1)  # Flatten coordinates into 2D array

    unique_clusters = np.unique(cluster_labels[cluster_labels != -1])  # Exclude noise
    medoids = []
    medoid_indices = []

    for cluster in unique_clusters:
        # Extract points belonging to the current cluster
        cluster_indices = np.where(cluster_labels == cluster)[0]
        cluster_points = coords[cluster_indices]

        # Compute pairwise distances within the cluster
        pairwise_distances = np.linalg.norm(
            cluster_points[:, None, :] - cluster_points[None, :, :], axis=-1
        )
        medoid_idx = np.argmin(pairwise_distances.sum(axis=1))  # Index of the medoid in cluster_points
        medoid_indices.append(cluster_indices[medoid_idx])  # Map back to original flattened index
        medoids.append(cluster_points[medoid_idx])  # Store the medoid's coordinates

    return np.array(medoids), np.array(medoid_indices)

