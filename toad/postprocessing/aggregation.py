from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from toad.clustering import sorted_cluster_labels
from toad.utils import get_unique_variable_name


class Aggregation:
    """
    Aggregation methods for TOAD objects.
    """

    def __init__(self, toad):
        self.td = toad

    def cluster_occurrence_rate(
        self,
        cluster_vars: list[str] | None = None,
    ) -> xr.DataArray:
        """Calculate the normalized occurrence rate of points being part of any cluster.

        For each point in space, calculates how many times it is part of a cluster
        (not noise) across different clustering variables, normalized by the total
        number of clusterings. This is done by checking if each point was ever part
        of a cluster (cluster label > -1) for each clustering variable, summing these
        occurrences, and dividing by the total number of clustering variables.

        Args:
            cluster_vars: List of clustering variable names to consider. If None,
                uses all clustering variables in the TOAD object. Each variable should
                contain cluster labels where -1 indicates noise points and values >= 0
                indicate cluster membership.

        Returns:
            DataArray containing the normalized cluster occurrence rate for each point.
            Values range from 0 (never in a cluster) to 1 (always in a cluster).
            The output variable name will be "cluster_occurence_rate" with a numeric
            suffix if that name already exists in the dataset.

        Example:
            If a point is part of a cluster in 2 out of 3 clustering variables,
            its occurrence rate would be 2/3 ≈ 0.67.
        """
        # Determine clustering variables
        cluster_vars = cluster_vars if cluster_vars else self.td.cluster_vars

        # Normalize by the total number of clusterings
        num_clusterings = len(cluster_vars)
        cluster_normalized = xr.where(
            self.td.data[cluster_vars[0]].max(dim=self.td.time_dim) > -1,
            1.0 / num_clusterings,
            0,
        )
        # in-place summation to conserve memory
        for cluster_var in cluster_vars[1:]:
            cluster_normalized += xr.where(
                self.td.data[cluster_var].max(dim=self.td.time_dim) > -1,
                1.0 / num_clusterings,
                0,
            )

        # Set name
        output_label = get_unique_variable_name(
            "cluster_occurence_rate", self.td.data, self.td.logger
        )
        cluster_normalized = cluster_normalized.rename(output_label)

        # Add attributes
        cluster_normalized.attrs.update(
            {
                "cluster_vars": cluster_vars,
                "description": "Normalized occurrence rate of points being part of any cluster",
            }
        )

        return cluster_normalized

    def cluster_consistency(self, cluster_vars: list[str] | None = None):
        """
        Evaluate the spatial consistency of cluster membership for each grid cell
        across multiple clustering variables (e.g., from different models).

        This function measures how stable the *spatial neighborhood* of each grid cell's
        cluster is across clustering variables, using the Jaccard similarity.

        For each grid cell:
        1. Identify which cluster it belongs to in each clustering variable.
        2. For every pair of clusterings, retrieve the full set of grid cells that were
        in the same cluster, and compute the Jaccard similarity between these sets.
        (Jaccard = |A ∩ B| / |A ∪ B|)
        3. Average the Jaccard scores over all clustering pairs to obtain a consistency score.

        Interpretation:
        - A score near 1.0 means the cell consistently clusters with the same spatial
        neighborhood across different clustering setups.
        - A score near 0.0 means the cell’s cluster context varies substantially.
        - NaN is returned if the cell is unclustered (noise) in all clustering variables.

        Args:
            td: TOAD object containing clustering results.
            cluster_vars: Optional list of cluster variable names. If None, uses td.cluster_vars.

        Returns:
            xr.DataArray: Stability scores per grid cell, with the same spatial shape
                        as the input data and values in [0, 1] or NaN.
        """
        # get all clsuter vars if nothing is provided
        if cluster_vars is None:
            cluster_vars = list(self.td.cluster_vars)

        n_vars = len(cluster_vars)

        # Get grid dimensions from first clustering
        data0 = self.td.data[cluster_vars[0]].isel({self.td.time_dim: 0})
        N = data0.size
        grid_shape = data0.shape

        # Cache which grid cells belonged to each cluster
        membership_lookup = precompute_spatial_memberships(self.td, cluster_vars)

        # For each grid cell, get its cluster ID in each clustering
        # Take max over time since cluster IDs are consistent
        cluster_maps = np.stack(
            [
                self.td.data[cvar].max(dim=self.td.time_dim).values.flatten()
                for cvar in cluster_vars
            ],
            axis=1,
        )  # shape: (N, n_vars)

        # Compute stability for each grid cell
        stability_scores = np.zeros(N, dtype=np.float32)
        for i in range(N):
            jaccards = []
            # Compare each pair of clusterings
            for v1, v2 in combinations(range(n_vars), 2):
                cid1 = cluster_maps[i, v1]
                cid2 = cluster_maps[i, v2]
                if cid1 < 0 or cid2 < 0:
                    continue  # Skip if cell was noise in either clustering

                # Get spatial extent of both clusters
                members1 = membership_lookup.get((cluster_vars[v1], cid1), set())
                members2 = membership_lookup.get((cluster_vars[v2], cid2), set())
                jaccards.append(jaccard_similarity(members1, members2))

            # Average similarities, or NaN if cell was noise in all comparisons
            stability_scores[i] = np.mean(jaccards) if jaccards else np.nan

        return xr.DataArray(
            stability_scores.reshape(grid_shape),
            coords=data0.coords,
            dims=data0.dims,
            name="Jaccard similarity",
        )

    def cluster_consensus_spatial(
        self,
        cluster_vars: List[str] | None = None,
        min_consensus: float = 0.5,
        top_n_clusters: int | None = None,
        neighbor_connectivity: int = 8,
    ) -> Tuple[xr.Dataset, pd.DataFrame]:
        """
        This function implements a consensus aggregation closely related to evidence accumulation clustering (EAC) from [Fred+Jain2005]_,
        but reformulated for spatial grid data. Instead of dense all-pairs co-association, we accumulate "votes" only between spatially neighboring cells,
        yielding a scalable sparse adjacency graph from which consensus regions are formed.

        Builds a spatial consensus map from multiple clustering results by collapsing time within each input,
        constructing a pixel-adjacency co-association graph, thresholding by agreement, and labeling the resulting
        connected components.

        The function produces robust, spatially coherent regions that persist across clustering choices/variables
        by combining clusterings through a graph-based consensus method.

        Args:
            cluster_vars (List[str] or None): List of clustering variable names to include in the consensus.
                If None, uses all cluster variables in self.td.cluster_vars.
            min_consensus (float): Minimum fraction (in [0,1]) of clusterings that must support an edge
                (pixel adjacency) for it to be included in the consensus graph. Higher values = stricter consensus.
            top_n_clusters (int or None): If set, only top N largest clusters (per clustering) are used when voting for edges.
                If None, all clusters are included.
            neighbor_connectivity (int): Neighborhood connectivity for spatial adjacency, either 4 (Von Neumann) or 8 (Moore, default).

        Returns:
            Tuple[xr.Dataset, pd.DataFrame]:
                - xr.Dataset with:
                    * consensus_clusters (int32, shape: (y, x)): Consensus cluster/component labels; -1 indicates noise or unassigned.
                    * consensus_consistency (float32, shape: (y, x)): Local mean of co-association edge weights around each pixel,
                      reflecting neighborhood agreement across input cluster maps.
                - pd.DataFrame: One row per consensus cluster with columns:
                    * cluster_id
                    * mean_consistency
                    * size
                    * mean_{space_dim0}, mean_{space_dim1} (average spatial coordinates for the cluster)

        Algorithm Overview:
            1. Collapse time in each clustering map: mark a pixel as "clustered" if it is ever assigned to a cluster at any time.
            2. For each clustering, obtain the spatial footprint of each cluster. Optionally, restrict to the top N clusters.
            3. For each cluster, increment votes for each pair of adjacent (connected) pixels within that cluster.
            4. Accumulate edge votes across all clusterings, then normalize by the number of clustering maps.
            5. Retain only those edges (pixel adjacencies) present in at least `min_consensus` fraction of clusterings.
            6. Construct an undirected sparse graph with surviving edges; run connected components labeling.
            7. Relabel clusters in order of descending size for interpretability; assign -1 to isolated (noise) pixels.
            8. Compute, for each pixel, the mean strength (consistency) of its incident consensus edges.

        Notes:
            * Adjacency can be 4- or 8-connected; 8-neighborhood is default for spatial coherence.
            * Consensus clusters represent regions whose internal edges are repeatedly co-clustered
              across the inputs and may be chained via single-link paths.
            * Large, non-compact clusters can form if consensus is too lenient; increase `min_consensus` or
              apply additional filtering for tighter components if needed.
            * Suitable for identifying robust tipping regions or domains unaffected by clustering noise.

        Example:
            >>> ds, summary_df = obj.cluster_consensus_spatial(
                    cluster_vars=['clust_a', 'clust_b'], min_consensus=0.7, neighbor_connectivity=8)

        Raises:
            ValueError: If neighbor_connectivity is not 4 or 8.
            AssertionError: If no cluster_vars are found.
        """
        # Get list of cluster variables if not provided
        if cluster_vars is None:
            cluster_vars = list(self.td.cluster_vars)
        assert len(cluster_vars) > 0, "No cluster variables provided/found."

        # Check if neighbor connectivity is valid
        if neighbor_connectivity not in (4, 8):
            raise ValueError(
                f"`neighbor_connectivity` must be 4 or 8, but got {neighbor_connectivity}."
            )

        # Get dimensions from first clustering
        sample = self.td.data[cluster_vars[0]]
        spatial_dims = self.td.space_dims

        # Get array sizes
        y_len = sample.sizes[spatial_dims[0]]
        x_len = sample.sizes[spatial_dims[1]]

        # Create flattened index array for 2D grid
        N = y_len * x_len
        flat_idx_2d = np.arange(N, dtype=np.int64).reshape((y_len, x_len))

        # Store coordinates for output arrays (include 2D coords like latitude/longitude)
        coords_spatial = {
            name: coord
            for name, coord in sample.coords.items()
            if (len(coord.dims) > 0) and set(coord.dims).issubset(spatial_dims)
        }
        # Ensure the index coordinates for each spatial dim are present
        for d in spatial_dims:
            coords_spatial.setdefault(d, sample[d])

        # Lists to store graph edges between adjacent cells
        edge_rows, edge_cols = [], []

        # Process each clustering
        for cvar in cluster_vars:
            labels = self.td.data[cvar].values  # (T, Y, X)

            # Get unique cluster IDs, optionally taking only top N largest
            unique_ids = self.td.get_cluster_ids(cvar)
            if unique_ids.size == 0:
                continue
            if top_n_clusters is not None and top_n_clusters > 0:
                unique_ids = unique_ids[:top_n_clusters]

            # Per-map deduplication of edges
            map_edges: set[tuple[int, int]] = set()

            # For each cluster, find adjacent cells that were ever in it
            for cid in unique_ids:
                mask2d = (labels == cid).any(axis=0)  # (Y, X)
                add_adjacent_true_pairs(
                    mask2d, map_edges, flat_idx_2d, neighbor_connectivity == 8
                )

            if map_edges:
                r, c = zip(*map_edges)
                edge_rows.extend(r)
                edge_cols.extend(c)

        # If no edges found, return all cells as noise
        if len(edge_rows) == 0:
            da_consensus_labels = xr.DataArray(
                np.full((y_len, x_len), -1, dtype=np.int32),
                coords=coords_spatial,
                dims=spatial_dims,
                name="consensus_clusters",
            )
            da_consistency = xr.DataArray(
                np.full((y_len, x_len), 0, dtype=np.float32),
                coords=coords_spatial,
                dims=spatial_dims,
                name="consensus_consistency",
            )
            ds_out = xr.Dataset(
                {
                    "consensus_clusters": da_consensus_labels,
                    "consensus_consistency": da_consistency,
                }
            )
            summary_df = build_consensus_summary_df(
                da_consensus_labels, da_consistency, self.td, spatial_dims
            )
            return ds_out, summary_df

        # Create sparse adjacency matrix
        rows = np.array(edge_rows, dtype=np.int64)
        cols = np.array(edge_cols, dtype=np.int64)
        data = np.ones(len(rows), dtype=np.float32)
        M = len(cluster_vars)

        # Convert to normalized adjacency matrix (fraction of maps supporting each undirected edge)
        coo = coo_matrix((data, (rows, cols)), shape=(N, N))
        csr = coo.tocsr()
        csr.sum_duplicates()
        csr.data = np.divide(csr.data, float(M))

        # Symmetrize by taking maximum to ensure undirected adjacency
        csr = csr.maximum(csr.T)

        # Remove edges below consensus threshold
        mask_keep = csr.data >= float(min_consensus)
        csr.data = np.where(mask_keep, csr.data, 0).astype(csr.data.dtype, copy=False)
        csr.eliminate_zeros()

        # If no edges remain after thresholding, return all noise
        if csr.nnz == 0:
            da_consensus_labels = xr.DataArray(
                np.full((y_len, x_len), -1, dtype=np.int32),
                coords=coords_spatial,
                dims=spatial_dims,
                name="consensus_clusters",
            )
            da_consistency = xr.DataArray(
                np.full((y_len, x_len), 0, dtype=np.float32),
                coords=coords_spatial,
                dims=spatial_dims,
                name="consensus_consistency",
            )
            ds_out = xr.Dataset(
                {
                    "consensus_clusters": da_consensus_labels,
                    "consensus_consistency": da_consistency,
                }
            )
            summary_df = build_consensus_summary_df(
                da_consensus_labels, da_consistency, self.td, spatial_dims
            )
            return ds_out, summary_df

        # Compute per-node average edge weight
        node_sum = np.array(csr.sum(axis=1)).ravel()
        node_deg = np.array(csr.count_nonzero(axis=1)).ravel().astype(np.float32)
        consensus_consistency = np.divide(
            node_sum, node_deg, out=np.zeros_like(node_sum), where=node_deg > 0
        ).reshape((y_len, x_len))

        # Find connected components in thresholded graph
        bin_adj = csr.copy()
        bin_adj.data[:] = 1.0
        bin_adj = bin_adj.maximum(bin_adj.T)
        _, labels_flat = connected_components(
            bin_adj, directed=False, return_labels=True
        )

        # Reshape labels back to 2D and mark isolated points as noise
        labels_2d = labels_flat.reshape((y_len, x_len))
        deg = np.array(bin_adj.getnnz(axis=1)).reshape((y_len, x_len))
        labels_2d[deg == 0] = -1

        # Sort cluster labels by size
        flat = labels_2d.flatten()
        flat_sorted = sorted_cluster_labels(flat)
        labels_2d = flat_sorted.reshape((y_len, x_len))

        # Create output DataArrays
        da_consensus_labels = xr.DataArray(
            labels_2d,
            coords=coords_spatial,
            dims=spatial_dims,
            name="consensus_clusters",
        )
        da_consensus_labels.attrs.update(
            {
                "cluster_vars": cluster_vars,
                "min_consensus": min_consensus,
                "top_n_clusters": top_n_clusters,
                "neighbor_connectivity": neighbor_connectivity,
                "description": "Spatial consensus clusters (time-collapsed).",
            }
        )

        da_consistency = xr.DataArray(
            consensus_consistency,
            coords=coords_spatial,
            dims=spatial_dims,
            name="consensus_consistency",
        )

        ds_out = xr.Dataset(
            {
                "consensus_clusters": da_consensus_labels,
                "consensus_consistency": da_consistency,
            }
        )

        summary_df = build_consensus_summary_df(
            da_consensus_labels, da_consistency, self.td, spatial_dims
        )
        return ds_out, summary_df


def jaccard_similarity(set_a, set_b):
    """
    Compute Jaccard similarity between two sets: |A ∩ B| / |A ∪ B|

    Args:
        set_a, set_b: Input sets to compare

    Returns:
        float: Similarity score in [0,1]. 1.0 means identical sets,
            0.0 means no overlap. Returns 1.0 if both sets are empty.
    """
    a = set(set_a)
    b = set(set_b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def precompute_spatial_memberships(td, cluster_vars):
    """
    Precompute flattened membership sets for each (cluster_var, cluster_id) pair.
    For each cluster in each clustering, stores which grid cells were ever part of that cluster.

    Args:
        td: TOAD instance containing the cluster variables
        cluster_vars: List of cluster variable names to process

    Returns:
        dict: Maps (cluster_var, cluster_id) tuples to sets of flattened grid cell indices.
            Only includes non-noise clusters (cluster_id >= 0).
    """
    lookup = {}
    for cvar in cluster_vars:
        clusters = td.get_clusters(cvar)  # shape: (time, lat, lon)
        cids = td.get_cluster_ids(cvar)

        for cid in cids:
            if cid < 0:  # Skip noise points (labeled as -1)
                continue
            # Find grid cells that were ever part of this cluster
            mask = (clusters == cid).any(dim=td.time_dim)
            flat_idxs = np.flatnonzero(mask.values.flatten())
            lookup[(cvar, cid)] = set(flat_idxs)

    return lookup


def add_adjacent_true_pairs(
    mask2d: np.ndarray,
    edge_set: set[tuple[int, int]],
    flat_idx_2d: np.ndarray,
    use_eight: bool,
) -> None:
    """Adds undirected neighbor edges for True cells in a 2D mask.

    For each cell in a 2D boolean mask, this function adds undirected edges between
    all pairs of adjacent True cells to the provided edge set. Adjacency is determined
    by 4- or 8-connected neighborhoods.

    Args:
        mask2d (np.ndarray): A 2D boolean array indicating active cells (True).
        edge_set (set[tuple[int, int]]): A set to which undirected edge tuples (i, j)
            will be added, where i and j are flattened pixel indices. Edges are deduplicated
            such that (i, j) and (j, i) are treated as the same.
        flat_idx_2d (np.ndarray): A 2D array of flattened indices, shape (y_len, x_len).
        use_eight (bool): If True, consider diagonally adjacent neighbors (8-connectivity).
            If False, only include horizontal and vertical neighbors (4-connectivity).

    Returns:
        None
    """
    # Horizontal neighbors
    common = mask2d[:, :-1] & mask2d[:, 1:]
    if common.any():
        a = flat_idx_2d[:, :-1][common].ravel()
        b = flat_idx_2d[:, 1:][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))
    # Vertical neighbors
    common = mask2d[:-1, :] & mask2d[1:, :]
    if common.any():
        a = flat_idx_2d[:-1, :][common].ravel()
        b = flat_idx_2d[1:, :][common].ravel()
        for i, j in zip(a.tolist(), b.tolist()):
            edge_set.add((i, j) if i < j else (j, i))
    if use_eight:
        # Diagonal neighbors: top-left to bottom-right
        common = mask2d[:-1, :-1] & mask2d[1:, 1:]
        if common.any():
            a = flat_idx_2d[:-1, :-1][common].ravel()
            b = flat_idx_2d[1:, 1:][common].ravel()
            for i, j in zip(a.tolist(), b.tolist()):
                edge_set.add((i, j) if i < j else (j, i))
        # Diagonal neighbors: top-right to bottom-left
        common = mask2d[:-1, 1:] & mask2d[1:, :-1]
        if common.any():
            a = flat_idx_2d[:-1, 1:][common].ravel()
            b = flat_idx_2d[1:, :-1][common].ravel()
            for i, j in zip(a.tolist(), b.tolist()):
                edge_set.add((i, j) if i < j else (j, i))


def build_consensus_summary_df(
    labels2d: xr.DataArray,
    consistency2d: xr.DataArray,
    td,
    spatial_dims: Tuple[str, str],
) -> pd.DataFrame:
    """Builds a summary DataFrame of cluster statistics from 2D label and consistency arrays.

    Computes descriptive statistics for spatial clusters, including mean consistency,
    cluster size, spatial centroids, transition-time statistics, and model spread.

    Args:
        labels2d (xr.DataArray): A 2D array containing integer cluster labels for each spatial cell.
            Cells labeled as -1 are considered noise and ignored.
        consistency2d (xr.DataArray): A 2D array of consistency scores for each spatial cell.
        td: TOAD object containing data and cluster variables.
        spatial_dims (Tuple[str, str]): Tuple of (dim0, dim1) spatial dimension names.

    Returns:
        pd.DataFrame: A DataFrame with one row per cluster (excluding noise), containing columns:
            - cluster_id (int): The cluster label.
            - mean_consistency (float): Mean consistency score within the cluster.
            - size (int): Number of cells belonging to the cluster.
            - mean_<spatial_dim_0> (float): Mean location in the first spatial dimension.
            - mean_<spatial_dim_1> (float): Mean location in the second spatial dimension.
            - mean_transition_time (float): Mean transition time averaged across models and spatial cells.
            - std_transition_time (float): Standard deviation of transition times across models.
            - mean_within_model_spread (float): Mean spatial spread of transition times within models.
            - std_within_model_spread (float): Standard deviation of within-model spread across models.

    Notes:
        - If all cells are noise (i.e., all labels are -1), an empty DataFrame with the proper columns is returned.
        - Transition-time and spread statistics are filled with NaN if unavailable.
    """
    sd0, sd1 = spatial_dims
    dim = labels2d.name if labels2d.name else "cluster"
    cluster_map = labels2d.where(labels2d != -1)

    if np.all(labels2d.values == -1):
        cols = [
            "mean_consistency",
            "size",
            f"mean_{sd0}",
            f"mean_{sd1}",
            "mean_transition_time",
            "std_transition_time",
            "mean_within_model_spread",
            "std_within_model_spread",
        ]
        return pd.DataFrame({c: [] for c in cols})

    # === Base metrics from xarray groupby ===
    mean_consistency = consistency2d.groupby(cluster_map).mean(skipna=True)
    cluster_sizes = (
        xr.ones_like(cluster_map)
        .where(cluster_map.notnull())
        .groupby(cluster_map)
        .sum(skipna=True)
    )
    space_dim0_mean = (
        td.data[sd0].where(cluster_map >= 0).groupby(cluster_map).mean(skipna=True)
    )
    space_dim1_mean = (
        td.data[sd1].where(cluster_map >= 0).groupby(cluster_map).mean(skipna=True)
    )

    df = pd.DataFrame(
        {
            "cluster_id": mean_consistency[dim].values.astype(int),
            "mean_consistency": mean_consistency.values.astype(np.float32),
            "size": cluster_sizes.values.astype(np.int32),
            f"mean_{sd0}": space_dim0_mean.values.astype(np.float32),
            f"mean_{sd1}": space_dim1_mean.values.astype(np.float32),
        }
    )

    # === Transition-time metrics (vectorized, readable) ===
    # Build per-model transition-time maps (threshold=0 selects all transitions)
    transition_time_maps = []
    for cluster_var in td.cluster_vars:
        shift_var = td.data[cluster_var].shifts_variable
        transition_time_maps.append(
            td.cluster_stats(shift_var).time.compute_transition_time(
                shift_threshold=0.0
            )
        )

    if len(transition_time_maps) == 0:
        # No models available: return NaNs for transition-time statistics
        df_transitions = pd.DataFrame(
            {
                "cluster_id": df["cluster_id"].values.astype(int),
                "mean_transition_time": np.nan,
                "std_transition_time": np.nan,
                "mean_within_model_spread": np.nan,
                "std_within_model_spread": np.nan,
            }
        )
    else:
        # Stack models into a single array along a named 'cluster_var' axis
        cluster_var_index = pd.Index(td.cluster_vars, name="cluster_var")
        transition_time_stack = xr.concat(transition_time_maps, dim=cluster_var_index)

        # For each cluster id: compute per-model spatial mean/std (reduces 'y','x')
        per_cluster_per_model_mean = transition_time_stack.groupby(cluster_map).mean(
            skipna=True
        )
        per_cluster_per_model_std = transition_time_stack.groupby(cluster_map).std(
            skipna=True
        )

        # Aggregate across models for each cluster id
        mean_transition_time = per_cluster_per_model_mean.mean(
            dim="cluster_var", skipna=True
        )
        std_transition_time_by = per_cluster_per_model_mean.std(
            dim="cluster_var", skipna=True
        )
        mean_within_model_spread = per_cluster_per_model_std.mean(
            dim="cluster_var", skipna=True
        )
        std_within_model_spread = per_cluster_per_model_std.std(
            dim="cluster_var", skipna=True
        )

        # Build DataFrame with the same grouping dimension as above
        group_dim = mean_consistency.dims[0]
        df_transitions = pd.DataFrame(
            {
                "cluster_id": mean_transition_time[group_dim].values.astype(int),
                "mean_transition_time": mean_transition_time.values.astype(np.float32),
                "std_transition_time": std_transition_time_by.values.astype(np.float32),
                "mean_within_model_spread": mean_within_model_spread.values.astype(
                    np.float32
                ),
                "std_within_model_spread": std_within_model_spread.values.astype(
                    np.float32
                ),
            }
        )

    # Merge side-by-side
    df = df.merge(df_transitions, on="cluster_id", how="left")

    return df
