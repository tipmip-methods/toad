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
        min_consensus: float = 0.3,
        top_n_clusters: int | None = None,
        neighbor_connectivity: int = 8,
    ) -> Tuple[xr.Dataset, pd.DataFrame]:
        """
        Build a **spatial consensus map** from multiple clustering results by
        collapsing time within each input, constructing a pixel-adjacency
        **co-association graph**, thresholding by agreement, and labeling the
        resulting **connected components**.

        --------------------
        Data & assumptions
        --------------------
        - Time is **collapsed per cluster** inside each map using `any(axis=time)`,
        i.e., a pixel belongs to cluster *c* in that map if it was *ever* in *c*
        at any time. Consensus is therefore purely **spatial**.
        - Adjacency is **8-neighborhood** on the (y, x) grid (Moore) by default.
          Set `neighbor_connectivity=4` to use 4-neighborhood (Von Neumann).
        - Output label -1 marks pixels not connected to any consensus component.

        -------------
        Key tunables
        -------------
        - `min_consensus` ∈ [0,1]: minimum fraction of maps that must support an
        edge between two neighboring pixels for that edge to be kept. Higher
        values = fewer edges, more fragmentation; lower = more edges, possible
        chaining/merging.
        - `top_n_clusters`: if set, only the N largest clusters (per map) are used
        when voting for edges. This can focus consensus on the dominant features.
        - `neighbor_connectivity`: 4 or 8 (default 8). Spatial neighborhood used for
          co-association edges.

        --------
        Outputs
        --------
        Returns an (xr.Dataset, pandas.DataFrame) where the Dataset contains:
        - `consensus_clusters` (int32, `(y, x)`): Consensus component IDs; -1 = noise.
        - `consensus_consistency` (float32, `(y, x)`): Local mean of co-association edge weights
          around each pixel, reflecting how strongly its neighborhood co-occurs across input
          cluster maps.
        And the DataFrame contains one row per cluster with columns:
        - `cluster_id`, `mean_consistency`, `size`, `mean_{space_dim0}`, `mean_{space_dim1}`.

        ---------------------
        Algorithm (step-by-step)
        ---------------------
        High-level overview:
        1. Collapse time in each clustering map: mark a pixel as "clustered" if it is ever clustered at any time.
        2. Record pixel occurrence: count in how many maps each pixel is clustered.
        3. Vote for spatial edges: whenever two adjacent pixels appear in the same cluster in a map, add a vote for that edge.
        4. Normalize edge votes: convert votes to fractions of how many maps agree on each adjacency.
        5. Consensus thresholding: keep only edges agreed upon by at least min_consensus fraction of maps.
        6. Connected-components labeling: group pixels into consensus clusters using the surviving edges.
        7. Sort clusters by size and label noise pixels as -1.

        Detailed steps:
        1) **Select inputs**: if `cluster_vars` is None, use all available in
        `self.td.cluster_vars`. Read grid dims `(time, y, x)` from the first map.

        2) **Initialize containers**:
        - Build a flattened index array `I[y, x] -> {0..y*x-1}` for graph nodes.

        3) **Per-map spatial footprints & edge votes**:
        For each `cvar` in `cluster_vars`:
            a) Read labels `L(time, y, x)`.
            b) Get the set of cluster IDs (optionally keep only `top_n_clusters`).
            c) For each cluster ID `cid`:
                - Form the **2D footprint** `mask2d(y, x) = (L == cid).any(axis=time)`.
                - For every pair of **adjacent True pixels** in `mask2d`
                    (left-right and up-down), add one **vote** to the corresponding
                    edge `(i, j)` of the co-association graph.

        Remark: Edge votes are **accumulated across maps and clusters**.

        4) **Normalize edge weights**:
        Convert the vote counts to **fractions** by dividing by the number of
        maps `M = len(cluster_vars)`, so each edge weight `w_ij ∈ [0, 1]`.

        5) **Consensus thresholding**:
        Keep only edges with `w_ij >= min_consensus`. Remove all others,
        producing a sparse, undirected graph of “sufficiently agreed” adjacencies.

        6) **Connected components**:
        Binarize the remaining adjacency and run `connected_components` to assign
        a component ID to every pixel. Pixels with zero surviving degree are set
        to `-1` (noise / unassigned).

        7) **Label normalization (optional for readability)**:
        Reassign component IDs in descending order of component size so that
        the largest consensus region gets label 0, the next largest 1, etc.

        8) **Finalize arrays**:
        - Return the 2D label map `(y, x)`.

        ----------------
        Interpretation
        ----------------
        - A consensus component represents a spatial region whose **internal edges**
        (adjacent pixel pairs) are repeatedly co-clustered across the input maps.
        - Large regions can form via **single-link chaining**: a sequence of
        locally agreed edges may connect areas never jointly present in any
        single map. If undesirable, consider raising `min_consensus`, masking
        by a minimum `occurrence_rate`, using 8-neighborhood, or applying a
        post-processing bridge filter (e.g., k-core >= 2).

        -------------------
        Computational notes
        -------------------
        - Time and memory scale with the number of pixels (nodes) and
        the number of adjacency relations (edges). Using local (4- or 8-neighbor)
        connectivity keeps the graph sparse and the algorithm fast.
        - Graph operations are done in scipy sparse format (COO/CSR).
        - Although the consensus is purely spatial, the output `consensus_clusters`
        is repeated along the time dimension to maintain compatibility
        with the original data structure `(time, y, x)`.

        --------------------------
        Relation to tipping analysis
        --------------------------
        This procedure yields **robust, spatially coherent regions** that persist
        across clustering choices/variables. Such regions form natural units for
        aggregating resilience metrics (e.g., critical slowing down indicators) and
        for mapping candidate tipping domains with reduced sensitivity to any
        single clustering configuration.
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

        # Store coordinates for output arrays
        coords_spatial = {d: sample[d] for d in spatial_dims}

        def add_adjacent_true_pairs(
            mask2d: np.ndarray, edge_set: set[tuple[int, int]], use_eight: bool
        ) -> None:
            """Add undirected neighbor edges for True cells on a 2D mask (y,x), deduplicated per map.

            If `use_eight` is True, include diagonal neighbors (8-neighborhood). Otherwise, only 4-neighborhood.
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

        def _build_summary_ds(
            labels2d: xr.DataArray, consistency2d: xr.DataArray
        ) -> pd.DataFrame:
            """Compute 1D per-cluster summary variables and return as a DataFrame.

            Returns a DataFrame with columns:
            - mean_consistency
            - size
            - mean_{space_dim0}
            - mean_{space_dim1}
            and index cluster_id.
            """
            cluster_map = labels2d.where(labels2d != -1)
            sd0, sd1 = spatial_dims
            dim = cluster_map.name if cluster_map.name else "cluster"
            cname = "cluster_id"

            # If no clusters present, return empty DataFrame
            if np.all(labels2d.values == -1):
                cols = ["mean_consistency", "size", f"mean_{sd0}", f"mean_{sd1}"]
                return pd.DataFrame({c: [] for c in cols})

            mean_consistency = (
                consistency2d.groupby(cluster_map)
                .mean(skipna=True)
                .rename({dim: cname})
                .astype(np.float32)
            )
            cluster_sizes = (
                xr.full_like(cluster_map, 1, dtype=np.int32)
                .where(cluster_map.notnull())
                .groupby(cluster_map)
                .sum(skipna=True)
                .rename({dim: cname})
                .astype(np.int32)
            )
            space_dim0_mean = (
                self.td.data[sd0]
                .where(cluster_map >= 0)
                .groupby(cluster_map)
                .mean(skipna=True)
                .rename({dim: cname})
                .astype(np.float32)
            )
            space_dim1_mean = (
                self.td.data[sd1]
                .where(cluster_map >= 0)
                .groupby(cluster_map)
                .mean(skipna=True)
                .rename({dim: cname})
                .astype(np.float32)
            )

            ds = xr.Dataset(
                {
                    "mean_consistency": mean_consistency,
                    "size": cluster_sizes,
                    f"mean_{sd0}": space_dim0_mean,
                    f"mean_{sd1}": space_dim1_mean,
                }
            )
            return ds.to_dataframe().reset_index()

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
                add_adjacent_true_pairs(mask2d, map_edges, neighbor_connectivity == 8)

            if map_edges:
                r, c = zip(*map_edges)
                edge_rows.extend(r)
                edge_cols.extend(c)

        # If no edges found, return all cells as noise
        if len(edge_rows) == 0:
            final_name = get_unique_variable_name(
                "consensus_clusters", self.td.data, self.td.logger
            )
            da_consensus_labels = xr.DataArray(
                np.full((y_len, x_len), -1, dtype=np.int32),
                coords=coords_spatial,
                dims=spatial_dims,
                name=final_name,
            )
            da_consistency = xr.DataArray(
                np.full((y_len, x_len), 0, dtype=np.float32),
                coords=coords_spatial,
                dims=spatial_dims,
                name="consensus_consistency",
            )
            ds_out = xr.Dataset(
                {
                    final_name: da_consensus_labels,
                    "consensus_consistency": da_consistency,
                }
            )
            summary_df = _build_summary_ds(da_consensus_labels, da_consistency)
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

        # Compute per-node average edge weight
        node_sum = np.array(csr.sum(axis=1)).ravel()
        node_deg = np.array(csr.count_nonzero(axis=1)).ravel().astype(np.float32)
        consensus_consistency = np.divide(
            node_sum, node_deg, out=np.zeros_like(node_sum), where=node_deg > 0
        ).reshape((y_len, x_len))

        # If no edges remain after thresholding, return all noise
        if csr.nnz == 0:
            final_name = get_unique_variable_name(
                "consensus_clusters", self.td.data, self.td.logger
            )
            da_consensus_labels = xr.DataArray(
                np.full((y_len, x_len), -1, dtype=np.int32),
                coords=coords_spatial,
                dims=spatial_dims,
                name=final_name,
            )
            da_consistency = xr.DataArray(
                consensus_consistency,
                coords=coords_spatial,
                dims=spatial_dims,
                name="consensus_consistency",
            )
            ds_out = xr.Dataset(
                {
                    final_name: da_consensus_labels,
                    "consensus_consistency": da_consistency,
                }
            )
            summary_df = _build_summary_ds(da_consensus_labels, da_consistency)
            return ds_out, summary_df

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
        final_name = get_unique_variable_name(
            "consensus_clusters", self.td.data, self.td.logger
        )
        da_consensus_labels = xr.DataArray(
            labels_2d, coords=coords_spatial, dims=spatial_dims, name=final_name
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
            {final_name: da_consensus_labels, "consensus_consistency": da_consistency}
        )
        summary_df = _build_summary_ds(da_consensus_labels, da_consistency)
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
