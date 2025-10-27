from itertools import combinations
from typing import List, Tuple

import numpy as np
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
    ) -> Tuple[xr.DataArray, xr.DataArray]:
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
        - Adjacency is **4-neighborhood** on the (y, x) grid (Von Neumann).
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

        --------
        Outputs
        --------
        Returns a tuple:
        (1) `consensus_clusters` (xr.DataArray[int32], shape `(time, y, x)`):
            Consensus component IDs repeated along time for convenience; -1 = noise.
        (2) `consensus_consistency` (xr.DataArray[float], shape (y, x)):
            Local mean of co-association edge weights around each pixel,
            reflecting how strongly its neighborhood co-occurs across input cluster maps.

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
        - Repeat the 2D label map along `time` to return `(time, y, x)` labels.

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

        # Get dimensions from first clustering
        time_dim = self.td.time_dim
        sample = self.td.data[cluster_vars[0]]
        dims = list(sample.dims)
        spatial_dims = self.td.space_dims

        # Get array sizes
        t_len = sample.sizes[time_dim]
        y_len = sample.sizes[spatial_dims[0]]
        x_len = sample.sizes[spatial_dims[1]]

        # Create flattened index array for 2D grid
        N = y_len * x_len
        flat_idx_2d = np.arange(N, dtype=np.int64).reshape((y_len, x_len))

        # Store coordinates for output arrays
        coords = {d: sample[d] for d in dims}
        coords_spatial = {d: sample[d] for d in spatial_dims}

        def add_adjacent_true_pairs(
            mask2d: np.ndarray, rows: List[int], cols: List[int]
        ) -> None:
            """Add 4-neighbor edges for True cells on a 2D mask (y,x)."""
            # Find horizontally adjacent True cells
            common = mask2d[:, :-1] & mask2d[:, 1:]
            if common.any():
                a = flat_idx_2d[:, :-1][common]
                b = flat_idx_2d[:, 1:][common]
                rows.extend(a.tolist())
                cols.extend(b.tolist())
            # Find vertically adjacent True cells
            common = mask2d[:-1, :] & mask2d[1:, :]
            if common.any():
                a = flat_idx_2d[:-1, :][common]
                b = flat_idx_2d[1:, :][common]
                rows.extend(a.tolist())
                cols.extend(b.tolist())

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

            # For each cluster, find adjacent cells that were ever in it
            for cid in unique_ids:
                mask2d = (labels == cid).any(axis=0)  # (Y, X)
                add_adjacent_true_pairs(mask2d, edge_rows, edge_cols)

        # If no edges found, return all cells as noise
        if len(edge_rows) == 0:
            final_name = get_unique_variable_name(
                "consensus_clusters", self.td.data, self.td.logger
            )
            da_consensus_labels = xr.DataArray(
                np.full((t_len, y_len, x_len), -1, dtype=np.int32),
                coords=coords,
                dims=dims,
                name=final_name,
            )
            da_consistency = xr.DataArray(
                np.full((y_len, x_len), 0, dtype=np.float32),
                coords=coords_spatial,
                dims=spatial_dims,
                name="consensus_consistency",
            )
            return da_consensus_labels, da_consistency

        # Create sparse adjacency matrix
        rows = np.array(edge_rows, dtype=np.int64)
        cols = np.array(edge_cols, dtype=np.int64)
        data = np.ones(len(rows), dtype=np.float32)
        M = len(cluster_vars)

        # Convert to normalized adjacency matrix
        coo = coo_matrix((data, (rows, cols)), shape=(N, N))
        csr = coo.tocsr()
        csr.data = np.divide(csr.data, float(M))

        # Remove edges below consensus threshold
        mask_keep = csr.data >= float(min_consensus)
        csr.data = csr.data * mask_keep.astype(csr.data.dtype)
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
                np.full((t_len, y_len, x_len), -1, dtype=np.int32),
                coords=coords,
                dims=dims,
                name=final_name,
            )
            da_consistency = xr.DataArray(
                consensus_consistency,
                coords=coords_spatial,
                dims=spatial_dims,
                name="consensus_consistency",
            )
            return da_consensus_labels, da_consistency

        # Find connected components in thresholded graph
        bin_adj = csr.copy()
        bin_adj.data[:] = 1.0
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

        # Repeat 2D labels along time dimension
        labels_3d = np.repeat(labels_2d[np.newaxis, ...], t_len, axis=0)

        # Create output DataArrays
        final_name = get_unique_variable_name(
            "consensus_clusters", self.td.data, self.td.logger
        )
        da_consensus_labels = xr.DataArray(
            labels_3d, coords=coords, dims=dims, name=final_name
        )
        da_consensus_labels.attrs.update(
            {
                "cluster_vars": cluster_vars,
                "min_consensus": min_consensus,
                "top_n_clusters": top_n_clusters,
                "description": "Spatial consensus clusters (time-collapsed).",
            }
        )

        da_consistency = xr.DataArray(
            consensus_consistency,
            coords=coords_spatial,
            dims=spatial_dims,
            name="consensus_consistency",
        )

        return da_consensus_labels, da_consistency


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
