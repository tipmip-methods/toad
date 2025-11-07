import logging
from itertools import combinations
from typing import List, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# from scipy.sparse import coo_matrix  # only used in utils helpers
from scipy.sparse.csgraph import connected_components

# from sklearn.neighbors import NearestNeighbors  # unused here; kept in utils
from toad.clustering import sorted_cluster_labels
from toad.regridding.base import BaseRegridder

# from toad.regridding.healpix import HealPixRegridder  # unused here; used in utils
from toad.utils import detect_latlon_names, get_unique_variable_name
from toad.utils.cluster_consensus_utils import (
    _add_adjacent_true_pairs,
    _build_consensus_summary_df,
    _build_empty_consensus_summary_df,
    _build_knn_edges_from_latlon,
    _build_knn_edges_from_regridder,
    _compute_weighted_consensus,
    _knn_edges_from_mask,
    _native_edges_from_mask,
)

logger = logging.getLogger("TOAD")


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

        **⚠️ Deprecated:** This function is conceptually superseded by `cluster_consensus()`.
        The Jaccard-based cluster consistency metric is retained for backwards compatibility
        but will be removed in a future release. The `consistency` field returned by
        `cluster_consensus()` provides a more efficient and interpretable measure of local
        co-association across runs.

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
        - A score near 0.0 means the cell's cluster context varies substantially.
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

    def cluster_consensus(
        self,
        cluster_vars: List[str] | None = None,
        min_consensus: float = 0.5,
        top_n_clusters: int | None = None,
        neighbor_connectivity: int = 8,
        regridder: BaseRegridder | None = None,
        k_neighbors: int = 8,
    ) -> Tuple[xr.Dataset, pd.DataFrame]:
        """Build a spatial consensus clustering from multiple clustering results.

        Implements a consensus aggregation method closely related to evidence accumulation clustering (EAC)
        from [Fred+Jain2005]_, but reformulated for spatial grid data. Instead of dense all-pairs
        co-association, we accumulate "votes" only between spatially neighboring cells, yielding a
        scalable sparse adjacency graph from which consensus regions are formed.

        The method produces robust, spatially coherent regions that persist across clustering
        choices/variables by combining clusterings through a graph-based consensus approach.

        Args:
            cluster_vars: List of clustering variable names to include in the consensus.
                If None, uses all cluster variables in self.td.cluster_vars.
            min_consensus: Minimum fraction (in [0,1]) of clusterings that must support an edge
                (pixel adjacency) for it to be included in the consensus graph. Higher values =
                stricter consensus. Default: 0.5.
            top_n_clusters: If set, only top N largest clusters (per clustering) are used when
                voting for edges. If None, all clusters are included. Default: None.
            neighbor_connectivity: Neighborhood connectivity for spatial adjacency when lat/lon
                coordinates are not available. Either 4 (Von Neumann, horizontal/vertical only)
                or 8 (Moore, including diagonals). Default: 8. This parameter controls index-based
                grid adjacency (not K-nearest neighbors) and is only used for grids without
                geographic coordinates; for lat/lon grids, see `k_neighbors`.
            regridder: Optional custom regridder. If None and data has regular lat/lon dimensions,
                HealPixRegridder will be used automatically. Default: None.
            k_neighbors: Number of nearest neighbors to consider for lat/lon grids using
                K-nearest neighbors on the sphere. Only applies when lat/lon coordinates are
                available. Higher values provide more connectivity but may be less spatially
                selective. Default: 8. For very high-resolution grids, consider increasing to
                12-16; for coarse grids, 4-6 may suffice.

        Returns:
            Tuple[xr.Dataset, pd.DataFrame]: A tuple containing:

            Dataset with two variables:
                - ``clusters`` (int32, shape (y, x)): Consensus cluster/component labels.
                  Values >= 0 indicate cluster membership; -1 indicates noise/unassigned.
                - ``consistency`` (float32, shape (y, x)): Local mean of co-association edge
                  weights around each pixel, reflecting neighborhood agreement across input
                  cluster maps.

            DataFrame with one row per consensus cluster, containing:
                - ``cluster_id`` (int32): Cluster identifier.
                - ``mean_consistency`` (float32): Mean consistency score for the cluster.
                - ``size`` (int32): Number of spatial grid cells in the cluster.
                - ``mean_{space_dim0}`` (float32): Average spatial coordinate for first dimension.
                - ``mean_{space_dim1}`` (float32): Average spatial coordinate for second dimension.
                - ``mean_mean_shift_time`` (float32): Central estimate of transition time,
                  averaged over space and clusterings.
                - ``std_mean_shift_time`` (float32): Variation in average shift time across
                  clusterings.
                - ``mean_std_shift_time`` (float32): Average spatial spread of shift timing.
                - ``std_std_shift_time`` (float32): Variation in spatial coherence across
                  clusterings.

        Notes:
            The algorithm proceeds as follows:

            1. Collapse time in each clustering map: mark a pixel as "clustered" if it is ever
               assigned to a cluster at any time.
            2. For each clustering, obtain the spatial footprint of each cluster. Optionally,
               restrict to the top N clusters.
            3. For each cluster, increment votes for each pair of adjacent (connected) pixels
               within that cluster.
            4. Accumulate edge votes across all clusterings, then normalize by the number of
               clustering maps.
            5. Retain only those edges (pixel adjacencies) present in at least `min_consensus`
               fraction of clusterings.
            6. Construct an undirected sparse graph with surviving edges; run connected components
               labeling.
            7. Relabel clusters in order of descending size for interpretability; assign -1 to
               isolated (noise) pixels.
            8. Compute, for each pixel, the mean strength (consistency) of its incident consensus
               edges.

            Additional implementation details:

            * Adjacency method depends on grid type:
              - For lat/lon grids: K-nearest neighbors on sphere using geodesic distance
                (controlled by `k_neighbors`, default 8). This uses coordinate-based spatial
                relationships rather than grid indices.
              - For non-geographic grids: Index-based 4- or 8-connectivity using grid array
                structure (controlled by `neighbor_connectivity`). This is not K-nearest
                neighbors—it connects cells based on their position in the 2D array (horizontal,
                vertical, and optionally diagonal neighbors in grid index space).
            * Consensus clusters represent regions whose internal edges are repeatedly co-clustered
              across the inputs and may be chained via single-link paths.
            * Large, non-compact clusters can form if consensus is too lenient; increase
              `min_consensus` or apply additional filtering for tighter components if needed.
            * Suitable for identifying robust tipping regions or domains unaffected by clustering noise.

        Example:
            >>> ds, summary_df = td.aggregation().cluster_consensus(
            ...     cluster_vars=['clust_a', 'clust_b'], min_consensus=0.7
            ... )
            >>> ds.clusters.plot()  # Visualize consensus clusters
            >>> summary_df.head()  # View cluster statistics

        Raises:
            ValueError: If neighbor_connectivity is not 4 or 8.
            AssertionError: If no cluster_vars are found.

        See Also:
            Evidence accumulation clustering (EAC) method from Fred & Jain (2005). This
            implementation uses spatial adjacency instead of dense all-pairs co-association
            for scalability.
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

        # Determine latitude/longitude names from dataset (e.g. lat, latitude, or None)
        lat_name, lon_name = detect_latlon_names(self.td.data)

        # check if dataset has lat/lon (as dims, or coords, or variables)
        has_latlon = lat_name is not None and lon_name is not None

        # Determine if this is a regular 1D lat/lon grid (i.e. dims are exactly lat, lon)
        is_latlon_dims = has_latlon and (self.td.space_dims == [lat_name, lon_name])

        # Recast naming for readability
        regrid_enabled = is_latlon_dims

        # TODO: fix this bug: looks like we are getting circular clusters around the poles... not sure the regridder is working correctly.
        if regrid_enabled:
            logger.warning(
                "There may be a bug here.. when using regridder, the consensus clusters may be incorrect."
            )

        # use knn if dataset has lat/lon
        if has_latlon:
            lat = sample[lat_name].values
            lon = sample[lon_name].values

            # if lat/lon are 1D, convert to 2D to keep consistent with 2D grids, i.e. irregular such as lat(i, j) and lon(i, j)
            if lat.ndim == 1 and lon.ndim == 1:
                lon, lat = np.meshgrid(lon, lat)

            if regrid_enabled:
                knn_rows, knn_cols, hp_index_flat = _build_knn_edges_from_regridder(
                    lat, lon, k=k_neighbors, regridder=regridder
                )
                # Compute HealPix pixel count once for consistency
                # Note: hp_index_flat should never be empty if regridding succeeded
                N_hp = int(hp_index_flat.max()) + 1
            else:
                knn_rows, knn_cols = _build_knn_edges_from_latlon(
                    lat, lon, k=k_neighbors
                )

            use_knn = True
        else:
            # Fallback to index-based adjacency
            use_knn = False

        # Collect per-map edges for numerator (votes) and denominator (availability)
        rows_V, cols_V = [], []
        rows_A, cols_A = [], []

        # Preallocate reusable arrays (if using regridding)
        if regrid_enabled:
            mask_hp = np.zeros(N_hp, dtype=bool)
            valid_hp = np.ones(N_hp, dtype=bool)
        else:
            # Preallocate availability mask for non-regrid case (same shape every iteration)
            present_mask2d = np.ones((y_len, x_len), dtype=bool)

        # Process each clustering
        for cvar in cluster_vars:
            # Get cluster IDs, optionally filter to top N largest (shared logic)
            unique_ids = self.td.get_cluster_ids(cvar)
            if unique_ids.size == 0:
                continue
            if top_n_clusters is not None and top_n_clusters > 0:
                unique_ids = unique_ids[:top_n_clusters]

            if regrid_enabled:
                labels3d = self.td.data[cvar].values  # (T, Y, X)

                # Build 2D mask: pixels in any of the selected clusters at any time
                labels_2d = np.logical_or.reduce(
                    [(labels3d == cid).any(axis=0) for cid in unique_ids]
                )  # (Y,X), boolean

                # Convert mask to HealPix indexing
                # hp_index_flat maps original pixels → HealPix pixels
                # Reuse preallocated mask, fill with False for this iteration
                mask_hp.fill(False)
                mask_hp[np.unique(hp_index_flat[labels_2d.ravel()])] = True

                # Availability: all HP pixels valid (same every iteration)
                rA, cA = _knn_edges_from_mask(valid_hp, knn_rows, knn_cols)
                rows_A.extend(rA)
                cols_A.extend(cA)

                # Votes: both endpoints in footprint
                rV, cV = _knn_edges_from_mask(mask_hp, knn_rows, knn_cols)
                rows_V.extend(rV)
                cols_V.extend(cV)
            else:
                labels = self.td.data[cvar].values  # (T, Y, X)

                # Per-map deduplication of edges (votes)
                map_edges_V: set[tuple[int, int]] = set()

                # build adjacency edges for each cluster footprint (votes)
                for cid in unique_ids:
                    mask2d = (labels == cid).any(axis=0)  # (Y, X)

                    if use_knn:
                        # cluster footprint mask (reuse mask2d computation)
                        mask_flat = mask2d.ravel()
                        both_true = mask_flat[knn_rows] & mask_flat[knn_cols]
                        for i, j in zip(knn_rows[both_true], knn_cols[both_true]):
                            map_edges_V.add((int(i), int(j)))
                    else:
                        _add_adjacent_true_pairs(
                            mask2d,
                            map_edges_V,
                            flat_idx_2d,
                            neighbor_connectivity == 8,
                        )

                if map_edges_V:
                    r, c = zip(*map_edges_V)
                    rows_V.extend(r)
                    cols_V.extend(c)

                # Availability per map (all pixels valid → all adjacency edges)
                # present_mask2d preallocated before loop since it's the same every iteration
                if use_knn:
                    rA, cA = _knn_edges_from_mask(
                        present_mask2d.ravel(), knn_rows, knn_cols
                    )
                else:
                    rA, cA = _native_edges_from_mask(
                        present_mask2d, flat_idx_2d, neighbor_connectivity == 8
                    )
                rows_A.extend(rA)
                cols_A.extend(cA)

        # If no edges found, return all cells as noise
        if len(rows_V) == 0:
            return _build_empty_consensus_summary_df(
                self.td, y_len, x_len, coords_spatial, spatial_dims
            )

        # Build weighted consensus
        shape = (N_hp, N_hp) if regrid_enabled else (N, N)
        W = _compute_weighted_consensus(
            rows_V, cols_V, rows_A, cols_A, shape, min_consensus
        )

        # If no edges remain after thresholding, return all noise
        if W.nnz == 0:
            return _build_empty_consensus_summary_df(
                self.td, y_len, x_len, coords_spatial, spatial_dims
            )

        # Compute per-node average edge weight
        node_sum = np.array(W.sum(axis=1)).ravel()
        node_deg = np.array(W.count_nonzero(axis=1)).ravel().astype(np.float32)

        if regrid_enabled:
            consistency_hp = np.divide(
                node_sum, node_deg, out=np.zeros_like(node_sum), where=node_deg > 0
            )

            # map back to original grid
            consistency_orig = consistency_hp[hp_index_flat]  # shape: (N_orig,)
            consistency = consistency_orig.reshape(lat.shape)  # shape: (Y, X)
        else:
            consistency = np.divide(
                node_sum, node_deg, out=np.zeros_like(node_sum), where=node_deg > 0
            ).reshape((y_len, x_len))

        # Find connected components in thresholded graph
        bin_adj = W.copy()
        bin_adj.data[:] = 1.0
        bin_adj = bin_adj.maximum(bin_adj.T)
        _, labels_flat = connected_components(
            bin_adj, directed=False, return_labels=True
        )

        # Reshape labels back to 2D and mark isolated points as noise
        if regrid_enabled:
            labels_flat_orig = labels_flat[hp_index_flat]  # shape (N,)
            labels_2d = labels_flat_orig.reshape(lat.shape)  # restored (Y,X)
            deg_hp = np.array(bin_adj.getnnz(axis=1))
            deg_orig = deg_hp[hp_index_flat]
            deg_2d = deg_orig.reshape(lat.shape)
            labels_2d[deg_2d == 0] = -1
        else:
            labels_2d = labels_flat.reshape((y_len, x_len))
            deg = np.array(bin_adj.getnnz(axis=1)).reshape((y_len, x_len))
            labels_2d[deg == 0] = -1

        # Sort cluster labels by size
        labels_2d = sorted_cluster_labels(labels_2d.flatten()).reshape(labels_2d.shape)

        # Create output DataArrays
        da_consensus_labels = xr.DataArray(
            labels_2d,
            coords=coords_spatial,
            dims=spatial_dims,
            name="clusters",
        )
        da_consensus_labels.attrs.update(
            {
                "cluster_vars": cluster_vars,
                "min_consensus": min_consensus,
                "top_n_clusters": top_n_clusters,
                "neighbor_connectivity": neighbor_connectivity,
                "k_neighbors": k_neighbors,
                "description": "Spatial consensus clusters (time-collapsed).",
            }
        )

        da_consistency = xr.DataArray(
            consistency,
            coords=coords_spatial,
            dims=spatial_dims,
            name="consistency",
        )

        ds_out = xr.Dataset(
            {
                "clusters": da_consensus_labels,
                "consistency": da_consistency,
            }
        )

        summary_df = _build_consensus_summary_df(
            self.td, da_consensus_labels, da_consistency, spatial_dims
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
