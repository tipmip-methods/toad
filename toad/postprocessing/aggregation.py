import logging
from itertools import combinations
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

# from scipy.sparse import coo_matrix  # only used in utils helpers
from toad._version import __version__

# from sklearn.neighbors import NearestNeighbors  # unused here; kept in utils
from toad.regridding.healpix import HealPixRegridder
from toad.utils import _attrs, get_latlon_info, get_unique_variable_name
from toad.utils.cluster_consensus_utils import (
    _build_consensus_summary_df,
    _build_knn_edges_from_latlon,
    _build_knn_edges_from_regridder,
    _collect_consensus_edges,
    _compute_weighted_consensus,
    _create_consensus_output_arrays,
    _EdgeCollectionContext,
    _graph_to_labels_and_consistency,
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
            The output variable name will be "cluster_occurrence_rate" with a numeric
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
            "cluster_occurrence_rate", self.td.data, self.td.logger
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
        print(
            "This function is deprecated and will be removed in a future release. Use cluster_consensus() instead."
        )
        # get all cluster vars if nothing is provided
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
        min_consensus: float = 0.75,
        min_cluster_size: int = 5,
        top_n_clusters: int | None = None,
        neighbor_connectivity: int = 8,
        regridder: HealPixRegridder | None = None,
        k_neighbors: int = 8,
        show_progress: bool = True,
        overwrite: bool = False,
    ) -> pd.DataFrame:
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
            min_cluster_size: Minimum number of grid cells for a consensus cluster to be retained.
                Clusters smaller than this are relabelled to -1 (noise). Default: 5.
            top_n_clusters: If set, only top N largest clusters (per clustering) are used when
                voting for edges. If None, all clusters are included. Default: None.
            neighbor_connectivity: Neighborhood connectivity for spatial adjacency when lat/lon
                coordinates are not available. Either 4 (Von Neumann, horizontal/vertical only)
                or 8 (Moore, including diagonals). Default: 8. This parameter controls index-based
                grid adjacency (not K-nearest neighbors) and is only used for grids without
                geographic coordinates; for lat/lon grids, see `k_neighbors`.
            regridder: Optional custom regridder. If None and data has regular lat/lon dimensions,
                HealPixRegridder will be used automatically. Default: None.
                **Note:** Currently only HealPixRegridder is supported for consensus clustering.
                Other regridders will raise a ValueError.
            k_neighbors: Number of nearest neighbors to consider for lat/lon grids using
                K-nearest neighbors on the sphere. Only applies when lat/lon coordinates are
                available. Higher values provide more connectivity but may be less spatially
                selective. Default: 8. For very high-resolution grids, consider increasing to
                12-16; for coarse grids, 4-6 may suffice.
            show_progress: Whether to show the progress bar. Default: True.
            overwrite: If True and output variables already exist, replace them. If False,
                unique names (e.g. consensus_clusters_1, consensus_consistency_1) are used. Default: False.

        Returns:
            pd.DataFrame: Summary statistics for each consensus cluster (cluster_id,
            mean_consistency, size, etc.). The consensus ``consensus_clusters`` and
            ``consensus_consistency`` variables are merged in-place into ``td.data``.

            The merged Dataset has two new variables:
                - ``consensus_clusters`` (float32, shape (y, x)): Consensus cluster/component labels.
                  NaN = no abrupt shifts detected in any input clustering; -1 = shifts
                  detected but not in any consensus cluster; values >= 0 = cluster membership.
                - ``consensus_consistency`` (float32, shape (y, x)): Local mean of co-association edge
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
            8. Demote clusters smaller than min_cluster_size to -1 (noise).
            9. Compute, for each pixel, the mean strength (consistency) of its incident consensus
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
            >>> summary_df = td.aggregate.cluster_consensus(
            ...     cluster_vars=['clust_a', 'clust_b'], min_consensus=0.7
            ... )
            >>> td.data.consensus_clusters.plot()  # Visualize consensus clusters
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
        if min_cluster_size < 1:
            raise ValueError(
                f"`min_cluster_size` must be >= 1, but got {min_cluster_size}."
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

        # Determine latitude/longitude names and grid type from dataset
        lat_name, lon_name, has_latlon, is_latlon_dims = get_latlon_info(
            self.td.data, self.td.space_dims
        )

        # Recast naming for readability
        regrid_enabled = is_latlon_dims

        # Build mask of grid cells that were ever in any cluster (label >= 0) in any
        # input clustering. Used to differentiate: NaN = no abrupt shifts detected;
        # -1 = shifts detected but not in consensus cluster (matches regular clustering).
        ever_clustered = np.zeros((y_len, x_len), dtype=bool)
        for cvar in cluster_vars:
            labels = self.td.data[cvar].values  # (T, Y, X)
            ever_clustered |= (labels >= 0).any(axis=0)

        # Initialize variables that may be conditionally defined (for type checking)
        lat: np.ndarray | None = None
        knn_rows: np.ndarray | None = None
        knn_cols: np.ndarray | None = None
        hp_index_flat: np.ndarray | None = None
        N_hp: int = 0
        mask_hp: np.ndarray | None = None
        valid_hp: np.ndarray | None = None
        present_mask2d: np.ndarray | None = None

        # use knn if dataset has lat/lon
        # Note: regrid_enabled can only be True if has_latlon is True (since is_latlon_dims requires has_latlon)
        if has_latlon:
            lat = sample[lat_name].values
            lon = sample[lon_name].values
            # Type narrowing assertion: helps type checker understand lat/lon are arrays (not None) after assignment
            assert lat is not None and lon is not None

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
            assert N_hp > 0, "N_hp must be set when regrid_enabled is True"
            assert hp_index_flat is not None, (
                "hp_index_flat must be set when regrid_enabled is True"
            )
            mask_hp = np.zeros(N_hp, dtype=bool)
            valid_hp = np.ones(N_hp, dtype=bool)
        else:
            present_mask2d = np.ones((y_len, x_len), dtype=bool)

        ctx = _EdgeCollectionContext(
            spatial_dims=tuple(spatial_dims),
            y_len=y_len,
            x_len=x_len,
            flat_idx_2d=flat_idx_2d,
            regrid_enabled=regrid_enabled,
            use_knn=use_knn,
            neighbor_connectivity=neighbor_connectivity,
            top_n_clusters=top_n_clusters,
            show_progress=show_progress,
            hp_index_flat=hp_index_flat,
            N_hp=N_hp,
            mask_hp=mask_hp if regrid_enabled else None,
            valid_hp=valid_hp if regrid_enabled else None,
            knn_rows=knn_rows,
            knn_cols=knn_cols,
            present_mask2d=present_mask2d if not regrid_enabled else None,
        )
        rows_V, cols_V, rows_A, cols_A = _collect_consensus_edges(
            self.td, cluster_vars, ctx
        )

        # If no edges found, return empty summary without modifying td.data
        if len(rows_V) == 0:
            return pd.DataFrame()

        # Build weighted consensus
        if regrid_enabled:
            assert N_hp > 0, "N_hp must be set when regrid_enabled is True"
            shape = (N_hp, N_hp)
        else:
            shape = (N, N)
        W = _compute_weighted_consensus(
            rows_V, cols_V, rows_A, cols_A, shape, min_consensus
        )

        # If no edges remain after thresholding, return empty summary without modifying td.data
        if W.nnz == 0:
            return pd.DataFrame()

        labels_2d, consistency = _graph_to_labels_and_consistency(
            W,
            ever_clustered,
            y_len,
            x_len,
            min_cluster_size,
            regrid_enabled,
            hp_index_flat,
            lat.shape if (regrid_enabled and lat is not None) else None,
        )

        # Get unique variable names for clusters and consistency
        cluster_label = "consensus_clusters"
        consistency_label = "consensus_consistency"
        if not overwrite:
            cluster_label = get_unique_variable_name(
                cluster_label, self.td.data, self.td.logger
            )
            consistency_label = get_unique_variable_name(
                consistency_label, self.td.data, self.td.logger
            )
        elif overwrite:
            if cluster_label in self.td.data:
                self.td.data = self.td.data.drop_vars(cluster_label)
            if consistency_label in self.td.data:
                self.td.data = self.td.data.drop_vars(consistency_label)

        shared_attrs = {
            "cluster_vars": cluster_vars,
            "min_consensus": min_consensus,
            "min_cluster_size": min_cluster_size,
            "top_n_clusters": top_n_clusters,
            "neighbor_connectivity": neighbor_connectivity,
            "k_neighbors": k_neighbors,
            _attrs.TIME_DIM: self.td.time_dim,
            _attrs.METHOD_NAME: "cluster_consensus",
            _attrs.TOAD_VERSION: __version__,
        }
        da_consensus_labels, da_consistency = _create_consensus_output_arrays(
            labels_2d,
            consistency,
            coords_spatial,
            spatial_dims,
            shared_attrs,
            cluster_name=cluster_label,
            consistency_name=consistency_label,
        )

        summary_df = _build_consensus_summary_df(
            self.td, da_consensus_labels, da_consistency, spatial_dims
        )

        # Log and merge
        unique_ids = np.unique(labels_2d[(labels_2d >= 0) & np.isfinite(labels_2d)])
        n_clusters = len(unique_ids)
        n_noise = int(np.sum(labels_2d == -1))
        n_nan = int(np.sum(np.isnan(labels_2d)))
        valid_cons = np.isfinite(consistency)
        c_min = float(consistency[valid_cons].min()) if valid_cons.any() else np.nan
        c_mean = float(consistency[valid_cons].mean()) if valid_cons.any() else np.nan
        c_max = float(consistency[valid_cons].max()) if valid_cons.any() else np.nan

        self.td.logger.info(
            f"New consensus variable \033[1m{cluster_label}\033[0m: {n_clusters} clusters, "
            f"{n_noise} noise, {n_nan} NaN. Consistency \033[1m{consistency_label}\033[0m: "
            f"min/mean/max={c_min:.3f}/{c_mean:.3f}/{c_max:.3f}"
        )

        self.td.data = xr.merge(
            [self.td.data, da_consensus_labels, da_consistency],
            combine_attrs="override",
            compat="override",
        )

        return summary_df


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
