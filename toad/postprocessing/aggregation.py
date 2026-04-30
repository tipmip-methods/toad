import logging
from dataclasses import dataclass
from typing import Any, List, Literal, cast

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse.csgraph import connected_components
from tqdm import tqdm

from toad._version import __version__
from toad.clustering import sorted_cluster_labels
from toad.utils import _attrs, get_unique_variable_name
from toad.utils.cluster_consensus_utils import (
    _build_consensus_summary_df_spacetime,
    _build_empty_consensus_time_resolved,
    _build_spacetime_graph_edges,
    _compute_weighted_consensus,
    _consensus_input_support_mask,
    _dilate_cluster_labels_spacetime,
    _largest_cluster_ids,
    _native_edges_from_mask,
    _trim_spacetime_consensus_to_original_support,
    consensus_shift_time_distribution,
    consensus_shift_time_distributions,
    label_field_shift_time_distributions,
    label_field_shift_time_samples,
)

logger = logging.getLogger("TOAD")


def _format_consensus_summary(output_label: str, labels: np.ndarray) -> str:
    """One-line summary for logging after :meth:`Aggregation.compute_consensus`.

    Mirrors :func:`toad.clustering._format_cluster_summary` in tone and ANSI styling.
    """
    flat = np.asarray(labels, dtype=np.float64).ravel()
    n = int(flat.size)
    if n == 0:
        return f"{output_label}: empty consensus grid"

    n_no_shift = int(np.count_nonzero(~np.isfinite(flat)))
    noise = int(np.count_nonzero(np.isfinite(flat) & (flat == -1)))
    # Two decimals: with .1f, sub-percent shares (e.g. 0.05%) round to "0.0%" and
    # look inconsistent with a nonzero (noise:,) count.
    pct_noise = 100.0 * noise / n
    pct_no_shift = 100.0 * n_no_shift / n
    pos = flat[np.isfinite(flat) & (flat >= 0)]
    n_clusters = int(np.unique(pos).size) if pos.size else 0

    clusters_text = f"{n_clusters} {'consensus cluster' if n_clusters == 1 else 'consensus clusters'}"
    return (
        f"New consensus variable \033[1m{output_label}\033[0m: Identified \033[1m{clusters_text}\033[0m "
        f"over {n:,} spacetime cells; {pct_noise:.2f}% shift noise / not in consensus ({noise:,} cells); "
        f"{pct_no_shift:.2f}% no abrupt shift ({n_no_shift:,} cells NaN)."
    )


def _filter_consensus_labels_min_size(
    da_labels: xr.DataArray,
    da_consistency: xr.DataArray,
    min_cluster_area: int,
    *,
    time_dim: str,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Remove consensus clusters whose spatial footprint is too small.

    For each cluster id, ``area`` is the number of spatial cells where that id appears at
    any time along ``time_dim`` (same definition as the ``area`` column in
    :func:`toad.utils.cluster_consensus_utils._build_consensus_summary_df_spacetime`).
    Clusters with ``area < min_cluster_area`` are set to noise ``-1``; their consistency
    values become NaN. Remaining clusters are renumbered by
    :func:`toad.clustering.sorted_cluster_labels` (largest id 0, etc.).
    """
    if min_cluster_area <= 0:
        return da_labels, da_consistency
    if time_dim not in da_labels.dims:
        raise ValueError(
            f"`time_dim` {time_dim!r} must be a dimension of `da_labels`, "
            f"got dims={tuple(da_labels.dims)}."
        )
    lab = np.asarray(da_labels.data, dtype=np.float64)
    flat = lab.ravel()
    uniq = np.unique(flat[np.isfinite(flat) & (flat >= 0)])
    remove_list: list[int] = []
    for k in uniq:
        footprint = (da_labels == float(k)).any(dim=time_dim)
        area = int(footprint.sum(skipna=True).item())
        if area < min_cluster_area:
            remove_list.append(int(k))
    remove = np.asarray(remove_list, dtype=np.int64)
    if remove.size == 0:
        return da_labels, da_consistency

    flat = flat.copy()
    fin = np.isfinite(flat)
    flat[fin & np.isin(flat, remove.astype(np.float64))] = -1.0
    flat = sorted_cluster_labels(flat)
    lab_out = flat.reshape(lab.shape)
    cons = np.asarray(da_consistency.data, dtype=np.float32).copy()
    cons_r = cons.ravel()
    flat_out = lab_out.ravel()
    cons_r[(flat_out == -1) | ~np.isfinite(flat_out)] = np.nan
    cons_out = cons_r.reshape(lab.shape)
    da_l = xr.DataArray(
        lab_out,
        coords=da_labels.coords,
        dims=da_labels.dims,
        attrs=da_labels.attrs,
        name=da_labels.name,
    )
    da_c = xr.DataArray(
        cons_out,
        coords=da_consistency.coords,
        dims=da_consistency.dims,
        attrs=da_consistency.attrs,
        name=da_consistency.name,
    )
    return da_l, da_c


@dataclass(frozen=True)
class _SpacetimeConsensusContext:
    spatial_dims: tuple[str, str]
    time_dim: str
    T: int
    y_len: int
    x_len: int
    coords_spatial: dict[str, Any]
    time_coord: xr.DataArray
    n_space: int
    spatial_er: np.ndarray
    spatial_ec: np.ndarray
    st_er: np.ndarray
    st_ec: np.ndarray


@dataclass(frozen=True)
class _SpacetimeVoteData:
    rows_V: np.ndarray
    cols_V: np.ndarray
    n_contributing_maps: int
    original_support_flat: np.ndarray


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
        """Share of clusterings (label fields) in which a grid cell is ever a cluster.

        This is an **aggregation** over several clustering results, not a statistic
        tied to a single :attr:`TOAD.cluster_vars` entry. For each space point it counts
        how many of the given label variables assign a non-noise id at any time, then
        divides by the number of those variables (not part of the consensus graph API).

        Args:
            cluster_vars: Label variables to include. If None, uses all
                :attr:`TOAD.cluster_vars`. Each must use ``-1`` for noise and ``>= 0`` for
                cluster membership.

        Returns:
            2D DataArray in ``[0, 1]`` named ``cluster_occurrence_rate`` or a uniquified
            name if that variable already exists in the dataset.
        """
        cluster_vars = cluster_vars if cluster_vars else self.td.cluster_vars
        if not cluster_vars:
            raise ValueError(
                "cluster_vars is empty; add cluster label variables or pass a non-empty list."
            )

        num_clusterings = len(cluster_vars)
        cluster_normalized = xr.where(
            self.td.data[cluster_vars[0]].max(dim=self.td.time_dim) > -1,
            1.0 / num_clusterings,
            0,
        )
        for cluster_var in cluster_vars[1:]:
            cluster_normalized = cluster_normalized + xr.where(
                self.td.data[cluster_var].max(dim=self.td.time_dim) > -1,
                1.0 / num_clusterings,
                0,
            )

        output_label = get_unique_variable_name(
            "cluster_occurrence_rate", self.td.data, self.td.logger
        )
        cluster_normalized = cluster_normalized.rename(output_label)
        cluster_normalized.attrs.update(
            {
                "cluster_vars": list(cluster_vars),
                "description": "Normalized occurrence rate of points being part of any cluster",
            }
        )
        return cluster_normalized

    def compute_consensus(
        self,
        cluster_vars: List[str] | None = None,
        *,
        min_consensus: float,
        temporal_tolerance: int,
        spatial_tolerance: int,
        top_n_clusters: int | None = None,  # TODO rename?
        stitch_meridian: bool = False,
        show_progress: bool = True,
        output_label_suffix: str = "",
        output_label: str | None = None,
        overwrite: bool = False,
        min_cluster_area: int | None = 2,
    ) -> None:
        """Build a spacetime consensus clustering and merge it into ``self.td.data``.

        Writes two variables: consensus labels (``variable_type=consensus_cluster``) and
        a companion consistency field (``variable_type=consensus_consistency``). The list of
        input clustering variables is stored on both as ``cluster_vars``.

        **Label encoding** (same idea as :func:`toad.clustering.compute_clusters`): ``NaN`` where
        every input cluster field is ``NaN`` (no abrupt shift at that spacetime cell); ``-1`` where
        at least one input had a defined label (including noise ``-1``) but the voxel is not in a
        consensus component; non-negative integers for consensus cluster ids. Consistency is
        ``NaN`` wherever the label is ``NaN``.

        Args:
            cluster_vars: List of clustering variable names to include in the consensus.
                If None, uses all cluster variables in ``self.td.cluster_vars``.
            min_consensus: Minimum fraction (in [0,1]) of clusterings that must support an edge
                for it to be included in the consensus graph after ``W = V/A``. Higher values =
                stricter consensus. Required (callers must choose explicitly).
            temporal_tolerance: Non-negative integer (required; no implicit default).
                Dilation radius applied to peak-event voxel labels before voting. If a cell is
                labelled at time ``t`` in an input clustering, it is treated as active for that
                same cluster id at all times ``t'`` with ``|t' - t| <= temporal_tolerance`` when
                computing consensus votes. ``0`` means no dilation (exact-time peak voxels only).
                This pools timing jitter before the standard same-time spatial / consecutive-time
                consensus graph is evaluated. The returned ``clusters`` mask is then trimmed back
                to voxels that had support in at least one original undilated input clustering, so
                tolerance affects matching but does not directly thicken the public output mask.
                This is a local agreement rule, not a global cap on the final cluster time span:
                connected components may still extend over more than ``temporal_tolerance`` timesteps
                via transitive chains.
            spatial_tolerance: Non-negative integer (required; no implicit default).
                Spatial dilation radius measured in graph hops on the spatial adjacency used
                by the consensus graph. ``0`` means exact spatial support only. Positive values
                let nearby spatially displaced detections support the same local consensus edge.
                As with ``temporal_tolerance``, this affects matching only; the returned mask is
                trimmed back to original undilated support afterwards.
            top_n_clusters: If set, only the largest N clusters by actual spacetime size in
                each input clustering are used when voting for edges. This is computed inside
                consensus and does not rely on the stored cluster-id order. If None, all
                clusters are included. Default: None.
            stitch_meridian: Whether to stitch the first and last native-grid columns into a
                wrapped seam. This only affects native-grid adjacency (including curvilinear
                grids that keep their original ``y/x`` or ``i/j`` topology) and is useful for
                domains split at the meridian. Default: False.
            show_progress: Whether to show the progress bar. Default: True.
            output_label_suffix: Suffix appended to the default output label ``cluster_consensus``.
            output_label: Explicit name for the consensus labels variable. If None, uses
                ``"cluster_consensus" + output_label_suffix``.
            overwrite: If True, replace existing variables with the same names. If False,
                append ``_1``, ``_2``, … when a name is already taken (same convention as
                :func:`toad.clustering.compute_clusters`).
            min_cluster_area: Post-filter on spatial footprint: consensus clusters whose
                spatial extent (number of grid cells where the cluster appears at any time)
                is **strictly below** this threshold are dropped; those voxels become noise
                (``-1``) and remaining cluster ids are re-sorted by size. Default: ``2`` (drop
                single-cell clusters). Set to ``None`` to disable this post-filter entirely.
                Non-negative integers are allowed; ``0`` disables filtering (same effect as
                choosing a threshold that never removes a cluster).

        Notes:
            The method builds one graph on ``(time × space)`` nodes (see
            ``_build_spacetime_graph_edges``), accumulates ``V`` /
            ``A`` on that edge set, thresholds, then one connected-components pass. Optional
            ``temporal_tolerance`` and ``spatial_tolerance`` first dilate peak-event labels in
            time and space for matching, so nearby events can agree through the standard graph
            rather than by adding new edge families. After trimming back to original support,
            labels are re-sorted **globally** by final spacetime component size so output
            slices share stable ids.

            Additional implementation details:

            * Spatial adjacency is **index-based 8-neighbour** on the label grid, with
              optional first/last-column seam stitching via ``stitch_meridian``.
            * Consensus clusters represent regions whose internal edges are repeatedly co-clustered
              across the inputs and may be chained via single-link paths.
            * Large, non-compact clusters can form if consensus is too lenient; increase
              `min_consensus` or apply additional filtering for tighter components if needed.
            * Suitable for identifying robust tipping regions or domains unaffected by clustering noise.
            * The per-cluster summary table is not built during this method; call
              :meth:`consensus_summary` when you need it. (The internal solver used to
              construct a summary DataFrame that was then unused; that extra work was removed.)

        Example:
            >>> td.compute_consensus(
            ...     cluster_vars=['clust_a', 'clust_b'],
            ...     min_consensus=0.7,
            ...     temporal_tolerance=0,
            ...     spatial_tolerance=0,
            ... )
            >>> td.plot.consensus_overview()
            >>> td.aggregate.consensus_summary().head()

        Raises:
            ValueError: If a tolerance is negative, or if ``min_cluster_area`` is invalid.
            AssertionError: If no cluster_vars are found.

        See Also:
            Evidence accumulation clustering (EAC) method from Fred & Jain (2005). This
            implementation uses spatial adjacency instead of dense all-pairs co-association
            for scalability.
        """
        if cluster_vars is None:
            cluster_vars = list(self.td.cluster_vars)
        assert len(cluster_vars) > 0, "No cluster variables provided/found."

        if temporal_tolerance < 0:
            raise ValueError(
                f"`temporal_tolerance` must be >= 0, got {temporal_tolerance}."
            )
        if spatial_tolerance < 0:
            raise ValueError(
                f"`spatial_tolerance` must be >= 0, got {spatial_tolerance}."
            )
        if min_cluster_area is not None and min_cluster_area < 0:
            raise ValueError(
                f"`min_cluster_area` must be >= 0 or None, got {min_cluster_area}."
            )

        new_output_label = (
            output_label if output_label else f"cluster_consensus{output_label_suffix}"
        )
        if not overwrite:
            new_output_label = get_unique_variable_name(
                new_output_label, self.td.data, logger
            )
        else:
            if new_output_label in self.td.data:
                self.td.data = self.td.data.drop_vars(new_output_label)
            consistency_drop = f"{new_output_label}_consistency"
            if consistency_drop in self.td.data:
                self.td.data = self.td.data.drop_vars(consistency_drop)

        ds_out = self._cluster_consensus_spacetime(
            cluster_vars=cluster_vars,
            min_consensus=min_consensus,
            top_n_clusters=top_n_clusters,
            stitch_meridian=stitch_meridian,
            show_progress=show_progress,
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
        )

        da_labels = ds_out["clusters"].rename(new_output_label)
        da_consistency = ds_out["consistency"].rename(f"{new_output_label}_consistency")

        if min_cluster_area is not None and min_cluster_area > 0:
            da_labels, da_consistency = _filter_consensus_labels_min_size(
                da_labels,
                da_consistency,
                min_cluster_area,
                time_dim=self.td.time_dim,
            )
            da_labels.attrs["min_cluster_area"] = int(min_cluster_area)
            da_consistency.attrs["min_cluster_area"] = int(min_cluster_area)

        lab = np.asarray(da_labels.data, dtype=np.float64)
        da_labels.attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_CONSENSUS_CLUSTER
        u = np.unique(lab[np.isfinite(lab) & (lab >= 0)])
        da_labels.attrs[_attrs.CLUSTER_IDS] = (
            u.astype(int) if u.size else np.array([], dtype=int)
        )
        da_labels.attrs[_attrs.CLUSTER_VARS] = list(cluster_vars)
        da_labels.attrs[_attrs.TOAD_VERSION] = __version__

        da_consistency.attrs[_attrs.VARIABLE_TYPE] = _attrs.TYPE_CONSENSUS_CONSISTENCY
        da_consistency.attrs[_attrs.CONSENSUS_LABELS_VAR] = new_output_label
        da_consistency.attrs[_attrs.CLUSTER_VARS] = list(cluster_vars)
        da_consistency.attrs[_attrs.TOAD_VERSION] = __version__

        self.td.data = xr.merge(
            [self.td.data, da_labels, da_consistency],
            combine_attrs="override",
            compat="override",
        )

        logger.info(_format_consensus_summary(new_output_label, lab))

    def consensus_summary(self, consensus_var: str | None = None) -> pd.DataFrame:
        """Rebuild the per-cluster summary table from stored consensus label and consistency arrays.

        Args:
            consensus_var: Name of the consensus labels variable (``variable_type=consensus_cluster``).
                If None, infers the variable when exactly one consensus label exists on the dataset
                (same resolution rules as :meth:`toad.core.TOAD._resolve_consensus_var`).

        Returns:
            DataFrame with one row per consensus cluster. Includes ``median_median_shift_time``
            (median of per-input spatial medians), related between-input std columns, and
            ``pooled_median_shift_time`` / ``pooled_std_shift_time`` over all pooled event-time
            samples; see :func:`toad.utils.cluster_consensus_utils._build_consensus_summary_df_spacetime`.
        """
        consensus_var = self.td._resolve_consensus_var(consensus_var)
        consistency_var = f"{consensus_var}_consistency"
        if consistency_var not in self.td.data:
            raise ValueError(
                f"No matching consistency variable {consistency_var!r} for {consensus_var!r}."
            )
        labels = self.td.data[consensus_var]
        consistency = self.td.data[consistency_var]
        spatial_dims = tuple(self.td.space_dims)
        time_dim = self.td.time_dim
        return _build_consensus_summary_df_spacetime(
            self.td, labels, consistency, spatial_dims, time_dim
        )

    def consensus_shift_time_distribution(
        self,
        da_clusters: xr.DataArray,
        *,
        spatial_dims: tuple[str, str] | None = None,
        time_dim: str | None = None,
        shift_threshold: float = 0.0,
    ) -> tuple[xr.Dataset, pd.DataFrame]:
        """Export event-time samples behind the consensus summary shift columns.

        Wraps :func:`toad.utils.cluster_consensus_utils.consensus_shift_time_distribution`.
        Returns an ``xr.Dataset`` (per-cluster × ``cluster_var`` means/stds) and a long
        ``DataFrame`` of per-cell transition times for histograms.

        Args:
            da_clusters: Consensus labels from :meth:`compute_consensus` / :meth:`TOAD.compute_consensus`.
            spatial_dims: Defaults to ``self.td.space_dims``.
            time_dim: Inferred from ``self.td.time_dim`` when present on ``da_clusters``.
            shift_threshold: Unused for spacetime consensus. Kept only for API compatibility.

        Returns:
            ``(xr.Dataset, pandas.DataFrame)`` — see the utility docstring.
        """
        return consensus_shift_time_distribution(
            self.td,
            da_clusters,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
            shift_threshold=shift_threshold,
        )

    def consensus_shift_time_distributions(
        self,
        da_clusters: xr.DataArray,
        *,
        spatial_dims: tuple[str, str] | None = None,
        time_dim: str | None = None,
        shift_threshold: float = 0.0,
        distribution_result: tuple[xr.Dataset, pd.DataFrame] | None = None,
        source_input_cluster_var: str | None = None,
    ) -> dict[int, np.ndarray]:
        """Return transition-time samples grouped by consensus cluster id.

        This is a plotting-friendly wrapper around
        :meth:`consensus_shift_time_distribution`. It aggregates across all input
        ``cluster_var`` values and returns ``{cluster_id: shift_times}``, which is
        convenient for violin plots of consensus-cluster timing distributions.

        Args:
            da_clusters: Consensus labels from :meth:`compute_consensus` / :meth:`TOAD.compute_consensus`.
            spatial_dims: Defaults to ``self.td.space_dims``.
            time_dim: Inferred from ``self.td.time_dim`` when present on ``da_clusters``.
            shift_threshold: Passed to ``compute_transition_time`` (default ``0.0``).
            distribution_result: If provided, a prior ``(dataset, dataframe)`` tuple from
                :meth:`consensus_shift_time_distribution` for the same inputs; recomputation
                of the long table is skipped.
            source_input_cluster_var: If set, use only long-form rows from this input
                clustering (see :func:`toad.utils.cluster_consensus_utils.consensus_shift_time_distributions`).

        Returns:
            Mapping from consensus cluster id to a 1D array of transition times.
            In spacetime mode the samples match the summary pipeline exactly, so the
            same spatial cell may appear multiple times if it belongs to the same
            consensus component at multiple timesteps.
        """
        return consensus_shift_time_distributions(
            self.td,
            da_clusters,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
            shift_threshold=shift_threshold,
            distribution_result=distribution_result,
            source_input_cluster_var=source_input_cluster_var,
        )

    def label_shift_time_samples(
        self,
        label_data: str | xr.DataArray,
        *,
        spatial_dims: tuple[str, str] | None = None,
        time_dim: str | None = None,
    ) -> pd.DataFrame:
        """Per-cell event times for a single 3D label field (non-consensus or any labels).

        Wraps :func:`toad.utils.cluster_consensus_utils.label_field_shift_time_samples`.
        Pass a variable name in ``self.td.data`` or a :class:`xarray.DataArray` of labels.
        """
        da = self.td.data[label_data] if isinstance(label_data, str) else label_data
        return label_field_shift_time_samples(
            self.td,
            da,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
        )

    def label_shift_time_distributions(
        self,
        label_data: str | xr.DataArray,
        *,
        spatial_dims: tuple[str, str] | None = None,
        time_dim: str | None = None,
    ) -> dict[int, np.ndarray]:
        """Transition-time samples per cluster id in one time-resolved label field.

        For violin-style plots of a **normal** clustering, pass that cluster map’s
        name. Same convention as the consensus long table (event time at each
        labelled spacetime cell).

        See :func:`toad.utils.cluster_consensus_utils.label_field_shift_time_distributions`.
        """
        da = self.td.data[label_data] if isinstance(label_data, str) else label_data
        return label_field_shift_time_distributions(
            self.td,
            da,
            spatial_dims=spatial_dims,
            time_dim=time_dim,
        )

    def consensus_extraction_mask_2d(
        self,
        da_clusters: xr.DataArray,
        consensus_cluster_id: int,
        cluster_var: str,
    ) -> xr.DataArray:
        """2D boolean mask of grid cells used by :meth:`consensus_cluster_timeseries`.

        A cell is True iff at some time (if applicable) the consensus label matches
        ``consensus_cluster_id`` and the input ``cluster_var`` is supported
        (non-noise in the allowed set). Same construction as the spatial mask passed
        to :meth:`toad.core.TOAD.get_timeseries` in :meth:`consensus_cluster_timeseries`.

        Args:
            da_clusters: Consensus label field.
            consensus_cluster_id: Target consensus cluster id.
            cluster_var: One input clustering variable name.

        Returns:
            DataArray of bool with dimensions ``self.td.space_dims``.
        """
        cluster_mask = da_clusters == consensus_cluster_id
        time_dim = self.td.time_dim
        has_time_dim = time_dim in da_clusters.dims
        support_mask = _consensus_input_support_mask(
            self.td,
            da_clusters,
            cluster_var,
            spatial_dims=tuple(self.td.space_dims),
            time_dim=time_dim if has_time_dim else None,
        )
        if has_time_dim:
            return (cluster_mask & support_mask).any(dim=time_dim)
        return cluster_mask & support_mask

    def consensus_cluster_extraction_n_cells_2d(
        self,
        da_clusters: xr.DataArray,
        consensus_cluster_id: int,
        cluster_var: str,
    ) -> int:
        """Count of True cells in :meth:`consensus_extraction_mask_2d`."""
        m = self.consensus_extraction_mask_2d(
            da_clusters, consensus_cluster_id, cluster_var
        )
        return int(m.sum().item())

    def consensus_cluster_timeseries(
        self,
        da_clusters: xr.DataArray,
        consensus_cluster_id: int,
        *,
        var: str | None = None,
        cluster_vars: list[str] | None = None,
        aggregation: Literal[
            "raw", "mean", "sum", "std", "median", "percentile", "max", "min"
        ]
        | str = "raw",
        percentile: float | None = None,
        normalize: Literal["max", "max_each"] | str | None = None,
        keep_full_timeseries: bool = True,
    ) -> dict[str, xr.DataArray]:
        """Extract per-input timeseries for one consensus cluster.

        This mirrors :meth:`toad.core.TOAD.get_timeseries`, but uses a consensus
        cluster from :meth:`compute_consensus` as the mask and returns one result
        per input ``cluster_var``. The output is convenient for overlaying model
        trajectories from the same consensus region.

        Args:
            da_clusters: Consensus labels from :meth:`compute_consensus` / :meth:`TOAD.compute_consensus`.
            consensus_cluster_id: Consensus cluster id to extract.
            var: Optional base variable override used for all returned timeseries.
                If omitted, each ``cluster_var`` is mapped to its own base variable.
            cluster_vars: Optional subset of input clustering variables. Defaults to
                ``da_clusters.attrs["cluster_vars"]`` when available, else ``self.td.cluster_vars``.
            aggregation: Same as :meth:`toad.core.TOAD.get_timeseries`.
            percentile: Required when ``aggregation="percentile"``.
            normalize: Same as :meth:`toad.core.TOAD.get_timeseries`.
            keep_full_timeseries: If ``False`` and ``da_clusters`` is time-resolved,
                values outside the consensus cluster's overall start/end window are
                masked out.

        Returns:
            Mapping ``{cluster_var: xr.DataArray}`` for input clusterings that actually
            support the requested consensus cluster. Timeseries are extracted from the
            supported spatial footprint of the consensus cluster, i.e. cells that both
            belong to the consensus cluster and are supported by the given input
            clustering at least once in time. Each value has the same output form as
            :meth:`toad.core.TOAD.get_timeseries` for the chosen aggregation.
        """
        if cluster_vars is None:
            cluster_vars_attr = da_clusters.attrs.get("cluster_vars")
            if cluster_vars_attr is not None:
                cluster_vars = list(cluster_vars_attr)
            else:
                cluster_vars = list(self.td.cluster_vars)
        if len(cluster_vars) == 0:
            raise ValueError("No cluster variables available for consensus timeseries.")

        cluster_mask = da_clusters == consensus_cluster_id
        if not bool(cluster_mask.any().item()):
            present_ids = np.unique(da_clusters.values)
            present_ids = present_ids[np.isfinite(present_ids) & (present_ids >= 0)]
            raise ValueError(
                f"Consensus cluster id {consensus_cluster_id} not found. "
                f"Available ids: {present_ids.astype(int).tolist()}"
            )

        time_dim = self.td.time_dim
        has_time_dim = time_dim in da_clusters.dims
        time_window_mask: xr.DataArray | None = None
        if has_time_dim and not keep_full_timeseries:
            cluster_present_over_time = cluster_mask.any(dim=self.td.space_dims)
            active_idx = np.flatnonzero(cluster_present_over_time.values)
            if active_idx.size > 0:
                in_window = np.zeros(da_clusters.sizes[time_dim], dtype=bool)
                in_window[active_idx[0] : active_idx[-1] + 1] = True
                time_window_mask = xr.DataArray(
                    in_window,
                    dims=[time_dim],
                    coords={time_dim: da_clusters[time_dim]},
                    name="consensus_time_window",
                )

        out: dict[str, xr.DataArray] = {}
        for cluster_var in cluster_vars:
            mask_for_data = self.consensus_extraction_mask_2d(
                da_clusters, consensus_cluster_id, cluster_var
            )
            if not bool(mask_for_data.any().item()):
                continue

            base_var = var
            if base_var is None:
                base_var = self.td.data[cluster_var].attrs.get(_attrs.BASE_VARIABLE)
                if base_var is None:
                    raise ValueError(
                        f"Cluster variable '{cluster_var}' has no BASE_VARIABLE attribute. "
                        "Pass `var=...` explicitly."
                    )

            data = self.td.data[base_var].where(mask_for_data)
            if time_window_mask is not None:
                data = data.where(time_window_mask)

            series = self.td._aggregate_spatial(data, aggregation, percentile)
            if not np.isfinite(np.asarray(series.values, dtype=float)).any():
                continue
            if normalize:
                if normalize == "max":
                    series = self.td._normalize_timeseries(
                        series, float(series.max()), normalize
                    )
                elif normalize == "max_each":
                    norm_val = (
                        series.max(dim=self.td.time_dim)
                        if "cell_xy" in series.dims
                        else float(series.max())
                    )
                    series = self.td._normalize_timeseries(series, norm_val, normalize)
                else:
                    raise ValueError(f"Unknown normalization method: {normalize}")

            series = series.copy()
            series.name = f"{cluster_var}_consensus_cluster_{consensus_cluster_id}"
            series.attrs.update(
                {
                    "consensus_cluster_id": int(consensus_cluster_id),
                    "cluster_var": cluster_var,
                    "base_var": base_var,
                    "aggregation": aggregation,
                    "keep_full_timeseries": keep_full_timeseries,
                }
            )
            out[cluster_var] = series

        if not out:
            raise ValueError(
                f"No input clusterings support consensus cluster id {consensus_cluster_id} "
                "under the current extraction settings."
            )
        return out

    def _labels_tyx(
        self,
        cvar_name: str,
        time_dim: str,
        spatial_dims: tuple[str, str],
    ) -> np.ndarray:
        """Return cluster labels in canonical ``(time, y, x)`` order."""
        da = self.td.data[cvar_name]
        return da.transpose(time_dim, spatial_dims[0], spatial_dims[1]).values

    @staticmethod
    def _flatten_spacetime_labels(labels_tyx: np.ndarray) -> np.ndarray:
        """Flatten ``(time, y, x)`` labels to match spacetime node indexing."""
        return np.asarray(labels_tyx).reshape(-1)

    @staticmethod
    def _reshape_spacetime_consensus_outputs(
        *,
        labels_st: np.ndarray,
        cons_flat: np.ndarray,
        deg: np.ndarray,
        T: int,
        n_space: int,
        y_len: int,
        x_len: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map flat spacetime consensus results back to ``(time, y, x)`` arrays."""
        clusters_out = np.full((T, y_len, x_len), -1, dtype=np.int32)
        consistency_out = np.zeros((T, y_len, x_len), dtype=np.float32)

        for t in range(T):
            sl = slice(t * n_space, (t + 1) * n_space)
            labels_2d = labels_st[sl].reshape((y_len, x_len))
            cons_2d = cons_flat[sl].reshape((y_len, x_len))
            deg_2d = deg[sl].reshape((y_len, x_len))
            labels_2d[deg_2d == 0] = -1
            clusters_out[t] = labels_2d.astype(np.int32)
            consistency_out[t] = cons_2d
        return clusters_out, consistency_out

    def _build_spacetime_consensus_context(
        self,
        *,
        sample: xr.DataArray,
        stitch_meridian: bool,
    ) -> _SpacetimeConsensusContext:
        """Build grid and graph context for spacetime consensus (native 8-neighbour)."""
        spatial_dims = tuple(self.td.space_dims)
        time_dim = self.td.time_dim
        T = int(sample.sizes[time_dim])
        y_len = sample.sizes[spatial_dims[0]]
        x_len = sample.sizes[spatial_dims[1]]
        N = y_len * x_len
        flat_idx_2d = np.arange(N, dtype=np.int64).reshape((y_len, x_len))

        coords_spatial = {
            name: coord
            for name, coord in sample.coords.items()
            if (len(coord.dims) > 0) and set(coord.dims).issubset(spatial_dims)
        }
        for d in spatial_dims:
            coords_spatial.setdefault(d, sample[d])
        time_coord = sample[time_dim]

        present_mask2d = np.ones((y_len, x_len), dtype=bool)
        er, ec = _native_edges_from_mask(
            present_mask2d, flat_idx_2d, stitch_longitude=stitch_meridian
        )
        spatial_er = np.asarray(er, dtype=np.int64)
        spatial_ec = np.asarray(ec, dtype=np.int64)
        n_space = N

        st_er, st_ec = _build_spacetime_graph_edges(T, n_space, spatial_er, spatial_ec)
        return _SpacetimeConsensusContext(
            spatial_dims=spatial_dims,
            time_dim=time_dim,
            T=T,
            y_len=y_len,
            x_len=x_len,
            coords_spatial=cast(dict[str, Any], coords_spatial),
            time_coord=time_coord,
            n_space=n_space,
            spatial_er=spatial_er,
            spatial_ec=spatial_ec,
            st_er=st_er,
            st_ec=st_ec,
        )

    def _accumulate_spacetime_votes(
        self,
        *,
        cluster_vars: List[str],
        top_n_clusters: int | None,
        temporal_tolerance: int,
        spatial_tolerance: int,
        show_progress: bool,
        context: _SpacetimeConsensusContext,
    ) -> _SpacetimeVoteData | None:
        """Collect sparse vote edges and undilated support across all input maps."""
        n_st = context.T * context.n_space
        rows_V_parts: list[np.ndarray] = []
        cols_V_parts: list[np.ndarray] = []
        n_contributing_maps = 0
        original_support_flat = np.zeros(n_st, dtype=bool)

        cvar_iter = tqdm(
            cluster_vars, disable=not show_progress, desc="spacetime consensus"
        )
        for cvar in cvar_iter:
            allowed = _largest_cluster_ids(self.td, cvar, top_n_clusters)
            if allowed.size == 0:
                continue
            n_contributing_maps += 1

            labels_orig = self._labels_tyx(
                cvar_name=cvar,
                time_dim=context.time_dim,
                spatial_dims=context.spatial_dims,
            )
            orig_flat = self._flatten_spacetime_labels(labels_orig)
            original_support_flat |= (
                np.isfinite(orig_flat) & (orig_flat >= 0) & np.isin(orig_flat, allowed)
            )

            labels_dilated = _dilate_cluster_labels_spacetime(
                orig_flat.reshape(context.T, context.n_space),
                allowed,
                temporal_tolerance=temporal_tolerance,
                spatial_tolerance=spatial_tolerance,
                spatial_rows=context.spatial_er,
                spatial_cols=context.spatial_ec,
            )
            labels_flat = labels_dilated.reshape(-1)

            lu = labels_flat[context.st_er]
            lv = labels_flat[context.st_ec]
            keep_votes = (
                (lu == lv)
                & (lu >= 0)
                & np.isfinite(lu)
                & np.isfinite(lv)
                & np.isin(lu, allowed)
            )
            if np.any(keep_votes):
                rows_V_parts.append(context.st_er[keep_votes])
                cols_V_parts.append(context.st_ec[keep_votes])

        if not rows_V_parts:
            return None

        return _SpacetimeVoteData(
            rows_V=np.concatenate(rows_V_parts).astype(np.int64, copy=False),
            cols_V=np.concatenate(cols_V_parts).astype(np.int64, copy=False),
            n_contributing_maps=n_contributing_maps,
            original_support_flat=original_support_flat,
        )

    @staticmethod
    def _solve_spacetime_consensus_components(
        *,
        min_consensus: float,
        context: _SpacetimeConsensusContext,
        vote_data: _SpacetimeVoteData,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Solve the weighted consensus graph and extract trimmed components."""
        n_st = context.T * context.n_space
        W = _compute_weighted_consensus(
            vote_data.rows_V,
            vote_data.cols_V,
            context.st_er,
            context.st_ec,
            (n_st, n_st),
            min_consensus,
            data_A=np.full(
                context.st_er.shape[0], vote_data.n_contributing_maps, dtype=np.float32
            ),
        )
        if W.nnz == 0:
            return None

        node_sum = np.array(W.sum(axis=1)).ravel()
        node_deg = np.array(W.count_nonzero(axis=1)).ravel().astype(np.float32)
        cons_flat = np.divide(
            node_sum, node_deg, out=np.zeros_like(node_sum), where=node_deg > 0
        ).astype(np.float32)

        bin_adj = W.copy()
        bin_adj.data[:] = 1.0
        bin_adj = bin_adj.maximum(bin_adj.T)
        _, labels_st = connected_components(bin_adj, directed=False, return_labels=True)
        deg = np.array(bin_adj.getnnz(axis=1))
        labels_st = labels_st.astype(np.int64, copy=False)
        labels_st[deg == 0] = -1
        labels_st = sorted_cluster_labels(labels_st)
        labels_st, cons_flat = _trim_spacetime_consensus_to_original_support(
            labels_st, cons_flat, vote_data.original_support_flat
        )
        labels_st = sorted_cluster_labels(labels_st)
        return labels_st, cons_flat, deg

    def _all_inputs_no_shift_mask_flat(
        self,
        cluster_vars: List[str],
        context: _SpacetimeConsensusContext,
    ) -> np.ndarray:
        """True in ``clusters.values.ravel()`` order where every input label is NaN (no shift).

        Built on the native ``(time, y, x)`` grid (same layout as the stored consensus
        label field).
        """
        T, y_len, x_len = context.T, context.y_len, context.x_len
        all_nan = np.ones((T, y_len, x_len), dtype=bool)
        for cvar in cluster_vars:
            lab_tyx = self._labels_tyx(cvar, context.time_dim, context.spatial_dims)
            if lab_tyx.shape != (T, y_len, x_len):
                raise ValueError(
                    f"Label field {cvar!r} has shape {lab_tyx.shape}, expected "
                    f"({T}, {y_len}, {x_len}) for consensus no-shift mask."
                )
            all_nan &= np.isnan(np.asarray(lab_tyx, dtype=np.float64))
        return all_nan.ravel()

    def _mark_no_shift_nan_on_consensus_dataset(
        self,
        ds: xr.Dataset,
        cluster_vars: List[str],
        context: _SpacetimeConsensusContext,
    ) -> xr.Dataset:
        """Set consensus label NaN and consistency NaN where all inputs have no shift (aligns with compute_clusters)."""
        all_none = self._all_inputs_no_shift_mask_flat(cluster_vars, context)
        da_c = ds["clusters"]
        T, y_len, x_len = context.T, context.y_len, context.x_len
        flat = np.asarray(da_c.data, dtype=np.float64).ravel()
        m = (flat == -1) & all_none
        flat = flat.copy()
        flat[m] = np.nan
        new_lab = flat.reshape((T, y_len, x_len))
        new_lab_f = new_lab.ravel()
        cons = np.asarray(ds["consistency"].data, dtype=np.float32).copy().reshape(-1)
        cons[~np.isfinite(new_lab_f)] = np.nan
        cons = cons.reshape((T, y_len, x_len))
        return xr.Dataset(
            {
                "clusters": xr.DataArray(
                    new_lab,
                    coords=da_c.coords,
                    dims=da_c.dims,
                    attrs=da_c.attrs,
                    name=da_c.name,
                ),
                "consistency": xr.DataArray(
                    cons,
                    coords=ds["consistency"].coords,
                    dims=ds["consistency"].dims,
                    attrs=ds["consistency"].attrs,
                    name=ds["consistency"].name,
                ),
            }
        )

    def _empty_spacetime_consensus_result(
        self,
        context: _SpacetimeConsensusContext,
        cluster_vars: List[str],
    ) -> xr.Dataset:
        """Return an all-noise spacetime consensus result with matching metadata shape."""
        ds = _build_empty_consensus_time_resolved(
            context.T,
            context.y_len,
            context.x_len,
            context.coords_spatial,
            context.spatial_dims,
            context.time_coord,
            context.time_dim,
        )
        return self._mark_no_shift_nan_on_consensus_dataset(ds, cluster_vars, context)

    def _build_spacetime_consensus_result(
        self,
        *,
        cluster_vars: List[str],
        min_consensus: float,
        top_n_clusters: int | None,
        stitch_meridian: bool,
        temporal_tolerance: int,
        spatial_tolerance: int,
        context: _SpacetimeConsensusContext,
        labels_st: np.ndarray,
        cons_flat: np.ndarray,
        deg: np.ndarray,
    ) -> xr.Dataset:
        """Assemble spacetime consensus xarray outputs (summary is built in :meth:`compute_consensus`)."""
        clusters_out, consistency_out = self._reshape_spacetime_consensus_outputs(
            labels_st=labels_st,
            cons_flat=cons_flat,
            deg=deg,
            T=context.T,
            n_space=context.n_space,
            y_len=context.y_len,
            x_len=context.x_len,
        )
        if not np.any(clusters_out >= 0):
            return self._empty_spacetime_consensus_result(context, cluster_vars)

        da_clusters = xr.DataArray(
            clusters_out,
            coords={context.time_dim: context.time_coord, **context.coords_spatial},
            dims=[context.time_dim, context.spatial_dims[0], context.spatial_dims[1]],
            name="clusters",
        )
        _cluster_attrs: dict[str, object] = {
            "cluster_vars": cluster_vars,
            "min_consensus": min_consensus,
            "top_n_clusters": top_n_clusters,
            "stitch_meridian": stitch_meridian,
            "spatial_adjacency": "native_grid_8_connected",
            "spacetime_consensus": True,
            "temporal_tolerance": temporal_tolerance,
            "spatial_tolerance": spatial_tolerance,
            "description": (
                "Spacetime lattice consensus (time × space graph). "
                "Cluster ids are global across time (same id = same connected component). "
                "temporal_tolerance and spatial_tolerance dilate peak-event labels for "
                "matching, but the returned mask is trimmed to original undilated event "
                "support; "
                "summary aggregates all timesteps; see compute_consensus docstring."
            ),
        }
        da_clusters.attrs.update(_cluster_attrs)
        da_consistency = xr.DataArray(
            consistency_out,
            coords={context.time_dim: context.time_coord, **context.coords_spatial},
            dims=[context.time_dim, context.spatial_dims[0], context.spatial_dims[1]],
            name="consistency",
        )
        return self._mark_no_shift_nan_on_consensus_dataset(
            xr.Dataset({"clusters": da_clusters, "consistency": da_consistency}),
            cluster_vars,
            context,
        )

    def _cluster_consensus_spacetime(
        self,
        cluster_vars: List[str],
        min_consensus: float,
        top_n_clusters: int | None,
        stitch_meridian: bool,
        show_progress: bool,
        temporal_tolerance: int,
        spatial_tolerance: int,
    ) -> xr.Dataset:
        """Spacetime lattice consensus on the native 8-neighbour graph.

        Invariants:
        - Node flat index ``t * n_space + s`` matches ``labels.reshape(T, -1).ravel()`` order
          with ``labels`` in ``(time, y, x)`` layout (transposed from disk order if needed).
        - ``sorted_cluster_labels`` is applied only to the full ``labels_st`` vector, not per
          time slice, so cluster ids are stable across ``time_dim``.
        """
        sample = self.td.data[cluster_vars[0]]
        context = self._build_spacetime_consensus_context(
            sample=sample,
            stitch_meridian=stitch_meridian,
        )
        vote_data = self._accumulate_spacetime_votes(
            cluster_vars=cluster_vars,
            top_n_clusters=top_n_clusters,
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
            show_progress=show_progress,
            context=context,
        )
        if vote_data is None:
            return self._empty_spacetime_consensus_result(context, cluster_vars)

        solved = self._solve_spacetime_consensus_components(
            min_consensus=min_consensus,
            context=context,
            vote_data=vote_data,
        )
        if solved is None:
            return self._empty_spacetime_consensus_result(context, cluster_vars)

        labels_st, cons_flat, deg = solved
        return self._build_spacetime_consensus_result(
            cluster_vars=cluster_vars,
            min_consensus=min_consensus,
            top_n_clusters=top_n_clusters,
            stitch_meridian=stitch_meridian,
            temporal_tolerance=temporal_tolerance,
            spatial_tolerance=spatial_tolerance,
            context=context,
            labels_st=labels_st,
            cons_flat=cons_flat,
            deg=deg,
        )
