# Consensus Clustering Flow

This document describes the current **spacetime consensus** implementation, invoked via
`td.compute_consensus()` on `toad.core.TOAD` (which delegates to
`td.aggregate.compute_consensus()` on `toad.postprocessing.Aggregation`).

The implementation provides:

- spacetime consensus only
- native 8-neighbour grid connectivity as the default
- optional meridian stitching on native grids
- optional explicit HealPix regridding
- temporal and spatial tolerance as pre-vote dilation rules
- optional post-filter on minimum spatial cluster footprint

## Overview

`compute_consensus` combines multiple input cluster label maps into a single consensus
clustering in spacetime, then **merges the result into** ``td.data`` as normal variables (not
a separate returned dataset).

The main idea is:

1. Treat every labelled spacetime voxel as a possible part of a consensus event.
2. Build a sparse spacetime graph with local spatial and temporal edges.
3. Let each input clustering vote for edges where neighbouring voxels belong to the same input cluster.
4. Normalise vote counts by the number of contributing input maps.
5. Keep only edges whose support exceeds `min_consensus`.
6. Run connected components on the surviving graph.
7. Trim the result back to the original undilated support.
8. Optionally filter tiny clusters and renumber labels; attach metadata as TOAD ``attrs``.

The public result is one **3D** consensus label field (plus a consistency field) on the original
model grid, stored on the `TOAD` object’s `xarray.Dataset`.

## Public API

Consensus is a **void** method: it **writes** into ``td.data``. Typical usage:

```python
td.compute_consensus(
    cluster_vars=None,  # or an explicit list of input cluster map names
    min_consensus=0.75,
    temporal_tolerance=0,
    spatial_tolerance=0,
    top_n_clusters=None,
    stitch_meridian=False,
    regridder=None,
    show_progress=True,
    # optional naming / safety:
    output_label_suffix="",
    output_label=None,
    overwrite=False,
    min_cluster_area=2,
)
```

The default name for the consensus **labels** variable is ``"cluster_consensus"`` (plus your
suffix), unless ``output_label`` is set. A companion **consistency** array is written as
``f"{label_name}_consistency"``.

**Summary table (per–consensus cluster statistics)** is not the return value of
`compute_consensus`; call `td.aggregate.consensus_summary()` (or
`toad.postprocessing.Aggregation.consensus_summary`) when you need the
`pandas.DataFrame` after the run.

## Input Expectations

The method expects multiple clustering variables in `td.data`.

Each clustering variable should be a labelled spacetime field:

- same time axis
- same spatial grid
- `-1` for clustered input noise and `NaN` for voxels with no detected event support
- non-negative integers for cluster ids

Consensus is always computed across the supplied input cluster maps, not from raw variables directly.

## Core Flow

### 1. Select input cluster maps

If `cluster_vars` is `None`, TOAD uses all cluster variables in `td.cluster_vars`.

If `top_n_clusters` is set, only the largest `N` clusters by actual spacetime size from each input clustering are allowed to vote.

### 2. Choose spatial graph mode

Consensus always works on a spacetime graph, but the spatial part of that graph can be built in different ways.

There are currently two modes.

#### Mode A: Native grid mode

This is the default when `regridder=None`.

The spatial graph is built directly on the original grid using 8-neighbour connectivity:

- horizontal neighbours
- vertical neighbours
- diagonal neighbours

This applies to:

- regular `lat/lon` grids
- curvilinear `i/j` grids with 2D latitude/longitude coordinates
- other native grids without geographic coordinates

If `stitch_meridian=True`, the first and last grid columns are also connected across the seam, including diagonal seam neighbours.

This is useful for domains that are split at the meridian.

#### Mode B: Explicit regridded HealPix mode

This is used only when `regridder` is provided.

Currently only `HealPixRegridder` is supported.

In this mode:

1. original grid cells are mapped to HealPix pixels
2. the spatial graph is built on native HealPix neighbour relations
3. consensus voting happens in HealPix space
4. the final result is mapped back to the original model grid

Important:

- passing `regridder=HealPixRegridder(...)` is an explicit opt-in
- this works both for regular `lat/lon` grids and curvilinear `i/j` grids, as long as latitude/longitude coordinates exist
- if no latitude/longitude coordinates exist, TOAD raises `ValueError`

### 3. Build the spacetime graph

After the spatial graph is chosen, TOAD builds one graph on `(time x space)` nodes.

Each node is one spacetime voxel.

The graph contains:

- spatial edges within each time slice
- temporal chain edges between the same spatial node at consecutive times

So the consensus graph is fully spacetime-resolved.

### 4. Dilate each input clustering before voting

For each input clustering, TOAD reshapes the labels into `(time, space)` and optionally dilates them before voting.

This dilation is controlled by:

- `temporal_tolerance`
- `spatial_tolerance`

These do not add new edge types to the graph.
Instead, they temporarily expand labelled support before evaluating the standard local spacetime graph.

#### Temporal tolerance

`temporal_tolerance=k` means:

- if a voxel is labelled at time `t`
- it is treated as active for the same cluster id at all times `t'` with `|t' - t| <= k`

This pools timing jitter before voting.

#### Spatial tolerance

`spatial_tolerance=k` means:

- if a voxel is labelled at one spatial node
- it is treated as active for the same cluster id up to `k` spatial graph hops away

This pools spatial jitter before voting.

The hop distance is measured on the chosen spatial graph:

- native 8-neighbour grid in native mode
- HealPix neighbour graph in regridded mode

#### Conflict handling during dilation

If different cluster ids overlap after dilation within one input clustering, the overlap is marked as conflict and is not allowed to vote as a clean cluster member.

This prevents ambiguous regions from contributing misleading support.

### 5. Vote for local spacetime edges

For each input clustering, TOAD checks every edge in the spacetime graph.

An edge receives a vote from that clustering if both edge endpoints:

- are active after dilation
- have the same cluster id
- belong to an allowed input cluster

Votes are accumulated across all input clusterings.

### 6. Normalise by the number of contributing maps

The weighted consensus matrix is computed as:

```text
W = V / A
```

where:

- `V` is the number of maps that voted for an edge
- `A` is the number of contributing input maps

In other words, `min_consensus` is interpreted as:

> keep edges that are supported by at least this fraction of the input cluster maps

### 7. Threshold the consensus graph

Edges with:

```text
W >= min_consensus
```

are kept.

All weaker edges are discarded.

### 8. Solve connected components

Connected components on the surviving thresholded graph become the provisional consensus clusters.

### 9. Trim back to original support

Temporal and spatial tolerance affect matching, but they do not directly thicken the public output.

After connected components are found, TOAD trims the result back to voxels that had support in at least one original undilated input clustering.
The remaining clusters are then re-sorted by final spacetime size so that output ids reflect the trimmed public result.

This is very important:

- tolerance changes which regions can agree
- tolerance does not directly define the final public mask thickness

## Output

`compute_consensus` **returns ``None``** and **merges** the following into ``td.data`` (names depend on
``output_label`` / ``output_label_suffix``; defaults below use the built-in label name):

### 1. Consensus label field

- **Default variable name** `"cluster_consensus"` (unless renamed): integer consensus labels
  on the original grid, with `variable_type=consensus_cluster` and
  `cluster_vars` stored on `attrs`.

### 2. Consistency field

- **Default name** ``"cluster_consensus_consistency"``: mean consensus weight of the
  surviving threshold-passing edges touching each voxel, with ``variable_type`` for consistency
  and a pointer to the label variable.

The data layout is always the original TOAD grid:

- native mode: original grid in, original grid out
- HealPix regridded mode: original grid in, HealPix used internally, original grid out

So even when regridding is used internally, the **stored** label field is mapped back onto the original model grid.

If multiple original cells map to the same HealPix pixel, they receive the same returned consensus label for that timestep.

### 3. Summary table (on demand)

A `pandas.DataFrame` with one row per consensus cluster (including
`cluster_id`, `mean_consistency`, `area`, `volume`, means of spatial
coordinates, shift-time statistics, etc.) is produced by
`td.aggregate.consensus_summary()` from the **stored**
label and consistency fields—call it after `compute_consensus` when you need the table.

For the spacetime consensus:

- `area` is the spatial footprint size
- `volume` is the number of labelled spacetime voxels

These statistics are computed after trimming to original support, and (if enabled) after the
optional minimum-area post-filter.

## Parameter Reference

### `cluster_vars`

Which input cluster label variables to aggregate.

- `None`: use all cluster variables in `td.cluster_vars`
- list of names: use only those cluster maps

### `min_consensus`

Edge support threshold in `[0, 1]`.

Typical interpretation:

- `0.5`: keep edges supported by at least half the maps
- `0.75`: keep edges supported by at least three quarters of the maps
- `1.0`: require unanimous support

Higher values give stricter, usually smaller, more conservative consensus clusters.

### `top_n_clusters`

Limit voting to the largest `N` clusters in each input clustering, ranked by their actual spacetime voxel count.

This does not rely on the stored cluster-id order. Smaller excluded clusters do not vote and do not count towards the retained original-support mask.

### `stitch_meridian`

Only affects native grid mode.

If `True`, connect the first and last columns of the original grid.

Use this when the grid is split at the meridian and these columns are true neighbours geographically.

Do not use it for regional domains that do not wrap around.

### `regridder`

Explicitly switch consensus into regridded mode.

Currently supported:

- `HealPixRegridder(...)`

Behaviour:

- `None`: use native grid adjacency
- provided: use HealPix adjacency internally and map the result back to the original grid

This is the switch that determines whether consensus happens on the original grid topology or on HealPix.

### `show_progress`

Show or hide the progress bar while input clusterings are processed.

### `temporal_tolerance`

Non-negative integer temporal dilation radius.

- `0`: exact-time voting only
- `1`: allow plus/minus one timestep timing mismatch
- larger values: tolerate broader timing jitter

This is a local agreement rule, not a cap on the total final cluster duration.
Large connected components can still extend far in time through transitive chains.

If `temporal_tolerance` is larger than the available time axis, it effectively saturates.
In that case a cluster id is treated as active at all timesteps where that input map exists,
so the matching semantics become close to a collapsed-time interpretation. This still does
not recreate the old collapsed consensus output, because the public result remains a trimmed
3D spacetime mask on the new spacetime graph.

### `spatial_tolerance`

Non-negative integer spatial dilation radius in graph hops.

- `0`: exact spatial support only
- `1`: allow one-hop spatial mismatch
- larger values: allow broader spatial jitter

This is evaluated on the chosen spatial graph, so its meaning depends on the active mode:

- native mode: hops on the native 8-neighbour grid
- regridded mode: hops on the HealPix neighbour graph

If `spatial_tolerance` is larger than the effective graph diameter of the relevant domain,
it also saturates and the dilation can reach most or all of the connected spatial graph.
This is usually both conceptually undesirable and computationally expensive: unlike temporal
tolerance on a 1D axis, large spatial tolerance requires broad graph-hop expansion and can
become very slow on large grids.

### `output_label_suffix` / `output_label` / `overwrite`

Control how the new variables are named and whether existing names are replaced or uniquified
(same idea as for `toad.clustering.compute_clusters`).

### `min_cluster_area`

After consensus, drop clusters whose **spatial** footprint (distinct cells that ever carry that
id) is below this threshold; those cells become noise ``-1`` and ids are re-sorted. Use ``None``
to disable, or ``0`` for the same effect as “no size filter” in the implementation.

## Current Modes Summary

### Native mode

Use this when:

- you trust the original model topology
- the grid is native `x/y` or `i/j`
- you want spatial tolerance measured on the original grid

Configuration:

```python
td.compute_consensus(
    min_consensus=0.75,
    temporal_tolerance=0,
    spatial_tolerance=0,
    stitch_meridian=False,
    regridder=None,
)
```

### Native mode with seam stitching

Use this when:

- the original grid wraps across the first/last column
- you want native topology plus explicit seam closure

Configuration:

```python
td.compute_consensus(
    min_consensus=0.75,
    temporal_tolerance=0,
    spatial_tolerance=0,
    stitch_meridian=True,
    regridder=None,
)
```

### Explicit HealPix regridded mode

Use this when:

- you want consensus neighbourhoods defined on a common spherical grid
- you want curvilinear `i/j` grids to be handled through geographic reprojection
- you want spatial tolerance measured in HealPix neighbour hops

Configuration:

```python
from toad.regridding import HealPixRegridder

td.compute_consensus(
    min_consensus=0.75,
    temporal_tolerance=0,
    spatial_tolerance=0,
    regridder=HealPixRegridder(nside=16),
)
```

## Important Behaviour Changes Relative to Older Versions

- The old collapsed-time consensus mode has been removed.
- Consensus is always spacetime-resolved.
- KNN-based geographic adjacency has been removed.
- Native grids now use fixed 8-neighbour connectivity.
- HealPix adjacency now uses native HealPix neighbours instead of KNN on pixel centres.
- Regridding is now explicit rather than automatic.
- The API is integrated into the main TOAD workflow via `compute_consensus` (in-place on ``td.data``)
  instead of returning a free-standing dataset from a `cluster_consensus` method.

## Practical Interpretation

This algorithm is best understood as a way to find robust spacetime event regions whose internal local connectivity is repeatedly supported across multiple input clusterings.

It does not ask:

> which voxels are globally clustered together in most inputs?

It asks:

> which local spacetime neighbour relations are repeatedly supported strongly enough that they form robust connected components?

That is why:

- `min_consensus` acts on local graph edges
- tolerance affects local matching before voting
- large final clusters can still form via transitive paths

## Common Pitfalls

### `regridder` changes the computation, not the output grid

If you pass `regridder=HealPixRegridder(...)`, consensus is computed in HealPix space internally, but the **saved** label field in ``td.data`` still lives on the original model grid.

### `temporal_tolerance` does not mean final clusters are only `2k+1` timesteps long

Tolerance is local.
Connected components can span much longer periods through chains of supported edges.

### `spatial_tolerance` can merge nearby events

If it is too large, nearby but distinct events can become connected through the dilated support.

### `stitch_meridian=True` should only be used when the domain really wraps

It is not a generic option for all native grids.

## Suggested Usage Patterns

### Conservative consensus

```python
td.compute_consensus(
    min_consensus=0.8,
    temporal_tolerance=0,
    spatial_tolerance=0,
)
```

### Allow modest timing and spatial jitter

```python
td.compute_consensus(
    min_consensus=0.6,
    temporal_tolerance=1,
    spatial_tolerance=1,
)
```

### Force geographic regridding before consensus

```python
from toad.regridding import HealPixRegridder

td.compute_consensus(
    min_consensus=0.6,
    temporal_tolerance=1,
    spatial_tolerance=1,
    regridder=HealPixRegridder(nside=16),
)
```

## Related helper functions

Once consensus has been computed, the following methods on `td.aggregate` are useful for
analysing the result (they take the **stored** consensus label array, by variable name):

- `consensus_shift_time_distribution`
- `consensus_shift_time_distributions` (for violin-style pools; you may pass
  `distribution_result=...` from a prior `consensus_shift_time_distribution` call to avoid
  duplicate work)
- `consensus_cluster_timeseries`

These work from the final consensus clusters and use support-aware filtering so that summary
statistics and extracted diagnostics come only from input clusterings that actually support the
final consensus region.
