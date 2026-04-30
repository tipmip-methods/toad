# Spacetime consensus clustering

`td.compute_consensus()` (on `toad.core.TOAD`, delegating to
`toad.postprocessing.Aggregation.compute_consensus`) merges several input **cluster label maps**
into one **3D** consensus field on the same time axis and grid. Voting happens on a **spacetime
graph** (space and time as one grid). Results are written in place into `td.data`; there is no
separate return dataset.

## Contents

- [At a glance](#at-a-glance)
- [Public API](#public-api)
- [How to read the rest](#how-to-read-the-rest)
- [Inputs in TOAD](#inputs-in-toad)
- [How it works (pipeline)](#how-it-works-pipeline)
- [Core flow (detailed)](#core-flow-detailed)
- [Output](#output)
- [Parameter reference](#parameter-reference)
- [Configuration examples](#configuration-examples)
- [Changes from older TOAD versions](#changes-from-older-toad-versions)
- [Interpreting the result](#interpreting-the-result)
- [Common pitfalls](#common-pitfalls)
- [Related helpers](#related-helpers)

## At a glance


| Role           | Content                                                                                                       |
| -------------- | ------------------------------------------------------------------------------------------------------------- |
| **Input**      | Two or more cluster variables on `td` (same time × space).                                                    |
| **Parameters** | `min_consensus`, `temporal_tolerance`, `spatial_tolerance` (required), plus optional graph and post-filters.  |
| **Output**     | Consensus label + consistency on the **original** grid; optional per-cluster table via `consensus_summary()`. |


**Idea in one sentence:** build local spacetime **edges** between neighbours, count how often each
edge is supported across inputs, keep edges with enough support, then take **connected
components**—not a global “majority label per cell” vote.

```mermaid
flowchart TD
  s1[1. Choose input maps, build spacetime graph: spatial edges each time, plus t→t+1 in time at each site] --> s2[2. Dilate each input: temporal and spatial tolerances (optional)]
  s2 --> s3[3. Each edge: if dilated support agrees, add to V, A; then W = V/A]
  s3 --> s4{4. W >= min_consensus ?}
  s4 -->|yes| s5[5. Surviving graph: components, global ids, trim to undilated support]
  s4 -->|no| s6[6. Edge discarded (below threshold)]
```



Step **1** in the figure also applies optional **topn** (largest input clusters per map) before
voting. **Step 2** only relabels **where** each cluster id is active for matching on the **fixed**
graph from step 1; it does not add new edge types (see [How it works](#how-it-works-pipeline)).

## Public API

The method is **void**; it **writes** into `td.data`.

```python
td.compute_consensus(
    cluster_vars=None,  # or explicit list of input cluster map names
    min_consensus=0.75,
    temporal_tolerance=0,
    spatial_tolerance=0,
    top_n_clusters=None,
    stitch_meridian=False,
    show_progress=True,
    output_label_suffix="",
    output_label=None,
    overwrite=False,
    min_cluster_area=2,
)
```

- Default **labels** name: `"cluster_consensus"` (plus suffix), unless `output_label` is set.
- Companion **consistency** variable: `f"{label_name}_consistency"`.
- The **per-cluster summary table** is not built here; call
`td.aggregate.consensus_summary()` when you need a `DataFrame` after the run.

## How to read the rest

- Skim [Output](#output) if you need encoding rules for the **consensus result** (inputs follow normal TOAD cluster fields).
- Read **How it works** and **Core flow (detailed)** to understand the algorithm.
- Use **Parameter reference** as a dictionary while you tune runs.
- Check **Pitfalls** before interpreting surprising clusters.

## Inputs in TOAD

`compute_consensus` takes existing **cluster** variables on your `td` (for example from
`toad.clustering.compute_clusters`). They are the same 3D fields and the same in-package label
conventions you already use elsewhere. This page documents the **consensus layer**; see
[Output](#output) for how the **consensus** label and consistency fields are encoded (that output
is not identical to a single input map).

## How it works (pipeline)

1. (Optional) Restrict each input to its **largest** `N` clusters if `top_n_clusters` is set.
2. Fix the **spatial** part of the graph: **native** 8-neighbour on the label grid, optionally
  **meridian** stitched; adjacency is always in the dataset's index / `(y, x)` layout.
3. One **spacetime** graph: nodes are `(time, space)` voxels, edges are **in-slice spatial**
  neighbours + **in-time** links between the same space cell at consecutive times.
4. For **each** input map, optionally **dilate** where each cluster id is “active” using
  `temporal_tolerance` and `spatial_tolerance` (see below). Dilation can mark **conflicts**
   where two ids would claim the same node—those do not vote as clean same-id support.
5. For each spacetime **edge** and each input, record whether that map **votes** (both endpoints
  active, same id, allowed cluster).
6. Build **V** and **A** on that edge set: **W = V / A**; keep edges with
  **W ≥ min_consensus**.
7. **Connected components** on the surviving graph → provisional consensus cluster ids; **trim**
  to voxels that had support in at least one **undilated** input; optionally **filter** small
   spatial footprints (**min_cluster_area**) and renumber by size.

Tolerances **do not** add new edge types. They only change **who counts as agreeing** on the
fixed spacetime graph. They also **do not** define a fixed thickness of the final mask: matching
is local, and long structures can still arise through **transitive** chains of edges.

## Core flow (detailed)

The subsections follow the same order as the pipeline above.

### 1. Select input cluster maps

- `cluster_vars is None` → all `td.cluster_vars`.
- With `top_n_clusters`, only the **largest N** clusters by **spacetime voxel count** in each
input are allowed to vote (order of stored ids does not matter).

### 2. Native spatial graph (8-neighbour)

- **8-neighbour** (horizontal, vertical, diagonal) in **index** space: `(i, j)` neighbours on the
stored label array.
- Works for **lat/lon**, **i/j** with 2D lat/lon as coords, or grids **without** geographic
coordinates: the graph follows the **array layout**, not great-circle distance.
- `stitch_meridian=True`: also connect the **first and last columns** (and diagonal seam
connections). Use only for domains that really wrap; not for small regional windows.

### 3. Spacetime graph

One graph on `(time × space)` nodes. Edges: **same-time spatial** neighbours, plus **time**
links `(t, s)`–`(t+1, s)` for every space node `s`.

### 4. Dilation (temporal and spatial tolerance)

For each input, labels are viewed on `(time, space)` and optionally **dilated** before votes:


| Parameter                  | Meaning                                                                                                                                       |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **temporal_tolerance = k** | If a cell has a cluster id at time `t`, that id is active for the same id at all times within **k** steps of `t` (clipped to the time range). |
| **spatial_tolerance = k**  | The same id is treated as active up to **k** hops on the **8-neighbour** spatial graph.                                                       |


**Conflicts:** if two different cluster ids’ dilations would occupy the same node, that node is
marked as conflict; it does **not** supply clean “same id on both sides” support for an edge in
that input.

### 5. Voting on edges

For each spacetime edge, each input map casts at most one “agreement” if both endpoints are
active and share the same allowed cluster id.

### 6. Weight, threshold, components

- Aggregated **V** and **A** on the **same** undirected edge set, **W = V / A** per edge
(`A` is **per edge**—local availability, not a single global integer).
- Keep edges with **W ≥ min_consensus**; run **connected components** on the graph that
remains.
- Provisional cluster labels, then **trim** to undilated support and **re-sort** ids by final
spacetime size so labels are global across time.

**Important:** tolerance changes **where** agreement can occur; the **output mask** is still
**trimmed** to where at least one input had an undilated label.

## Output

`compute_consensus` returns `**None`** and updates `td.data` (names follow `output_label` /
`output_label_suffix`; defaults use `"cluster_consensus"`).

### 1. Consensus label field

- **Default name** `"cluster_consensus"`; `variable_type=consensus_cluster`; `cluster_vars` in
`attrs`.

**Encoding** (same idea as `toad.clustering.compute_clusters`):

- `**NaN`** — every input field is `**NaN**` at that cell (no abrupt shift in any map).
- `**-1**` — at least one input had a defined label, but the cell is **not** in a consensus
component (noise / not in consensus).
- **Non-negative integers** — consensus cluster id, stable in time after trimming and
re-sorting.

Dtype may be **float** where `NaN` is present.

### 2. Consistency field

- Default: `"cluster_consensus_consistency"`: mean of surviving edge weights at each **labelled**
voxel. `**NaN`** where the label is `**NaN**` (all-input no shift).

**Grid:** results are on the same **(time, y, x)** layout as the input cluster fields.

### 3. Summary table (on demand)

`td.aggregate.consensus_summary()` builds a `**DataFrame`** from the stored label +
consistency (e.g. `cluster_id`, `mean_consistency`, **area** = spatial footprint, **volume** =
spacetime voxel count, shift-time stats, …) **after** trim and, if enabled, **min_cluster
area** filtering.

## Parameter reference

**Always required (no defaults in the API):** `min_consensus`, `temporal_tolerance`,
`spatial_tolerance`.


| Parameter                                            | Role                                                                                                                                                                                  |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cluster_vars`                                       | Which input cluster maps; `None` = all `td.cluster_vars`.                                                                                                                             |
| `min_consensus`                                      | Edge **fraction** in `[0, 1]`; higher → stricter, usually smaller regions.                                                                                                            |
| `top_n_clusters`                                     | Per input, only the **largest N** spacetime clusters may vote.                                                                                                                        |
| `stitch_meridian`                                    | Connect **first/last column** on the index grid (native mode).                                                                                                                        |
| `show_progress`                                      | Progress bar.                                                                                                                                                                         |
| `output_label` / `output_label_suffix` / `overwrite` | Naming and replace vs uniquify, like `compute_clusters`.                                                                                                                              |
| `min_cluster_area`                                   | Drop clusters with spatial footprint < threshold (`None` to disable; `0` = no filter).                                                                                                |
| `temporal_tolerance`                                 | Time dilation **radius** (integer); local rule, not a cap on final cluster **duration** (chains can be long). Very large **k** can behave like time-saturated matching on short axes. |
| `spatial_tolerance`                                  | Spatial hop **radius**; large values can be slow and over-merge.                                                                                                                      |


## Configuration examples

**Two native configurations (all other parameters can match):**

```python
# Default native grid
td.compute_consensus(
    min_consensus=0.75,
    temporal_tolerance=0,
    spatial_tolerance=0,
    stitch_meridian=False,
)

# Native + meridian wrap
td.compute_consensus(
    min_consensus=0.75,
    temporal_tolerance=0,
    spatial_tolerance=0,
    stitch_meridian=True,
)
```

**Tuning agreement vs jitter:**

```python
# Stricter local edges, no time/space blur
td.compute_consensus(
    min_consensus=0.8,
    temporal_tolerance=0,
    spatial_tolerance=0,
)

# Softer edge threshold, allow ±1 step in time and space
td.compute_consensus(
    min_consensus=0.6,
    temporal_tolerance=1,
    spatial_tolerance=1,
)
```

## Changes from older TOAD versions

- **Collapsed-time** consensus and the old **KNN** geographic adjacency are **removed**; the only
path is the **3D** spacetime graph described here.
- **Native** space uses **fixed 8-neighbour** topology in index space (not KNN on pixel centres).
- The workflow is `**compute_consensus` in place** on `td.data`, not a free-standing
`cluster_consensus`-style return.

## Interpreting the result

The method finds **spacetime regions** whose **local** internal edges (after dilation) are
repeatedly supported. It does **not** ask “which voxels are in the same label in a majority of
maps globally”.

It does ask: **which neighbour relations** along the fixed spacetime graph survive the **W**
threshold so strongly that they form a **component**? That is why:

- `min_consensus` is about **edges**, not per-cell mode labels;
- tolerances only change **local** matching;
- the final object can be **long in time** or **large in space** through **chained** edges.

## Common pitfalls

- `**temporal_tolerance = k`** does **not** cap cluster **length** to `2k+1`; components can
extend much farther along time through a chain of surviving edges.
- **Large `spatial_tolerance`** can connect distinct nearby events; it can also be **expensive**
to compute on big grids.
- `**stitch_meridian=True`** is wrong for regional, non-wrapping domains.
- **Grid structure:** 8-neighbour adjacency is tied to the **data array layout**; cell aspect ratio
and zonal vs meridional spacing on lat/lon grids are **not** equalized in this consensus path.

## Related helpers

On `td.aggregate` (use the **stored** consensus variable name):

- `consensus_shift_time_distribution`
- `consensus_shift_time_distributions` (optional `distribution_result=...` to avoid recomputing)
- `consensus_cluster_timeseries`

These use **support-aware** filtering: diagnostics refer to input runs that **actually** support
each consensus region.