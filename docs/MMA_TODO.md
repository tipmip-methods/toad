
## MMA review – summary

### Documentation updates

- **MMA_WORKFLOW.md**: Updated to cover
  - `cluster_occurrence_rate()`
  - `get_consensus_summary()`
  - Native plotting (both formats)
  - Plotting `cluster_occurrence_rate`
  - Method reference table
- **api_ref.rst**: Added MMA section with autosummary and link to the workflow doc.
- Removed reference to the old `cluster_consensus` aggregation.

---

## Critical issues (none)

No critical bugs were found. Current behaviour:

- Consensus logic in `consensus_utils` is consistent.
- HealPix and native paths are implemented and tested.
- Export/import flow is consistent.
- `cluster_occurrence_rate` can be called before `run_consensus`.
- Methods that use `self.data` correctly require `run_consensus`.

---

## Gaps and recommendations

### 1. No built-in `plot_cluster_occurrence_rate`

`plot_consensus_clusters` exists, but `cluster_occurrence_rate` has no equivalent. Users must plot manually (as in the doc examples). A `plot_cluster_occurrence_rate()` method would match `plot_consensus_clusters` and simplify usage.

### 2. `cluster_occurrence_rate` not stored in `mma.data`

`cluster_occurrence_rate()` returns a `DataArray` and does not add it to `mma.data`. Optional merging into `mma.data` (or storing in a separate attribute) could be useful when saving results.

### 3. HealPix: `get_healpix_latlon()` vs `run_consensus`

`get_healpix_latlon()` does not require `run_consensus` (it only needs `nside`). This is intentional and correct.

### 4. Native export: time dimension handling

Native export keeps `clusters.dims`; the time dimension is preserved. Behaviour appears correct, but it may be worth documenting that all exports must share the same time grid if times are used later.

### 5. Empty models

If a model has no non-noise clusters (all pixels noise), its masks list is empty. In `cluster_occurrence_rate` this yields 0/`n_models` for those points, which is correct. In consensus, `ever_clustered` stays False for points never in any cluster, which is also correct.

### 6. Sphinx and `MMA_WORKFLOW`

`MMA_WORKFLOW.md` sits in `docs/` and is not in the Sphinx build. The api_ref links to the GitHub file. For rendered docs, consider either moving it to `docs/source/` and adding it to the toctree, or using MyST to include it in the build.



---

## Architecture overview

| Component | Purpose |
|-----------|---------|
| `MMA` | Main interface; loads exports, runs consensus, exposes results |
| `consensus_utils` | KNN construction, weighted consensus, shared core logic |
| `compute_clusters(export_for_mma=..., mma_grid=...)` | Exports HealPix or native cluster labels for MMA |

The separation between load/consensus/mapping and the core consensus logic is clear.

---

## Test coverage

- HealPix and native exports
- MMA init (HealPix vs native)
- `run_consensus`, `get_shift_times_per_consensus_cluster`, `get_consensus_summary`, `cluster_occurrence_rate`
- `plot_consensus_clusters` for both formats

No missing tests were identified for core behaviour.

---

**Bottom line:** MMA is ready for use. The main improvement would be adding `plot_cluster_occurrence_rate()` and optionally integrating `cluster_occurrence_rate` into `mma.data` when saving results.