# MMA – development notes

## Current architecture

| Component | Purpose |
|-----------|---------|
| `MMA` | Loads per-model exports, runs consensus, exposes results and plotting |
| `healpix_member_support_consensus` | Member-support consensus on `(time, hp_pixel)` HEALPix grids |
| `member_support_consensus` | Member-support consensus on native `(time, y, x)` grids (used via `TOAD.compute_consensus` for native MMA) |
| `compute_clusters(export_for_mma=..., mma_grid=...)` | Exports spacetime cluster labels for MMA |

MMA uses the **same member-support algorithm** as `td.compute_consensus()`, not the old edge-vote method.

---

## Implemented

- HealPix and native export via `export_for_mma`
- Member-support consensus on both formats
- Spacetime consensus output (`time × space`)
- `cluster_occurrence_rate()` (ever-in diagnostic, independent of consensus)
- `get_consensus_summary()`, shift-time extraction, map plotting
- Tests in `tests/mma/` and `tests/postprocessing/test_healpix_member_support_consensus.py`

---

## Optional improvements

### 1. `plot_cluster_occurrence_rate()`

`plot_consensus_clusters` exists; occurrence rate still needs manual plotting. A dedicated method would mirror the consensus plot API.

### 2. Store `cluster_occurrence_rate` in `mma.data`

Currently returned as a standalone `DataArray`. Could be merged when saving results.

### 3. Sphinx integration

`MMA_WORKFLOW.md` lives in `docs/` and is linked from `api_ref.rst` via GitHub. Consider adding it to the Sphinx toctree (e.g. via MyST).

### 4. Shared time grid validation

Exports should share the same time axis across models. MMA checks this on load; document clearly for users.

---

## Removed (stale)

- `toad/utils/consensus_utils.py` — old edge-vote / ever-in consensus (superseded by member-support)
