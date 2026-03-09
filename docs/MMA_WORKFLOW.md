# Multi-Model Aggregation (MMA) Workflow

This document describes the full workflow for running consensus clustering across multiple models or runs using TOAD's MMA pipeline.

## Overview

1. **Per model**: Compute shifts → cluster → export cluster labels (HealPix or native)
2. **MMA**: Load exported files → run consensus clustering → inspect results
3. *(Optional)*: Summary table with original dataset paths (deferred)

---

## Step 1: Per-Model Pipeline (Export for MMA)

For each model or run, compute shifts and clusters, then export cluster labels for MMA. Use a **fixed `nside`** when exporting HealPix so all files share the same grid.

### Option A: HealPix export (recommended for mixed grids)

```python
from pathlib import Path
from toad import TOAD
from toad.regridding import HealPixRegridder
from toad.shifts import ASDETECT
from sklearn.cluster import HDBSCAN

NSIDE = 32  # Use same nside for all models

# Per model
model_paths = ["model_a.nc", "model_b.nc", "model_c.nc"]
export_paths = []

for i, path in enumerate(model_paths):
    td = TOAD(path, time_dim="time")
    td.compute_shifts(td.base_vars[0], method=ASDETECT())
    td.compute_clusters(
        method=HDBSCAN(min_cluster_size=25),
        time_weight=1.0,
        shift_threshold=0.8,
        regridder=HealPixRegridder(nside=NSIDE),
        export_for_mma=f"clusters_model_{i}.nc",
        mma_grid="healpix",
    )
    export_paths.append(f"clusters_model_{i}.nc")
```

### Option B: Native export (same grid across models)

```python
from pathlib import Path
from toad import TOAD
from toad.shifts import ASDETECT
from sklearn.cluster import HDBSCAN

# Per model — requires lat/lon coordinates
model_paths = ["model_a.nc", "model_b.nc", "model_c.nc"]
export_paths = []

for i, path in enumerate(model_paths):
    td = TOAD(path, time_dim="time")
    td.compute_shifts(td.base_vars[0], method=ASDETECT())
    td.compute_clusters(
        method=HDBSCAN(min_cluster_size=25),
        time_weight=1.0,
        shift_threshold=0.8,
        export_for_mma=f"clusters_model_{i}.nc",
        mma_grid="native",
    )
    export_paths.append(f"clusters_model_{i}.nc")
```

**Note:** `mma_grid="healpix"` requires `regridder=HealPixRegridder(nside=...)`. Without it, an error is raised.

---

## Step 2: Run MMA Consensus

```python
from toad import MMA

# Paths to exported cluster label files
paths = ["clusters_model_0.nc", "clusters_model_1.nc", "clusters_model_2.nc"]

# nside must match the HealPix files (or be used when MMA regrids native files)
mma = MMA(paths, nside=NSIDE)  # use same NSIDE as export

# Run consensus
ds = mma.run_consensus(
    min_consensus=0.5,
    min_cluster_size=2,
    k_neighbors=8,
    top_n_clusters=10,  # optional: limit clusters per model
    show_progress=True,
)

# Results in mma.data
print(mma.data)
# consensus_clusters, consensus_consistency (both on HealPix)
```

---

## Step 3: Inspect Results

```python
import numpy as np

# Consensus clusters (1D on HealPix — can plot with healpy)
clusters = mma.data["consensus_clusters"].values
consistency = mma.data["consensus_consistency"].values

# Number of consensus clusters
n_clusters = len(set(c for c in clusters if c >= 0 and not np.isnan(c)))
print(f"Consensus: {n_clusters} clusters")

# Save for later use
mma.data.to_netcdf("consensus_result.nc")
```

---

## Complete Minimal Example

```python
from toad import MMA, TOAD
from toad.regridding import HealPixRegridder
from toad.shifts import ASDETECT
from sklearn.cluster import HDBSCAN

NSIDE = 32

# 1. Per model: compute and export
for i, name in enumerate(["model_a", "model_b", "model_c"]):
    td = TOAD(f"{name}.nc", time_dim="time")
    td.compute_shifts(td.base_vars[0], method=ASDETECT())
    td.compute_clusters(
        method=HDBSCAN(min_cluster_size=20),
        regridder=HealPixRegridder(nside=NSIDE),
        export_for_mma=f"{name}_clusters.nc",
        mma_grid="healpix",
    )

# 2. MMA consensus
mma = MMA(
    ["model_a_clusters.nc", "model_b_clusters.nc", "model_c_clusters.nc"],
    nside=NSIDE,
)
ds = mma.run_consensus(min_consensus=0.5)

# 3. Use results
mma.data.to_netcdf("consensus.nc")
```

---

## File Formats

| Format   | When to use            | Regridder required              |
|----------|------------------------|----------------------------------|
| `healpix`| Mixed/native grids     | `HealPixRegridder(nside=...)`   |
| `native` | All models same grid   | No                              |

- **HealPix**: All exported files must use the same `nside`. MMA loads them directly.
- **Native**: MMA regrids each file to a common HealPix grid using lat/lon before consensus. Specify `nside` when constructing `MMA(...)`.

---

## Summary Table (TODO)

A `summary_table(original_dataset_paths)` method that maps consensus clusters back to each model's native grid for per-cluster statistics (e.g. violin plots of shift distributions) is planned for a future release.
