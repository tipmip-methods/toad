# Multi-Model Aggregation (MMA) Workflow

This document describes the full workflow for running consensus clustering across multiple models or runs using TOAD's MMA pipeline. It also serves as the main development reference for the MMA process.

## Overview

1. **Per model**: Compute shifts → cluster → export cluster labels (HealPix or native) + cluster variable in original dims
2. **MMA**: Load exported files → run consensus clustering (uses full 3D `cluster`, ever-in semantics) → inspect results
3. **Shift time stats**: Extract shift time distributions per consensus cluster (from exports only; no original dataset needed)

**Time collapse:** MMA uses *ever-in* semantics: a pixel participates in a consensus cluster if it was ever assigned to that cluster at any time. This matches the old `cluster_consensus` aggregation and preserves multi-regime behaviour (a pixel can contribute to multiple clusters across time).

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

**Note:** HealPix requires `regridder=HealPixRegridder(nside=...)` and expects lat/lon. For projected grids (e.g. x,y in km), use Option B instead.

### Option B: Native export (x,y or lat/lon, same grid across models)

```python
from toad import TOAD
from toad.shifts import ASDETECT
from sklearn.cluster import HDBSCAN

# Per model — works with x,y (km), lat/lon, or any 2D spatial coords
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

**Note:** All native exports must have the same spatial grid. Works with x,y in km (e.g. South Polar Stereographic), lat/lon, or any 2D coordinates.

---

## Step 2: Run MMA Consensus

```python
from toad import MMA

# Paths to exported cluster label files
paths = ["clusters_model_0.nc", "clusters_model_1.nc", "clusters_model_2.nc"]

mma = MMA(paths, nside=NSIDE)  # HealPix: same NSIDE as export. Native: use nside=None

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

## Step 3: Inspect and Plot Results

```python
import numpy as np

# Consensus clusters (1D on HealPix)
clusters = mma.data["consensus_clusters"].values
consistency = mma.data["consensus_consistency"].values

# Number of consensus clusters
n_clusters = len(set(c for c in clusters if c >= 0 and not np.isnan(c)))
print(f"Consensus: {n_clusters} clusters")

# Save for later use
mma.data.to_netcdf("consensus_result.nc")
```

### Plotting consensus clusters (HealPix)

**Built-in method** (recommended)

```python
import cartopy.crs as ccrs

fig, ax = mma.plot_consensus_clusters(
    map_style={"projection": "mollweide", "continent_shading": True},
)

# Or with custom projection and options
fig, ax = mma.plot_consensus_clusters(
    map_style={"projection": ccrs.Orthographic(-40, 15), "continent_shading": True},
    s=10,
    show_noise=True,  # plot noise pixels as grey
)
```

**Manual plotting** (e.g. to reuse an existing axes from `td.plot.map()`)

```python
import cartopy.crs as ccrs

fig, ax = td.plot.map(map_style={"projection": ccrs.Orthographic(-40, 15)})
ax.set_global()
fig, ax = mma.plot_consensus_clusters(ax=ax, s=1)
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

MMA accepts **HealPix** or **native** format. Format is auto-detected from the first file.

| Format   | When to use            | MMA init              |
|----------|------------------------|-----------------------|
| HealPix  | Mixed grids, global    | `MMA(paths, nside=32)`|
| Native   | Same grid, x,y or lat/lon | `MMA(paths, nside=None)` |

**HealPix:** Use `mma_grid='healpix'` and `regridder=HealPixRegridder(nside=...)`. Requires lat/lon.

**Native:** Use `mma_grid='native'`. Works with x,y (km), lat/lon, or any 2D coords. All files must have the same grid.

---

## Shift time distributions per consensus cluster

Uses HealPix index lookup: (lat, lon) → pixel index → consensus ID. See `map_consensus_to_coords` / `map_consensus_to_dataset`.

**Option A: From export files only** (no original dataset)

```python
from toad import MMA
import matplotlib.pyplot as plt

# After run_consensus(): aggregate shift times per consensus cluster across all exports
times_by_cluster = mma.get_shift_times_per_consensus_cluster()
for cid, times in times_by_cluster.items():
    plt.hist(times, bins=30, alpha=0.5, label=f"Cluster {cid}")
plt.xlabel("Shift time")
plt.legend()
plt.show()
```

For a single cluster from a single file: `mma.get_shift_times_from_export(path, consensus_cluster_id=0)`.

**Option B: With TOAD (original dataset)**

```python
from toad import MMA, TOAD

td = TOAD("model_a.nc", time_dim="time")
consensus_ids = mma.map_consensus_to_dataset(td.data)
mask = (consensus_ids == 0) & consensus_ids.notnull()
times = td.get_cluster_times_in_region(mask, cluster_var=td.cluster_vars[0])
```

**Methods:**
- `get_shift_times_from_export(path, consensus_cluster_id)` — extract times from export file; no TOAD needed
- `map_consensus_to_coords(lat, lon)` — raw numpy lookup; returns cluster IDs with same shape as lat/lon
- `map_consensus_to_dataset(ds)` — map consensus onto xarray dataset; uses auto-detected lat/lon, returns DataArray with matching spatial dims
