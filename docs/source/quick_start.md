# Quick Start

## Simple use case

```python
from toad import TOAD
from toad.shifts import ASDETECT
from sklearn.cluster import HDBSCAN

# init TOAD object
td = TOAD("data.nc")

# Compute shifts for variable 'tas' using the method ASDETECT (Boulton & Lenton, 2019)
td.compute_shifts("tas", method=ASDETECT())

# Compute clusters for points that have shifts larger than 0.8 using HDBSCAN (McInnes, 2017)
td.compute_clusters(
    var="tas",
    method=HDBSCAN(min_cluster_size=10),
)

# Visualize results
td.plot.overview("tas");
```

For more details, please see the [tutorial](https://github.com/tipmip-methods/toad/blob/main/tutorials/basics.ipynb).
