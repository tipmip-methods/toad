# Quick Start

## Simple use case

```python
from toad import TOAD
from toad.shifts import ASDETECT
from sklearn.cluster import HDBSCAN


# init TOAD object
td = TOAD("data.nc")

# Compute shifts for variable 'temp' using the method ASDETECT (Boulton & Lenton, 2019)
td.compute_shifts("temp", method=ASDETECT())

# Compute clusters for points that have shifts larger than 0.8 using HDBSCAN (McInnes, 2017)
td.compute_clusters(
    var="temp",
    method=HDBSCAN(min_cluster_size=10),
)

# Visualize results
td.plot.overview("temp");
```

For more details, please see the [tutorial](https://github.com/tipmip-methods/toad/blob/main/tutorials/basics.ipynb).
