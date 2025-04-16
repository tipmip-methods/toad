# Quick Start

## Simple use case

``` python
from toad import TOAD
from toad.shifts_detection.methods import ASDETECT
from sklearn.cluster import HDBSCAN


# init TOAD object
td = TOAD("data.nc")

# Compute shifts for variable 'temp' using the method ASDETECT (Boulton & Lenton, 2019)
td.compute_shifts("temp", method=ASDETECT())

# Compute clusters for points that have shifts larger than 0.8 using HDBSCAN (McInnes, 2017)
td.compute_clusters(
    var="temp",
    shifts_filter_func=lambda x: np.abs(x)>0.8, 
    method=HDBSCAN(min_cluster_size=25),        
)

# Visualise results
td.plotter().plot_clusters_on_map("temp");
```

For more details, please see the [tutorial](https://github.com/tipmip-methods/toad_torial/blob/main/tutorials/basics.ipynb). 
