
from toad.shifts_detection.methods import ASDETECT as ASDETECT
from toad import TOAD

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
import time

# user input
fp = "tutorials/test_data/garbe_2020_antarctica.nc"
var = "thk"
c = 1                                               # coarsenin of spatial dimensions
chunk_sizes = [None, 5, 20, 50, 70, 95 ,100, 150]   # chunk sizes for spatial dimensions
sample_size = 1                                    # number of samples for time measurement

# load data
data = xr.open_dataset(fp)
# coarse data
spatial_dims = list(data[var].dims)
spatial_dims.remove("time")
c_dict = {dim: c for dim in spatial_dims}
c_dict["time"] = 3
data = data.coarsen(**c_dict,
                    boundary="trim").reduce(np.mean)
data_dim = dict(data.sizes)
print(f"Dimensions after coarsening: {dict(data_dim)}")

# create TOAD object
td = TOAD(data)
#results = [0] * (len(chunk_sizes) + 1)          # add one for no chunking

# call compute_shift once to let dask create the lazy dataframe object
print("> Creating lazy dataframe object...")
td.compute_shifts(var,
                  method=ASDETECT(),
                  overwrite=True,
                  chunk_size=None,)
print("DONE.")

# run test
print("> Running test...")
results = [0] * len(chunk_sizes)                    # create space for results
for i in range(len(chunk_sizes)):
    print(f">> Chunk size: {chunk_sizes[i]}")
    size = chunk_sizes[i]
    for j in range(sample_size):
        print(f">>> Sample {j+1}/{sample_size}...",end="\n")
        # get test data
        td = TOAD(data)

        # Time the execution
        start_time = time.time()
        td.compute_shifts(var,
                          method=ASDETECT(),
                          overwrite=True,
                          chunk_size=size,)
        elapsed = time.time() - start_time

        results[i] += elapsed
    results[i] /= sample_size

# store results in pandas dataframe
print("Storing results in pandas dataframe...",end="")
chunk_str = [str(i) for i in chunk_sizes]
df = pd.DataFrame(results, columns=["Time (s)"])
df["Chunk Size"] = chunk_str
#df["Chunk Size"] = df["Chunk Size"].astype(str)
print(df)

df.to_pickle("chunk_size_performance.pkl")
print("DONE.")

# plot results
print("Plotting results...",end="")
plt.figure(figsize=(10, 5))
plt.bar(df["Chunk Size"], df["Time (s)"], color="lightblue")
plt.xlabel("Chunk Size")
plt.ylabel("Time (s)")
plt.title(f"Chunk Size Performance\nData Size: {data_dim}\n Time Sample Size: {sample_size}")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("chunk_size_performance.png")
print("DONE.")
