"""
This script generates sample datasets to be stored locally in NetCDF format.
The data points are generated and structured in a 3D array with dimensions (time, latitude, longitude)
using the synthetic data generator of the toad package.

coded by: Lukas Röhrich @ PIK
June 2025
"""

import numpy as np
import xarray as xr
import dask.array as da
import os

from toad.utils import create_global_dataset

# Parameters
shape = (24,24,24)#(100, 1000, 1000)
tim = np.arange(shape[0])
lat = np.linspace(-90, 90, shape[1])
lon = np.linspace(-180, 180, shape[2])
var = "var"
sample_size = 10
output_dir = "datasets_synth"
os.makedirs(output_dir, exist_ok=True)

print(f"> Generating {sample_size} datasets and saving to NetCDF...")
for i in range(sample_size):
    print(f">> Dataset {i+1}/{sample_size}")

    # Generate regular NumPy array (for chunk_size=None)
    array = np.random.random(shape).astype(np.float32)
    data_ds, labels_xr, shift_params = create_global_dataset(
        lat_size = shape[1],
        lon_size = shape[2],
        time_size = shape[0],
        n_shifts = 3,
        random_seed = i,            # Different seed for each sample
        background_noise = 0.01,    # Example noise level
        background_trend = 0.0,     # No trend
    )

    # Save to NetCDF — lazy Dask arrays will be computed during saving
    data_ds.to_netcdf(f"{output_dir}/sample_{i}.nc")
