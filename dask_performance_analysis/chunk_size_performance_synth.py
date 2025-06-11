"""
This script take sample datasets and runs them trough the ASDETECT method to compute shifts.
Each sample is run with different chunk sizes to compare performance.
The total time over all samples is averaged for each chunk size.

coded by: Lukas Röhrich @ PIK
June 2025
"""


from toad.shifts_detection.methods import ASDETECT as ASDETECT
from toad import TOAD

import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
import time
import os
import mpi4py.MPI as MPI
import sys

shape = (100, 500, 500)
tim = np.arange(shape[0])
lat = np.linspace(-90, 90, shape[1])
lon = np.linspace(-180, 180, shape[2])
var = "value"
sample_size = 1#10
chunk_sizes = [None, 5, 250]
scheduler = "synchronous" #"synchronous" #"threads" #"processes"
data_dir = "dataset_500"
filename_out = "dask_500_" + scheduler

results = []
memory_chunk = []
chunk_labels = []

# parallelisation
rank = MPI.COMM_WORLD.Get_rank()  # Each process gets a unique rank
sys.stdout = open(f"log/{scheduler}_{rank}.out", "w")
sys.stderr = open(f"log/{scheduler}_{rank}.err", "w")

print("\n\n> Running benchmark...")
for size in chunk_sizes:
    print(f">> Chunk size: {size}")
    times = []

    for i in range(sample_size):
        print(f">>> Sample {i+1}/{sample_size}")

        # Load appropriate data
        if size is None:
            ds = xr.open_dataset(f"{data_dir}/sample_{i}.nc")
            td = TOAD(ds)

            start = time.time()
            td.compute_shifts(var,
                              method=ASDETECT(),
                              overwrite=True,
                              chunk_size=None,
                              dask_compute=False,
                              scheduler=scheduler)
            elapsed = time.time() - start

            output_path = os.path.join(data_dir, f"sample_{i}_out_none_{scheduler}_r{rank}.nc")
            td.data.to_netcdf(output_path)

        else:
            ds = xr.open_dataset(f"{data_dir}/sample_{i}.nc",
                                 chunks={"lat": size, "lon": size},
                                 )
            td = TOAD(ds)

            # Persist data
            td.data[var] = td.data[var].persist()

            start = time.time()
            td.compute_shifts(var,
                                method=ASDETECT(),
                                overwrite=True,
                                chunk_size=size,
                                dask_compute=True,
                                scheduler=scheduler)
            elapsed = time.time() - start

            output_path = os.path.join(data_dir, f"sample_{i}_out_{str(size)}_{scheduler}_r{rank}.nc")
            td.data.to_netcdf(output_path)

        times.append(elapsed)

        # Clean up to safe memory
        del td, ds

    avg_time = np.mean(times)   # average per chunk size
    results.append(avg_time)
    print(f">> Average time for chunk size {size}: {avg_time:.2f} seconds")

    if size is None:
        mem = 'FULL'#float(np.prod(shape) * np.dtype(np.float32).itemsize) / 1e6
    else:
        mem = float(np.prod([shape[0], size, size]) * np.dtype(np.float32).itemsize) / 1e6
    memory_chunk.append(mem)
    print("Memory Chunk:", memory_chunk)

# Store and plot results
    chunk_labels.append(str(size) if size is not None else "None")
    df = pd.DataFrame({
        "Chunk Size": chunk_labels,
        "Time (s)": results,
        "Memory Size (MB)": memory_chunk
    })
    df.to_pickle(filename_out + ".pkl")

    plt.figure(figsize=(10, 5))
    plt.bar(df["Chunk Size"], df["Time (s)"], color="lightblue")
    plt.xlabel("Chunk Memory Size [MB]")
    plt.ylabel("Time [s]")
    full_mem = float(np.prod(shape) * np.dtype(np.float32).itemsize) / 1e6
    plt.title(f"Chunk Size Performance\nData Shape: {shape}, Total Memory: {full_mem} MB\nSamples: {sample_size}, Scheduler: {scheduler}")
    plt.xticks(df["Chunk Size"].unique(), df["Memory Size (MB)"].unique(), rotation=45)

    # second x-axis label
    ax = plt.gca()
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("\n\nChunking per Spatial Axis [data points]", color="black")
    ax2.tick_params(axis="x", colors="black")
    xticks = ax.get_xticks()
    ax2.set_xticks(xticks)
    ax2.set_xticklabels([str(x) for x in df["Chunk Size"]], rotation=45, color="black")

    for y in ax.get_yticks():
        ax.axhline(y, color='grey', linestyle='--', linewidth=0.5, zorder=0)
    plt.tight_layout()
    plt.savefig(filename_out + ".png")
print("> DONE.")
