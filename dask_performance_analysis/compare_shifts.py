"""
This script looks at the results from the chunk size performance tests and compares the shift values
from different chunk sizes.

# coded by: Lukas Röhrich @ PIK
# June 2025
"""

import numpy as np
import os
import xarray as xr

dirname = "datasets_synth"
samples = list()
chunk_sizes = list()
shifts = list()

nc_files = [
    os.path.join(dirname, f)
    for f in os.listdir(dirname)
    if f.endswith(".nc") and "out" in f
]
nc_files.sort()

print("#### Comparing shift values from different chunk sizes ####")
print(f"\nFound {len(nc_files)} files in {dirname}.")
# collect all samples-indices, chunks-sizes and shift values
for f in nc_files:
    s = f.split("/")[1].split("_")[1]
    samples.append(s)
    cs = f.split("/")[1].split("_")[-1][:-3]
    chunk_sizes.append(cs)

    #ds = xr.open_dataset(f)
    #shifts.append(ds["value_dts"].values)

print(f"Comparing shift-values for {len(set(samples))} samples and {len(set(chunk_sizes))} chunk sizes...")
# iterate over all samples and define benchmark shift values (chunk size "none")
for i in set(samples):
    print(f"\nSample {i}:")
    ix_s = np.where(np.array(samples) == i)[0]
    ix_n = np.where(np.array(chunk_sizes) == "none")[0]
    ix_bench = np.intersect1d(ix_s, ix_n)
    try:
        ds_bench = xr.open_dataset(nc_files[ix_bench[0]])
    except:
        raise ValueError(f"Error reading file {nc_files[ix_bench[0]]}")
    shift_bench = ds_bench["value_dts"].values if (2 > len(ix_bench) > 0) else None

    # iterate over all chunk sizes and compare shift values with benchmark
    for j in set(chunk_sizes):
        if j == "none":
            continue
        ix_c = np.where(np.array(chunk_sizes) == j)[0]
        ix_compare = np.intersect1d(ix_s, ix_c)
        try:
            ds_compare = xr.open_dataset(nc_files[ix_compare[0]])
            shift_compare = ds_compare["value_dts"].values if (2 > len(ix_compare) > 0) else None

            array_equal = np.array_equal(shift_bench, shift_compare) if shift_bench is not None and shift_compare is not None else False

            print(f"> Chunk size {j}: {array_equal}")
            del ds_compare, shift_compare
        except:
            print(f"> Chunk size {j}: Error reading file {nc_files[ix_compare[0]]}")

    del ds_bench, shift_bench
print("Done.")