import time
import numpy as np
import xarray as xr
import pandas as pd
from toad import TOAD

# load example dataset
data = xr.open_dataset("tutorials/test_data/garbe_2020_antarctica.nc")
data = data.drop_vars(["lat", "lon"])  # drop lat/lon to use the native coordinates

# lower the resolution to speed up computation
data = data.coarsen(x=10, y=10, time=2, boundary="trim").reduce(np.mean)
td = TOAD(data)

from toad.shifts import ASDETECT
from toad.shifts.methods.asswdetect import ASSWDETECT

# benchmarking parameters
n_runs = 3  # number of timed runs per method

results = []

def benchmark(method_callable, ov, label, warmup=True):
    # optional warmup run (to exclude numba compilation overhead)
    if warmup:
        method_callable()
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        method_callable()
        end = time.perf_counter()
        times.append(end - start)
    mean_time = np.mean(times)
    results.append({
        "method": label,
        "overlap": ov,
        #"mean_time": mean_time,
        "all_times": times
    })
    # Print results
    print(f"{label}: mean {mean_time:.4f} s over {n_runs} runs and {ov} overlap ")

# --- Method 1: ASDETECT ---
def run_asdetect():
    td.compute_shifts("thk", method=ASDETECT(), overwrite=True)

benchmark(run_asdetect, ov="NaN", label="ASDETECT")

# --- Method 2: ASSWDETECT with different overlaps ---
overlaps = [0.0, 0.1, 0.25, 0.5, 0.75]

for ov in overlaps:
    def run_assw():
        td.compute_shifts("thk", method=ASSWDETECT(overlap=ov), overwrite=True)
    benchmark(run_assw, ov=ov, label=f"ASSWDETECT")

# Save results to CSV
rows = []
for r in results:
    method = r["method"]
    #mean_time = r["mean_time"]
    overlap = r["overlap"]
    for i, t in enumerate(r["all_times"], start=1):
        rows.append({
            "method": method,
            "run": i,
            "overlap": overlap,
            "time": t,
            #"mean_time": mean_time
        })

df = pd.DataFrame(rows)
df.to_csv("benchmark_results.csv", index=False)
print("Results saved to benchmark_results.csv")