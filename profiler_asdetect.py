"""
March 2025 - Lukas Röhrich @ PIK, Germany

This script is used to profile the ASDETECT method for shifts detection in TOAD. The test data is thk-data from antarctica and has a
quite high resolution. The data is coarsened before the shifts are detected. A coarsening of (5,5,3) leads to a single runtime of ~60s. (10,10,3) has a single run time ~10s.
Both times on my personal laptop. The profiling is done using the cProfile module.

Create a profile using this command:

>> python -m cProfile -o asdetect.prof profiler_asdetect.py

View profile using snakeviz:

>> snakeviz program.prof
"""

import toad
from toad.shifts_detection.methods import ASDETECT

fp = "./tutorials/test_data/garbe_2020_antarctica.nc"
td = toad.TOAD(fp)

# Setup
lat_coarsen = 5
lon_coarsen = 5
time_coarsen = 3
td.data = td.data.coarsen(
    x=lat_coarsen,
    y=lon_coarsen,
    time=time_coarsen,
    boundary="trim",
).mean()

td.compute_shifts("thk", ASDETECT(), overwrite=True)

shifts = td.get_shifts("thk")
mean = shifts.mean().values
std = shifts.std().values