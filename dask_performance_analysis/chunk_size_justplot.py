import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

filename_in = "20250605_chunk_size_performance_real"
filename_out = filename_in
shape = (86, 324, 324)
sample_size = 5
col2 = 'black'

df = pd.read_pickle(filename_in + ".pkl")

plt.figure(figsize=(10, 5))
plt.bar(df["Chunk Size"], df["Time (s)"], color="lightgreen")
plt.xlabel("Chunk Memory Size [MB]")
plt.ylabel("Time [s]")
full_mem = float(np.prod(shape) * np.dtype(np.float32).itemsize) / 1e6
plt.title(f"Chunk Size Performance - Real Data\nData Shape: {shape}, Total Memory: {full_mem} MB\nSamples: {sample_size}")
plt.xticks(df["Chunk Size"].unique(), df["Memory Size (MB)"].unique(), rotation=45)

# second x-axis label
ax = plt.gca()
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
ax2.set_xlabel("\n\nChunking per Spatial Axis [data points]", color=col2)
ax2.tick_params(axis='x', colors=col2)
xticks = ax.get_xticks()
ax2.set_xticks(xticks)
ax2.set_xticklabels([str(x) for x in df["Chunk Size"]], rotation=45, color=col2)

for y in ax.get_yticks():
    ax.axhline(y, color='grey', linestyle='--', linewidth=0.5, zorder=0)
plt.tight_layout()
plt.savefig(filename_out + ".png")