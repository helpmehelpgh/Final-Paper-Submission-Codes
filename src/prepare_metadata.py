import pandas as pd
from pathlib import Path

chunk1_csv = Path("/home/ayr0001/STEAD/chunk1.csv")
chunk2_csv = Path("/home/ayr0001/STEAD/chunk2.csv")
out_dir = Path("/home/ayr0001/DLHW2Par8/results/stead_metadata")
out_dir.mkdir(parents=True, exist_ok=True)

# Read
df_noise = pd.read_csv(chunk1_csv, low_memory=False)
df_eq = pd.read_csv(chunk2_csv, low_memory=False)

# Keep only needed columns
keep_cols = [
    "trace_name",
    "trace_category",
    "source_magnitude",
    "source_distance_km",
    "p_arrival_sample",
    "s_arrival_sample",
    "snr_db",
]

df_noise = df_noise[keep_cols].copy()
df_eq = df_eq[keep_cols].copy()

# Binary task: balanced subset
n = min(len(df_noise), len(df_eq), 50000)
df_noise_bin = df_noise.sample(n=n, random_state=42).copy()
df_eq_bin = df_eq.sample(n=n, random_state=42).copy()
df_binary = pd.concat([df_noise_bin, df_eq_bin], ignore_index=True)
df_binary = df_binary.sample(frac=1, random_state=42).reset_index(drop=True)

# Magnitude task: earthquakes only
df_mag = df_eq.dropna(subset=["source_magnitude"]).copy()

# Distance task: earthquakes only
df_dist = df_eq.dropna(subset=["source_distance_km"]).copy()

# Save
df_binary.to_csv(out_dir / "binary_eq_noise.csv", index=False)
df_mag.to_csv(out_dir / "magnitude_only.csv", index=False)
df_dist.to_csv(out_dir / "distance_only.csv", index=False)

print("Saved:")
print(out_dir / "binary_eq_noise.csv", df_binary.shape)
print(out_dir / "magnitude_only.csv", df_mag.shape)
print(out_dir / "distance_only.csv", df_dist.shape)
