import pandas as pd
from pathlib import Path
from config_full import CSV_PATH, METADATA_DIR


out_dir = Path(METADATA_DIR)
out_dir.mkdir(parents=True, exist_ok=True)


def magnitude_class(m):
    if m < 2.0:
        return 0
    elif m < 3.0:
        return 1
    elif m < 4.0:
        return 2
    else:
        return 3


def distance_class(d):
    if d < 50:
        return 0
    elif d < 100:
        return 1
    elif d < 200:
        return 2
    else:
        return 3


print("Reading full STEAD CSV...")
df = pd.read_csv(CSV_PATH, low_memory=False)

keep_cols = [
    "trace_name",
    "trace_category",
    "source_magnitude",
    "source_distance_km",
    "p_arrival_sample",
    "s_arrival_sample",
    "snr_db",
]

df = df[keep_cols].copy()

print("\nTrace category counts:")
print(df["trace_category"].value_counts(dropna=False))

# -----------------------------
# 1. Binary balanced metadata
# -----------------------------
df_noise = df[df["trace_category"] == "noise"].copy()
df_eq = df[df["trace_category"] == "earthquake_local"].copy()

n_binary = min(len(df_noise), len(df_eq))

df_noise_bin = df_noise.sample(n=n_binary, random_state=42)
df_eq_bin = df_eq.sample(n=n_binary, random_state=42)

df_binary = pd.concat([df_noise_bin, df_eq_bin], ignore_index=True)
df_binary = df_binary.sample(frac=1, random_state=42).reset_index(drop=True)

df_binary.to_csv(out_dir / "binary_full_balanced.csv", index=False)

print("\nSaved binary full balanced:")
print(df_binary.shape)
print(df_binary["trace_category"].value_counts())

# -----------------------------
# 2. Magnitude balanced metadata
# -----------------------------
df_mag = df_eq.dropna(subset=["source_magnitude", "trace_name"]).copy()
df_mag["source_magnitude"] = pd.to_numeric(df_mag["source_magnitude"], errors="coerce")
df_mag = df_mag.dropna(subset=["source_magnitude"]).copy()
df_mag["mag_class"] = df_mag["source_magnitude"].apply(magnitude_class)

print("\nOriginal magnitude class counts:")
print(df_mag["mag_class"].value_counts().sort_index())

min_mag = df_mag["mag_class"].value_counts().min()

mag_parts = []
for c in sorted(df_mag["mag_class"].unique()):
    part = df_mag[df_mag["mag_class"] == c].sample(n=min_mag, random_state=42)
    mag_parts.append(part)

df_mag_bal = pd.concat(mag_parts, ignore_index=True)
df_mag_bal = df_mag_bal.sample(frac=1, random_state=42).reset_index(drop=True)

df_mag_bal.to_csv(out_dir / "magnitude_full_balanced.csv", index=False)

print("\nSaved magnitude full balanced:")
print(df_mag_bal.shape)
print(df_mag_bal["mag_class"].value_counts().sort_index())

# -----------------------------
# 3. Distance balanced metadata
# -----------------------------
df_dist = df_eq.dropna(subset=["source_distance_km", "trace_name"]).copy()
df_dist["source_distance_km"] = pd.to_numeric(df_dist["source_distance_km"], errors="coerce")
df_dist = df_dist.dropna(subset=["source_distance_km"]).copy()
df_dist["dist_class"] = df_dist["source_distance_km"].apply(distance_class)

print("\nOriginal distance class counts:")
print(df_dist["dist_class"].value_counts().sort_index())

min_dist = df_dist["dist_class"].value_counts().min()

dist_parts = []
for c in sorted(df_dist["dist_class"].unique()):
    part = df_dist[df_dist["dist_class"] == c].sample(n=min_dist, random_state=42)
    dist_parts.append(part)

df_dist_bal = pd.concat(dist_parts, ignore_index=True)
df_dist_bal = df_dist_bal.sample(frac=1, random_state=42).reset_index(drop=True)

df_dist_bal.to_csv(out_dir / "distance_full_balanced.csv", index=False)

print("\nSaved distance full balanced:")
print(df_dist_bal.shape)
print(df_dist_bal["dist_class"].value_counts().sort_index())

print("\nAll metadata files saved in:")
print(out_dir)
