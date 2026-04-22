import pandas as pd
import h5py
from config_full import CSV_PATH, HDF5_PATH

df = pd.read_csv(CSV_PATH, low_memory=False)

print("CSV shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nTrace category counts:")
print(df["trace_category"].value_counts(dropna=False))

print("\nMagnitude summary:")
print(pd.to_numeric(df["source_magnitude"], errors="coerce").describe())

print("\nDistance summary:")
print(pd.to_numeric(df["source_distance_km"], errors="coerce").describe())

with h5py.File(HDF5_PATH, "r") as f:
    print("\nHDF5 keys:", list(f.keys()))
    print("Number of traces:", len(f["data"]))

    first_trace = df["trace_name"].iloc[0]
    print("First trace:", first_trace)

    x = f["data"][first_trace][:]
    print("Waveform shape:", x.shape)
