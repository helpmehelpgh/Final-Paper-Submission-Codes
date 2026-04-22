import pandas as pd
import h5py

csv_path = "/home/ayr0001/STEAD/chunk2.csv"
hdf5_path = "/home/ayr0001/STEAD/chunk2.hdf5"

df = pd.read_csv(csv_path)
print("CSV shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nTrace category counts:")
print(df["trace_category"].value_counts(dropna=False))

with h5py.File(hdf5_path, "r") as f:
    print("\nTop-level keys:", list(f.keys()))
    first_trace = df["trace_name"].iloc[0]
    print("First trace name:", first_trace)

    x = f["data"][first_trace][:]
    print("Waveform shape:", x.shape)

    print("\nAttributes of first trace:")
    for k, v in f["data"][first_trace].attrs.items():
        print(k, ":", v)

