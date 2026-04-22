import h5py
import torch
from torch.utils.data import Dataset


class STEADBinaryDataset(Dataset):
    def __init__(self, metadata_df, chunk1_hdf5, chunk2_hdf5):
        self.df = metadata_df.reset_index(drop=True)
        self.chunk1_hdf5 = chunk1_hdf5
        self.chunk2_hdf5 = chunk2_hdf5
        self.h5_noise = None
        self.h5_eq = None

    def __len__(self):
        return len(self.df)

    def _open_hdf5(self):
        if self.h5_noise is None:
            self.h5_noise = h5py.File(self.chunk1_hdf5, "r")
        if self.h5_eq is None:
            self.h5_eq = h5py.File(self.chunk2_hdf5, "r")

    def __getitem__(self, idx):
        self._open_hdf5()

        row = self.df.iloc[idx]
        trace_name = row["trace_name"]
        category = row["trace_category"]

        if category == "noise":
            x = self.h5_noise["data"][trace_name][:]
            y = 0
        else:
            x = self.h5_eq["data"][trace_name][:]
            y = 1

        # shape: (6000, 3) -> (3, 6000)
        x = torch.tensor(x, dtype=torch.float32).T

        # normalize each component
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        x = (x - mean) / std

        y = torch.tensor(y, dtype=torch.long)
        return x, y
