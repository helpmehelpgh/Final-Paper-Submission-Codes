import h5py
import torch
from torch.utils.data import Dataset


class STEADFullDataset(Dataset):
    def __init__(self, metadata_df, hdf5_path, label_column, task_type="binary"):
        self.df = metadata_df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.label_column = label_column
        self.task_type = task_type
        self.h5 = None

    def __len__(self):
        return len(self.df)

    def _open_hdf5(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.hdf5_path, "r")

    def __getitem__(self, idx):
        self._open_hdf5()

        row = self.df.iloc[idx]
        trace_name = row["trace_name"]

        x = self.h5["data"][trace_name][:]
        x = torch.tensor(x, dtype=torch.float32).T  # (6000, 3) -> (3, 6000)

        # channel-wise normalization
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        x = (x - mean) / std

        if self.task_type == "binary":
            y = 0 if row[self.label_column] == "noise" else 1
        else:
            y = int(row[self.label_column])

        y = torch.tensor(y, dtype=torch.long)
        return x, y
