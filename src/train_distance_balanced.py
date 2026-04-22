import h5py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from model import CNN1DClassifier


CSV_PATH = "/home/ayr0001/STEAD/chunk2.csv"
HDF5_PATH = "/home/ayr0001/STEAD/chunk2.hdf5"

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3


def distance_class(d):
    if d < 50:
        return 0
    elif d < 100:
        return 1
    elif d < 200:
        return 2
    else:
        return 3


class STEADDistanceDataset(Dataset):
    def __init__(self, df, hdf5_path):
        self.df = df.reset_index(drop=True)
        self.hdf5_path = hdf5_path
        self.h5 = None

    def __len__(self):
        return len(self.df)

    def _open_hdf5(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.hdf5_path, "r")

    def __getitem__(self, idx):
        self._open_hdf5()
        row = self.df.iloc[idx]

        x = self.h5["data"][row["trace_name"]][:]
        x = torch.tensor(x, dtype=torch.float32).T
        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-8)

        y = torch.tensor(int(row["dist_class"]), dtype=torch.long)
        return x, y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["trace_category"] == "earthquake_local"].copy()
    df["source_distance_km"] = pd.to_numeric(df["source_distance_km"], errors="coerce")
    df = df.dropna(subset=["source_distance_km", "trace_name"]).copy()

    df["dist_class"] = df["source_distance_km"].apply(distance_class)

    print("Original distance class counts:")
    print(df["dist_class"].value_counts().sort_index())

    min_count = df["dist_class"].value_counts().min()

    balanced_parts = []
    for c in sorted(df["dist_class"].unique()):
        part = df[df["dist_class"] == c].sample(n=min_count, random_state=42)
        balanced_parts.append(part)

    df_bal = pd.concat(balanced_parts, ignore_index=True)
    df_bal = df_bal.sample(frac=1, random_state=42).reset_index(drop=True)

    print("\nBalanced distance class counts:")
    print(df_bal["dist_class"].value_counts().sort_index())

    dataset = STEADDistanceDataset(df_bal, HDF5_PATH)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = CNN1DClassifier(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    Path("results/models").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_loss = total_loss / total
        train_acc = correct / total

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)

                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "results/models/best_distance_balanced_cnn.pt")

    print("Best validation accuracy:", best_val_acc)
    print("Saved model: results/models/best_distance_balanced_cnn.pt")


if __name__ == "__main__":
    main()
