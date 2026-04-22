import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path

from model import CNN1DClassifier


CSV_PATH = "/home/ayr0001/STEAD/chunk2.csv"
HDF5_PATH = "/home/ayr0001/STEAD/chunk2.hdf5"
MODEL_PATH = "results/models/best_distance_balanced_cnn.pt"


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
        trace_name = row["trace_name"]

        x = self.h5["data"][trace_name][:]
        x = torch.tensor(x, dtype=torch.float32).T

        # Normalize each component
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True) + 1e-8
        x = (x - mean) / std

        y = torch.tensor(int(row["dist_class"]), dtype=torch.long)

        return x, y


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["trace_category"] == "earthquake_local"].copy()

    df["source_distance_km"] = pd.to_numeric(
        df["source_distance_km"], errors="coerce"
    )

    df = df.dropna(subset=["source_distance_km", "trace_name"]).copy()
    df["dist_class"] = df["source_distance_km"].apply(distance_class)

    print("Original distance class counts:")
    print(df["dist_class"].value_counts().sort_index())

    # Balance classes using the smallest class count
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
    _, val_ds = random_split(
        dataset, [train_size, val_size], generator=generator
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    model = CNN1DClassifier(num_classes=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1).cpu().numpy()

            all_pred.extend(pred)
            all_true.extend(y.numpy())

    target_names = [
        "d<50",
        "50<=d<100",
        "100<=d<200",
        "d>=200"
    ]

    print("\nClassification Report:")
    print(
        classification_report(
            all_true,
            all_pred,
            target_names=target_names,
            zero_division=0
        )
    )

    cm = confusion_matrix(all_true, all_pred)

    print("\nConfusion Matrix:")
    print(cm)

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix: Balanced Distance Classes")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1, 2, 3], target_names, rotation=20)
    plt.yticks([0, 1, 2, 3], target_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        "results/figures/distance_balanced_confusion_matrix.png",
        dpi=300
    )
    plt.close()

    print("\nSaved figure:")
    print("results/figures/distance_balanced_confusion_matrix.png")


if __name__ == "__main__":
    main()
