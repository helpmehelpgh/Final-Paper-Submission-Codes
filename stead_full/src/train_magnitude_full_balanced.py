import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

sys.path.append("stead_full/src")

from config_full import HDF5_PATH, MODELS_DIR
from dataset_full import STEADFullDataset
from model import CNN1DClassifier


METADATA_PATH = "stead_full/results/metadata/magnitude_full_balanced.csv"

BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(METADATA_PATH, low_memory=False)

    print("Dataset shape:", df.shape)
    print("Magnitude class counts:")
    print(df["mag_class"].value_counts().sort_index())

    dataset = STEADFullDataset(
        metadata_df=df,
        hdf5_path=HDF5_PATH,
        label_column="mag_class",
        task_type="multiclass"
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = CNN1DClassifier(num_classes=4).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    Path(MODELS_DIR).mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0

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
            torch.save(
                model.state_dict(),
                f"{MODELS_DIR}/best_magnitude_full_balanced_cnn.pt"
            )

    print("Best validation accuracy:", best_val_acc)
    print(f"Saved model: {MODELS_DIR}/best_magnitude_full_balanced_cnn.pt")


if __name__ == "__main__":
    main()
