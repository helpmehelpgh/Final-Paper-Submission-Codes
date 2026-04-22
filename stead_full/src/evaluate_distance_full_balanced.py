import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

sys.path.append("stead_full/src")

from config_full import HDF5_PATH, FIGURES_DIR
from dataset_full import STEADFullDataset
from model import CNN1DClassifier


METADATA_PATH = "stead_full/results/metadata/distance_full_balanced.csv"
MODEL_PATH = "stead_full/results/models/best_distance_full_balanced_cnn.pt"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(METADATA_PATH, low_memory=False)

    print("Dataset shape:", df.shape)
    print("Distance class counts:")
    print(df["dist_class"].value_counts().sort_index())

    dataset = STEADFullDataset(
        metadata_df=df,
        hdf5_path=HDF5_PATH,
        label_column="dist_class",
        task_type="multiclass"
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(42)
    _, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

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

    target_names = ["d<50", "50<=d<100", "100<=d<200", "d>=200"]

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

    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix: Full Balanced Distance Classes")
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
        f"{FIGURES_DIR}/distance_full_balanced_confusion_matrix.png",
        dpi=300
    )
    plt.close()

    print("\nSaved figure:")
    print(f"{FIGURES_DIR}/distance_full_balanced_confusion_matrix.png")


if __name__ == "__main__":
    main()
