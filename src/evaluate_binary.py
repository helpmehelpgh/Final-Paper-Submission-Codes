import pandas as pd
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from dataset import STEADBinaryDataset
from model import CNN1DClassifier


metadata_path = Path("results/stead_metadata/binary_eq_noise.csv")
chunk1_hdf5 = "/home/ayr0001/STEAD/chunk1.hdf5"
chunk2_hdf5 = "/home/ayr0001/STEAD/chunk2.hdf5"
model_path = "results/models/best_binary_cnn.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

df = pd.read_csv(metadata_path)

dataset = STEADBinaryDataset(
    metadata_df=df,
    chunk1_hdf5=chunk1_hdf5,
    chunk2_hdf5=chunk2_hdf5
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

generator = torch.Generator().manual_seed(42)
_, val_ds = random_split(dataset, [train_size, val_size], generator=generator)

val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=2)

model = CNN1DClassifier(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
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

print("\nClassification Report:")
print(classification_report(all_true, all_pred, target_names=["Noise", "Earthquake"]))

cm = confusion_matrix(all_true, all_pred)
print("\nConfusion Matrix:")
print(cm)

Path("results/figures").mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix: Earthquake vs Noise")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks([0, 1], ["Noise", "Earthquake"])
plt.yticks([0, 1], ["Noise", "Earthquake"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.colorbar()
plt.tight_layout()
plt.savefig("results/figures/binary_confusion_matrix.png", dpi=300)
plt.close()

print("\nSaved figure:")
print("results/figures/binary_confusion_matrix.png")
