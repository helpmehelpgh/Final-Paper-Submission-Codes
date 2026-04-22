import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path

from dataset import STEADBinaryDataset
from model import CNN1DClassifier


metadata_path = Path("results/stead_metadata/binary_eq_noise.csv")
chunk1_hdf5 = "/home/ayr0001/STEAD/chunk1.hdf5"
chunk2_hdf5 = "/home/ayr0001/STEAD/chunk2.hdf5"

batch_size = 64
epochs = 5
lr = 1e-3

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
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

model = CNN1DClassifier(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

best_val_acc = 0.0
Path("results/models").mkdir(parents=True, exist_ok=True)

for epoch in range(1, epochs + 1):
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

    print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "results/models/best_binary_cnn.pt")

print("Best validation accuracy:", best_val_acc)
print("Saved model: results/models/best_binary_cnn.pt")
