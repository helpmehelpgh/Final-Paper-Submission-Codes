from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


def preprocess_pair(wheel_file: str, acc_file: str, k: int = 10) -> pd.DataFrame:
    """
    Preprocess one matched pair of:
    - front-left wheel speed file
    - ACC status file

    Steps:
    1. keep only Time and Message
    2. convert wheel speed from km/h to m/s
    3. binarize ACC label: 1 if status == 6, else 0
    4. align ACC labels to wheel-speed timestamps using zero-order hold
    5. create lagged features v_t, v_t-1, ..., v_t-k
    """
    wheel = pd.read_csv(wheel_file, usecols=["Time", "Message"]).copy()
    acc = pd.read_csv(acc_file, usecols=["Time", "Message"]).copy()

    wheel = wheel.rename(columns={"Message": "speed_kmh"})
    acc = acc.rename(columns={"Message": "acc_status"})

    wheel = wheel.sort_values("Time").reset_index(drop=True)
    acc = acc.sort_values("Time").reset_index(drop=True)

    # duplicated ACC timestamps are safe to remove
    acc = acc.drop_duplicates(subset=["Time"], keep="first").reset_index(drop=True)

    # km/h -> m/s
    wheel["speed_ms"] = wheel["speed_kmh"] / 3.6

    # binary label: ACC enabled iff status == 6
    acc["label"] = (acc["acc_status"] == 6).astype(int)

    # zero-order hold alignment
    merged = pd.merge_asof(
        wheel[["Time", "speed_ms"]].sort_values("Time"),
        acc[["Time", "label"]].sort_values("Time"),
        on="Time",
        direction="backward",
    )

    merged["label"] = merged["label"].fillna(0).astype(int)

    # lag features
    merged["v_t"] = merged["speed_ms"]
    for i in range(1, k + 1):
        merged[f"v_t-{i}"] = merged["speed_ms"].shift(i)

    merged = merged.dropna().reset_index(drop=True)
    return merged


def build_full_acc_dataframe(base_dir: str, k: int = 10) -> pd.DataFrame:
    """
    Build one combined dataframe from all matched experiments.
    Uses only:
    - *_wheel_speed_fl.csv
    - matching *_acc_status.csv
    """
    files = sorted(os.listdir(base_dir))
    wheel_files = [f for f in files if f.endswith("_wheel_speed_fl.csv")]

    all_data = []

    for wf in wheel_files:
        prefix = wf.replace("_wheel_speed_fl.csv", "")
        af = prefix + "_acc_status.csv"

        wheel_path = os.path.join(base_dir, wf)
        acc_path = os.path.join(base_dir, af)

        if not os.path.exists(acc_path):
            continue

        df = preprocess_pair(wheel_path, acc_path, k=k)
        df["experiment"] = prefix
        all_data.append(df)

    if not all_data:
        raise FileNotFoundError("No matched wheel_speed_fl / acc_status pairs were found.")

    full_df = pd.concat(all_data, ignore_index=True)
    return full_df


def prepare_acc_data(
    base_dir: str,
    k: int = 10,
    sample_size: Optional[int] = 300000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Full preprocessing pipeline:
    1. build combined dataframe
    2. create feature matrix X and labels y
    3. optional random sampling
    4. train/test split
    5. feature scaling
    """
    full_df = build_full_acc_dataframe(base_dir=base_dir, k=k)

    feature_cols = ["v_t"] + [f"v_t-{i}" for i in range(1, k + 1)]
    X = full_df[feature_cols].to_numpy(dtype=np.float32)
    y = full_df["label"].to_numpy(dtype=np.float32)

    if sample_size is not None and len(full_df) > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(full_df), size=sample_size, replace=False)
        X = X[idx]
        y = y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return {
        "full_df": full_df,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train.astype(np.float32),
        "y_test": y_test.astype(np.float32),
        "scaler": scaler,
    }


class ACCDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class ACCNet(nn.Module):
    """
    Binary classifier for ACC state.
    Input: [v_t, v_t-1, ..., v_t-10]
    Output: 1 logit
    """

    def __init__(self, in_features: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(inplace=True),

            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ACCTrainer:
    model: nn.Module
    train_loader: DataLoader
    test_loader: DataLoader
    epoch: int = 20
    eta: float = 1e-3
    loss: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    device: Optional[torch.device] = None
    print_every: int = 50

    train_loss_vector: List[float] = field(default_factory=list)
    train_accuracy_vector: List[float] = field(default_factory=list)
    test_loss_vector: List[float] = field(default_factory=list)
    test_accuracy_vector: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)

        if self.loss is None:
            self.loss = nn.BCEWithLogitsLoss()

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.eta)

    def _run_train_epoch(self, ep: int) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (Xb, yb) in enumerate(self.train_loader):
            Xb = Xb.to(self.device)
            yb = yb.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(Xb)
            loss_val = self.loss(logits, yb)
            loss_val.backward()
            self.optimizer.step()

            probs = torch.sigmoid(logits.detach())
            preds = (probs >= 0.5).float()

            running_loss += loss_val.item() * Xb.size(0)
            running_correct += (preds == yb).sum().item()
            running_total += Xb.size(0)

            if (batch_idx + 1) % self.print_every == 0:
                batch_acc = (preds == yb).float().mean().item()
                print(
                    f"Epoch [{ep+1}/{self.epoch}] "
                    f"Batch [{batch_idx+1}/{len(self.train_loader)}] "
                    f"Loss: {loss_val.item():.6f} "
                    f"Acc: {batch_acc:.4f}"
                )

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        return epoch_loss, epoch_acc

    def _run_test_epoch(self) -> Tuple[float, float]:
        self.model.eval()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        with torch.no_grad():
            for Xb, yb in self.test_loader:
                Xb = Xb.to(self.device)
                yb = yb.to(self.device)

                logits = self.model(Xb)
                loss_val = self.loss(logits, yb)

                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                running_loss += loss_val.item() * Xb.size(0)
                running_correct += (preds == yb).sum().item()
                running_total += Xb.size(0)

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        return epoch_loss, epoch_acc

    def train(self) -> Dict[str, List[float]]:
        for ep in range(self.epoch):
            train_loss, train_acc = self._run_train_epoch(ep)
            test_loss, test_acc = self._run_test_epoch()

            self.train_loss_vector.append(train_loss)
            self.train_accuracy_vector.append(train_acc)
            self.test_loss_vector.append(test_loss)
            self.test_accuracy_vector.append(test_acc)

            print(
                f"Epoch [{ep+1}/{self.epoch}] completed | "
                f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f} | "
                f"Test Loss: {test_loss:.6f}, Test Acc: {test_acc:.4f}"
            )

        return {
            "train_loss": self.train_loss_vector,
            "train_accuracy": self.train_accuracy_vector,
            "test_loss": self.test_loss_vector,
            "test_accuracy": self.test_accuracy_vector,
        }

    def save(self, file_name: str = "accnet.onnx") -> Path:
        self.model.eval()

        out_path = Path(file_name).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        dummy = torch.randn(1, 11, device=self.device, dtype=torch.float32)

        torch.onnx.export(
            self.model,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["features"],
            output_names=["logits"],
            dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}},
        )
        return out_path

    def plot_history(self, save_prefix: str = "results/acc") -> None:
        out_dir = Path(save_prefix).expanduser().resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.plot(self.train_loss_vector, label="Train Loss")
        plt.plot(self.test_loss_vector, label="Test Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss History")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(self.train_accuracy_vector, label="Train Accuracy")
        plt.plot(self.test_accuracy_vector, label="Test Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Accuracy History")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_accuracy.png", dpi=300, bbox_inches="tight")
        plt.close()