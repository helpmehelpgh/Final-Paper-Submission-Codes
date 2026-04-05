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

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


def _read_signal_csv(file_path: str, value_name: str) -> pd.DataFrame:
    """
    Read a decoded CAN csv and keep only Time + Message.
    Rename Message to the requested value_name.
    """
    df = pd.read_csv(file_path, usecols=["Time", "Message"]).copy()
    df = df.rename(columns={"Message": value_name})
    df = df.sort_values("Time").reset_index(drop=True)
    df = df.drop_duplicates(subset=["Time"], keep="first").reset_index(drop=True)
    return df


def _merge_signal_zoh(base_df: pd.DataFrame, signal_df: pd.DataFrame, value_name: str) -> pd.DataFrame:
    """
    Merge one signal onto the base timeline using zero-order hold.
    """
    merged = pd.merge_asof(
        base_df.sort_values("Time"),
        signal_df[["Time", value_name]].sort_values("Time"),
        on="Time",
        direction="backward",
    )
    return merged


def preprocess_experiment_v2(prefix: str, base_dir: str, k: int = 10) -> pd.DataFrame:
    """
    Build one multi-signal dataframe for a single experiment.

    Signals used:
    - wheel_speed_fl  -> converted from km/h to m/s
    - relative_vel
    - lead_distance
    - accely
    - acc_status      -> binary target: 1 if status == 6 else 0

    Features created:
    - current + lagged values for all main signals
    - first differences for key signals
    - rolling statistics
    """
    wheel_file = os.path.join(base_dir, f"{prefix}_wheel_speed_fl.csv")
    relvel_file = os.path.join(base_dir, f"{prefix}_relative_vel.csv")
    lead_file = os.path.join(base_dir, f"{prefix}_lead_distance.csv")
    accely_file = os.path.join(base_dir, f"{prefix}_accely.csv")
    acc_status_file = os.path.join(base_dir, f"{prefix}_acc_status.csv")

    required = [wheel_file, relvel_file, lead_file, accely_file, acc_status_file]
    for fp in required:
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing required file: {fp}")

    wheel = _read_signal_csv(wheel_file, "speed_kmh")
    relvel = _read_signal_csv(relvel_file, "relative_vel")
    lead = _read_signal_csv(lead_file, "lead_distance")
    accely = _read_signal_csv(accely_file, "accely")
    acc_status = _read_signal_csv(acc_status_file, "acc_status")

    # Convert wheel speed from km/h to m/s
    wheel["speed_ms"] = wheel["speed_kmh"] / 3.6

    # Binary ACC label
    acc_status["label"] = (acc_status["acc_status"] == 6).astype(int)

    # Use wheel-speed timestamps as the base timeline
    merged = wheel[["Time", "speed_ms"]].copy()

    merged = _merge_signal_zoh(merged, relvel, "relative_vel")
    merged = _merge_signal_zoh(merged, lead, "lead_distance")
    merged = _merge_signal_zoh(merged, accely, "accely")
    merged = _merge_signal_zoh(merged, acc_status[["Time", "label"]], "label")

    # Fill early missing values before first observation
    merged["relative_vel"] = merged["relative_vel"].bfill().fillna(0.0)
    merged["lead_distance"] = merged["lead_distance"].bfill().fillna(0.0)
    merged["accely"] = merged["accely"].bfill().fillna(0.0)
    merged["label"] = merged["label"].fillna(0).astype(int)

    # Current-value aliases
    merged["speed_t"] = merged["speed_ms"]
    merged["relvel_t"] = merged["relative_vel"]
    merged["lead_t"] = merged["lead_distance"]
    merged["accely_t"] = merged["accely"]

    # Lag features
    for i in range(1, k + 1):
        merged[f"speed_t-{i}"] = merged["speed_ms"].shift(i)
        merged[f"relvel_t-{i}"] = merged["relative_vel"].shift(i)
        merged[f"lead_t-{i}"] = merged["lead_distance"].shift(i)
        merged[f"accely_t-{i}"] = merged["accely"].shift(i)

    # First-difference features
    merged["dspeed_t"] = merged["speed_ms"].diff()
    merged["drelvel_t"] = merged["relative_vel"].diff()
    merged["dlead_t"] = merged["lead_distance"].diff()
    merged["daccely_t"] = merged["accely"].diff()

    # Rolling features
    merged["speed_roll_mean_5"] = merged["speed_ms"].rolling(window=5).mean()
    merged["speed_roll_std_5"] = merged["speed_ms"].rolling(window=5).std()
    merged["relvel_roll_mean_5"] = merged["relative_vel"].rolling(window=5).mean()
    merged["lead_roll_mean_5"] = merged["lead_distance"].rolling(window=5).mean()

    merged["experiment"] = prefix

    merged = merged.dropna().reset_index(drop=True)
    return merged


def build_full_acc_dataframe_v2(base_dir: str, k: int = 10) -> pd.DataFrame:
    """
    Build one combined dataframe using multi-signal preprocessing.
    """
    files = sorted(os.listdir(base_dir))
    wheel_files = [f for f in files if f.endswith("_wheel_speed_fl.csv")]

    prefixes = [f.replace("_wheel_speed_fl.csv", "") for f in wheel_files]
    all_data = []

    for prefix in prefixes:
        try:
            df = preprocess_experiment_v2(prefix=prefix, base_dir=base_dir, k=k)
            all_data.append(df)
        except FileNotFoundError:
            continue

    if not all_data:
        raise FileNotFoundError("No valid experiment groups were found for ACCDataset v2.")

    full_df = pd.concat(all_data, ignore_index=True)
    return full_df


def split_by_experiment(
    full_df: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Split by experiment names, not random rows.
    """
    experiments = sorted(full_df["experiment"].unique().tolist())

    rng = np.random.default_rng(random_state)
    shuffled = experiments.copy()
    rng.shuffle(shuffled)

    n_test = max(1, int(round(len(shuffled) * test_ratio)))
    test_experiments = shuffled[:n_test]
    train_experiments = shuffled[n_test:]

    train_df = full_df[full_df["experiment"].isin(train_experiments)].reset_index(drop=True)
    test_df = full_df[full_df["experiment"].isin(test_experiments)].reset_index(drop=True)

    return train_df, test_df, train_experiments, test_experiments


def _get_feature_columns_v2(k: int = 10) -> List[str]:
    feature_cols = [
        "speed_t", "relvel_t", "lead_t", "accely_t",
        "dspeed_t", "drelvel_t", "dlead_t", "daccely_t",
        "speed_roll_mean_5", "speed_roll_std_5",
        "relvel_roll_mean_5", "lead_roll_mean_5",
    ]

    for i in range(1, k + 1):
        feature_cols.extend([
            f"speed_t-{i}",
            f"relvel_t-{i}",
            f"lead_t-{i}",
            f"accely_t-{i}",
        ])

    return feature_cols


def prepare_acc_data_v2(
    base_dir: str,
    k: int = 10,
    sample_size: Optional[int] = None,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """
    Full v2 pipeline:
    - multi-signal preprocessing
    - split by experiment
    - optional sampling inside train and test
    - scaling
    """
    full_df = build_full_acc_dataframe_v2(base_dir=base_dir, k=k)

    train_df, test_df, train_experiments, test_experiments = split_by_experiment(
        full_df=full_df,
        test_ratio=test_ratio,
        random_state=random_state,
    )

    if sample_size is not None:
        rng = np.random.default_rng(random_state)

        if len(train_df) > sample_size:
            train_idx = rng.choice(len(train_df), size=sample_size, replace=False)
            train_df = train_df.iloc[train_idx].reset_index(drop=True)

        test_sample_size = max(1, int(sample_size * test_ratio / max(1e-8, 1 - test_ratio)))
        if len(test_df) > test_sample_size:
            test_idx = rng.choice(len(test_df), size=test_sample_size, replace=False)
            test_df = test_df.iloc[test_idx].reset_index(drop=True)

    feature_cols = _get_feature_columns_v2(k=k)

    X_train = train_df[feature_cols].to_numpy(dtype=np.float32)
    X_test = test_df[feature_cols].to_numpy(dtype=np.float32)
    y_train = train_df["label"].to_numpy(dtype=np.float32)
    y_test = test_df["label"].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return {
        "full_df": full_df,
        "train_df": train_df,
        "test_df": test_df,
        "train_experiments": train_experiments,
        "test_experiments": test_experiments,
        "feature_cols": feature_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
    }


class ACCDatasetV2(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class ACCNetV2(nn.Module):
    """
    Larger MLP for richer multi-signal feature input.
    """

    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class ACCTrainerV2:
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

    def save(self, file_name: str = "accnet_v2.onnx", in_features: int = 52) -> Path:
        self.model.eval()

        out_path = Path(file_name).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        dummy = torch.randn(1, in_features, device=self.device, dtype=torch.float32)

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

    def plot_history(self, save_prefix: str = "results/acc_v2") -> None:
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