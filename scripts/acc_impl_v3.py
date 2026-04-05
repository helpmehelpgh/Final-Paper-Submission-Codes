from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mchnpkg.deepl import prepare_acc_data_v3, ACCDatasetV3, ACCNetV3, ACCTrainerV3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACC state classifier v3 training script")
    parser.add_argument("--data_dir", type=str, default="/data/CPE_487-587/ACCDataset")
    parser.add_argument("--output_dir", type=str, default="results/acc_v3")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=300000)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Preparing ACC dataset v3...")
    data = prepare_acc_data_v3(
        base_dir=args.data_dir,
        k=args.k,
        sample_size=args.sample_size,
        test_ratio=args.test_ratio,
        random_state=args.seed,
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    train_experiments = data["train_experiments"]
    test_experiments = data["test_experiments"]
    feature_cols = data["feature_cols"]

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("Feature count:", len(feature_cols))
    print("Train positive ratio:", float(y_train.mean()))
    print("Test positive ratio:", float(y_test.mean()))
    print("Train experiments:", train_experiments)
    print("Test experiments:", test_experiments)

    train_dataset = ACCDatasetV3(X_train, y_train)
    test_dataset = ACCDatasetV3(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ACCNetV3(in_features=X_train.shape[1])

    trainer = ACCTrainerV3(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epoch=args.epochs,
        eta=args.lr,
        print_every=50,
    )

    print("Starting v3 training...")
    history = trainer.train()

    trainer.plot_history(save_prefix=str(output_dir / "acc_v3"))
    onnx_path = trainer.save(
        file_name=str(output_dir / "accnet_v3.onnx"),
        in_features=X_train.shape[1],
    )

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"X_train shape: {X_train.shape}\n")
        f.write(f"X_test shape: {X_test.shape}\n")
        f.write(f"y_train shape: {y_train.shape}\n")
        f.write(f"y_test shape: {y_test.shape}\n")
        f.write(f"Feature count: {len(feature_cols)}\n")
        f.write(f"Train positive ratio: {float(y_train.mean())}\n")
        f.write(f"Test positive ratio: {float(y_test.mean())}\n")
        f.write(f"Train experiments: {train_experiments}\n")
        f.write(f"Test experiments: {test_experiments}\n")
        f.write(f"ONNX path: {onnx_path}\n")
        f.write(f"Final train loss: {history['train_loss'][-1]}\n")
        f.write(f"Final train accuracy: {history['train_accuracy'][-1]}\n")
        f.write(f"Final test loss: {history['test_loss'][-1]}\n")
        f.write(f"Final test accuracy: {history['test_accuracy'][-1]}\n")

    print(f"Saved ONNX model to: {onnx_path}")
    print(f"Saved summary to: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()