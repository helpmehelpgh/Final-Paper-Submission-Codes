from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mchnpkg.deepl import prepare_acc_data, ACCDataset, ACCNet, ACCTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ACC state classifier training script")
    parser.add_argument("--data_dir", type=str, default="/data/CPE_487-587/ACCDataset")
    parser.add_argument("--output_dir", type=str, default="results/acc_final")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sample_size", type=int, default=300000)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
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

    print("Preparing ACC dataset...")
    data = prepare_acc_data(
        base_dir=args.data_dir,
        k=args.k,
        sample_size=args.sample_size,
        test_size=args.test_size,
        random_state=args.seed,
    )

    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("Train positive ratio:", float(y_train.mean()))
    print("Test positive ratio:", float(y_test.mean()))

    train_dataset = ACCDataset(X_train, y_train)
    test_dataset = ACCDataset(X_test, y_test)

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

    model = ACCNet(in_features=X_train.shape[1])

    trainer = ACCTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epoch=args.epochs,
        eta=args.lr,
        print_every=50,
    )

    print("Starting training...")
    history = trainer.train()

    trainer.plot_history(save_prefix=str(output_dir / "acc"))
    onnx_path = trainer.save(file_name=str(output_dir / "accnet.onnx"))

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"X_train shape: {X_train.shape}\n")
        f.write(f"X_test shape: {X_test.shape}\n")
        f.write(f"y_train shape: {y_train.shape}\n")
        f.write(f"y_test shape: {y_test.shape}\n")
        f.write(f"Train positive ratio: {float(y_train.mean())}\n")
        f.write(f"Test positive ratio: {float(y_test.mean())}\n")
        f.write(f"Feature count: {X_train.shape[1]}\n")
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