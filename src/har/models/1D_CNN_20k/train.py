"""Train: python src/har/models/1D_CNN_20k/train.py --epochs 50.

Dependencies: numpy, torch. Validation subjects come only from the training
split. The official test split is evaluated once, after model selection.
"""

import argparse
import copy
import sys
from datetime import datetime, timezone
from pathlib import Path

# Make the root-level loader available when running this nested script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset

from data_loader import CHANNELS, DEFAULT_DIRECTORY, DataLoader
if __package__:
    from .model import ActivityCNN
    from .report import save_report
else:
    from model import ActivityCNN
    from report import save_report


CLASS_NAMES = ["walking", "upstairs", "downstairs", "sitting", "standing", "laying"]


def make_batches(X, y, mean, std, batch_size, shuffle=False):
    # Existing loader: (N, time, channels), labels 1–6.
    # PyTorch Conv1d: (N, channels, time), labels 0–5.
    inputs = torch.from_numpy(((X - mean) / std).transpose(0, 2, 1).copy())
    targets = torch.from_numpy(y - 1)
    return TorchDataLoader(
        TensorDataset(inputs, targets), batch_size=batch_size, shuffle=shuffle
    )


def run_epoch(model, batches, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss = 0.0
    confusion = torch.zeros(6, 6, dtype=torch.int64)
    with torch.set_grad_enabled(optimizer is not None):
        for inputs, targets in batches:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            indices = (targets * 6 + logits.argmax(dim=1)).detach().cpu()
            confusion += torch.bincount(indices, minlength=36).reshape(6, 6)
    accuracy = confusion.diag().sum().item() / confusion.sum().item()
    return total_loss / len(batches.dataset), accuracy, confusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_ROOT / "outputs/1D_CNN_20k/activity_cnn.pt",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=None,
        help="Report folder (default: reports/1D_CNN_20k/<UTC run timestamp>)",
    )
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.patience) < 1 or args.lr <= 0:
        parser.error("epochs, batch-size, patience, and lr must be positive")
    started_at = datetime.now(timezone.utc)
    report_dir = args.report_dir or (
        PROJECT_ROOT / "reports/1D_CNN_20k" / started_at.strftime("%Y%m%dT%H%M%S%fZ")
    )

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test, y_train, y_test = DataLoader(args.data_dir).load_data()

    subjects = np.loadtxt(args.data_dir / "train/subject_train.txt", dtype=int)
    if len(subjects) != len(y_train):
        raise ValueError("Subject IDs must align with the training windows")
    unique_subjects = rng.permutation(np.unique(subjects))
    val_subjects = unique_subjects[:int(np.ceil(0.2 * len(unique_subjects)))]
    is_val = np.isin(subjects, val_subjects)
    # Statistics use training subjects only, preserving within-window means.
    mean = X_train[~is_val].mean(axis=(0, 1), keepdims=True)
    std = X_train[~is_val].std(axis=(0, 1), keepdims=True).clip(min=1e-6)
    train_batches = make_batches(
        X_train[~is_val], y_train[~is_val], mean, std, args.batch_size, shuffle=True
    )
    val_batches = make_batches(
        X_train[is_val], y_train[is_val], mean, std, args.batch_size
    )

    model = ActivityCNN(input_channels=len(CHANNELS)).to(device)
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count < 20_000, f"Model exceeds parameter budget: {count}"
    print(f"Device: {device}; trainable parameters: {count:,}")
    print(f"Validation subjects: {sorted(val_subjects.tolist())}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_loss, stale_epochs = float("inf"), 0
    best_state = None
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, _ = run_epoch(
            model, train_batches, criterion, device, optimizer
        )
        val_loss, val_acc, _ = run_epoch(model, val_batches, criterion, device)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
        })
        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f}, acc {train_acc:.3f}"
            f" | val loss {val_loss:.4f}, acc {val_acc:.3f}"
        )
        if val_loss < best_loss:
            best_loss, stale_epochs = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping.")
                break

    if best_state is None:
        raise RuntimeError("Training produced no finite validation loss")
    model.load_state_dict(best_state)
    test_batches = make_batches(X_test, y_test, mean, std, args.batch_size)
    test_loss, test_acc, confusion = run_epoch(model, test_batches, criterion, device)
    f1 = 2 * confusion.diag().float() / (
        confusion.sum(dim=0) + confusion.sum(dim=1)
    ).clamp(min=1)
    print(f"\nBest epoch: {best_epoch}")
    print(f"Test accuracy: {test_acc:.4f}; macro F1: {f1.mean().item():.4f}")
    print("Class order:", ", ".join(CLASS_NAMES))
    print("Per-class F1:", [round(value, 4) for value in f1.tolist()])
    print("Confusion matrix (rows = true, columns = predicted):")
    print(confusion.numpy())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {k: v.cpu() for k, v in best_state.items()},
            "mean": torch.from_numpy(mean),
            "std": torch.from_numpy(std),
            "channels": list(CHANNELS),
            "validation_subjects": val_subjects.tolist(),
            "best_epoch": best_epoch,
            "seed": args.seed,
        },
        args.output,
    )
    print(f"Saved model and normalization statistics to {args.output}")
    save_report(report_dir, {
        "schema_version": 1,
        "started_at_utc": started_at.isoformat(),
        "model": "ActivityCNN",
        "trainable_parameters": count,
        "device": str(device),
        "config": {
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "patience": args.patience,
            "seed": args.seed,
            "optimizer": "Adam",
            "data_directory": str(args.data_dir.resolve()),
            "checkpoint_path": str(args.output.resolve()),
            "channels": list(CHANNELS),
        },
        "split": {
            "training_subjects": sorted(np.unique(subjects[~is_val]).tolist()),
            "validation_subjects": sorted(val_subjects.tolist()),
            "training_windows": int((~is_val).sum()),
            "validation_windows": int(is_val.sum()),
            "test_windows": len(y_test),
        },
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "selection_metric": "minimum validation loss",
        "history": history,
        "history_note": "Training metrics accumulate during weight updates; validation metrics use evaluation mode after each epoch.",
        "class_names": CLASS_NAMES,
        "test": {
            "evaluated_epoch": best_epoch,
            "loss": test_loss,
            "accuracy": test_acc,
            "macro_f1": f1.mean().item(),
            "per_class_f1": dict(zip(CLASS_NAMES, f1.tolist())),
            "confusion_matrix": confusion.tolist(),
            "confusion_matrix_orientation": "rows=true, columns=predicted; raw counts",
        },
    })
    print(f"Saved JSON report and plotting CSV files to {report_dir}")


if __name__ == "__main__":
    main()
