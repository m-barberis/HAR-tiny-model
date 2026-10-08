"""Evaluate a saved 1D CNN or hybrid CNN-LSTM on the official test split."""

import argparse
import importlib.util
from pathlib import Path

import torch

from data_loader import CHANNELS, DEFAULT_DIRECTORY, DataLoader

ROOT = Path(__file__).resolve().parent

CLASS_NAMES = ["walking", "upstairs", "downstairs", "sitting", "standing", "laying"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "outputs/1D_CNN_20k/best.pt")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIRECTORY)
    args = parser.parse_args()

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if tuple(checkpoint["channels"]) != CHANNELS:
        parser.error("Checkpoint channels do not match the dataset loader")
    if "gravity_mean" in checkpoint:
        parser.error("Gravity-branch checkpoints are not supported")
    is_hybrid = "lstm.weight_ih_l0" in checkpoint["model_state_dict"]
    folder = "Hybrid_CNN_LSTM" if is_hybrid else "1D_CNN_20k"
    spec = importlib.util.spec_from_file_location(
        "har_model", ROOT / "src/har/models" / folder / "model.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if is_hybrid:
        # Dropout is disabled in evaluation mode, including for older checkpoints
        # that did not save their dropout settings.
        model = module.HybridCNNLSTM_Dropout(
            input_channels=len(CHANNELS),
            dropout_cnn=checkpoint.get("dropout_cnn", 0.2),
            dropout_fc=checkpoint.get("dropout_fc", 0.2),
        )
    else:
        model = module.ActivityCNN(
            input_channels=len(CHANNELS),
            AvgPool1d=checkpoint.get("avg_pool", False),
            dropout=checkpoint.get("dropout", 0.0),
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    X, y = DataLoader(args.data_dir).load_split("test")
    inputs = (torch.from_numpy(X) - checkpoint["mean"]) / checkpoint["std"]
    inputs = inputs.transpose(1, 2).contiguous()
    targets = torch.from_numpy(y - 1)
    confusion = torch.zeros(6, 6, dtype=torch.int64)
    with torch.inference_mode():
        for start in range(0, len(targets), 64):
            predicted = model(inputs[start:start + 64]).argmax(dim=1)
            indices = targets[start:start + 64] * 6 + predicted
            confusion += torch.bincount(indices, minlength=36).reshape(6, 6)

    accuracy = confusion.diag().sum().item() / confusion.sum().item()
    f1 = 2 * confusion.diag().float() / (
        confusion.sum(dim=0) + confusion.sum(dim=1)
    ).clamp(min=1)
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro F1: {f1.mean().item():.4f}")
    for name, score in zip(CLASS_NAMES, f1.tolist()):
        print(f"  {name} F1: {score:.4f}")
    print("Confusion matrix (rows=true, columns=predicted):")
    print("Class order:", ", ".join(CLASS_NAMES))
    print(confusion.numpy())


if __name__ == "__main__":
    main()
