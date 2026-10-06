"""Run a saved HAR model on the official test split and report CPU performance.

Usage: python3 tests/benchmark_inference.py outputs/1D_CNN_20k/activity_cnn.pt
"""

import argparse
import importlib.util
import json
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_loader import CHANNELS, DEFAULT_DIRECTORY, DataLoader


def load_model(checkpoint):
    weights = checkpoint["model_state_dict"]
    is_hybrid = "cnn.0.weight" in weights
    has_gravity = "gravity_mean" in checkpoint
    folder = (
        "Hybrid_GravityBranch" if is_hybrid and has_gravity else
        "Hybrid_CNN_LSTM" if is_hybrid else
        "1D_CNN_20k_GravityBranch" if has_gravity else
        "1D_CNN_20k"
    )
    class_name = (
        "HybridCNNLSTM_GravityBranch" if is_hybrid and has_gravity else
        "HybridCNNLSTM_Dropout" if is_hybrid else
        "ActivityCNN_GravityBranch" if has_gravity else
        "ActivityCNN"
    )
    spec = importlib.util.spec_from_file_location("har_benchmark_model", ROOT / "src/har/models" / folder / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = getattr(module, class_name)(input_channels=len(CHANNELS))
    model.load_state_dict(weights)
    return model.eval(), has_gravity


def peak_rss_mib():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)


def single_window_memory(model, inputs, gravity_inputs=None):
    """Return stored tensor bytes and peak extra CPU tensor bytes in one forward pass."""
    stored = sum(t.numel() * t.element_size() for t in (*model.parameters(), *model.buffers()))
    window = inputs[:1]
    gravity_window = gravity_inputs[:1] if gravity_inputs is not None else None
    input_bytes = window.numel() * window.element_size()
    if gravity_window is not None:
        input_bytes += gravity_window.numel() * gravity_window.element_size()

    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], profile_memory=True
    ) as profiler:
        if gravity_window is None:
            output = model(window)
        else:
            output = model(window, gravity_window)
        del output

    # Memory events carry the live total. Summing operator memory fields would
    # double count allocations because nested operators report the same tensors.
    with tempfile.TemporaryDirectory() as directory:
        trace = Path(directory) / "trace.json"
        profiler.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text())["traceEvents"]
    peak_extra = max(
        (event["args"]["Total Allocated"] for event in events if event.get("name") == "[memory]"),
        default=0,
    )
    return stored, input_bytes, peak_extra


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path, help="checkpoint saved by a training script")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DIRECTORY)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--single-window-memory", action="store_true",
                        help="profile PyTorch tensor memory during one CPU forward pass")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    if tuple(checkpoint["channels"]) != CHANNELS:
        parser.error("checkpoint channels do not match the dataset loader")
    model, has_gravity = load_model(checkpoint)
    _, X_test, _, _ = DataLoader(args.data_dir).load_data()
    mean = checkpoint["mean"].numpy()
    std = checkpoint["std"].numpy()
    inputs = torch.from_numpy(((X_test - mean) / std).transpose(0, 2, 1).copy())

    if has_gravity:
        total = [CHANNELS.index(f"total_acc_{axis}") for axis in "xyz"]
        body = [CHANNELS.index(f"body_acc_{axis}") for axis in "xyz"]
        gravity = X_test[:, :, total] - X_test[:, :, body]
        features = np.concatenate((gravity.mean(axis=1), gravity.std(axis=1)), axis=1)
        gravity_inputs = torch.from_numpy(
            ((features - checkpoint["gravity_mean"].numpy()) / checkpoint["gravity_std"].numpy()).astype(np.float32)
        )

    # Warm up the model before timing the entire test-set forward pass.
    with torch.inference_mode():
        if has_gravity:
            model(inputs[:1], gravity_inputs[:1])
        else:
            model(inputs[:1])
        start = time.perf_counter()
        for offset in range(0, len(inputs), args.batch_size):
            batch = inputs[offset:offset + args.batch_size]
            if has_gravity:
                model(batch, gravity_inputs[offset:offset + args.batch_size])
            else:
                model(batch)
        elapsed = time.perf_counter() - start

    print(f"Test windows: {len(inputs)}")
    print(f"Batch size: {args.batch_size}")
    print(f"CPU inference time: {elapsed:.3f} s")
    print(f"Inference speed: {len(inputs) / elapsed:.1f} windows/s")
    print(f"Peak process memory: {peak_rss_mib():.1f} MiB (maximum RSS, including model and data)")
    if args.single_window_memory:
        stored, input_bytes, peak_extra = single_window_memory(
            model, inputs, gravity_inputs if has_gravity else None
        )
        print(f"Model parameters and buffers: {stored / 1024:.1f} KiB")
        print(f"One input window: {input_bytes / 1024:.1f} KiB")
        print(f"Peak extra tensors in one forward pass: {peak_extra / 1024:.1f} KiB")
        print(f"Estimated peak tensor footprint: {(stored + input_bytes + peak_extra) / 1024:.1f} KiB")
        print("Tensor estimate excludes Python/PyTorch runtime and native library workspaces.")


if __name__ == "__main__":
    main()
