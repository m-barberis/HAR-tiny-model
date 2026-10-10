"""Plot training and validation loss and accuracy from a learning_curve.csv."""

import argparse
import csv
import math
from pathlib import Path


DEFAULT_CSV = Path(__file__).resolve().parent / "reports/1D_CNN_20k/learning_curve.csv"
COLUMNS = ("epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy")


def load_history(path):
    """Read the numeric history and validate the required columns."""
    history = {column: [] for column in COLUMNS}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        for row in reader:
            for column in COLUMNS:
                try:
                    value = float(row[column])
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"{path}, line {reader.line_num}: invalid {column}"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(f"{path}, line {reader.line_num}: non-finite {column}")
                history[column].append(value)
    if not history["epoch"]:
        raise ValueError(f"{path}: no training history rows")
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path", nargs="?", type=Path, default=DEFAULT_CSV,
        help="History CSV (default: reports/1D_CNN_20k/learning_curve.csv)",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output image path (default: training_history.png beside the CSV)",
    )
    parser.add_argument("--show", action="store_true", help="Also open the plot")
    args = parser.parse_args()
    history = load_history(args.csv_path)
    best_index = min(range(len(history["val_loss"])), key=history["val_loss"].__getitem__)
    selected_epoch = history["epoch"][best_index]

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split, name, color in (
            ("train", "Training", "#2878B5"),
            ("val", "Validation", "#D47724"),
        ):
            ax.plot(history["epoch"], history[f"{split}_{metric}"],
                    label=name, color=color, linewidth=2)
        ax.axvline(
            selected_epoch, color="#555555", linestyle="--", linewidth=1.5,
            label=f"Selected epoch: {selected_epoch:g}",
        )
        ax.set_title(metric.title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.title())
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend()
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1))
    fig.suptitle(f"Training history: {args.csv_path.parent.name}")
    output = args.output or args.csv_path.with_name("training_history.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved plot to {output.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
