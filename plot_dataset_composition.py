"""Plot activity counts in the official UCI HAR train and test splits."""

import argparse
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = (
    ROOT / "data/raw/human+activity+recognition+using+smartphones/UCI HAR Dataset"
)


def load_counts(data_dir):
    """Read activity names and count labeled windows, without loading signals."""
    labels = {}
    for line in (data_dir / "activity_labels.txt").read_text().splitlines():
        if line.strip():
            label, name = line.split(maxsplit=1)
            labels[int(label)] = name
    if not labels:
        raise ValueError("No activity labels found")

    counts = {}
    for split in ("train", "test"):
        path = data_dir / split / f"y_{split}.txt"
        counts[split] = Counter(int(value) for value in path.read_text().split())
        if not counts[split]:
            raise ValueError(f"{path}: no labeled windows found")
        unknown = counts[split].keys() - labels.keys()
        if unknown:
            raise ValueError(f"{path}: unknown activity IDs {sorted(unknown)}")
    return dict(sorted(labels.items())), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "reports/dataset_composition.png",
        help="Output image path (default: reports/dataset_composition.png)",
    )
    parser.add_argument("--show", action="store_true", help="Also open the plot")
    args = parser.parse_args()

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, counts = load_counts(args.data_dir)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True, layout="constrained")
    maximum = max(max(split.values()) for split in counts.values())
    positions = list(range(len(labels)))
    names = [name.replace("_", "\n").title() for name in labels.values()]

    for ax, (split, color) in zip(axes, (("train", "#2878B5"), ("test", "#D47724"))):
        values = [counts[split][label] for label in labels]
        total = sum(values)
        bars = ax.bar(positions, values, color=color, width=0.7)
        ax.bar_label(
            bars,
            labels=[f"{count:,}\n({count / total:.1%})" for count in values],
            padding=4, fontsize=9,
        )
        ax.set_title(f"{split.title()} - {total:,} windows")
        ax.set_xticks(positions, names)
        ax.set_xlabel("Activity label")
        ax.set_ylim(0, maximum * 1.2)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Number of windows")
    fig.suptitle("Dataset composition: percentages within each split")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Saved plot to {args.output.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
