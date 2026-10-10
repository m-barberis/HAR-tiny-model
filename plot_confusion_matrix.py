"""Plot a labeled confusion matrix CSV produced by the HAR training scripts."""

import argparse
import csv
from pathlib import Path


DEFAULT_CSV = Path(__file__).resolve().parent / "reports/1D_CNN_20k/confusion_matrix.csv"


def load_matrix(path):
    """Return true labels, predicted labels, and nonnegative integer counts."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = [row for row in csv.reader(handle) if row]
    if len(rows) < 2 or len(rows[0]) < 2:
        raise ValueError(f"{path}: expected a header and labeled count rows")
    predicted_labels = rows[0][1:]
    true_labels, counts = [], []
    for line, row in enumerate(rows[1:], start=2):
        if len(row) != len(predicted_labels) + 1:
            raise ValueError(f"{path}, row {line}: wrong number of columns")
        try:
            values = [int(value) for value in row[1:]]
        except ValueError as exc:
            raise ValueError(f"{path}, row {line}: counts must be integers") from exc
        if any(value < 0 for value in values):
            raise ValueError(f"{path}, row {line}: counts must be nonnegative")
        true_labels.append(row[0])
        counts.append(values)
    return true_labels, predicted_labels, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv_path", nargs="?", type=Path, default=DEFAULT_CSV,
        help="Input CSV (default: reports/1D_CNN_20k/confusion_matrix.csv)",
    )
    parser.add_argument("--output", type=Path, help="Output image path")
    parser.add_argument("--show", action="store_true", help="Also open the plot")
    parser.add_argument(
        "--normalize", action="store_true",
        help="Show percentages within each true-label row instead of counts",
    )
    args = parser.parse_args()
    true_labels, predicted_labels, counts = load_matrix(args.csv_path)

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    totals = [sum(row) for row in counts]
    values = (
        [[count / total if total else 0 for count in row]
         for row, total in zip(counts, totals)]
        if args.normalize else counts
    )
    maximum = 1 if args.normalize else max(1, max(map(max, counts)))
    fig, ax = plt.subplots(figsize=(8, 6.5), layout="constrained")
    heatmap = ax.imshow(values, cmap="Blues", vmin=0, vmax=maximum)
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            label = f"{value:.1%}" if args.normalize else f"{value:,}"
            if args.normalize and totals[i] == 0:
                label = "N/A"
            ax.text(j, i, label, ha="center", va="center", fontsize=11,
                    color="white" if value > maximum * 0.5 else "#222222")
    ax.set_xticks(range(len(predicted_labels)),
                  [label.replace("_", " ").title() for label in predicted_labels],
                  rotation=35, ha="right")
    ax.set_yticks(range(len(true_labels)),
                  [label.replace("_", " ").title() for label in true_labels])
    ax.set_xlabel("Predicted activity", fontsize=12)
    ax.set_ylabel("True activity", fontsize=12)
    ax.set_title(f"Confusion matrix: {args.csv_path.parent.name}", fontsize=14, pad=14)
    colorbar = fig.colorbar(heatmap, ax=ax, shrink=0.85)
    colorbar.set_label("Percentage within true activity" if args.normalize else "Number of windows")
    if args.normalize:
        colorbar.ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    filename = "confusion_matrix_normalized.png" if args.normalize else "confusion_matrix.png"
    output = args.output or args.csv_path.with_name(filename)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f"Saved plot to {output.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
