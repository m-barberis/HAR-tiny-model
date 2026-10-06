"""Save training results as JSON and plotting-friendly CSV files."""

import csv
import json
from pathlib import Path


def save_report(directory, report):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")

    with (directory / "learning_curve.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"]
        )
        writer.writeheader()
        writer.writerows(report["history"])

    with (directory / "confusion_matrix.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["true_activity / predicted_activity", *report["class_names"]])
        for name, row in zip(report["class_names"], report["test"]["confusion_matrix"]):
            writer.writerow([name, *row])
