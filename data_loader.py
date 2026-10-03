"""Load UCI HAR acceleration and gyroscope windows (requires numpy).

Each X array has shape (windows, 128 time steps, 9 sensor channels).
Channels are body_acc_x/y/z, body_gyro_x/y/z, total_acc_x/y/z.
Each y array contains the matching activity IDs (1–6).
"""

from pathlib import Path

import numpy as np


DEFAULT_DIRECTORY = (
    Path(__file__).resolve().parent
    / "data/raw/human+activity+recognition+using+smartphones/UCI HAR Dataset"
)
CHANNELS = tuple(
    f"{sensor}_{axis}"
    for sensor in ("body_acc", "body_gyro", "total_acc")
    for axis in "xyz"
) # all 9 channels used


class DataLoader:
    def __init__(self, directory=DEFAULT_DIRECTORY):
        self.directory = Path(directory)

    def load_data(self):
        """Return X_train, X_test, y_train, y_test as NumPy arrays."""

        def load_split(split):
            folder = self.directory / split
            signals = [
                np.loadtxt(
                    folder / "Inertial Signals" / f"{channel}_{split}.txt",
                    dtype=np.float32,
                )
                for channel in CHANNELS
            ]
            X = np.stack(signals, axis=-1)
            y = np.loadtxt(folder / f"y_{split}.txt", dtype=np.int64)
            if X.shape[0] != len(y) or X.shape[1] != 128:
                raise ValueError(f"{split}: unexpected signal or label shape")
            return X, y

        X_train, y_train = load_split("train")
        X_test, y_test = load_split("test")
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = DataLoader().load_data()
    print(f"Training: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing:  X={X_test.shape}, y={y_test.shape}")
    print(f"Channels: {CHANNELS}")
