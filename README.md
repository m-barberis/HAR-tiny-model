# HAR-tiny-model

## Folder conventions

- `data_loader.py`: shared dataset loader; its default data path is relative to this file.
- `src/har/loader/`: data inspection notebooks and loader utilities.
- `src/har/models/<model_name>/`: each model's definition and training script.
- `data/`: raw and processed datasets.
- `outputs/<model_name>/`: generated checkpoints and model results.
- `reports/`: written analysis and reusable training reports.
- `tests/`: automated tests.

Place future files in the corresponding folder above.

## Train the CNN

From the project root, with NumPy and PyTorch installed:

```bash
python3 src/har/models/1D_CNN_20k/train.py --epochs 50
```

The default dataset and checkpoint paths are independent of the working directory.
The checkpoint is saved to `outputs/1D_CNN_20k/activity_cnn.pt`.
Explicit `--data-dir` and `--output` relative paths resolve from the working directory.

Each completed training run also saves a separate folder under
`reports/1D_CNN_20k/<UTC run timestamp>/`:

- `report.json`: run settings, split information, full learning history, best
  epoch, and test loss, accuracy, macro/per-class F1, and confusion matrix.
- `learning_curve.csv`: epoch, train/validation loss, and train/validation accuracy.
- `confusion_matrix.csv`: labeled raw counts, with true classes in rows and
  predicted classes in columns.

Values retain their full precision; accuracy and F1 use the 0–1 scale.
The test results correspond to the checkpoint with the lowest validation loss.
Use `--report-dir reports/my_experiment` to choose a folder explicitly (files
in that folder are overwritten if it is reused). Reports use only the Python
standard library and add no dependencies.

For example, load a report later to plot its learning curves or confusion matrix:

```python
import json
from pathlib import Path

report = json.loads(Path("reports/1D_CNN_20k/<run timestamp>/report.json").read_text())
epochs = [row["epoch"] for row in report["history"]]
val_loss = [row["val_loss"] for row in report["history"]]
confusion_matrix = report["test"]["confusion_matrix"]
class_names = report["class_names"]
```

Open `src/har/loader/inspect_windows.ipynb` with the notebook kernel's working
directory set to the project root or any folder inside it.
