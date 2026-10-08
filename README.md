# Human activity recognition

A small 1D CNN that classifies activity from accelerometer and gyroscope
time-series windows in the UCI HAR dataset. The main model has 19,734 trainable
parameters and does not use the dataset's precomputed feature table.

Complete experiments version.

## Setup

Run these commands from the project folder:

```bash
python3 -m pip install -r requirements.txt
```

Place the extracted `UCI HAR Dataset` folder inside
`data/raw/human+activity+recognition+using+smartphones/`.
The scripts also accept `--data-dir` pointing to the extracted dataset folder.

## Evaluate the saved model

```bash
python3 evaluate.py
```

This loads `outputs/1D_CNN_20k/best.pt`, applies the saved normalization,
and runs the model on the official test set. It prints accuracy, macro F1,
per-class F1, and the confusion matrix. Evaluation runs on the CPU.

The script automatically recognizes the 1D CNN and hybrid CNN-LSTM from the
checkpoint. For example, to evaluate the hybrid CV model:

```bash
python3 evaluate.py --weights outputs/Hybrid_CNN_LSTM_CV/best.pt
```

## Train

```bash
python3 src/har/models/1D_CNN_20k/train.py --epochs 50 \
  --output outputs/my_run/best.pt --report-dir reports/my_run
```

Choose new output paths for each run: training stops if the checkpoint or
report folder already exists. Without these options, the defaults are
`outputs/1D_CNN_20k/best.pt` and `reports/1D_CNN_20k/`.

Training and validation are split by subject. Normalization is fitted on the
training subjects, and the checkpoint with the lowest validation loss is saved.
The test set is evaluated after training. Add `--avg-pool` to use average pooling
instead of max pooling.

Each report folder contains `report.json`, `learning_curve.csv`, and
`confusion_matrix.csv`.

## Files

- `data_loader.py`: loads the sensor windows and labels.
- `evaluate.py`: evaluates saved 1D CNN and hybrid CNN-LSTM models (without gravity branches).
- `src/har/models/1D_CNN_20k/`: main CNN and training scripts; `train_cv.py` runs the dropout cross-validation experiment.
- Other folders in `src/har/models/`: hybrid CNN-LSTM and gravity-branch experiments.
- `src/har/inspection/inspect_windows.ipynb`: data inspection notebook.
- `outputs/`: saved checkpoints.
- `reports/`: experiment settings and results.
- `tests/benchmark_inference.py`: optional CPU inference timing.
