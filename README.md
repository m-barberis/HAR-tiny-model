# Human activity recognition

A small 1D CNN that classifies activity from accelerometer and gyroscope
time-series windows in the UCI HAR dataset. The main model has 19,734 trainable
parameters and does not use the dataset's precomputed feature table.

Further experiments that have been made during the week, like the insertion of gravity features in the classifier or training runs with different seeds, are not included to keep the results' submission clear and simple.
Only the source code will be submitted, so you will need to train the models again to evaluate them. 

## Setup

Run these commands from the project folder:

```bash
python -m pip install -r requirements.txt
```

Place the extracted `UCI HAR Dataset` folder inside
`data/raw/`.
The scripts also accept `--data-dir` pointing to the extracted dataset folder.

## Train

```bash
python src/har/models/1D_CNN_20k/train.py --epochs 50 \
  --output outputs/my_run/best.pt --report-dir reports/my_run
```

Choose new output paths for each run: training stops if the checkpoint or
report folder already exists. Without these options, the defaults are
`outputs/1D_CNN_20k/best.pt` and `reports/1D_CNN_20k/`.

**The reports from my runs are included in the submission, so using the command without changing --report-dir will cause an error.**

Training and validation are split by subject. Normalization is fitted on the
training subjects, and the checkpoint with the lowest validation loss is saved.
The test set is evaluated after training. Add `--avg-pool` to use average pooling
instead of max pooling.

Each report folder contains `report.json`, `learning_curve.csv`, and
`confusion_matrix.csv`.

## Evaluate the saved model

```bash
python evaluate.py --weights outputs/my_run/best.pt
```

This loads `outputs/1D_CNN_20k/best.pt` by default, applies the saved normalization,
and runs the model on the official test set. It prints accuracy, macro F1,
per-class F1, and the confusion matrix. Evaluation runs on the CPU.

The script automatically recognizes the 1D CNN and hybrid CNN-LSTM from the
checkpoint. For example, to evaluate the hybrid CV model:

```bash
python evaluate.py --weights outputs/Hybrid_CNN_LSTM_CV/best.pt
```

## Files

- `data_loader.py`: loads the sensor windows and labels.
- `evaluate.py`: evaluates saved 1D CNN and hybrid CNN-LSTM models (without gravity branches).
- `src/har/models/1D_CNN_20k/`: main CNN and training scripts; `train_cv.py` runs the dropout cross-validation experiment.
- `src/har/models/Hybrid_CNN_LSTM`: hybrid CNN-LSTM model and training scripts.
- `outputs/`: saved checkpoints.
- `reports/`: experiment settings and results.
