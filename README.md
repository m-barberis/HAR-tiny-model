
# Human Activity Recognition: 1D CNN vs CNN-LSTM

A PyTorch project comparing two compact neural networks for human activity recognition from multivariate sensor time series:

- A **1D CNN** with 19,734 trainable parameters.
- A **hybrid CNN-LSTM** with 16,982 trainable parameters.

Both models classify six activities using the nine inertial signal channels provided by the UCI HAR dataset. They operate on sensor windows rather than the dataset’s precomputed feature table.

## Dataset

Each input window contains **128 time steps and nine channels**:

- Body acceleration: x, y, z.
- Body angular velocity: x, y, z.
- Total acceleration: x, y, z.

The six activity classes are walking, walking upstairs, walking downstairs, sitting, standing, and laying.

The dataset loader returns arrays with shape `(windows, time_steps, channels)`. Inputs are transposed to `(batch, channels, time_steps)` for the models.

## Models

### 1D CNN

Three convolutional blocks with batch normalization and ReLU activations, followed by global average pooling and a linear classifier.

The implementation supports configurable dropout and a choice between max and average pooling.

### CNN-LSTM

Two convolutional blocks extract local temporal features and reduce the sequence length. A single-layer LSTM processes the resulting sequence, and its final hidden state feeds a linear classifier.

Dropout is applied in the convolutional blocks and before the classifier.

## Evaluation protocol

- The official UCI HAR training and test splits are preserved.
- Training and validation subjects are disjoint.
- Per-channel normalization statistics are fitted on training subjects only.
- Models are trained with Adam and cross-entropy loss.
- Early stopping and checkpoint selection use validation loss.
- Dropout selection uses five-fold cross-validation over training subjects, testing rates from 0.0 to 0.5 in increments of 0.1.
- After dropout selection, a final model is trained using a subject-disjoint validation split and evaluated on the official test set.

Reported metrics include accuracy, macro-F1, per-class F1, and the confusion matrix.

## Results

The following results come from the saved experiment reports. All four runs use seed 42.

| Model | Dropout selection | Parameters | Test accuracy | Test macro-F1 |
|---|---|---:|---:|---:|
| 1D CNN | No dropout | 19,734 | 93.72% | 0.9380 |
| 1D CNN | Five-fold CV: 0.2 | 19,734 | 91.65% | 0.9161 |
| CNN-LSTM | Fixed dropout | 16,982 | 91.82% | 0.9201 |
| CNN-LSTM | Five-fold CV: 0.5 | 16,982 | 90.02% | 0.9003 |

In these experiments, the CNN achieves higher test scores than the hybrid CNN-LSTM. Dropout selected through cross-validation does not improve test performance over the corresponding initial configuration.

These are **final test scores, not cross-validation averages**. Cross-validation is used to select dropout based on mean validation loss. The available reports use a single seed, so the results do not establish a general ranking of the architectures.

## Setup

Install dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

Place the extracted dataset at:

```text
data/raw/human+activity+recognition+using+smartphones/UCI HAR Dataset/
```

Alternatively, pass its location through `--data-dir`.

Saved experiment reports are included. Model checkpoints and dataset files must be supplied or generated locally.

## Training

Run commands from the project root. Use a new checkpoint path and report directory for each experiment: training scripts refuse to overwrite existing outputs.

### Train the 1D CNN

```bash
python src/har/models/1D_CNN_20k/train.py \
  --epochs 50 \
  --output outputs/cnn_run/best.pt \
  --report-dir reports/cnn_run
```

Add `--avg-pool` to use average pooling instead of max pooling.

### Train the CNN-LSTM

```bash
python src/har/models/Hybrid_CNN_LSTM/train.py \
  --epochs 50 \
  --output outputs/hybrid_run/best.pt \
  --report-dir reports/hybrid_run
```

### Select dropout using cross-validation

For the CNN:

```bash
python src/har/models/1D_CNN_20k/train_cv.py \
  --epochs 50 \
  --output outputs/cnn_cv_run/best.pt \
  --report-dir reports/cnn_cv_run
```

For the CNN-LSTM:

```bash
python src/har/models/Hybrid_CNN_LSTM/train_cv.py \
  --epochs 50 \
  --output outputs/hybrid_cv_run/best.pt \
  --report-dir reports/hybrid_cv_run
```

The cross-validation scripts select a dropout rate, train a final model, and save its checkpoint and evaluation report.

## Evaluate a saved checkpoint

```bash
python evaluate.py --weights outputs/cnn_run/best.pt
```

Or, for the hybrid model:

```bash
python evaluate.py --weights outputs/hybrid_run/best.pt
```

The evaluation script automatically identifies the architecture, restores the saved normalization statistics, and evaluates the official test split on CPU.

It prints:

- Accuracy.
- Macro-F1.
- Per-class F1.
- Confusion matrix, with rows representing true classes and columns representing predictions.

## Project structure

```text
.
├── data_loader.py
├── evaluate.py
├── requirements.txt
├── data/
│   └── raw/
├── src/har/models/
│   ├── 1D_CNN_20k/
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── train_cv.py
│   │   └── report.py
│   └── Hybrid_CNN_LSTM/
│       ├── model.py
│       ├── train.py
│       ├── train_cv.py
│       └── report.py
├── outputs/
└── reports/
```

Each experiment report directory contains:

- `report.json`: configuration, subject splits, training history, and evaluation metrics.
- `learning_curve.csv`: training and validation metrics by epoch.
- `confusion_matrix.csv`: test-set confusion matrix.

## Scope and limitations

This project studies **time-series classification**, rather than forecasting. Its focus is implementing compact models and comparing them using subject-disjoint validation.

The current experiments cover one dataset and one recorded random seed. Broader conclusions would require repeated runs and additional datasets. Parameter counts describe model size; inference latency and memory consumption have not been benchmarked here.
