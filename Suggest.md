# Crypto Multi-Asset TFT Pipeline

Python project for training a Temporal Fusion Transformer (TFT) to forecast short-term Ethereum direction using multi-asset crypto market data and engineered technical indicators.

## Prerequisites
- Python 3.10+
- GPU with CUDA support (optional but recommended)
- Recommended Python packages:
  ```sh
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # adjust for your CUDA version
  pip install numpy pandas scikit-learn matplotlib
  ```

## Repository Layout
- `main.py` – end-to-end training and evaluation script, including plotting and CSV export.
- `preprocess.py` – data ingestion, feature engineering, windowed dataset builders, and loaders.
- `model.py` – Temporal Fusion Transformer, loss functions, and training utilities.
- `dataset/` – raw CSV history for each tracked asset (e.g., `dataset/Ethereum/*.csv`). CSVs are expected to be semicolon-separated.
- `log/` – training artifacts (plots, prediction logs) are written here.

## Data Expectations
1. Place historical OHLCV CSV files under `dataset/<AssetName>/`.
2. Each CSV should include the columns listed in `preprocess.KEEP_COLS`. Timestamps are parsed from `timeClose` or `timestamp`.
3. The script will build a panel joining all assets defined in `ASSET_DIRS` (`ETH`, `LINK`, `DAI`, `UNI`) and create lagged/technical features automatically.

## Running the Pipeline
1. (Optional) Set environment variables or edit constants near the top of `main.py` to tweak hyperparameters and data splits.
2. Launch training:
```sh
python main.py
```
3. Monitor console output for per-epoch metrics and threshold selection.

### Outputs
- `log/training_metrics.png` – loss and validation metric trajectories.
- `log/eth_test_price.png` – ETH close price with signal annotations over the test window.
- `log/test_predictions.csv` – probabilities, predictions, and returns for the test split.

## Key Configuration Flags (in `main.py`)
- `SAMPLE_MODE`: switch between random sampling (`True`) and chronological splits (`False` via `split_by_time`).
- `DATASET_CONFIG`: window length, batch sizes, and shuffling behaviour for loaders.
- `MODEL_CONFIG`: TFT architecture parameters (`d_model`, `nhead`, `num_layers`, etc.).
- `TRAINING_CONFIG`: epochs, gradient clipping, early stopping patience, and minimum delta.
- `CALIBRATE_PROBS`: enables isotonic calibration on validation probabilities.
- `THRESHOLD_METRIC`: controls which metric (`balanced_accuracy`, `sharpe`, `f1`, etc.) is used to pick the classification threshold.

## Reproducibility & Tips
- Randomness is controlled via the global `SEED`.
- Ensure `dataset` timestamps are sorted and non-duplicated; the loader drops rows with missing ETH close prices or targets.
- When adding new assets, extend `ASSET_DIRS` and confirm that the feature engineering in `preprocess.add_prefixed_features` supports the desired columns.

