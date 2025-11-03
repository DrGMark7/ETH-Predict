import math
import random
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import dates as mdates
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, roc_auc_score

from preprocess import (
    KEEP_COLS,
    SampledData,
    ScaledData,
    add_eth_target,
    build_multi_asset_panel,
    enrich_panel_features,
    make_loaders,
    make_sampled_loaders,
    prepare_features_and_scale,
    prepare_sampled_data,
    split_by_time,
)
from model import (
    BCELossWithLabelSmoothing,
    FocalLoss,
    TemporalFusionTransformer,
    compute_class_stats,
    run_epoch,
    select_threshold,
    safe_metrics,
    trade_statistics,
)


# ===== Global Configuration =====
DATA_ROOT = Path("dataset")
ASSET_DIRS = {
    "ETH": "Ethereum",
    "LINK": "Chainlink",
    "DAI": "Dai",
    "UNI": "Uniswap",
}

SPLIT_CONFIG = dict(
    train_start="2017-01-01",
    train_end="2023-12-31",
    val_start="2024-01-01",
    val_end="2024-12-31",
    test_start="2025-01-01",
    test_end=None,
)

DATASET_CONFIG = dict(
    window=90,
    batch_size_train=128,
    batch_size_eval=256,
    shuffle_train=True,
    drop_last_train=True,
    use_weighted_sampler=False,
)

SAMPLE_MODE = True
SAMPLE_SPLIT = dict(train=0.7, val=0.15, test=0.15)

MODEL_CONFIG = dict(
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_ff=512,
    dropout=0.1,
    max_pos_len=1500,
)

OPTIMIZER_CONFIG = dict(
    lr=3e-4,
    weight_decay=5e-5,
)

TRAINING_CONFIG = dict(
    epochs=100,
    grad_clip=1.0,
    patience=20,
    min_delta=1e-4,
)

USE_SCHEDULER = True
SCHEDULER_CONFIG = dict(eta_min=1e-5)

USE_POS_WEIGHT = True
USE_FOCAL_LOSS = True
FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 2.0
THRESHOLD_METRIC = "balanced_accuracy"
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)
WARMUP_FRACTION = 0.1
CALIBRATE_PROBS = True
TRANSACTION_COST = 0.0005
MARGIN_TAU = 0.003
NEUTRAL_POLICY = "drop"
USE_LABEL_SMOOTHING = True
LABEL_SMOOTHING_EPS = 0.05
LOG_DIR = Path("log")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337


# ===== Utility Functions =====
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_log_dir(path: Path = LOG_DIR) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def smooth_curve(values, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 2 or window <= 1:
        return arr
    window = min(window, arr.size)
    if window % 2 == 0:
        window = window - 1 if window == arr.size else window + 1
    pad = window // 2
    kernel = np.ones(window, dtype=float) / window
    padded = np.pad(arr, pad, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: arr.size]


def plot_training_history(history: Dict[str, list], out_dir: Path) -> None:
    ensure_log_dir(out_dir)
    epochs = history.get("epoch", [])
    if not epochs:
        return

    epochs_arr = np.asarray(epochs)
    grid_style = dict(color="#e3ecff", linestyle="--", linewidth=0.8, alpha=0.8)

    def plot_metric(ax, key: str, label: str, marker: str) -> None:
        series = history.get(key, [])
        if not series:
            return
        series_arr = np.asarray(series, dtype=float)
        epochs_slice = epochs_arr[: series_arr.size]
        smoothed = smooth_curve(series_arr)
        smoothed = smoothed[: epochs_slice.size]
        local_marker_every = max(1, epochs_slice.size // 12)
        ax.plot(epochs_slice, smoothed, label=label,
                marker=marker, markevery=local_marker_every, linewidth=1.6)

    plt.figure(figsize=(12, 8))
    ax = plt.subplot(2, 2, 1)
    plot_metric(ax, "train_loss", "train_loss", "o")
    plot_metric(ax, "val_loss", "val_loss", "s")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(**grid_style)

    ax = plt.subplot(2, 2, 2)
    plot_metric(ax, "val_bal", "val_bal", "o")
    plot_metric(ax, "val_acc", "val_acc", "s")
    ax.set_title("Validation Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(**grid_style)

    ax = plt.subplot(2, 2, 3)
    plot_metric(ax, "val_f1", "val_f1", "o")
    plot_metric(ax, "val_mcc", "val_mcc", "s")
    ax.set_title("Validation F1 / MCC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(**grid_style)

    ax = plt.subplot(2, 2, 4)
    plot_metric(ax, "val_roc", "val_roc", "o")
    plot_metric(ax, "lr", "lr", "s")
    ax.set_title("Validation ROC & LR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score / LR")
    ax.legend()
    ax.grid(**grid_style)

    plt.tight_layout()
    plt.savefig(out_dir / "training_metrics.png", dpi=200)
    plt.close()


def plot_test_price(panel: pd.DataFrame,
                    indices,
                    out_dir: Path,
                    title: str,
                    annotate_thresholds: bool,
                    probs: np.ndarray,
                    threshold: float) -> None:
    ensure_log_dir(out_dir)
    indices = np.asarray(indices)
    if len(indices) == 0:
        print("No test indices to plot.")
        return

    if np.issubdtype(indices.dtype, np.integer):
        sub_panel = panel.iloc[indices]
        plot_dates = sub_panel.index
    else:
        sub_panel = panel.loc[indices]
        plot_dates = sub_panel.index

    if "ETH_close" not in sub_panel.columns:
        print("ETH_close column missing; cannot plot price.")
        return

    prices = sub_panel["ETH_close"].astype(float)

    plt.figure(figsize=(14, 6))
    smoothed_prices = smooth_curve(prices, window=7) if len(prices) >= 7 else prices
    plt.plot(plot_dates, smoothed_prices, color="#1f77b4", linewidth=2.0,
             marker="o", markevery=max(1, len(plot_dates) // 30), label="ETH Close")
    plt.fill_between(plot_dates, prices.min(), prices.max(), color="#d6e9ff", alpha=0.1)

    if annotate_thresholds and probs is not None and threshold is not None:
        probs = np.asarray(probs)[:len(sub_panel)]
        long_mask = probs >= threshold
        plt.scatter(plot_dates[long_mask], prices.values[long_mask],
                    color="#2ca02c", s=30, label="Signal ≥ threshold", zorder=5, alpha=0.7)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    ax.grid(color="#e3ecff", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("ETH Close Price (USD)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "eth_test_price.png", dpi=200)
    plt.close()


def export_test_log(panel: pd.DataFrame,
                    indices,
                    probs: np.ndarray,
                    preds: np.ndarray,
                    targets: np.ndarray,
                    rets: np.ndarray,
                    out_dir: Path,
                    filename: str = "test_predictions.csv") -> None:
    ensure_log_dir(out_dir)
    indices = np.asarray(indices)
    if len(indices) == 0:
        print("No test indices to export.")
        return

    if np.issubdtype(indices.dtype, np.integer):
        sub_panel = panel.iloc[indices]
        index_vals = sub_panel.index
    else:
        sub_panel = panel.loc[indices]
        index_vals = sub_panel.index

    df = pd.DataFrame({
        "timestamp": index_vals,
        "ETH_close": sub_panel.get("ETH_close").values if "ETH_close" in sub_panel.columns else np.nan,
        "y_true": targets[:len(index_vals)],
        "y_pred": preds[:len(index_vals)],
        "prob": probs[:len(index_vals)],
        "log_return": rets[:len(index_vals)],
    })
    df["long_signal"] = (df["prob"] >= 0.5).astype(int)
    df["pnl_signal"] = df["long_signal"] * df["log_return"]
    df.to_csv(out_dir / filename, index=False)


# ===== Main Pipeline =====
def main() -> None:
    set_seed(SEED)
    panel = build_multi_asset_panel(DATA_ROOT, ASSET_DIRS, KEEP_COLS)
    panel = enrich_panel_features(panel, main_asset="ETH", assets=ASSET_DIRS.keys())
    panel = add_eth_target(panel, margin_tau=MARGIN_TAU, neutral_policy=NEUTRAL_POLICY)

    sample_info: SampledData | ScaledData
    if SAMPLE_MODE:
        sampled = prepare_sampled_data(
            panel,
            window=DATASET_CONFIG["window"],
            splits=SAMPLE_SPLIT,
            drop_cols=[],
            seed=SEED,
        )
        loaders = make_sampled_loaders(
            sampled,
            window=DATASET_CONFIG["window"],
            batch_size_train=DATASET_CONFIG["batch_size_train"],
            batch_size_eval=DATASET_CONFIG["batch_size_eval"],
            shuffle_train=DATASET_CONFIG["shuffle_train"],
        )

        def class_stats(indices: np.ndarray) -> tuple[float, float]:
            arr = sampled.y[indices]
            arr = arr[arr >= 0]
            if len(arr) == 0:
                return float("nan"), 0.0
            return float(arr.mean()), float(arr.sum())

        for name, idx in [("train", sampled.train_indices),
                          ("val", sampled.val_indices),
                          ("test", sampled.test_indices)]:
            rate, pos_count = class_stats(idx)
            print(f"{name} positive rate: {rate:.3%} ({pos_count:.0f}/{len(idx)})")

        pos_rate, _ = class_stats(sampled.train_indices)
        pos_weight = (1.0 - pos_rate) / max(pos_rate, 1e-6) if pos_rate < 1.0 else 1.0
        effective_pos_weight = pos_weight if USE_POS_WEIGHT else 1.0
        feature_dim = len(sampled.feature_cols)
        sample_info = sampled
        plot_indices = sampled.index[sampled.test_indices]
    else:
        splits = split_by_time(panel, neutral_policy=NEUTRAL_POLICY, **SPLIT_CONFIG)
        scaled = prepare_features_and_scale(splits, drop_cols=[])

        for name, part in [("train", splits.train), ("val", splits.val), ("test", splits.test)]:
            rate = float(part["y_up"].mean())
            print(f"{name} positive rate: {rate:.3%} ({part['y_up'].sum():.0f}/{len(part)})")

        loaders = make_loaders(scaled, **DATASET_CONFIG)

        pos_rate, pos_weight = compute_class_stats(scaled.y_train)
        effective_pos_weight = pos_weight if USE_POS_WEIGHT else 1.0
        feature_dim = len(scaled.feature_cols)
        sample_info = scaled
        plot_indices = splits.test.index

    print(
        f"Train positive rate: {pos_rate:.3%} → "
        f"pos_weight_raw={pos_weight:.2f} (applied={effective_pos_weight:.2f}, enabled={USE_POS_WEIGHT})"
    )

    print("Feature dim (F):", feature_dim)
    for name, loader in [("train", loaders.train), ("val", loaders.val), ("test", loaders.test)]:
        xb, yb, rb = next(iter(loader))
        print(f"{name}: X shape {xb.shape} (B,T,F), y shape {yb.shape}, ret shape {rb.shape}")

    model = TemporalFusionTransformer(in_dim=feature_dim, **MODEL_CONFIG).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), **OPTIMIZER_CONFIG)
    pos_weight_tensor = torch.tensor(effective_pos_weight, dtype=torch.float32, device=DEVICE)
    if USE_FOCAL_LOSS:
        criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, pos_weight=pos_weight_tensor)
        print(f"Using focal loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
    else:
        if USE_LABEL_SMOOTHING:
            criterion = BCELossWithLabelSmoothing(
                smoothing=LABEL_SMOOTHING_EPS,
                pos_weight=pos_weight_tensor,
            )
            print(f"Using BCE with label smoothing (eps={LABEL_SMOOTHING_EPS:.3f})")
        else:
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    criterion = criterion.to(DEVICE)

    if USE_SCHEDULER:
        total_epochs = TRAINING_CONFIG["epochs"]
        warmup_epochs = max(1, int(total_epochs * WARMUP_FRACTION))
        cosine_epochs = max(1, total_epochs - warmup_epochs)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=SCHEDULER_CONFIG.get("eta_min", 1e-5),
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    else:
        scheduler = None

    best_state = None
    best_val_loss = float("inf")
    best_epoch = -1
    patience = TRAINING_CONFIG.get("patience")
    min_delta = TRAINING_CONFIG.get("min_delta", 0.0)
    epochs_no_improve = 0
    last_val_loss = float("inf")
    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_bal": [],
        "val_f1": [],
        "val_mcc": [],
        "val_roc": [],
        "lr": [],
    }

    for epoch in range(TRAINING_CONFIG["epochs"]):
        train_loss, _, _, _ = run_epoch(
            model,
            loaders.train,
            criterion,
            optimizer=optimizer,
            grad_clip=TRAINING_CONFIG["grad_clip"],
            device=DEVICE,
        )
        val_loss, val_probs, val_targets, val_rets = run_epoch(model, loaders.val, criterion, device=DEVICE)
        val_pred = (val_probs >= 0.5).astype(int)
        val_acc = accuracy_score(val_targets, val_pred)
        try:
            val_bal = balanced_accuracy_score(val_targets, val_pred)
        except ValueError:
            val_bal = float("nan")
        try:
            val_f1 = f1_score(val_targets, val_pred, zero_division=0)
        except ValueError:
            val_f1 = float("nan")
        try:
            val_mcc = matthews_corrcoef(val_targets, val_pred)
        except ValueError:
            val_mcc = float("nan")
        try:
            val_roc = roc_auc_score(val_targets, val_probs)
        except ValueError:
            val_roc = float("nan")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_bal"].append(val_bal)
        history["val_f1"].append(val_f1)
        history["val_mcc"].append(val_mcc)
        history["val_roc"].append(val_roc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"epoch {epoch:02d} | train {train_loss:.4f} | val {val_loss:.4f} | "
            f"acc {val_acc:.3f} | bal {val_bal:.3f} | f1 {val_f1:.3f} | mcc {val_mcc:.3f} | roc {val_roc:.3f} | "
            f"lr {optimizer.param_groups[0]['lr']:.2e}"
        )
        improved = np.isfinite(val_loss) and (val_loss < best_val_loss - min_delta)
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience and epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch:02d}.")
                last_val_loss = val_loss
                break

        if scheduler is not None:
            scheduler.step()

        last_val_loss = val_loss

    if best_state is None:
        print("Warning: best checkpoint not captured; falling back to last epoch weights.")
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        best_val_loss = last_val_loss
        best_epoch = epoch
    else:
        print(f"Best validation loss {best_val_loss:.4f} at epoch {best_epoch:02d}.")

    plot_training_history(history, LOG_DIR)

    best_threshold = 0.5
    prob_calibrator = None
    if best_state is not None:
        model.load_state_dict(best_state)
        _, val_probs, val_targets, val_rets = run_epoch(model, loaders.val, criterion, device=DEVICE)
        if CALIBRATE_PROBS:
            calibrator = IsotonicRegression(out_of_bounds="clip")
            val_probs = calibrator.fit_transform(val_probs, val_targets)
            prob_calibrator = calibrator
            print("Applied isotonic calibration on validation probabilities.")
        best_threshold = select_threshold(
            val_targets,
            val_probs,
            THRESHOLD_GRID,
            metric=THRESHOLD_METRIC,
            y_ret=val_rets,
            cost=TRANSACTION_COST,
        )

    test_loss, test_probs, test_targets, test_rets = run_epoch(model, loaders.test, criterion, device=DEVICE)
    if prob_calibrator is not None:
        test_probs = prob_calibrator.transform(test_probs)
    test_pred = (test_probs >= best_threshold).astype(int)
    test_acc = accuracy_score(test_targets, test_pred)
    try:
        test_bal = balanced_accuracy_score(test_targets, test_pred)
    except ValueError:
        test_bal = float("nan")
    test_f1 = f1_score(test_targets, test_pred, zero_division=0)
    try:
        test_mcc = matthews_corrcoef(test_targets, test_pred)
    except ValueError:
        test_mcc = float("nan")
    try:
        test_roc = roc_auc_score(test_targets, test_probs)
    except ValueError:
        test_roc = float("nan")
    trade_stats = trade_statistics(test_probs, test_rets, best_threshold, TRANSACTION_COST)
    test_metrics = dict(
        acc=test_acc,
        bal=test_bal,
        f1=test_f1,
        mcc=test_mcc,
        roc=test_roc,
        loss=test_loss,
        thr=best_threshold,
        sharpe=trade_stats["sharpe"],
        pnl_mean=trade_stats["mean"],
    )
    print("TEST:", test_metrics)

    safe = safe_metrics(test_targets, test_probs, thr=best_threshold)
    safe["loss"] = test_loss
    safe["thr"] = best_threshold
    safe["sharpe"] = trade_stats["sharpe"]
    safe["pnl_mean"] = trade_stats["mean"]
    print("TEST (safe):", safe)

    plot_test_price(
        panel,
        plot_indices,
        LOG_DIR,
        title="ETH Close Price - Test Period",
        annotate_thresholds=True,
        probs=test_probs,
        threshold=best_threshold,
    )

    export_test_log(
        panel,
        plot_indices,
        probs=test_probs,
        preds=test_pred,
        targets=test_targets,
        rets=test_rets,
        out_dir=LOG_DIR,
    )


if __name__ == "__main__":
    main()
