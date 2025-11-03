import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.preprocessing import RobustScaler

KEEP_COLS = [
    "timeOpen", "timeClose", "timeHigh", "timeLow", "name",
    "open", "high", "low", "close", "volume",
    "marketCap", "circulatingSupply", "timestamp",
]


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    roll_up = up.ewm(alpha=1.0 / window, adjust=False).mean()
    roll_down = down.ewm(alpha=1.0 / window, adjust=False).mean()
    rs = roll_up / (roll_down + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff().fillna(0.0))
    return (direction * volume.fillna(0.0)).cumsum()


def _coerce_schema(df: pd.DataFrame, keep_cols: Iterable[str]) -> pd.DataFrame:
    cols = [c for c in keep_cols if c in df.columns]
    df = df[cols].copy()

    for c in ["timeOpen", "timeClose", "timeHigh", "timeLow", "timestamp"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce", utc=True)

    num_cols = [
        "open", "high", "low", "close", "volume",
        "marketCap", "circulatingSupply",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "timeClose" in df.columns:
        df = df.sort_values("timeClose").set_index("timeClose")
    elif "timestamp" in df.columns:
        df = df.sort_values("timestamp").set_index("timestamp")
    else:
        raise ValueError("No time column found (expected timeClose or timestamp).")

    return df[~df.index.duplicated(keep="last")]


def load_asset_folder(root: Path, asset_dirname: str, keep_cols: Iterable[str]) -> pd.DataFrame:
    base = root / asset_dirname
    csvs = sorted(base.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files under {base}")

    frames = []
    for path in csvs:
        try:
            tmp = pd.read_csv(path, sep=";", low_memory=False)
            frames.append(_coerce_schema(tmp, keep_cols))
        except Exception as exc:
            print(f"[WARN] skip {path}: {exc}")

    if not frames:
        raise RuntimeError(f"No valid CSV parsed for {asset_dirname}")

    return pd.concat(frames, axis=0).sort_index()


def add_prefixed_features(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = df.copy()
    close_price = out["close"]
    out[f"{prefix}_ret1"] = np.log(close_price / close_price.shift(1))
    out[f"{prefix}_ret5"] = np.log(close_price / close_price.shift(5))
    out[f"{prefix}_vol10"] = out[f"{prefix}_ret1"].rolling(10, min_periods=3).std()
    out[f"{prefix}_ma7"] = close_price.rolling(7, min_periods=3).mean()
    out[f"{prefix}_ma21"] = close_price.rolling(21, min_periods=5).mean()
    if {"high", "low"} <= set(out.columns):
        out[f"{prefix}_hlrng"] = (out["high"] - out["low"]) / close_price.shift(1)
        tr = _true_range(out["high"], out["low"], close_price)
        out[f"{prefix}_atr14"] = tr.rolling(14, min_periods=5).mean()
    if "volume" in out.columns:
        out[f"{prefix}_volchg1"] = np.log(out["volume"] / out["volume"].shift(1))
        out[f"{prefix}_obv"] = _obv(close_price, out["volume"])
        out[f"{prefix}_vol_z"] = (out["volume"] - out["volume"].rolling(20, min_periods=5).mean()) / (
            out["volume"].rolling(20, min_periods=5).std() + 1e-6
        )
    out[f"{prefix}_ema12"] = _ema(close_price, 12)
    out[f"{prefix}_ema26"] = _ema(close_price, 26)
    out[f"{prefix}_macd"] = out[f"{prefix}_ema12"] - out[f"{prefix}_ema26"]
    out[f"{prefix}_rsi14"] = _rsi(close_price, 14)
    out[f"{prefix}_mom10"] = close_price / close_price.shift(10) - 1.0
    return out


def build_multi_asset_panel(root: Path, assets_map: Dict[str, str], keep_cols: Iterable[str]) -> pd.DataFrame:
    prefixed_frames = []
    for prefix, dirname in assets_map.items():
        raw = load_asset_folder(root, dirname, keep_cols)
        features = add_prefixed_features(raw, prefix)

        base_cols = ["open", "high", "low", "close", "volume", "marketCap", "circulatingSupply"]
        rename_map = {c: f"{prefix}_{c}" for c in base_cols if c in features.columns}
        features = features.rename(columns=rename_map)

        has_cols = [f"{prefix}_{c}" for c in base_cols if f"{prefix}_{c}" in features.columns]
        features[f"{prefix}_mask"] = (~features[has_cols].isna().all(axis=1)).astype(float)

        cols_to_keep = [
            *rename_map.values(),
            f"{prefix}_ret1",
            f"{prefix}_ret5",
            f"{prefix}_vol10",
            f"{prefix}_ma7",
            f"{prefix}_ma21",
            f"{prefix}_atr14",
            f"{prefix}_ema12",
            f"{prefix}_ema26",
            f"{prefix}_macd",
            f"{prefix}_rsi14",
            f"{prefix}_mom10",
        ]
        optional_cols = [
            f"{prefix}_hlrng",
            f"{prefix}_volchg1",
            f"{prefix}_obv",
            f"{prefix}_vol_z",
        ]
        cols_to_keep.extend([c for c in optional_cols if c in features.columns])
        cols_to_keep.append(f"{prefix}_mask")

        prefixed_frames.append(features[cols_to_keep].dropna(how="all", axis=1))

    panel = prefixed_frames[0]
    for frame in prefixed_frames[1:]:
        panel = panel.join(frame, how="outer")

    return panel.sort_index()


def enrich_panel_features(panel: pd.DataFrame, main_asset: str, assets: Iterable[str]) -> pd.DataFrame:
    assets = list(assets)
    if main_asset not in assets:
        assets.append(main_asset)

    enriched = panel.copy()
    main_ret = enriched.get(f"{main_asset}_ret1")
    if main_ret is not None:
        for other in assets:
            if other == main_asset:
                continue
            other_ret = enriched.get(f"{other}_ret1")
            if other_ret is not None:
                enriched[f"{main_asset}_{other}_corr30"] = main_ret.rolling(30, min_periods=10).corr(other_ret)
                enriched[f"{main_asset}_{other}_corr90"] = main_ret.rolling(90, min_periods=30).corr(other_ret)

        rolling_std = main_ret.rolling(30, min_periods=15).std()
        q_low = np.nanquantile(rolling_std, 0.33)
        q_high = np.nanquantile(rolling_std, 0.66)
        enriched[f"{main_asset}_regime_low"] = (rolling_std <= q_low).astype(float)
        enriched[f"{main_asset}_regime_mid"] = ((rolling_std > q_low) & (rolling_std < q_high)).astype(float)
        enriched[f"{main_asset}_regime_high"] = (rolling_std >= q_high).astype(float)

    return enriched


def add_eth_target(panel: pd.DataFrame, margin_tau: float = 0.0, neutral_policy: str = "drop") -> pd.DataFrame:
    out = panel.copy()
    out["ETH_close_t"] = out["ETH_close"]
    out["ETH_close_t1"] = out["ETH_close"].shift(-1)
    out["ETH_ret_next"] = out["ETH_close_t1"] / out["ETH_close_t"] - 1.0
    out["ETH_logret_next"] = np.log(out["ETH_close_t1"] / out["ETH_close_t"])

    if margin_tau > 0:
        up = out["ETH_logret_next"] > margin_tau
        down = out["ETH_logret_next"] < -margin_tau
        out["y_up"] = np.where(up, 1.0, np.where(down, 0.0, np.nan))
    else:
        out["y_up"] = (out["ETH_logret_next"] > 0).astype(float)

    out = out.dropna(subset=["ETH_close_t", "ETH_close_t1"])
    if neutral_policy == "drop":
        out = out.dropna(subset=["y_up"])

    return out


@dataclass
class TimeSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_by_time(panel: pd.DataFrame,
                  train_start: str,
                  train_end: str,
                  val_start: str,
                  val_end: str,
                  test_start: str,
                  test_end: Optional[str] = None,
                  neutral_policy: str = "drop") -> TimeSplits:
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise ValueError("panel.index must be a DatetimeIndex")

    if panel.index.tz is not None:
        panel = panel.copy()
        panel.index = panel.index.tz_convert("UTC").tz_localize(None)

    train = panel.loc[train_start:train_end].copy()
    val = panel.loc[val_start:val_end].copy()
    test = panel.loc[test_start:(test_end or panel.index.max())].copy()

    for name, part in [("train", train), ("val", val), ("test", test)]:
        if "y_up" not in part.columns:
            raise ValueError("panel must contain 'y_up' column as label.")
        if neutral_policy == "drop":
            part.dropna(subset=["y_up"], inplace=True)
        if len(part) == 0:
            raise ValueError(f"{name} slice is empty. Check your date ranges.")

    return TimeSplits(train=train, val=val, test=test)


@dataclass
class ScaledData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    ret_train: np.ndarray
    ret_val: np.ndarray
    ret_test: np.ndarray
    feature_cols: list
    scaler: RobustScaler
    impute_median: np.ndarray


def _impute_with_median(X: np.ndarray, med: np.ndarray) -> np.ndarray:
    X = X.copy()
    mask = np.isnan(X)
    if mask.any():
        X[mask] = np.take(med, np.where(mask)[1])
    return np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)


def prepare_features_and_scale(splits: TimeSplits, drop_cols: Iterable[str] = None) -> ScaledData:
    drop_cols = set(drop_cols or [])
    drop_cols.update({"y_up", "ETH_close_t", "ETH_close_t1", "ETH_ret_next", "ETH_logret_next"})

    feature_cols = [c for c in splits.train.columns if c not in drop_cols]

    def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = df[feature_cols].astype(np.float32).values
        y = df["y_up"].astype(np.float32).values
        r = df["ETH_logret_next"].astype(np.float32).values
        return X, y, r

    X_train, y_train, r_train = to_xy(splits.train)
    X_val, y_val, r_val = to_xy(splits.val)
    X_test, y_test, r_test = to_xy(splits.test)

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    med = np.nanmedian(X_train, axis=0)
    X_train = _impute_with_median(X_train, med)
    X_val = _impute_with_median(X_val, med)
    X_test = _impute_with_median(X_test, med)

    return ScaledData(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        ret_train=r_train,
        ret_val=r_val,
        ret_test=r_test,
        feature_cols=feature_cols,
        scaler=scaler,
        impute_median=med,
    )


class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, ret: np.ndarray, window: int = 60, indices: Iterable[int] = None):
        if not (len(X) == len(y) == len(ret)):
            raise ValueError("X, y, and ret must have the same length")
        if len(y) < window:
            raise ValueError("Not enough samples to form one window.")
        self.X = X
        self.y = y
        self.ret = ret
        self.window = window
        base_indices = np.arange(window - 1, len(y))
        if indices is None:
            self.indices = base_indices
        else:
            indices = np.asarray(list(indices), dtype=int)
            indices = indices[(indices >= window - 1) & (indices < len(y))]
            self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        end = self.indices[idx]
        start = end - self.window + 1
        x = self.X[start:end + 1]
        y = self.y[end]
        r = self.ret[end]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32), torch.tensor(r, dtype=torch.float32)


@dataclass
class Loaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


@dataclass
class SampledData:
    X: np.ndarray
    y: np.ndarray
    ret: np.ndarray
    index: np.ndarray
    feature_cols: list
    scaler: RobustScaler
    impute_median: np.ndarray
    train_indices: np.ndarray
    val_indices: np.ndarray
    test_indices: np.ndarray


def make_loaders(scaled: ScaledData, window: int = 60,
                 batch_size_train: int = 128, batch_size_eval: int = 256,
                 shuffle_train: bool = True, drop_last_train: bool = True,
                 use_weighted_sampler: bool = False) -> Loaders:
    ds_train = WindowDataset(scaled.X_train, scaled.y_train, scaled.ret_train, window=window)
    ds_val = WindowDataset(scaled.X_val, scaled.y_val, scaled.ret_val, window=window)
    ds_test = WindowDataset(scaled.X_test, scaled.y_test, scaled.ret_test, window=window)

    sampler = None
    if use_weighted_sampler:
        labels = ds_train.y[ds_train.indices].astype(int)
        class_counts = np.bincount(labels, minlength=2).astype(float)
        class_counts[class_counts == 0] = 1.0
        class_weights = len(labels) / (class_counts * len(class_counts))
        sample_weights = class_weights[labels]
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        print(
            "Weighted sampler stats:",
            dict(class_counts=class_counts.tolist(), class_weights=class_weights.tolist()),
        )

    dl_train = DataLoader(
        ds_train,
        batch_size=batch_size_train,
        shuffle=shuffle_train if sampler is None else False,
        sampler=sampler,
        drop_last=drop_last_train if sampler is None else False,
    )
    dl_val = DataLoader(ds_val, batch_size=batch_size_eval, shuffle=False, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size_eval, shuffle=False, drop_last=False)
    return Loaders(train=dl_train, val=dl_val, test=dl_test)


def make_sampled_loaders(data: SampledData,
                         window: int,
                         batch_size_train: int,
                         batch_size_eval: int,
                         shuffle_train: bool = True) -> Loaders:
    ds_train = WindowDataset(data.X, data.y, data.ret, window=window, indices=data.train_indices)
    ds_val = WindowDataset(data.X, data.y, data.ret, window=window, indices=data.val_indices)
    ds_test = WindowDataset(data.X, data.y, data.ret, window=window, indices=data.test_indices)

    dl_train = DataLoader(ds_train, batch_size=batch_size_train, shuffle=shuffle_train, drop_last=False)
    dl_val = DataLoader(ds_val, batch_size=batch_size_eval, shuffle=False, drop_last=False)
    dl_test = DataLoader(ds_test, batch_size=batch_size_eval, shuffle=False, drop_last=False)
    return Loaders(train=dl_train, val=dl_val, test=dl_test)


def prepare_sampled_data(panel: pd.DataFrame,
                         window: int,
                         splits: Dict[str, float],
                         drop_cols: Iterable[str] = None,
                         seed: int = 1337) -> SampledData:
    drop_cols = set(drop_cols or [])
    drop_cols.update({"y_up", "ETH_close_t", "ETH_close_t1", "ETH_ret_next", "ETH_logret_next"})

    df = panel.dropna(subset=["ETH_logret_next"]).copy()
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X_raw = df[feature_cols].astype(np.float32).values
    y = df["y_up"].astype(np.float32).values
    ret = df["ETH_logret_next"].astype(np.float32).values

    valid_idx = np.where(~np.isnan(y))[0]
    valid_idx = valid_idx[valid_idx >= window - 1]
    if len(valid_idx) == 0:
        raise ValueError("No valid windows found for the given window size.")

    ratios = splits.copy()
    total_ratio = sum(ratios.values())
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-5):
        ratios = {k: v / total_ratio for k, v in ratios.items()}

    rng = np.random.default_rng(seed)
    shuffled = valid_idx.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * ratios.get("train", 0.7))
    n_val = int(n * ratios.get("val", 0.15))
    n_test = n - n_train - n_val
    train_indices = np.sort(shuffled[:n_train])
    val_indices = np.sort(shuffled[n_train:n_train + n_val])
    test_indices = np.sort(shuffled[n_train + n_val:])

    def gather_windows(indices: np.ndarray) -> np.ndarray:
        if len(indices) == 0:
            return np.empty((0, X_raw.shape[1]), dtype=np.float32)
        windows = [X_raw[i - window + 1:i + 1] for i in indices]
        return np.vstack(windows)

    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    train_windows = gather_windows(train_indices)
    if len(train_windows) == 0:
        raise ValueError("Training set is empty after sampling; adjust split ratios or window size.")
    train_windows_scaled = scaler.fit_transform(train_windows)

    X_scaled = scaler.transform(X_raw)
    med = np.nanmedian(train_windows_scaled, axis=0)
    X_scaled = _impute_with_median(X_scaled, med)

    return SampledData(
        X=X_scaled,
        y=np.nan_to_num(y, nan=-1.0),
        ret=ret,
        index=df.index.to_numpy(),
        feature_cols=feature_cols,
        scaler=scaler,
        impute_median=np.nan_to_num(med, nan=0.0),
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )
