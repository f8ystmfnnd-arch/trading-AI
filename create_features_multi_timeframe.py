"""
Create multi-timeframe features from resampled BTCUSDT OHLCV data.

Inputs:
    data/resampled/BTCUSDT_15m.csv
    data/resampled/BTCUSDT_1h.csv
    data/resampled/BTCUSDT_4h.csv

Output:
    data/processed/BTCUSDT_15m_features.csv

Run:
    python create_features_multi_timeframe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESAMPLED_DIR = BASE_DIR / "data" / "resampled"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_features.csv"

MAIN_15M_PATH = RESAMPLED_DIR / "BTCUSDT_15m.csv"
H1_PATH = RESAMPLED_DIR / "BTCUSDT_1h.csv"
H4_PATH = RESAMPLED_DIR / "BTCUSDT_4h.csv"

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]

MAIN_FEATURE_COLUMNS = [
    "return_1",
    "return_3",
    "return_5",
    "return_10",
    "ma_20",
    "ma_60",
    "ma_120",
    "ma_distance_20",
    "ma_distance_60",
    "volatility_20",
    "volatility_60",
    "rsi_14",
    "volume_change_1",
    "volume_ma_20",
    "high_low_range",
    "close_open_return",
]

H1_FEATURE_COLUMNS = [
    "h1_return_1",
    "h1_ma_20",
    "h1_ma_60",
    "h1_ma_distance_20",
    "h1_trend_direction",
]

H4_FEATURE_COLUMNS = [
    "h4_return_1",
    "h4_ma_20",
    "h4_ma_60",
    "h4_ma_distance_20",
    "h4_trend_direction",
]

TARGET_COLUMNS = ["target_return_next", "target_up_next"]
OUTPUT_COLUMNS = ["timestamp", *MAIN_FEATURE_COLUMNS, *H1_FEATURE_COLUMNS, *H4_FEATURE_COLUMNS, *TARGET_COLUMNS]


def require_file(path: Path, label: str) -> None:
    if path.exists():
        return

    raise FileNotFoundError(
        f"Missing {label} data file: {path}\n"
        "Please run the data pipeline first:\n"
        "  python collect_bybit_1m.py\n"
        "  python resample_ohlcv.py"
    )


def load_ohlcv(path: Path, label: str) -> pd.DataFrame:
    require_file(path, label)
    df = pd.read_csv(path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{label} file is missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")

    for column in ["open", "high", "low", "close", "volume", "turnover"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.reset_index(drop=True)
    print(f"[load] {label}: rows={len(df):,} path={path}")
    return df


def add_rsi(df: pd.DataFrame, column_name: str = "rsi_14", window: int = 14) -> pd.DataFrame:
    close_diff = df["close"].diff()
    gain = close_diff.clip(lower=0)
    loss = (-close_diff).clip(lower=0)

    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50)

    df[column_name] = rsi
    return df


def add_main_15m_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["return_1"] = df["close"].pct_change(1)
    df["return_3"] = df["close"].pct_change(3)
    df["return_5"] = df["close"].pct_change(5)
    df["return_10"] = df["close"].pct_change(10)

    df["ma_20"] = df["close"].rolling(20).mean()
    df["ma_60"] = df["close"].rolling(60).mean()
    df["ma_120"] = df["close"].rolling(120).mean()
    df["ma_distance_20"] = (df["close"] / df["ma_20"]) - 1
    df["ma_distance_60"] = (df["close"] / df["ma_60"]) - 1

    df["volatility_20"] = df["return_1"].rolling(20).std()
    df["volatility_60"] = df["return_1"].rolling(60).std()
    df = add_rsi(df)

    df["volume_change_1"] = df["volume"].pct_change(1)
    df["volume_ma_20"] = df["volume"].rolling(20).mean()
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"]
    df["close_open_return"] = (df["close"] / df["open"]) - 1

    df["target_return_next"] = df["close"].shift(-1) / df["close"] - 1
    df["target_up_next"] = (df["close"].shift(-1) > df["close"]).astype("Int64")

    print(f"[features] 15m main features created: {len(MAIN_FEATURE_COLUMNS)} columns")
    return df


def create_aux_features(df: pd.DataFrame, prefix: str, timeframe: pd.Timedelta) -> pd.DataFrame:
    df = df.copy()

    df[f"{prefix}_return_1"] = df["close"].pct_change(1)
    df[f"{prefix}_ma_20"] = df["close"].rolling(20).mean()
    df[f"{prefix}_ma_60"] = df["close"].rolling(60).mean()
    df[f"{prefix}_ma_distance_20"] = (df["close"] / df[f"{prefix}_ma_20"]) - 1
    df[f"{prefix}_trend_direction"] = (df[f"{prefix}_ma_20"] > df[f"{prefix}_ma_60"]).astype("Int64")

    # Resampled candle timestamps are candle-open times. Move them to candle-close
    # times so only completed 1h/4h candles can be merged into 15m rows.
    df["timestamp"] = df["timestamp"] + timeframe

    feature_columns = [column for column in df.columns if column.startswith(f"{prefix}_")]
    output = df[["timestamp", *feature_columns]].dropna().sort_values("timestamp")
    print(f"[features] {prefix} auxiliary features created: {len(feature_columns)} columns")
    return output.reset_index(drop=True)


def merge_aux_features(main_df: pd.DataFrame, aux_df: pd.DataFrame, label: str) -> pd.DataFrame:
    merged = pd.merge_asof(
        main_df.sort_values("timestamp"),
        aux_df.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )
    print(f"[merge] attached {label} features with backward merge_asof")
    return merged


def clean_output(df: pd.DataFrame) -> pd.DataFrame:
    before_rows = len(df)
    output_df = df[OUTPUT_COLUMNS].replace([np.inf, -np.inf], np.nan)
    output_df = output_df.dropna().sort_values("timestamp").drop_duplicates(subset="timestamp")
    output_df = output_df.reset_index(drop=True)

    dropped_rows = before_rows - len(output_df)
    print(f"[clean] dropped rows={dropped_rows:,}; final rows={len(output_df):,}")
    return output_df


def save_output(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[save] rows={len(df):,} path={OUTPUT_PATH}")


def print_feature_columns() -> None:
    feature_columns = [*MAIN_FEATURE_COLUMNS, *H1_FEATURE_COLUMNS, *H4_FEATURE_COLUMNS]
    print("[columns] feature columns:")
    for column in feature_columns:
        print(f"  - {column}")
    print("[columns] target columns:")
    for column in TARGET_COLUMNS:
        print(f"  - {column}")


def main() -> None:
    main_15m = load_ohlcv(MAIN_15M_PATH, "15m main")
    h1 = load_ohlcv(H1_PATH, "1h auxiliary")
    h4 = load_ohlcv(H4_PATH, "4h auxiliary")

    main_15m = add_main_15m_features(main_15m)
    h1_features = create_aux_features(h1, "h1", pd.Timedelta(hours=1))
    h4_features = create_aux_features(h4, "h4", pd.Timedelta(hours=4))

    merged = merge_aux_features(main_15m, h1_features, "1h")
    merged = merge_aux_features(merged, h4_features, "4h")

    output_df = clean_output(merged)
    print_feature_columns()
    save_output(output_df)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"[error] {error}")
        sys.exit(1)