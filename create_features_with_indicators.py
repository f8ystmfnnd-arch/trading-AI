"""Add technical indicator features to the multi-timeframe feature dataset.

Inputs:
    data/processed/BTCUSDT_15m_features.csv
    data/resampled/BTCUSDT_15m.csv

Output:
    data/processed/BTCUSDT_15m_features_with_indicators.csv

Run:
    py -3 create_features_with_indicators.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FEATURE_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_features.csv"
OHLCV_PATH = BASE_DIR / "data" / "resampled" / "BTCUSDT_15m.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_features_with_indicators.csv"

REQUIRED_FEATURE_COLUMNS = ["timestamp"]
REQUIRED_OHLCV_COLUMNS = ["timestamp", "high", "low", "close"]

ADDED_FEATURE_COLUMNS = [
    "bb_middle_20",
    "bb_upper_20",
    "bb_lower_20",
    "bb_width_20",
    "bb_percent_b_20",
    "bb_breakout_up_20",
    "bb_breakout_down_20",
    "atr_14",
    "atr_ratio_14",
    "ma_20_slope",
    "ma_60_slope",
    "ma_120_slope",
    "rsi_14_slope",
]


def require_file(path: Path, label: str) -> None:
    if path.exists():
        return
    raise FileNotFoundError(f"Missing {label} file: {path}")


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing_columns = [column for column in columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{label} is missing required columns: {missing_columns}")


def load_csv(path: Path, label: str, required_columns: list[str]) -> pd.DataFrame:
    require_file(path, label)
    df = pd.read_csv(path)
    require_columns(df, required_columns, label)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="last")
    return df.reset_index(drop=True)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = safe_divide(avg_gain, avg_loss)
    return 100 - (100 / (1 + rs))


def calculate_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv[["timestamp", "high", "low", "close"]].copy()
    for column in ["high", "low", "close"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    close = df["close"]
    high = df["high"]
    low = df["low"]

    bb_middle_20 = close.rolling(window=20, min_periods=20).mean()
    bb_std_20 = close.rolling(window=20, min_periods=20).std(ddof=0)
    bb_upper_20 = bb_middle_20 + (2 * bb_std_20)
    bb_lower_20 = bb_middle_20 - (2 * bb_std_20)
    bb_range_20 = bb_upper_20 - bb_lower_20

    df["bb_middle_20"] = bb_middle_20
    df["bb_upper_20"] = bb_upper_20
    df["bb_lower_20"] = bb_lower_20
    df["bb_width_20"] = safe_divide(bb_range_20, bb_middle_20)
    df["bb_percent_b_20"] = safe_divide(close - bb_lower_20, bb_range_20)
    df["bb_breakout_up_20"] = (close > bb_upper_20).astype(int)
    df["bb_breakout_down_20"] = (close < bb_lower_20).astype(int)

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(window=14, min_periods=14).mean()
    df["atr_ratio_14"] = safe_divide(df["atr_14"], close)

    ma_20 = close.rolling(window=20, min_periods=20).mean()
    ma_60 = close.rolling(window=60, min_periods=60).mean()
    ma_120 = close.rolling(window=120, min_periods=120).mean()
    df["ma_20_slope"] = safe_divide(ma_20 - ma_20.shift(5), ma_20.shift(5))
    df["ma_60_slope"] = safe_divide(ma_60 - ma_60.shift(5), ma_60.shift(5))
    df["ma_120_slope"] = safe_divide(ma_120 - ma_120.shift(5), ma_120.shift(5))

    rsi_14 = calculate_rsi(close, window=14)
    df["rsi_14_slope"] = safe_divide(rsi_14 - rsi_14.shift(5), rsi_14.shift(5))

    return df[["timestamp", *ADDED_FEATURE_COLUMNS]]


def main() -> None:
    features = load_csv(FEATURE_PATH, "feature dataset", REQUIRED_FEATURE_COLUMNS)
    ohlcv = load_csv(OHLCV_PATH, "15m OHLCV dataset", REQUIRED_OHLCV_COLUMNS)

    print(f"input rows: {len(features):,}")

    indicators = calculate_indicators(ohlcv)
    merged = features.merge(indicators, on="timestamp", how="inner")

    print(f"rows after merge: {len(merged):,}")
    print("added features:")
    for column in ADDED_FEATURE_COLUMNS:
        print(f"  - {column}")

    print("missing values by added feature:")
    missing_counts = merged[ADDED_FEATURE_COLUMNS].isna().sum()
    for column, missing_count in missing_counts.items():
        print(f"  - {column}: {missing_count:,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUTPUT_PATH, index=False)

    print(f"final saved rows: {len(merged):,}")
    print(f"saved path: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
