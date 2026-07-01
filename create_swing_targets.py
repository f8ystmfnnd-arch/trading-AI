"""
Create swing-trading risk targets from BTCUSDT 15-minute multi-timeframe features.

Input:
    data/processed/BTCUSDT_15m_features.csv

Output:
    data/processed/BTCUSDT_15m_swing_targets.csv

Run:
    py -3 create_swing_targets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_features.csv"
RESAMPLED_15M_PATH = BASE_DIR / "data" / "resampled" / "BTCUSDT_15m.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_swing_targets.csv"

BIG_MOVE_THRESHOLD_16 = 0.02
DROP_THRESHOLD_16 = 0.02
PUMP_THRESHOLD_16 = 0.02

BIG_MOVE_THRESHOLD_96 = 0.04
DROP_THRESHOLD_96 = 0.04
PUMP_THRESHOLD_96 = 0.04

HIGH_VOL_QUANTILE = 0.70
HORIZONS = [16, 96]

NEW_TARGET_COLUMNS = [
    "target_return_next_16",
    "target_abs_return_next_16",
    "target_volatility_next_16",
    "target_big_move_next_16",
    "target_drop_next_16",
    "target_pump_next_16",
    "target_volatility_high_next_16",
    "target_return_next_96",
    "target_abs_return_next_96",
    "target_volatility_next_96",
    "target_big_move_next_96",
    "target_drop_next_96",
    "target_pump_next_96",
    "target_volatility_high_next_96",
]

BINARY_TARGET_COLUMNS = [
    "target_big_move_next_16",
    "target_drop_next_16",
    "target_pump_next_16",
    "target_volatility_high_next_16",
    "target_big_move_next_96",
    "target_drop_next_96",
    "target_pump_next_96",
    "target_volatility_high_next_96",
]

PIPELINE_STEPS = [
    "py -3 collect_bybit_1m.py --update",
    "py -3 resample_ohlcv.py",
    "py -3 create_features_multi_timeframe.py",
    "py -3 create_swing_targets.py",
]


def pipeline_hint() -> str:
    return "Please run the data pipeline in this order:\n  " + "\n  ".join(PIPELINE_STEPS)


def calculate_future_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.shift(-horizon) / close - 1


def calculate_future_volatility(close: pd.Series, horizon: int) -> pd.Series:
    returns = close.pct_change()
    return returns.shift(-1).rolling(horizon).std().shift(-(horizon - 1))


def merge_close_from_resampled(df: pd.DataFrame) -> tuple[pd.DataFrame, bool, int]:
    if not RESAMPLED_15M_PATH.exists():
        raise FileNotFoundError(
            f"close column is required, but resampled 15m file was not found: {RESAMPLED_15M_PATH}\n"
            f"{pipeline_hint()}"
        )

    close_df = pd.read_csv(RESAMPLED_15M_PATH, usecols=["timestamp", "close"])
    close_df["timestamp"] = pd.to_datetime(close_df["timestamp"], utc=True, errors="coerce")
    close_df["close"] = pd.to_numeric(close_df["close"], errors="coerce")
    close_df = close_df.dropna(subset=["timestamp", "close"])
    close_df = close_df.sort_values("timestamp").drop_duplicates(subset="timestamp")

    merged = df.merge(close_df, on="timestamp", how="left", validate="one_to_one")
    missing_close = int(merged["close"].isna().sum())
    print(f"[merge] close merged from {RESAMPLED_15M_PATH}")
    print(f"[merge] rows without close after exact timestamp merge: {missing_close:,}")
    return merged, True, missing_close


def load_features() -> tuple[pd.DataFrame, bool, int, int]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input feature file was not found: {INPUT_PATH}\n{pipeline_hint()}")

    print(f"[input] {INPUT_PATH}")
    print(f"[output] {OUTPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    initial_rows = len(df)
    print(f"[load] initial rows: {initial_rows:,}")

    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required")

    # Existing target_return_next and target_up_next are preserved as labels.
    # close is used only to create targets and must be excluded from later model features.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    before_dedup = len(df)
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)
    removed_duplicates = before_dedup - len(df)

    close_merged = False
    missing_close = 0
    if "close" not in df.columns:
        df, close_merged, missing_close = merge_close_from_resampled(df)

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    print(f"[clean] duplicate timestamps removed: {removed_duplicates:,}")
    print(f"[load] close merge used: {close_merged}")
    return df, close_merged, removed_duplicates, missing_close


def thresholds_for_horizon(horizon: int) -> tuple[float, float, float]:
    if horizon == 16:
        return BIG_MOVE_THRESHOLD_16, DROP_THRESHOLD_16, PUMP_THRESHOLD_16
    if horizon == 96:
        return BIG_MOVE_THRESHOLD_96, DROP_THRESHOLD_96, PUMP_THRESHOLD_96
    raise ValueError(f"Unsupported horizon: {horizon}")


def add_horizon_targets(df: pd.DataFrame, horizon: int) -> tuple[pd.DataFrame, float]:
    output = df.copy()
    big_move_threshold, drop_threshold, pump_threshold = thresholds_for_horizon(horizon)

    return_col = f"target_return_next_{horizon}"
    abs_col = f"target_abs_return_next_{horizon}"
    vol_col = f"target_volatility_next_{horizon}"
    big_col = f"target_big_move_next_{horizon}"
    drop_col = f"target_drop_next_{horizon}"
    pump_col = f"target_pump_next_{horizon}"
    high_vol_col = f"target_volatility_high_next_{horizon}"

    output[return_col] = calculate_future_return(output["close"], horizon)
    output[abs_col] = output[return_col].abs()
    output[vol_col] = calculate_future_volatility(output["close"], horizon)

    volatility_threshold = float(output[vol_col].quantile(HIGH_VOL_QUANTILE))
    output[big_col] = (output[abs_col] >= big_move_threshold).astype(int)
    output[drop_col] = (output[return_col] <= -drop_threshold).astype(int)
    output[pump_col] = (output[return_col] >= pump_threshold).astype(int)
    output[high_vol_col] = (output[vol_col] >= volatility_threshold).astype(int)

    print(f"[target] next_{horizon} volatility high threshold q={HIGH_VOL_QUANTILE:.2f}: {volatility_threshold:.8f}")
    return output, volatility_threshold


def add_swing_targets(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for horizon in HORIZONS:
        output, _ = add_horizon_targets(output, horizon)
    return output


def clean_output(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before_rows = len(df)
    output = df.replace([np.inf, -np.inf], np.nan)
    output = output.dropna().sort_values("timestamp").drop_duplicates(subset="timestamp")
    output = output.reset_index(drop=True)
    removed_rows = before_rows - len(output)
    print(f"[clean] rows removed after target creation: {removed_rows:,}")
    print(f"[clean] final rows: {len(output):,}")
    return output, removed_rows


def print_target_columns() -> None:
    print("[columns] added swing target columns:")
    for column in NEW_TARGET_COLUMNS:
        print(f"  - {column}")


def print_class_ratios(df: pd.DataFrame) -> None:
    print("[classes] binary target class ratios:")
    for column in BINARY_TARGET_COLUMNS:
        ratios = df[column].value_counts(normalize=True).sort_index()
        print(f"{column}:")
        for klass, ratio in ratios.items():
            print(f"  {int(klass)}    {ratio:.4f}")


def save_output(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[save] completed: {OUTPUT_PATH}")
    print(f"[save] rows: {len(df):,}")


def main() -> None:
    df, _, _, _ = load_features()
    df = add_swing_targets(df)
    output_df, _ = clean_output(df)
    print_target_columns()
    print_class_ratios(output_df)
    save_output(output_df)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"[error] {error}")
        sys.exit(1)
