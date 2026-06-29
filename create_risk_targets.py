"""
Create risk-management targets from BTCUSDT 15-minute multi-timeframe features.

Input:
    data/processed/BTCUSDT_15m_features.csv

Output:
    data/processed/BTCUSDT_15m_risk_targets.csv

Run:
    python create_risk_targets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_features.csv"
RESAMPLED_15M_PATH = BASE_DIR / "data" / "resampled" / "BTCUSDT_15m.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "BTCUSDT_15m_risk_targets.csv"

BIG_MOVE_THRESHOLD = 0.01
DROP_THRESHOLD = 0.01
PUMP_THRESHOLD = 0.01
HIGH_VOL_QUANTILE = 0.70
FORWARD_BARS = 4

NEW_TARGET_COLUMNS = [
    "target_return_next_4",
    "target_abs_return_next_4",
    "target_volatility_next_4",
    "target_big_move_next_4",
    "target_drop_next_4",
    "target_pump_next_4",
    "target_volatility_high_next_4",
]

BINARY_TARGET_COLUMNS = [
    "target_big_move_next_4",
    "target_drop_next_4",
    "target_pump_next_4",
    "target_volatility_high_next_4",
]

PIPELINE_STEPS = [
    "python collect_bybit_1m.py",
    "python resample_ohlcv.py",
    "python create_features_multi_timeframe.py",
    "python create_risk_targets.py",
]


def pipeline_hint() -> str:
    return "Please run the data pipeline in this order:\n  " + "\n  ".join(PIPELINE_STEPS)


def merge_close_from_resampled(df: pd.DataFrame) -> pd.DataFrame:
    if not RESAMPLED_15M_PATH.exists():
        raise FileNotFoundError(
            "close column is required for risk target creation, but resampled close "
            f"file was not found: {RESAMPLED_15M_PATH}\n{pipeline_hint()}"
        )

    close_df = pd.read_csv(RESAMPLED_15M_PATH, usecols=["timestamp", "close"])
    close_df["timestamp"] = pd.to_datetime(close_df["timestamp"], utc=True, errors="coerce")
    close_df["close"] = pd.to_numeric(close_df["close"], errors="coerce")
    close_df = close_df.dropna(subset=["timestamp", "close"])
    close_df = close_df.sort_values("timestamp").drop_duplicates(subset="timestamp")

    merged = df.merge(close_df, on="timestamp", how="left", validate="one_to_one")
    missing_close = int(merged["close"].isna().sum())
    if missing_close:
        raise ValueError(
            "close column is required for risk target creation, but some feature "
            f"timestamps could not be matched in {RESAMPLED_15M_PATH}. "
            f"missing_close={missing_close:,}"
        )

    print(f"[merge] close merged from {RESAMPLED_15M_PATH}; rows={len(merged):,}")
    return merged


def load_features() -> pd.DataFrame:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Input feature file was not found: {INPUT_PATH}\n{pipeline_hint()}")

    print(f"[load] reading features: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)

    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required for risk target creation")

    # Existing target columns are kept for analysis, but must not be used as model features.
    # target_return_next and target_up_next are labels, not feature columns.
    missing_existing_targets = [
        column for column in ["target_return_next", "target_up_next"] if column not in df.columns
    ]
    if missing_existing_targets:
        raise ValueError(f"Input file is missing existing target columns: {missing_existing_targets}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates(subset="timestamp").reset_index(drop=True)

    if "close" not in df.columns:
        print("[load] close column not found in features; merging close from resampled 15m data")
        df = merge_close_from_resampled(df)

    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    print(f"[load] rows after timestamp sort/dedup={len(df):,}")
    return df


def add_risk_targets(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    output = df.copy()
    close = output["close"]

    output["target_return_next_4"] = close.shift(-FORWARD_BARS) / close - 1
    output["target_abs_return_next_4"] = output["target_return_next_4"].abs()

    one_bar_return = close.pct_change()
    output["target_volatility_next_4"] = (
        one_bar_return.shift(-1).rolling(FORWARD_BARS).std().shift(-(FORWARD_BARS - 1))
    )

    output["target_big_move_next_4"] = (
        output["target_abs_return_next_4"] >= BIG_MOVE_THRESHOLD
    ).astype(int)
    output["target_drop_next_4"] = (output["target_return_next_4"] <= -DROP_THRESHOLD).astype(int)
    output["target_pump_next_4"] = (output["target_return_next_4"] >= PUMP_THRESHOLD).astype(int)

    volatility_threshold = float(output["target_volatility_next_4"].quantile(HIGH_VOL_QUANTILE))
    output["target_volatility_high_next_4"] = (
        output["target_volatility_next_4"] >= volatility_threshold
    ).astype(int)

    print(f"[target] volatility threshold q={HIGH_VOL_QUANTILE:.2f}: {volatility_threshold:.8f}")
    return output, volatility_threshold


def clean_output(df: pd.DataFrame) -> pd.DataFrame:
    before_rows = len(df)
    output = df.replace([np.inf, -np.inf], np.nan)
    output = output.dropna().sort_values("timestamp").drop_duplicates(subset="timestamp")
    output = output.reset_index(drop=True)
    dropped_rows = before_rows - len(output)
    print(f"[clean] dropped rows={dropped_rows:,}; final rows={len(output):,}")
    return output


def print_target_columns() -> None:
    print("[columns] added risk target columns:")
    for column in NEW_TARGET_COLUMNS:
        print(f"  - {column}")


def print_binary_target_ratios(df: pd.DataFrame) -> None:
    print("[classes] binary target ratios:")
    for column in BINARY_TARGET_COLUMNS:
        counts = df[column].value_counts().sort_index()
        ratios = df[column].value_counts(normalize=True).sort_index()
        zero_count = int(counts.get(0, 0))
        one_count = int(counts.get(1, 0))
        zero_ratio = float(ratios.get(0, 0.0))
        one_ratio = float(ratios.get(1, 0.0))
        print(
            f"  {column}: 0={zero_count:,} ({zero_ratio:.2%}), "
            f"1={one_count:,} ({one_ratio:.2%})"
        )


def save_output(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[save] rows={len(df):,} path={OUTPUT_PATH}")


def main() -> None:
    df = load_features()
    df, _ = add_risk_targets(df)
    output_df = clean_output(df)
    print_target_columns()
    print_binary_target_ratios(output_df)
    save_output(output_df)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        print(f"[error] {error}")
        sys.exit(1)
