"""
Resample raw BTCUSDT 1-minute OHLCV data into higher timeframes.

Input:
    data/raw/BTCUSDT_1m.csv

Outputs:
    data/resampled/BTCUSDT_5m.csv
    data/resampled/BTCUSDT_15m.csv
    data/resampled/BTCUSDT_1h.csv
    data/resampled/BTCUSDT_4h.csv
    data/resampled/BTCUSDT_1d.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RAW_PATH = BASE_DIR / "data" / "raw" / "BTCUSDT_1m.csv"
OUTPUT_DIR = BASE_DIR / "data" / "resampled"

REQUIRED_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
TIMEFRAMES = {
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
}


def load_raw_data() -> pd.DataFrame:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw 1m file was not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Raw file is missing required columns: {missing_columns}")

    df = df[REQUIRED_COLUMNS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp")

    for column in ["open", "high", "low", "close", "volume", "turnover"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df.reset_index(drop=True)


def resample_timeframe(df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    indexed = df.set_index("timestamp")
    resampled = indexed.resample(frequency, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "turnover": "sum",
        }
    )

    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    resampled = resampled.reset_index()
    resampled = resampled.drop_duplicates(subset="timestamp").sort_values("timestamp")
    return resampled[REQUIRED_COLUMNS].reset_index(drop=True)


def main() -> None:
    raw_df = load_raw_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] raw rows={len(raw_df):,} path={RAW_PATH}")

    for label, frequency in TIMEFRAMES.items():
        output_path = OUTPUT_DIR / f"BTCUSDT_{label}.csv"
        output_df = resample_timeframe(raw_df, frequency)
        output_df.to_csv(output_path, index=False)
        print(f"[save] {label} rows={len(output_df):,} path={output_path}")


if __name__ == "__main__":
    main()
